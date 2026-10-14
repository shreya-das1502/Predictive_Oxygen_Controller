"""One authoritative runner, fixed protocol, provenance and no expensive grid search."""
import argparse
import hashlib
from datetime import datetime, timezone, timedelta
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from . import config
from .dataset import validate_dataset
from .provenance import load_prepared
from .evaluation import outer_splits, grouped_oof, metrics, alarm_statistics
from .model import FittedModel
from .benchmark_latency import benchmark_all, environment


def run(output=None,rebuild=False,skip_benchmark=False,target='future_low'):
    definition=config.target_definition(target)
    output=Path(output) if output is not None else config.HARDENED_DIR / target
    historical=config.OUTPUT_DIR.resolve()
    if output.resolve()==historical or historical in output.resolve().parents:
        raise ValueError('Historical results/validated is protected; select a new output directory')
    output.mkdir(parents=True,exist_ok=True)
    begun=time.perf_counter()
    settings={k:getattr(config,k) for k in ['HORIZON','SPO2_THRESHOLD','MAX_NUMERIC_GAP','SEED','TARGET_RECALL','INNER_FOLDS','VERSION','FEATURES','MODEL_CONFIGS','TIMING','OUTPUT_FILES']}
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(config.DATA_DIR.glob('*.csv'))}
    code_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((config.ROOT/'src').glob('*.py'))}
    data,cache_receipt,reused=load_prepared(output,target,rebuild)
    print(f'Target={target}; prepared cache reused={reused}',flush=True)
    validate_dataset(data)
    X=data[config.FEATURES].to_numpy(float);y=data.Label.to_numpy(int);groups=data.Record.to_numpy(int)
    per=[];pred=[];thresholds=[];split_log=[];train_time={n:0.0 for n in config.MODEL_CONFIGS};validation=[];oof_parts=[]
    with threadpool_limits(limits=1):
        for fold,(tr,te) in enumerate(outer_splits(groups),1):
            test_record=int(groups[te][0]);fold_meta={'test_record':test_record,'outer_train_records':sorted(map(int,np.unique(groups[tr]))),'inner':None}
            for name in config.MODEL_CONFIGS:
                start=time.perf_counter()
                threshold,oof,table,inner=grouped_oof(X[tr],y[tr],groups[tr],name)
                validation.append({'Model':name,'Test_Record':test_record,**metrics(y[tr],oof,oof>=threshold),'Threshold':threshold})
                # Save validation provenance separately from held-out outcomes.
                op=data.iloc[tr][['sample_id','Record','time_s','Label']].copy();op['Model']=name;op['Outer_Test_Record']=test_record;op['Score']=oof;op['Threshold']=threshold;oof_parts.append(op)
                table['Model']=name;table['Test_Record']=test_record;table['Selected']=table.Threshold==threshold;thresholds.append(table)
                model=FittedModel(name).fit(X[tr],y[tr]);train_time[name]+=time.perf_counter()-start
                scores=model.score(X[te]);p=(scores>=threshold).astype(int)
                frame=data.iloc[te][['sample_id','Record','pulse_row','time_s','peak_time_s','spo2','Future_Min_SpO2_30s','Label']].copy()
                frame['Target']=target;frame['Model']=name;frame['Score']=scores;frame['Threshold']=threshold;frame['Prediction']=p;pred.append(frame)
                m={'Model':name,'Record':test_record,'Target':target,'Positive_Containing':bool(y[te].sum()),**metrics(y[te],scores,p),**alarm_statistics(frame),'Threshold':threshold}
                per.append(m)
                fold_meta['inner']=inner
                print(f'[{fold}/{len(np.unique(groups))}] {test_record:02d} {name}: recall={m["Recall"]:.3f} precision={m["Precision"]:.3f} FP={m["FP"]}',flush=True)
            split_log.append(fold_meta)
        predictions=pd.concat(pred,ignore_index=True);record_metrics=pd.DataFrame(per)
        # Models must evaluate exactly the same sample IDs once.
        expected=set(data.sample_id)
        for _,g in predictions.groupby('Model'):
            if set(g.sample_id)!=expected or g.sample_id.duplicated().any():raise AssertionError('Unmatched evaluation rows')
        predictions.to_csv(output/'per_record_predictions.csv',index=False)
        record_metrics.to_csv(output/'per_recording_metrics.csv',index=False)
        pd.DataFrame(validation).to_csv(output/'validation_metrics.csv',index=False)
        pd.concat(oof_parts,ignore_index=True).to_csv(output/'validation_oof_predictions.csv.gz',index=False,compression='gzip')
        # Every unique score threshold is saved once using vectorized counts (compressed due to size).
        pd.concat(thresholds,ignore_index=True).to_csv(output/'threshold_analysis.csv.gz',index=False,compression='gzip')
        pd.concat([t[t.Selected] for t in thresholds],ignore_index=True).to_csv(output/'threshold_analysis.csv',index=False)
        (output/'splits.json').write_text(json.dumps(split_log,indent=2))
        pooled=[]
        for name,g in predictions.groupby('Model',sort=False):
            r=record_metrics[record_metrics.Model==name]
            pooled.append({'Model':name,**metrics(g.Label,g.Score,g.Prediction),
                           'Macro_Recall':float(r.Recall.mean()),'Macro_Precision':float(r.Precision.mean()),'Macro_F1':float(r.F1.mean()),
                           'Macro_Recall_Defined_Recordings':int(r.Recall.notna().sum()),'Macro_Precision_Defined_Recordings':int(r.Precision.notna().sum()),
                           'Macro_PR_AUC_AP':float(r.PR_AUC_AP.mean()),'False_Alert_Episodes':int(r.False_Alert_Episodes.sum()),
                           'Exposure_Hours':float(r.Exposure_Hours.sum()),'False_Alerts_Per_Hour':r.False_Alert_Episodes.sum()/r.Exposure_Hours.sum(),
                           'FP_Timestamps_Per_Hour':r.FP.sum()/r.Exposure_Hours.sum(),
                           'Training_And_Inner_Validation_s':train_time[name],
                           'Held_Out_Recordings':len(r),'Positive_Containing_Recordings':int((r.Positives>0).sum()),
                           'Negative_Only_Recordings':int((r.Positives==0).sum()),'Target':target,'Metric_Scope':'pooled held-out timestamps; Macro_* over defined per-record values',
                           'Features':json.dumps(config.FEATURES),'Hyperparameters':json.dumps(config.MODEL_CONFIGS[name]),
                           'Threshold_Strategy':config.threshold_strategy(),
                           'Limitations':'retrospective recording-wise CV; target-specific eligibility; few transition events; pooled scores from different refits are uncalibrated'})
        pooled=pd.DataFrame(pooled);pooled.to_csv(output/'pooled_metrics.csv',index=False)
        # Predeclared demonstration default Logistic; do not choose it from outer outcomes.
        (output/'models').mkdir(exist_ok=True)
        for name in config.MODEL_CONFIGS:
            threshold,scores,table,inner=grouped_oof(X,y,groups,name)
            model=FittedModel(name).fit(X,y);model.threshold=threshold
            model.save(output/'models'/f'{name}.joblib',{'seed':config.SEED,'training_records':sorted(map(int,np.unique(groups))),
                       'target':definition,
                       'threshold_strategy':'full-data grouped OOF; demonstration refit, NOT independent test',
                       'validation_recall':float(metrics(y,scores,scores>=threshold)['Recall']),'inner_splits':inner,
                       'dataset_sha256':hashlib.sha256((output/'causal_dataset.csv').read_bytes()).hexdigest()})
    latency=None if skip_benchmark else benchmark_all(output)
    comparison=pooled.copy()
    if latency is not None:
        a=latency[latency.Component=='pulse_to_alert'][['Model','Median_ms','P95_ms','P99_ms']]
        comparison=comparison.merge(a,on='Model')
    comparison.to_csv(output/'model_comparison.csv',index=False)
    manifest={'configuration':settings,'protocol':f'leave-one-recording-out; inner {config.INNER_FOLDS}-fold GroupKFold; no tuning',
              'target_definition':definition,'threshold_selection':config.threshold_strategy(),
              'dataset_sha256':hashlib.sha256((output/'causal_dataset.csv').read_bytes()).hexdigest(),
              'dataset_cache_receipt':cache_receipt,'recording_splits':split_log,
              'figure_generation':{'automatic':not skip_benchmark,'directory':'figures','generator':'src/figures.py'},
              'default_demo_model':'Logistic (prespecified)','environment':environment(),'source_hashes':source_hashes,'code_hashes':code_hashes,
              'git_status':subprocess.run(['git','status','--short'],capture_output=True,text=True).stdout,
              'run_seconds':time.perf_counter()-begun,'benchmark_completed':latency is not None,
              'completed_at':datetime.now(timezone(timedelta(hours=5,minutes=30))).isoformat(),
              'input_dataset':str(config.INPUT_DATASET),'output_directory':str(output.resolve())}
    (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    from .target_reporting import generate_target_report
    generate_target_report(output)
    if not skip_benchmark:
        from .figures import generate_figures
        generate_figures(output)
    print(comparison[['Model','Recall','Precision','F1','PR_AUC_AP','FP','FN']].to_string(index=False),flush=True)
    return comparison


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--target',choices=config.TARGETS,default='future_low');p.add_argument('--rebuild',action='store_true');p.add_argument('--skip-benchmark',action='store_true');a=p.parse_args()
    try:run(a.output,a.rebuild,a.skip_benchmark,a.target)
    except (ValueError,FileNotFoundError) as e:p.exit(2,f'Experiment prerequisite/error: {e}\n')
if __name__=='__main__':main()
