"""Execute the frozen, small comparison. Never writes outside this experiment."""
import argparse
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from src import config
from src.model import FittedModel
from src.alarm_budget_experiment import topology as training_topology, candidate_statistics, choose_candidate
from src.reliability_evaluation import eligible_events
from .study import HERE, ROOT, FEATURE_SETS, sha, build_data, load_numeric, fixed_forecasts

OUT=HERE/'outputs'

def array_sha(values):
    import hashlib
    return hashlib.sha256(np.asarray(values,dtype='<f8').tobytes()).hexdigest()

def freeze():
    OUT.mkdir(exist_ok=True)
    scientific=[HERE/'study.py',HERE/'run.py',HERE/'config.json']
    scientific += [ROOT/'src'/n for n in ['config.py','model.py','preprocessing.py','features.py','labeling.py','evaluation.py','reliability_evaluation.py','alarm_budget_experiment.py']]
    inputs={str(p.relative_to(ROOT)):sha(p) for p in scientific+sorted((ROOT/'data').glob('*.csv'))}
    marker=OUT/'protocol.json'
    if marker.exists():
        protocol=json.loads(marker.read_text())
        if protocol['input_hashes']!=inputs:raise ValueError('Frozen code/config/raw inputs changed: use a new experiment, not an altered resume')
        return protocol
    protocol=json.loads((HERE/'config.json').read_text())
    protocol.update(input_hashes=inputs,feature_sets=FEATURE_SETS,classifier_parameters=config.MODEL_CONFIGS,
                    candidate_quantiles=json.loads((ROOT/'results/alarm_budget_experiment/protocol.json').read_text())['candidate_quantiles'],
                    historical_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    package_versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','joblib','threadpoolctl','matplotlib']})
    marker.write_text(json.dumps(protocol,indent=2));return protocol

def prepared():
    receipt=OUT/'prepared_receipt.json'
    if receipt.exists():
        for name,want in json.loads(receipt.read_text()).items():
            if sha(OUT/name)!=want:raise ValueError('Prepared artifact changed: '+name)
        return pd.read_csv(OUT/'matched_dataset.csv.gz',float_precision='round_trip'),load_numeric(),pd.read_csv(OUT/'raw_event_audit.csv')
    data,numeric,events=build_data(OUT)
    names=['matched_dataset.csv.gz','all_pulses.csv.gz','data_inventory.csv','raw_event_audit.csv','data_checks.json']
    receipt.write_text(json.dumps({n:sha(OUT/n) for n in names},indent=2));return data,numeric,events

def fit_classifier(frame,features,name):
    model=FittedModel(name);model.features=list(features)
    return model.fit(frame[features].to_numpy(float),frame.Label.to_numpy(int))

def training_selection(train,oof,numeric,events,protocol):
    allowed=set(map(int,train.Record.unique()));ev=events[events.Record.isin(allowed)];num={r:numeric[r] for r in allowed}
    g=train.assign(Score=oof)
    normal,previous,duration=training_topology(g,ev,num,allowed)
    eligible=eligible_events(ev,normal,num,30.)
    grid=candidate_statistics(normal,previous,duration,ev,eligible,protocol['candidate_quantiles'],30.)
    selected=choose_candidate(grid,protocol['research_budget_starts_per_hour'],protocol['research_active_fraction_cap'])
    return selected,grid

def complete_checkpoint(path):
    marker=path/'complete.json'
    if not marker.exists():return False
    for name,want in json.loads(marker.read_text())['files'].items():
        if sha(path/name)!=want:raise ValueError('Checkpoint changed: '+str(path/name))
    return True

def train_classification(data,numeric,events,protocol):
    parent=OUT/'classification_folds';parent.mkdir(exist_ok=True)
    for modality,features in FEATURE_SETS.items():
        for name in protocol['classifiers']:
            for record in sorted(data.Record.unique()):
                ck=parent/f'{modality}__{name}__{int(record):02d}'
                if complete_checkpoint(ck):continue
                if ck.exists():raise ValueError('Incomplete checkpoint must be preserved and diagnosed: '+str(ck))
                ck.mkdir();begun=time.perf_counter()
                train=data[data.Record.ne(record)].reset_index(drop=True);test=data[data.Record.eq(record)].reset_index(drop=True)
                if set(train.Record)&set(test.Record):raise AssertionError('Outer overlap')
                oof=np.full(len(train),np.nan);inner=[]
                for tr,va in GroupKFold(n_splits=3).split(train,train.Label,train.Record):
                    fit=train.iloc[tr];val=train.iloc[va]
                    assert not set(fit.Record)&set(val.Record)
                    model=fit_classifier(fit,features,name);score=model.score(val[features].to_numpy(float));oof[va]=score
                    inner.append(dict(train_records=sorted(map(int,fit.Record.unique())),validation_records=sorted(map(int,val.Record.unique())),fit_rows_sha256=array_sha(fit.sample_id),validation_rows_sha256=array_sha(val.sample_id),validation_scores_sha256=array_sha(score)))
                if not np.isfinite(oof).all():raise AssertionError('Incomplete OOF scores')
                winner,grid=training_selection(train,oof,numeric,events,protocol)
                # Threshold is frozen before this fold's held-out features are scored.
                threshold=float.fromhex(winner['Threshold_hex'])
                model=fit_classifier(train,features,name);model.threshold=threshold
                scores=model.score(test[features].to_numpy(float))
                pred=test[['sample_id','Record','time_s','Label','FutureMin','SpO2_Current']].copy()
                pred['Modality']=modality;pred['Model']=name;pred['Score']=scores;pred['Threshold']=threshold;pred['Prediction']=(scores>=threshold).astype(int);pred['Policy']='training_OOF_budget5_cap005'
                pred.to_csv(ck/'predictions.csv.gz',index=False,compression='gzip')
                oo=train[['sample_id','Record','time_s','Label']].copy();oo['Score']=oof;oo.to_csv(ck/'oof.csv.gz',index=False,compression='gzip')
                grid.to_csv(ck/'training_candidates.csv',index=False)
                joblib.dump(model,ck/'outer_model.joblib')
                meta=dict(Modality=modality,Model=name,Outer_Test_Record=int(record),Features=features,Train_Records=sorted(map(int,train.Record.unique())),Inner=inner,OOF_Scores_SHA256=array_sha(oof),Selected=winner,Fit_Seconds=time.perf_counter()-begun,Threshold_Selected_Before_Heldout_Scoring=True)
                (ck/'metadata.json').write_text(json.dumps(meta,indent=2))
                (ck/'complete.json').write_text(json.dumps({'files':{p.name:sha(p) for p in ck.iterdir() if p.is_file()}},indent=2))
                print('FIT',modality,name,int(record),'seconds',round(time.perf_counter()-begun,2),flush=True)
    predictions=[];metadata=[];grid=[]
    for ck in sorted(parent.iterdir()):
        if not complete_checkpoint(ck):raise ValueError('Unsealed fold')
        predictions.append(pd.read_csv(ck/'predictions.csv.gz',float_precision='round_trip'));meta=json.loads((ck/'metadata.json').read_text());metadata.append(meta)
        grid.append(pd.read_csv(ck/'training_candidates.csv').assign(Modality=meta['Modality'],Model=meta['Model'],Outer_Test_Record=meta['Outer_Test_Record']))
    assert len(metadata)==len(FEATURE_SETS)*len(protocol['classifiers'])*data.Record.nunique()
    pred=pd.concat(predictions,ignore_index=True)
    for name,forecast in fixed_forecasts(data).items():
        f=data[['sample_id','Record','time_s','Label','FutureMin','SpO2_Current']].copy();f['Modality']='SpO2_history';f['Model']=name;f['Score']=94-forecast;f['Threshold']=0.;f['Prediction']=(forecast<=94).astype(int);f['Policy']='fixed_forecast_le94_not_budget_selected';pred=pd.concat([pred,f],ignore_index=True)
    pred.to_csv(OUT/'classification_predictions.csv.gz',index=False,compression='gzip')
    pd.concat(grid,ignore_index=True).to_csv(OUT/'training_candidates.csv.gz',index=False,compression='gzip')
    (OUT/'fold_ledger.json').write_text(json.dumps(metadata,indent=2))
    pd.DataFrame([dict(Modality=m['Modality'],Model=m['Model'],Outer_Test_Record=m['Outer_Test_Record'],**m['Selected']) for m in metadata]).to_csv(OUT/'selected_thresholds.csv',index=False)
    (OUT/'classification_fit_receipt.json').write_text(json.dumps({n:sha(OUT/n) for n in ['classification_predictions.csv.gz','fold_ledger.json','selected_thresholds.csv','training_candidates.csv.gz']},indent=2))
    return pred

def forecast(data,protocol):
    receipt=OUT/'forecast_fit_receipt.json'
    if receipt.exists():
        for name,want in json.loads(receipt.read_text()).items():
            if sha(OUT/name)!=want:raise ValueError('Forecast output changed')
        return pd.read_csv(OUT/'forecast_predictions.csv.gz',float_precision='round_trip')
    parts=[];meta=[];parent=OUT/'regression_models';parent.mkdir(exist_ok=True)
    for modality,features in FEATURE_SETS.items():
        for record in sorted(data.Record.unique()):
            train=data[data.Record.ne(record)];held=data[data.Record.eq(record)]
            pipe=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=protocol['ridge_alpha']))
            pipe.fit(train[features],train.FutureMin);raw=pipe.predict(held[features]);model_name='Ridge_'+modality
            f=held[['sample_id','Record','time_s','FutureMin','SpO2_Current']].copy();f['Model']=model_name;f['Raw_Forecast']=raw;f['Forecast']=np.clip(raw,0,100);parts.append(f)
            filename=f'{modality}__{int(record):02d}.joblib';joblib.dump(pipe,parent/filename)
            meta.append(dict(Model=model_name,Outer_Test_Record=int(record),Train_Records=sorted(map(int,train.Record.unique())),Features=features,Alpha=protocol['ridge_alpha'],Model_File=filename,Model_SHA256=sha(parent/filename),Train_Rows_SHA256=array_sha(train.sample_id)))
    for name,values in fixed_forecasts(data).items():
        f=data[['sample_id','Record','time_s','FutureMin','SpO2_Current']].copy();f['Model']=name;f['Raw_Forecast']=values;f['Forecast']=np.clip(values,0,100);parts.append(f)
    result=pd.concat(parts,ignore_index=True);result.to_csv(OUT/'forecast_predictions.csv.gz',index=False,compression='gzip');(OUT/'regression_ledger.json').write_text(json.dumps(meta,indent=2))
    receipt.write_text(json.dumps({n:sha(OUT/n) for n in ['forecast_predictions.csv.gz','regression_ledger.json']},indent=2));return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','fit','analyse','all'],default='all');args=parser.parse_args()
    with threadpool_limits(limits=1):
        protocol=freeze();data,numeric,events=prepared()
        if args.stage=='prepare':return
        if args.stage in ['fit','all']:
            if (OUT/'classification_fit_receipt.json').exists():
                for n,w in json.loads((OUT/'classification_fit_receipt.json').read_text()).items():
                    if sha(OUT/n)!=w:raise ValueError('Classification artifact changed: '+n)
            else:train_classification(data,numeric,events,protocol)
            forecast(data,protocol)
        if args.stage in ['analyse','all']:
            from .analysis import analyse
            analyse(OUT,protocol,data,numeric,events)

if __name__=='__main__':main()
