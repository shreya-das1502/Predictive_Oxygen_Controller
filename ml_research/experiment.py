"""Fixed controlled LORO experiment. No search/selection among outer outcomes."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src import config
from src.labeling import future_labels,onset_labels
from src.evaluation import metrics
from src.reliability_evaluation import raw_events,current_normal,links,episodes,eligible_events,off_event_alarms
from src.alarm_budget_experiment import candidate_statistics,choose_candidate,evaluate_scoped_events
from .baseline import ROOT,sha,verify_snapshot
from .core import CONTEXT,add_context,training_scores,negative_rank,array_hash

KEYS=['Target','Horizon_s','FeatureSet','ScoreSpace','Model']
RECORDS=range(1,31)


def freeze(output):
    path=output/'protocol.json'
    if path.exists():raise ValueError('Protocol already frozen; never overwrite or rerun inside this directory')
    original=json.loads((ROOT/'results/alarm_budget_experiment/protocol.json').read_text())
    protocol=dict(version=1,targets=list(config.TARGETS),model_configs=config.MODEL_CONFIGS,
        settings=[dict(Horizon_s=h,FeatureSet='base10') for h in [5,10,15,30]]+[dict(Horizon_s=30,FeatureSet='context18')],
        score_spaces=['raw','negative_rank'],features=dict(base10=list(config.FEATURES),context18=list(config.FEATURES)+CONTEXT),
        budgets=[1.,5.,10.],active_cap=.05,quantiles=original['candidate_quantiles'],event_windows=[5,10,15,30],
        threshold_objective='existing audited one-to-one maximum training OOF detections at the TRAINED horizon; same budget/cap and deterministic ties',
        score_normalization='negative_rank uses empirical CDF of scores of NEGATIVE FIT rows, separately for each inner/outer refit; not probability calibration',
        population='Same historical complete-30s rows within each target, even for shorter trained horizons. No added tail rows. Event exposure always original strict current-normal 30s coverage.',
        selection='Outer LORO, inner 3 GroupKFold; all model/policy conditions declared, none selected with outer outcomes',
        matching='unchanged episode_start_one_to_one',candidate_families='Only existing four; no extra high-capacity model justified with six onset-bearing recordings',
        cohort='Only existing 30 recordings, retrospectively examined; no verified patient IDs',bootstrap_seed=20261015,bootstrap_repetitions=2000,
        implementation={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    path.write_text(json.dumps(protocol,indent=2));return protocol


def inputs():
    numeric={}
    for r in RECORDS:
        n=pd.read_csv(config.DATA_DIR/f'bidmc_{r:02d}_Numerics.csv');n.columns=n.columns.str.strip()
        numeric[r]=(n['Time [s]'].to_numpy(),n.SpO2.to_numpy())
    return numeric,raw_events(numeric)


def datasets(target,numeric):
    folder=ROOT/'results/final'/target
    data=pd.read_csv(folder/'causal_dataset.csv')
    raw=pd.concat([pd.read_csv(folder/f'raw_pulses_{r:02d}.csv') for r in RECORDS],ignore_index=True)
    context=add_context(raw)
    data=data.merge(context,on=['Record','pulse_row'],validate='one_to_one').sort_values('sample_id').reset_index(drop=True)
    variants={}
    for h in [5,10,15,30]:
        d=data.copy()
        for r,g in d.groupby('Record'):
            nt,s=numeric[r]
            y=(onset_labels if target=='new_onset' else future_labels)(g.time_s.to_numpy(),nt,s,horizon=h)[0]
            if not np.isfinite(y).all():raise ValueError('Historical complete-30s rows must remain horizon-valid')
            if h==30:np.testing.assert_array_equal(y,g.Label)
            d.loc[g.index,'Label']=y.astype(int)
        d['Label']=d.Label.astype(int);variants[h]=d
    return variants


def topology_cache(data,numeric,events):
    cache={}
    for r in sorted(numeric):
        g=data[data.Record.eq(r)].sort_values('time_s').copy()
        if len(g):g=g[current_normal(g,{r:numeric[r]})].reset_index(drop=True)
        edge=links(g,{r:numeric[r]})
        prev=np.r_[False,edge] if len(g) else np.zeros(0,bool)
        duration=np.r_[np.diff(g.time_s)*edge,0.] if len(g) else np.zeros(0)
        cache[r]=(g,prev,duration)
    return cache


def select(train,scores,cache,events,numeric,test,protocol,horizon):
    if test in set(train.Record):raise ValueError('Held-out record supplied for selection')
    allowed=set(map(int,train.Record.unique()));parts=[];prev=[];duration=[]
    score_map=pd.Series(scores,index=train.sample_id)
    for r in sorted(allowed):
        g,p,d=cache[r];g=g.copy();g['Score']=g.sample_id.map(score_map)
        parts.append(g);prev.extend(p);duration.extend(d)
    normal=pd.concat(parts,ignore_index=True)
    ev=events[events.Record.isin(allowed)];num={r:numeric[r] for r in allowed}
    eligible=eligible_events(ev,normal,num,horizon)
    grid=candidate_statistics(normal,np.asarray(prev,bool),np.asarray(duration,float),ev,eligible,protocol['quantiles'],horizon)
    winners=[dict(Budget=b,**choose_candidate(grid,b,protocol['active_cap'])) for b in protocol['budgets']]
    return winners,grid


def fit_phase(output,protocol,numeric,events):
    selections=[];ledger=[];feature_analysis=[];latency_models=[];counts=[]
    pred_path=output/'outer_scores_unlabeled.csv.gz';grid_path=output/'training_candidates.csv.gz'
    for target in protocol['targets']:
        variants=datasets(target,numeric);cache=topology_cache(variants[30],numeric,events)
        for spec in protocol['settings']:
            h,fs=spec['Horizon_s'],spec['FeatureSet'];data=variants[h];cols=protocol['features'][fs]
            counts.append(dict(Target=target,Horizon_s=h,FeatureSet=fs,Rows=len(data),Positives=int(data.Label.sum()),Recordings=data.Record.nunique(),Positive_Recordings=int(data.groupby('Record').Label.sum().gt(0).sum())))
            data[['sample_id','Record','pulse_row','time_s','Label']].to_csv(output/f'labels_{target}_{h}_{fs}.csv.gz',index=False,compression='gzip')
            for test in sorted(data.Record.unique()):
                train=data[data.Record.ne(test)].reset_index(drop=True)
                held=data[data.Record.eq(test)].reset_index(drop=True)
                allowed=sorted(map(int,train.Record.unique()))
                # Descriptive analysis ONLY on permitted training labels/features, no selection.
                for feature in cols:
                    pos=train.loc[train.Label.eq(1),feature];neg=train.loc[train.Label.eq(0),feature]
                    scale=train[feature].std()
                    feature_analysis.append(dict(Target=target,Horizon_s=h,FeatureSet=fs,Outer_Test_Record=int(test),Feature=feature,Train_Positive_Mean=pos.mean(),Train_Negative_Mean=neg.mean(),Standardized_Difference=(pos.mean()-neg.mean())/scale if scale>0 else np.nan,Train_Positives=len(pos)))
                for model_name in protocol['model_configs']:
                    oof,model,ref,splits=training_scores(train,cols,model_name,int(test))
                    raw=model.score(held[cols].to_numpy(float))
                    context=dict(Target=target,Horizon_s=h,FeatureSet=fs,Model=model_name)
                    for space,values in oof.items():
                        chosen,grid=select(train,values,cache,events,numeric,int(test),protocol,h)
                        for row in chosen:selections.append(context|dict(ScoreSpace=space,Outer_Test_Record=int(test),Allowed_Records=','.join(map(str,allowed)),**row))
                        grid=grid.assign(**context,ScoreSpace=space,Outer_Test_Record=int(test))
                        grid.to_csv(grid_path,mode='a',header=not grid_path.exists(),index=False,compression='gzip')
                    # No held-out labels or outcomes enter fitting, normalization or selection.
                    scored=held[['sample_id','Record','time_s','pulse_row']].copy()
                    scored['raw']=raw;scored['negative_rank']=negative_rank(ref,raw)
                    for k,v in context.items():scored[k]=v
                    scored.to_csv(pred_path,mode='a',header=not pred_path.exists(),index=False,compression='gzip')
                    ledger.append(context|dict(Outer_Test_Record=int(test),Allowed_Records=allowed,inner=splits,OOF_Raw_SHA256=array_hash(oof['raw']),OOF_Rank_SHA256=array_hash(oof['negative_rank']),Training_Sample_ID_SHA256=array_hash(train.sample_id),Negative_Reference_SHA256=array_hash(ref),Reference_Count=len(ref)))
                    if test==max(data.Record.unique()):latency_models.append((context|dict(Outer_Test_Record=int(test)),model,ref,train[cols].iloc[:1].to_numpy(float)))
                print('FITTED',target,h,fs,int(test),'train',len(train),'held features',len(held),flush=True)
    pd.DataFrame(selections).to_csv(output/'selected_thresholds.csv',index=False)
    pd.DataFrame(feature_analysis).to_csv(output/'training_feature_analysis.csv',index=False)
    pd.DataFrame(counts).to_csv(output/'dataset_counts.csv',index=False)
    (output/'fold_ledger.json').write_text(json.dumps(ledger,indent=2))
    (output/'selection_complete.json').write_text(json.dumps(dict(phase='ALL selections frozen before held-out outcome evaluation',thresholds_sha256=sha(output/'selected_thresholds.csv'),protocol_sha256=sha(output/'protocol.json'),outer_unlabeled_sha256=sha(pred_path)),indent=2))
    return latency_models


def run(output):
    output=Path(output).resolve()
    marker=json.loads((output/'baseline_complete.json').read_text())
    for name,w in marker.items():
        if sha(output/name)!=w:raise ValueError('Baseline audit was modified')
    verify_snapshot(json.loads((output/'preservation.json').read_text()))
    protocol=freeze(output);numeric,events=inputs();events.to_csv(output/'raw_events.csv',index=False)
    with threadpool_limits(limits=1):
        models=fit_phase(output,protocol,numeric,events)
        from .reporting import evaluate,benchmark,finish
        evaluate(output,protocol,numeric,events)
        benchmark(output,protocol,models)
        finish(output,protocol)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
