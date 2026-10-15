"""Read-only checks of isolated experiment output; no preservation exemptions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.evaluation import metrics
from .baseline import sha,verify_snapshot
from .reporting import KEYS,EKEYS


def verify(output):
    output=Path(output);m=json.loads((output/'manifest.json').read_text())
    if sha(output/'protocol.json')!=m['protocol_sha256']:raise ValueError('Protocol changed')
    for name,w in m['outputs'].items():
        if sha(output/name)!=w:raise ValueError('New output changed: '+name)
    protected=verify_snapshot(json.loads((output/'preservation.json').read_text()))
    selected=pd.read_csv(output/'selected_thresholds.csv')
    if not selected.False_Alarm_Starts_Per_Hour.le(selected.Budget+1e-12).all() or not selected.Alarm_Active_Fraction.le(.05+1e-12).all():raise ValueError('Training constraint violation')
    for row in selected.itertuples():
        if row.Outer_Test_Record in map(int,row.Allowed_Records.split(',')):raise ValueError('Outer selection leakage')
    for split in json.loads((output/'fold_ledger.json').read_text()):
        allowed=set(split['Allowed_Records']);test=split['Outer_Test_Record'];seen=[]
        if test in allowed:raise ValueError('Outer fit leakage')
        for inner in split['inner']:
            tr=set(inner['train_records']);va=set(inner['validation_records'])
            if tr&va or tr|va!=allowed:raise ValueError('Inner fit leakage')
            seen.extend(va)
        if sorted(seen)!=sorted(allowed):raise ValueError('OOF recording coverage')
    pred=pd.read_csv(output/'outer_predictions.csv.gz',float_precision='round_trip')
    np.testing.assert_array_equal(pred.Prediction,(pred.Score>=pred.Threshold).astype(int))
    t=pd.read_csv(output/'timestamp_summary.csv');ts=t[t.Metric_Scope.eq('full_target')].set_index(KEYS+['Budget'])
    for key,g in pred.groupby(KEYS+['Budget']):
        if g.sample_id.duplicated().any():raise ValueError('Duplicate held-out sample')
        want=ts.loc[key];got=metrics(g.Label,g.Score,g.Prediction)
        for name in ['TP','FP','FN','TN','Recall','Precision','Specificity','PR_AUC_AP']:
            if not np.isclose(got[name],want[name],equal_nan=True):raise ValueError('Timestamp mismatch '+name)
    per=pd.read_csv(output/'event_per_recording.csv');summary=pd.read_csv(output/'event_summary.csv')
    for key,g in per.groupby(EKEYS):
        if sorted(g.Record)!=list(range(1,31)):raise ValueError('Incomplete reporting roster')
    aggregate=per.groupby(EKEYS)[['Detected','Eligible_Events','Exposure_s','Alarm_Active_s','Off_Event_Alarm_Starts']].sum()
    for row in summary.itertuples():
        key=tuple(getattr(row,k) for k in EKEYS);a=aggregate.loc[key]
        for c in a.index:
            if not np.isclose(a[c],getattr(row,c)):raise ValueError('Event aggregation mismatch')
        if row.Eligible_Events!=(3 if row.Subset=='91_92' else 14 if row.Window_s==30 else 15):raise ValueError('Event denominator changed')
    ev=pd.read_csv(output/'event_matches.csv');hits=ev[ev.Detected.eq(True)]
    if hits.duplicated(KEYS+['Budget','Window_s','Record','Alarm_ID']).any():raise ValueError('Shared alarm credit')
    if not ((hits.Lead_Time_s>0)&(hits.Lead_Time_s<=hits.Window_s)).all():raise ValueError('Lead boundary error')
    for (target,h,fs,space,model,budget,w),g in ev.groupby(KEYS+['Budget','Window_s']):
        for subset in ['all','91_92']:
            valid=g[g.Evaluation_Status.eq('eligible')]
            if subset=='91_92':valid=valid[valid.Has_91_92]
            q=summary[(summary.Target==target)&summary.Horizon_s.eq(h)&summary.FeatureSet.eq(fs)&summary.ScoreSpace.eq(space)&summary.Model.eq(model)&summary.Budget.eq(budget)&summary.Window_s.eq(w)&summary.Subset.eq(subset)].iloc[0]
            if len(valid)!=q.Eligible_Events or valid.Detected.eq(True).sum()!=q.Detected:raise ValueError('Event match mismatch')
    lat=pd.read_csv(output/'latency.csv');raw=pd.read_csv(output/'latency_samples.csv.gz')
    for row in lat.itertuples():
        g=raw[raw.FeatureSet.eq(row.FeatureSet)&raw.Component.eq(row.Component)]
        if row.Component!='causal_pulse_features':
            g=g[g.Target.eq(row.Target)&g.Model.eq(row.Model)&g.ScoreSpace.eq(row.ScoreSpace)&g.Horizon_s.eq(row.Horizon_s)]
        if len(g)!=row.Repetitions or not np.isclose(g.Latency_ms.median(),row.Median_ms) or not np.isclose(g.Latency_ms.quantile(.95),row.P95_ms):raise ValueError('Latency mismatch')
    return dict(status='passed',protected_files=protected,predictions=len(pred),event_conditions=len(summary),new_thresholds=len(selected),clinical_validation=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);print(json.dumps(verify(p.parse_args().output),indent=2))
