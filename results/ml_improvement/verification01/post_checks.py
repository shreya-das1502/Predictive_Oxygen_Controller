"""Read-only cross-checks of completed outputs; writes only a new audit JSON."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
if a.receipt.exists():raise ValueError('Do not overwrite verification evidence')
keys=['Target','Horizon_s','FeatureSet','ScoreSpace','Model','Budget','Metric_Scope']
old=pd.read_csv(a.reference/'timestamp_summary.csv',float_precision='round_trip');new=pd.read_csv(a.output/'timestamp_summary.csv',float_precision='round_trip')
assert not old.duplicated(keys).any() and not new.duplicated(keys).any()
x=old.merge(new,on=keys,suffixes=('_old','_new'),validate='one_to_one');assert len(x)==len(old)
columns=['Samples','Positives','Negatives','TP','FP','FN','TN','Recall','Precision','Specificity','F1','MCC','PR_AUC_AP','ROC_AUC','Alert_Fraction','Macro_Recall','Macro_Precision','Macro_Specificity','Macro_PR_AUC_AP']
for c in columns:np.testing.assert_allclose(x[c+'_old'],x[c+'_new'],rtol=1e-10,atol=1e-12,equal_nan=True,err_msg=c)
checkpoints=sorted((a.output/'checkpoints').glob('*/complete.json'));assert len(checkpoints)==144
assert sorted(int(p.parent.name.split('_')[0]) for p in checkpoints)==list(range(144))
combined=hashlib.sha256();catalog={}
for path in checkpoints:
 r=json.loads(path.read_text());catalog[tuple(r['context'][k] for k in keys[:5])]=path.parent
 for name,w in r['files'].items():assert hashlib.sha256((path.parent/name).read_bytes()).hexdigest()==w
 with (path.parent/'predictions.csv.gz').open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):combined.update(block)
assert combined.hexdigest()==hashlib.sha256((a.output/'outer_predictions.csv.gz').read_bytes()).hexdigest()
lat=pd.read_csv(a.output/'latency.csv');raw=pd.read_csv(a.output/'latency_samples.csv.gz')
for row in lat.itertuples():
 g=raw[raw.Component.eq(row.Component)&raw.FeatureSet.eq(row.FeatureSet)]
 if row.Component!='causal_pulse_features':g=g[g.Target.eq(row.Target)&g.Horizon_s.eq(row.Horizon_s)&g.Model.eq(row.Model)&g.ScoreSpace.eq(row.ScoreSpace)]
 assert len(g)==row.Repetitions==1000
 for q,c in [(.5,'Median_ms'),(.95,'P95_ms'),(.99,'P99_ms')]:assert np.isclose(g.Latency_ms.quantile(q),getattr(row,c),rtol=1e-10,atol=1e-12)
matches=pd.read_csv(a.output/'event_matches.csv',float_precision='round_trip');hits=matches[matches.Detected.eq(True)]
header=pd.read_csv(a.output/'outer_predictions.csv.gz',nrows=0).columns.tolist();loaded={}
for row in hits.itertuples():
 key=tuple(getattr(row,k) for k in keys[:5]);folder=catalog[key]
 if key not in loaded:
  first=int(folder.name.split('_')[0])==0
  loaded[key]=pd.read_csv(folder/'predictions.csv.gz',header=0 if first else None,names=None if first else header,float_precision='round_trip')
 g=loaded[key];point=g[g.Budget.eq(row.Budget)&g.Record.eq(row.Record)&g.time_s.eq(row.First_Alert_s)]
 assert len(point)==1 and point.Prediction.iloc[0]==1 and point.Score.iloc[0]>=point.Threshold.iloc[0]
 assert np.isclose(row.Lead_Time_s,row.Onset_s-row.First_Alert_s,rtol=0,atol=1e-12)
result=dict(status='passed',prior_partial_timestamp_rows_reproduced=len(old),numeric_metrics_compared=columns,complete_conditions=len(checkpoints),checkpoint_gzip_concatenation_verified=True,latency_components_checked=len(lat),measured_repetitions_per_component=1000,matched_alert_timestamps_checked=len(hits),note='Window-specific copies of detections are not distinct physiological events.')
a.receipt.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
