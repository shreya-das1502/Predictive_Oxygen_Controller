"""Original 13-event clipped-window metric, identical rows/events for baselines."""
from pathlib import Path
import pandas as pd
from src.reliability_evaluation import baseline_predictions,independent_historical
numeric={}
for path in Path('data').glob('bidmc_*_Numerics.csv'):
    n=pd.read_csv(path);n.columns=n.columns.str.strip()
    numeric[int(path.stem.split('_')[1])]=(n['Time [s]'].to_numpy(),n.SpO2.to_numpy())
p=pd.read_csv('results/validated/per_record_predictions.csv');d=pd.read_csv('results/validated/causal_dataset.csv')
e=pd.read_csv('results/reliability/final/independent_raw_events.csv')
b=baseline_predictions(p[p.Model==p.Model.iloc[0]],d)
m=independent_historical(b,e,numeric,'future_low',old_dense=True)
m.to_csv('results/reliability/original_baseline_event_matches.csv',index=False)
old=pd.read_csv('results/reliability/final/original_lead_time_recomputation.csv')
rows=[]
for name,g in pd.concat([old,m]).groupby('Model'):
    for subset in ['all','91_92']:
        q=g[g.Evaluation_Status.eq('eligible')]
        if subset=='91_92':q=q[q.Has_91_92]
        rows.append({'Model':name,'Subset':subset,'Eligible_Events':len(q),'Detected':int(q.Detected.eq(True).sum()),
                     'Sensitivity':q.Detected.eq(True).mean(),'Median_Lead_s':q.Lead_Time_s.median()})
s=pd.DataFrame(rows);s.to_csv('results/reliability/original_baseline_comparison.csv',index=False)
print(s.to_string(index=False))
