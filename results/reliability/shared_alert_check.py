"""Recompute explicitly reusable historical positive timestamps; no model fitting."""
import pandas as pd
rows=[]
for target in ['future_low','new_onset']:
    p=pd.read_csv('results/final/'+target+'/per_record_predictions.csv')
    e=pd.read_csv('results/reliability/verified/historical_event_recomputation.csv')
    e=e[(e.Target==target)&(e.Evaluation_Status=='eligible')]
    for (model,record),g in p.groupby(['Model','Record']):
        q=g[g.Prediction.eq(1)&g.spo2.gt(94)]
        ev=e[(e.Model==model)&(e.Record==record)]
        for sample in q.itertuples():
            onsets=ev[(ev.Onset_s>sample.time_s)&(ev.Onset_s<=sample.time_s+30)]
            if len(onsets)>1:
                rows.append({'Target':target,'Model':model,'Record':record,'sample_id':sample.sample_id,'time_s':sample.time_s,
                             'Creditable_Event_Count':len(onsets),'Onsets_s':','.join(map(str,onsets.Onset_s))})
pd.DataFrame(rows).to_csv('results/reliability/shared_positive_timestamps.csv',index=False)
print('Positive timestamps reusable across multiple events:',len(rows))
