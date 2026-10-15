"""Checkpointed copy of frozen evaluator; same calculations, cached repeated decisions."""
import json
import time
import hashlib
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from src import config
from src.evaluation import metrics
from src.reliability_evaluation import episodes,match_events,off_event_alarms
from src.alarm_budget_experiment import evaluate_scoped_events
from src.local_30_evaluation import bootstrap_weights,bootstrap_metrics,interval,lead_summary,divide
from .baseline import ROOT,sha,verify_snapshot
from .core import negative_rank,ContextState

KEYS=['Target','Horizon_s','FeatureSet','ScoreSpace','Model']
EKEYS=KEYS+['Budget','Window_s','Subset']


class MetricCache:
    def __init__(self):self.values={}
    def __call__(self,y,s,p):
        key=hashlib.sha256(b''.join(np.asarray(x,dtype='<f8').tobytes() for x in [y,s,p])).digest()
        if key not in self.values:self.values[key]=metrics(y,s,p)
        return self.values[key].copy()


def event_key(target, record, window, predictions):
    values=np.asarray(predictions)
    if not np.isin(values,[0,1]).all():raise ValueError('Binary decisions required')
    return (target,int(record),int(window),values.astype(np.uint8).tobytes())


def evaluate(output,protocol,numeric,events):
    from .experiment import datasets,topology_cache
    marker=json.loads((output/'selection_complete.json').read_text())
    if marker['thresholds_sha256']!=sha(output/'selected_thresholds.csv'):raise ValueError('Selections changed')
    scores=pd.read_csv(output/'outer_scores_unlabeled.csv.gz',float_precision='round_trip')
    selected=pd.read_csv(output/'selected_thresholds.csv',float_precision='round_trip')
    old_sel=pd.read_csv(ROOT/'results/alarm_budget_experiment/run02/selected_thresholds.csv')
    per=[];evrows=[];times=[];parity=[];summaries_saved=[];chunks=[]
    metric=MetricCache();event_cache={};condition_number=0
    parent=output/'checkpoints';parent.mkdir(exist_ok=True)
    for target in protocol['targets']:
        variants=datasets(target,numeric);cache=topology_cache(variants[30],numeric,events)
        old=pd.read_csv(ROOT/'results/final'/target/'per_record_predictions.csv',float_precision='round_trip')
        # All estimators must reproduce original 30s/base10 raw held-out scores.
        for name,g in old.groupby('Model'):
            q=scores[(scores.Target==target)&scores.Model.eq(name)&scores.Horizon_s.eq(30)&scores.FeatureSet.eq('base10')].sort_values('sample_id')
            want=g.sort_values('sample_id');np.testing.assert_array_equal(q.sample_id,want.sample_id)
            np.testing.assert_allclose(q.raw,want.Score,rtol=1e-9,atol=1e-10)
            parity.append(dict(Target=target,Model=name,Rows=len(q),Max_Score_Difference=float(np.max(np.abs(q.raw.to_numpy()-want.Score.to_numpy())))))
        eligible_cache={}
        for r,(g,_,_) in cache.items():
            for w in protocol['event_windows']:
                eligible_cache[r,w]=evaluate_scoped_events(events[events.Record.eq(r)],g.assign(Prediction=0),{r:numeric[r]},[],w)[0]
        conditions=[]
        for (h,fs,name),g in scores[scores.Target==target].groupby(['Horizon_s','FeatureSet','Model']):
            for space in protocol['score_spaces']:conditions.append((h,fs,name,space,g.rename(columns={space:'Score'}).drop(columns=[v for v in protocol['score_spaces'] if v!=space])))
        for h in [5,10,15,30]:
            for name,g in old.groupby('Model'):conditions.append((h,'frozen30',name,'raw',g[['sample_id','Record','time_s','Score']]))
            template=old[old.Model.eq('ELM')][['sample_id','Record','time_s']]
            for name in ['Always_Alert','Never_Alert','Periodic_10s_1s','Pulse_Rate_GT100']:
                g=template.copy()
                g['Score']=(1. if name=='Always_Alert' else 0. if name=='Never_Alert' else
                    (np.mod(g.time_s,10)<1).astype(float) if name=='Periodic_10s_1s' else
                    (variants[h].set_index('sample_id').loc[g.sample_id,'Pulse_Rate_BPM'].to_numpy()>100).astype(float))
                conditions.append((h,'fixed_baseline',name,'binary',g))
        for h,fs,name,space,g in conditions:
            context=dict(Target=target,Horizon_s=int(h),FeatureSet=fs,ScoreSpace=space,Model=name,
                Trained_Horizon_s=30 if fs=='frozen30' else (0 if fs=='fixed_baseline' else int(h)),
                Evaluation_Split='outer_LORO' if fs!='fixed_baseline' else 'fixed_recording_baseline',
                Threshold_Method='frozen_original_30s_training_OOF' if fs=='frozen30' else 'fixed_binary' if fs=='fixed_baseline' else 'inner3_GroupKFold_budget')
            cid=condition_number;condition_number+=1
            complete=sorted(parent.glob(f'{cid:03d}_*/complete.json'))
            if len(complete)>1:raise ValueError('Duplicate complete checkpoint')
            if complete:
                ck=complete[0].parent;receipt=json.loads(complete[0].read_text())
                if receipt['context']!=context:raise ValueError('Checkpoint condition changed')
                for filename,want in receipt['files'].items():
                    if sha(ck/filename)!=want:raise ValueError('Checkpoint changed')
                per.extend(pd.read_csv(ck/'per.csv').to_dict('records'))
                times.extend(pd.read_csv(ck/'times.csv').to_dict('records'))
                evrows.append(pd.read_csv(ck/'events.csv'))
                summaries_saved.extend(pd.read_csv(ck/'summary.csv').to_dict('records'))
                chunks.append(ck/'predictions.csv.gz')
                print('CHECKPOINT REUSED',cid,target,h,fs,name,space,flush=True)
                continue
            attempt=len(list(parent.glob(f'{cid:03d}_*')))
            ck=parent/f'{cid:03d}_{attempt}';ck.mkdir()
            per_start=len(per);time_start=len(times);event_start=len(evrows);local_summary=[]
            labels=variants[int(h)][['sample_id','Label']]
            g=g.drop(columns=['Label'],errors='ignore').merge(labels,on='sample_id',validate='one_to_one')
            for budget in protocol['budgets']:
                full_parts=[];normal_parts=[];timestamp_per=[]
                for r in range(1,31):
                    full=g[g.Record.eq(r)].sort_values('time_s').copy()
                    if len(full):
                        if fs=='fixed_baseline':threshold=.5
                        else:
                            selection=old_sel if fs=='frozen30' else selected
                            choose=selection[(selection.Target==target)&selection.Model.eq(name)&selection.Outer_Test_Record.eq(r)&selection.Budget.eq(budget)]
                            if fs!='frozen30':choose=choose[choose.Horizon_s.eq(h)&choose.FeatureSet.eq(fs)&choose.ScoreSpace.eq(space)]
                            if len(choose)!=1:raise ValueError('Nonunique training-selected threshold')
                            threshold=float.fromhex(choose.Threshold_hex.iloc[0])
                        full['Prediction']=(full.Score>=threshold).astype(int);full['Threshold']=threshold
                    else:full['Prediction']=pd.Series(dtype=int);full['Threshold']=pd.Series(dtype=float)
                    cached,previous,duration=cache[r]
                    normal=full.set_index('sample_id').reindex(cached.sample_id).reset_index()
                    for scope,part in [('full_target',full),('current_normal',normal)]:
                        tm=metric(part.Label,part.Score,part.Prediction)
                        times.append(context|dict(Budget=budget,Record=r,Metric_Scope=scope,Positive_Containing=bool(tm['Positives']),**tm))
                    full_parts.append(full);normal_parts.append(normal)
                    alarms=[a|dict(Record=r) for a in episodes(normal,previous[1:])]
                    for w in protocol['event_windows']:
                        eligible=eligible_cache[r,w]
                        eventkey=event_key(target,r,w,normal.Prediction)
                        if eventkey not in event_cache:
                            matched,used=(match_events(eligible,normal,alarms,w,'episode_start_one_to_one') if len(eligible) else (eligible.copy(),set()))
                            event_cache[eventkey]=(matched,off_event_alarms(alarms,events[events.Record.eq(r)],w))
                        matched,off_event_count=event_cache[eventkey]
                        if len(matched):evrows.append(matched.assign(**context,Budget=budget))
                        valid=matched[matched.Evaluation_Status.eq('eligible')]
                        for subset in ['all','91_92']:
                            v=valid if subset=='all' else valid[valid.Has_91_92.eq(True)]
                            leads=v.loc[v.Detected.eq(True),'Lead_Time_s'].to_numpy(float)
                            per.append(context|dict(Budget=budget,Window_s=w,Subset=subset,Record=r,Metric_Scope='strict_current_normal_one_to_one_events',
                                Eligible_Events=len(v),Detected=int(v.Detected.eq(True).sum()),Missed_Events=int(v.Detected.eq(False).sum()),
                                Exposure_s=float(duration.sum()),Alarm_Active_s=float(duration@normal.Prediction.to_numpy()),
                                Samples=len(normal),Alert_Timestamps=int(normal.Prediction.sum()),
                                Off_Event_Alarm_Starts=off_event_count,Alarm_Starts=len(alarms),
                                **lead_summary(leads)))
                for scope,parts in [('full_target',full_parts),('current_normal',normal_parts)]:
                    f=pd.concat(parts,ignore_index=True)
                    pooled=metric(f.Label,f.Score,f.Prediction)
                    rmetrics=[metric(x.Label,x.Score,x.Prediction) for x in parts if len(x)]
                    for k in ['Recall','Precision','PR_AUC_AP','Specificity']:
                        defined=[v[k] for v in rmetrics if np.isfinite(v[k])]
                        pooled['Macro_'+k]=float(np.mean(defined)) if defined else np.nan
                        pooled['Macro_'+k+'_Defined_Recordings']=len(defined)
                    timestamp_per.append(context|dict(Budget=budget,Metric_Scope=scope,**pooled))
                local_summary.extend(timestamp_per)
                # Full target scores/labels and exact decisions; each condition's three operating points are saved.
                f=pd.concat(full_parts,ignore_index=True)[['sample_id','Record','time_s','Score','Threshold','Prediction','Label']]
                f=f.assign(**context,Budget=budget)
                f.to_csv(ck/'predictions.csv.gz',mode='a',header=cid==0 and not (ck/'predictions.csv.gz').exists(),index=False,compression='gzip')
            pd.DataFrame(per[per_start:]).to_csv(ck/'per.csv',index=False)
            pd.DataFrame(times[time_start:]).to_csv(ck/'times.csv',index=False)
            pd.concat(evrows[event_start:],ignore_index=True).to_csv(ck/'events.csv',index=False)
            pd.DataFrame(local_summary).to_csv(ck/'summary.csv',index=False)
            (ck/'complete.json').write_text(json.dumps(dict(context=context,files={p.name:sha(p) for p in ck.iterdir() if p.is_file()}),indent=2))
            summaries_saved.extend(local_summary);chunks.append(ck/'predictions.csv.gz')
            print('EVALUATED',target,h,fs,name,space,flush=True)
    pd.DataFrame(summaries_saved).to_csv(output/'timestamp_summary.csv',index=False)
    with (output/'outer_predictions.csv.gz').open('xb') as combined:
        for chunk in chunks:
            with chunk.open('rb') as handle:shutil.copyfileobj(handle,combined)
    pd.DataFrame(parity).to_csv(output/'original_score_parity.csv',index=False)
    per=pd.DataFrame(per);per['Sensitivity']=divide(per.Detected,per.Eligible_Events)
    per['False_Alarm_Starts_Per_Hour']=divide(3600*per.Off_Event_Alarm_Starts,per.Exposure_s)
    per['Alarm_Active_Fraction']=divide(per.Alarm_Active_s,per.Exposure_s)
    per.to_csv(output/'event_per_recording.csv',index=False)
    pd.DataFrame(times).to_csv(output/'timestamp_per_recording.csv',index=False)
    events_out=pd.concat(evrows,ignore_index=True);events_out.to_csv(output/'event_matches.csv',index=False)
    summaries=[];cis=[];influence=[]
    weights=bootstrap_weights(protocol['bootstrap_repetitions'],protocol['bootstrap_seed'])
    pd.DataFrame(weights,columns=range(1,31)).to_csv(output/'bootstrap_recording_weights.csv.gz',index=False)
    counts=['Detected','Eligible_Events','Missed_Events','Exposure_s','Alarm_Active_s','Samples','Alert_Timestamps','Off_Event_Alarm_Starts','Alarm_Starts']
    for key,g in per.groupby(EKEYS,sort=False):
        context=dict(zip(EKEYS,key));total={c:g[c].sum() for c in counts}
        q=events_out
        for k,v in context.items():
            if k!='Subset':q=q[q[k].eq(v)]
        if context['Subset']=='91_92':q=q[q.Has_91_92.eq(True)]
        leads=q.loc[q.Detected.eq(True),['Record','Lead_Time_s']]
        summary=context|total|dict(Sensitivity=float(divide(total['Detected'],total['Eligible_Events'])),
            False_Alarm_Starts_Per_Hour=float(divide(3600*total['Off_Event_Alarm_Starts'],total['Exposure_s'])),
            Alarm_Active_Fraction=float(divide(total['Alarm_Active_s'],total['Exposure_s'])),**lead_summary(leads.Lead_Time_s.to_numpy()))
        summary.update(Trained_Horizon_s=g.Trained_Horizon_s.iloc[0],Evaluation_Split=g.Evaluation_Split.iloc[0],Threshold_Method=g.Threshold_Method.iloc[0],Metric_Scope='pooled_one_to_one_events')
        summaries.append(summary)
        for metric,values in bootstrap_metrics(g[counts+['Record']],leads,weights).items():
            cis.append(context|dict(Metric=metric,**interval(values),Replicates=len(weights),Evaluation_Split=g.Evaluation_Split.iloc[0]))
        for row in g.itertuples():
            influence.append(context|dict(Omitted_Record=row.Record,Detected=total['Detected']-row.Detected,
                Eligible_Events=total['Eligible_Events']-row.Eligible_Events,
                Sensitivity=float(divide(total['Detected']-row.Detected,total['Eligible_Events']-row.Eligible_Events)),
                False_Alarm_Starts_Per_Hour=float(divide(3600*(total['Off_Event_Alarm_Starts']-row.Off_Event_Alarm_Starts),total['Exposure_s']-row.Exposure_s))))
    summary=pd.DataFrame(summaries);summary.to_csv(output/'event_summary.csv',index=False)
    pd.DataFrame(cis).to_csv(output/'recording_bootstrap_intervals.csv',index=False)
    pd.DataFrame(influence).to_csv(output/'recording_influence.csv',index=False)
    # Original 30s raw/base refits must reproduce the frozen budget selections/results too.
    old_summary=pd.read_csv(ROOT/'results/alarm_budget_experiment/run02/summary.csv')
    for target in protocol['targets']:
        q=summary[(summary.Target==target)&summary.Horizon_s.eq(30)&summary.FeatureSet.eq('base10')&summary.ScoreSpace.eq('raw')]
        for row in q.itertuples():
            old=old_summary[(old_summary.Target==target)&old_summary.Model.eq(row.Model)&old_summary.Budget.eq(row.Budget)&old_summary.Window_s.eq(row.Window_s)&old_summary.Subset.eq(row.Subset)].iloc[0]
            for c in ['Detected','Eligible_Events','Missed_Events','Exposure_s','Off_Event_Alarm_Starts','Alarm_Active_s']:
                if not np.isclose(getattr(row,c),old[c]):raise ValueError('Baseline event parity failed: '+str((target,row.Model,c)))

