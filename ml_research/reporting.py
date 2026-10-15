"""Saved-array evaluation, paired recording uncertainty, figures and evidence report."""
import json
import time
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


def evaluate(output,protocol,numeric,events):
    from .experiment import datasets,topology_cache
    marker=json.loads((output/'selection_complete.json').read_text())
    if marker['thresholds_sha256']!=sha(output/'selected_thresholds.csv'):raise ValueError('Selections changed')
    scores=pd.read_csv(output/'outer_scores_unlabeled.csv.gz',float_precision='round_trip')
    selected=pd.read_csv(output/'selected_thresholds.csv',float_precision='round_trip')
    old_sel=pd.read_csv(ROOT/'results/alarm_budget_experiment/run02/selected_thresholds.csv')
    per=[];evrows=[];times=[];parity=[]
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
                        tm=metrics(part.Label,part.Score,part.Prediction)
                        times.append(context|dict(Budget=budget,Record=r,Metric_Scope=scope,Positive_Containing=bool(tm['Positives']),**tm))
                    full_parts.append(full);normal_parts.append(normal)
                    alarms=[a|dict(Record=r) for a in episodes(normal,previous[1:])]
                    for w in protocol['event_windows']:
                        eligible=eligible_cache[r,w]
                        matched,used=(match_events(eligible,normal,alarms,w,'episode_start_one_to_one') if len(eligible) else (eligible.copy(),set()))
                        if len(matched):evrows.append(matched.assign(**context,Budget=budget))
                        valid=matched[matched.Evaluation_Status.eq('eligible')]
                        for subset in ['all','91_92']:
                            v=valid if subset=='all' else valid[valid.Has_91_92.eq(True)]
                            leads=v.loc[v.Detected.eq(True),'Lead_Time_s'].to_numpy(float)
                            per.append(context|dict(Budget=budget,Window_s=w,Subset=subset,Record=r,Metric_Scope='strict_current_normal_one_to_one_events',
                                Eligible_Events=len(v),Detected=int(v.Detected.eq(True).sum()),Missed_Events=int(v.Detected.eq(False).sum()),
                                Exposure_s=float(duration.sum()),Alarm_Active_s=float(duration@normal.Prediction.to_numpy()),
                                Samples=len(normal),Alert_Timestamps=int(normal.Prediction.sum()),
                                Off_Event_Alarm_Starts=off_event_alarms(alarms,events[events.Record.eq(r)],w),Alarm_Starts=len(alarms),
                                **lead_summary(leads)))
                for scope,parts in [('full_target',full_parts),('current_normal',normal_parts)]:
                    f=pd.concat(parts,ignore_index=True)
                    pooled=metrics(f.Label,f.Score,f.Prediction)
                    rmetrics=[metrics(x.Label,x.Score,x.Prediction) for x in parts if len(x)]
                    for k in ['Recall','Precision','PR_AUC_AP','Specificity']:
                        defined=[v[k] for v in rmetrics if np.isfinite(v[k])]
                        pooled['Macro_'+k]=float(np.mean(defined)) if defined else np.nan
                        pooled['Macro_'+k+'_Defined_Recordings']=len(defined)
                    timestamp_per.append(context|dict(Budget=budget,Metric_Scope=scope,**pooled))
                pd.DataFrame(timestamp_per).to_csv(output/'timestamp_summary.csv',mode='a',header=not (output/'timestamp_summary.csv').exists(),index=False)
                # Full target scores/labels and exact decisions; each condition's three operating points are saved.
                f=pd.concat(full_parts,ignore_index=True)[['sample_id','Record','time_s','Score','Threshold','Prediction','Label']]
                f=f.assign(**context,Budget=budget)
                f.to_csv(output/'outer_predictions.csv.gz',mode='a',header=not (output/'outer_predictions.csv.gz').exists(),index=False,compression='gzip')
            print('EVALUATED',target,h,fs,name,space,flush=True)
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


def benchmark(output,protocol,models):
    rows=[];raw_samples=[]
    selected=pd.read_csv(output/'selected_thresholds.csv')
    for context,model,reference,x in models:
        for space in protocol['score_spaces']:
            selection=selected[selected.Budget.eq(5)&selected.ScoreSpace.eq(space)]
            for key,value in context.items():selection=selection[selection[key].eq(value)]
            threshold=float.fromhex(selection.Threshold_hex.iloc[0])
            samples=[]
            for i in range(1050):
                begin=time.perf_counter_ns();score=model.score(x)
                if space=='negative_rank':score=negative_rank(reference,score)
                _=bool(score[0]>=threshold)
                if i>=50:samples.append((time.perf_counter_ns()-begin)/1e6)
            raw_samples.extend(context|dict(ScoreSpace=space,Component='transform_model_normalize_threshold',Repetition=i,Latency_ms=v) for i,v in enumerate(samples))
            rows.append(context|dict(ScoreSpace=space,Component='transform_model_normalize_threshold',Repetitions=len(samples),Median_ms=float(np.median(samples)),P95_ms=float(np.quantile(samples,.95)),P99_ms=float(np.quantile(samples,.99)),Threshold_Timing_Value=threshold,Note='Warm batch-one timing only; cutoff comparison cost independent of chosen cutoff; excludes waveform/feature acquisition'))
    from src.features import FeatureState
    raw=pd.read_csv(ROOT/'results/final/new_onset/raw_pulses_01.csv').to_dict('records')
    for fs in ['base10','context18']:
        times=[];base=FeatureState();ctx=ContextState()
        for i in range(1050):
            if i%len(raw)==0:base=FeatureState();ctx=ContextState()
            row=raw[i%len(raw)];start=time.perf_counter_ns();base.update(row)
            if fs=='context18':ctx.update(row)
            if i>=50:times.append((time.perf_counter_ns()-start)/1e6)
        raw_samples.extend(dict(FeatureSet=fs,Component='causal_pulse_features',Repetition=i,Latency_ms=v) for i,v in enumerate(times))
        rows.append(dict(FeatureSet=fs,Component='causal_pulse_features',Repetitions=len(times),Median_ms=np.median(times),P95_ms=np.quantile(times,.95),P99_ms=np.quantile(times,.99),Note='Actual record-1 pulse rows, state reset on replay wrap; not waveform filtering'))
    pd.DataFrame(rows).to_csv(output/'latency.csv',index=False)
    pd.DataFrame(raw_samples).to_csv(output/'latency_samples.csv.gz',index=False,compression='gzip')


def figures(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    directory=output/'figures';directory.mkdir(exist_ok=False)
    e=pd.read_csv(output/'event_summary.csv');e=e[e.Subset.eq('all')&e.Window_s.eq(e.Horizon_s)]
    for target in config.TARGETS:
        fig,axes=plt.subplots(2,4,figsize=(17,8),layout='constrained')
        for col,h in enumerate([5,10,15,30]):
            q=e[(e.Target==target)&e.Horizon_s.eq(h)&~e.FeatureSet.eq('fixed_baseline')]
            for (fs,space,model),g in q.groupby(['FeatureSet','ScoreSpace','Model']):
                label=f'{model}/{fs}/{space}';style='--' if fs=='frozen30' else '-'
                axes[0,col].plot(g.False_Alarm_Starts_Per_Hour,g.Sensitivity,'o'+style,label=label,alpha=.7)
                axes[1,col].plot(100*g.Alarm_Active_Fraction,g.Sensitivity,'o'+style,label=label,alpha=.7)
            baseline=e[(e.Target==target)&e.Horizon_s.eq(h)&e.FeatureSet.eq('fixed_baseline')].drop_duplicates('Model')
            for row in baseline.itertuples():
                axes[0,col].scatter(row.False_Alarm_Starts_Per_Hour,row.Sensitivity,marker='x',s=60,label=row.Model)
                axes[1,col].scatter(100*row.Alarm_Active_Fraction,row.Sensitivity,marker='x',s=60,label=row.Model)
            axes[0,col].set(xlabel='Observed off-event starts / evaluable hour',ylabel='Event sensitivity',title=f'{target}: label H={h}s, matching W={h}s',ylim=(-.03,1.03))
            axes[1,col].set(xlabel='Observed alarm-active time (%)',ylabel='Event sensitivity',ylim=(-.03,1.03));axes[1,col].axvline(5,color='black',ls=':')
        handles,labels=axes[0,-1].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=7)
        fig.savefig(directory/f'{target}_alarm_tradeoffs.png',dpi=150);plt.close(fig)
    p=pd.read_csv(output/'outer_predictions.csv.gz');p=p[p.Budget.eq(5)&p.FeatureSet.isin(['base10','context18','frozen30'])]
    # Same scores repeat across budgets; only one operating point used for ranking.
    for target in config.TARGETS:
        fig,axes=plt.subplots(1,4,figsize=(17,4),layout='constrained')
        for ax,h in zip(axes,[5,10,15,30]):
            for (fs,space,name),g in p[(p.Target==target)&p.Horizon_s.eq(h)].groupby(['FeatureSet','ScoreSpace','Model']):
                if fs=='frozen30':continue
                precision,recall,_=precision_recall_curve(g.Label,g.Score)
                ax.plot(recall,precision,label=f'{name}/{fs}/{space}',alpha=.7)
            prevalence=p[(p.Target==target)&p.Horizon_s.eq(h)].drop_duplicates('sample_id').Label.mean()
            ax.axhline(prevalence,color='black',ls=':',label='Prevalence');ax.set(xlabel='Timestamp recall',ylabel='Timestamp precision',title=f'{target}, label H={h}s',ylim=(0,1))
        handles,labels=axes[-1].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=7)
        fig.savefig(directory/f'{target}_precision_recall.png',dpi=150);plt.close(fig)
    q=e[(e.Target=='new_onset')&e.Budget.eq(5)&~e.FeatureSet.eq('fixed_baseline')].copy()
    q['Condition']=q.Model+'/'+q.FeatureSet+'/'+q.ScoreSpace+'/H'+q.Horizon_s.astype(str)
    fig,axes=plt.subplots(2,1,figsize=(15,10),layout='constrained')
    axes[0].bar(q.Condition,q.Detected);axes[0].set(ylabel='Detected eligible events',title='Onset: prespecified budget 5; W equals label H (not guaranteed minimum warning)')
    leads=q.Lead_Median_s;axes[1].scatter(q.Condition,leads);axes[1].set(ylabel='Detected-only median lead (s)',title='Undefined leads are omitted, never plotted as zero')
    for ax in axes:ax.tick_params(axis='x',rotation=90,labelsize=6)
    fig.savefig(directory/'event_counts_and_leads.png',dpi=150);plt.close(fig)


def finish(output,protocol):
    verify_snapshot(json.loads((output/'preservation.json').read_text()))
    figures(output)
    from .summarize import write_report
    write_report(output,protocol)
    from src.benchmark_latency import environment
    manifest=dict(environment=environment(),status='complete',protocol_sha256=sha(output/'protocol.json'),baseline_document_sha256=sha(ROOT/'docs/ML_IMPROVEMENT_BASELINE.md'),
        sources={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')},
        original_artifacts_unchanged=True,outputs={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    from .verify import verify
    result=verify(output)
    (output/'verification.json').write_text(json.dumps(result,indent=2))
    print('COMPLETE',output,flush=True)
