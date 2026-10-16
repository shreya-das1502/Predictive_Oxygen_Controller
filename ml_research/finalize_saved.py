"""Finish saved experiments without rerunning CV or selecting another policy."""
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .baseline import ROOT,sha
from .reporting import KEYS,EKEYS
from src.local_30_evaluation import bootstrap_weights,divide,interval,lead_summary

COUNTS=['Detected','Eligible_Events','Missed_Events','Exposure_s','Alarm_Active_s','Samples','Alert_Timestamps','Off_Event_Alarm_Starts','Alarm_Starts']


def bootstrap_fast(recording, leads, weights):
    """Same cluster sums and ordinary repeated-observation median as the audit."""
    g=recording.set_index('Record').reindex(range(1,31))
    if g[COUNTS].isna().any().any():raise ValueError('Incomplete recording roster')
    sums=weights@g[COUNTS].to_numpy(float);c={k:sums[:,i] for i,k in enumerate(COUNTS)}
    out=dict(Sensitivity=divide(c['Detected'],c['Eligible_Events']),
        False_Alarm_Starts_Per_Hour=divide(3600*c['Off_Event_Alarm_Starts'],c['Exposure_s']),
        Alarm_Active_Fraction=divide(c['Alarm_Active_s'],c['Exposure_s']),
        Pulse_Alert_Fraction=divide(c['Alert_Timestamps'],c['Samples']))
    med=np.full(len(weights),np.nan)
    if len(leads):
        ordered=leads.sort_values('Lead_Time_s');values=ordered.Lead_Time_s.to_numpy(float)
        multiplicity=weights[:,ordered.Record.to_numpy(int)-1];cumulative=np.cumsum(multiplicity,axis=1);n=cumulative[:,-1];valid=n>0
        lo=((n-1)//2)[:,None];hi=(n//2)[:,None]
        left=(cumulative>lo).argmax(axis=1);right=(cumulative>hi).argmax(axis=1)
        med[valid]=(values[left[valid]]+values[right[valid]])/2
    out['Lead_Median_s']=med
    return out


def aggregate(output,protocol):
    per=pd.read_csv(output/'event_per_recording.csv');events=pd.read_csv(output/'event_matches.csv')
    weights=pd.read_csv(output/'bootstrap_recording_weights.csv.gz').to_numpy(int)
    np.testing.assert_array_equal(weights,bootstrap_weights(protocol['bootstrap_repetitions'],protocol['bootstrap_seed']))
    hits=events[events.Evaluation_Status.eq('eligible')&events.Detected.eq(True)]
    lead_groups={k:g for k,g in hits.groupby(KEYS+['Budget','Window_s'],sort=False)}
    empty=hits.iloc[:0];summaries=[];cis=[];influence=[];boots={};groups={};memo={}
    for i,(key,g) in enumerate(per.groupby(EKEYS,sort=False)):
        g=g.sort_values('Record');groups[key]=g;context=dict(zip(EKEYS,key))
        q=lead_groups.get(key[:-1],empty)
        if key[-1]=='91_92':q=q[q.Has_91_92.eq(True)]
        leads=q[['Record','Lead_Time_s']];total={c:g[c].sum() for c in COUNTS}
        summary=context|total|dict(Sensitivity=float(divide(total['Detected'],total['Eligible_Events'])),False_Alarm_Starts_Per_Hour=float(divide(3600*total['Off_Event_Alarm_Starts'],total['Exposure_s'])),Alarm_Active_Fraction=float(divide(total['Alarm_Active_s'],total['Exposure_s'])),**lead_summary(leads.Lead_Time_s.to_numpy(float)))
        summary.update(Trained_Horizon_s=g.Trained_Horizon_s.iloc[0],Evaluation_Split=g.Evaluation_Split.iloc[0],Threshold_Method=g.Threshold_Method.iloc[0],Metric_Scope='pooled_one_to_one_events')
        summaries.append(summary)
        signature=(g[COUNTS].to_numpy(float).tobytes(),leads.to_numpy(float).tobytes())
        if signature not in memo:memo[signature]=bootstrap_fast(g,leads,weights)
        boot=memo[signature];boots[key]=boot
        for metric,values in boot.items():cis.append(context|dict(Metric=metric,**interval(values),Replicates=len(weights),Evaluation_Split=g.Evaluation_Split.iloc[0]))
        for row in g.itertuples():influence.append(context|dict(Omitted_Record=row.Record,Detected=total['Detected']-row.Detected,Eligible_Events=total['Eligible_Events']-row.Eligible_Events,Sensitivity=float(divide(total['Detected']-row.Detected,total['Eligible_Events']-row.Eligible_Events)),False_Alarm_Starts_Per_Hour=float(divide(3600*(total['Off_Event_Alarm_Starts']-row.Off_Event_Alarm_Starts),total['Exposure_s']-row.Exposure_s))))
        if i%400==0:print('AGGREGATED',i,flush=True)
    summary=pd.DataFrame(summaries)
    old=pd.read_csv(ROOT/'results/alarm_budget_experiment/run02/summary.csv').set_index(['Target','Model','Budget','Window_s','Subset'])
    base=summary[summary.Horizon_s.eq(30)&summary.FeatureSet.eq('base10')&summary.ScoreSpace.eq('raw')]
    for row in base.itertuples():
        want=old.loc[(row.Target,row.Model,row.Budget,row.Window_s,row.Subset)]
        for c in ['Detected','Eligible_Events','Missed_Events','Exposure_s','Off_Event_Alarm_Starts','Alarm_Active_s']:np.testing.assert_allclose(getattr(row,c),want[c],rtol=1e-9,atol=1e-9,err_msg=str((row.Target,row.Model,c)))
    summary.to_csv(output/'event_summary.csv',index=False);pd.DataFrame(cis).to_csv(output/'recording_bootstrap_intervals.csv',index=False);pd.DataFrame(influence).to_csv(output/'recording_influence.csv',index=False)
    comparisons=[];paired=[];indexed=summary.set_index(EKEYS)
    for row in summary[summary.FeatureSet.isin(['base10','context18'])].itertuples():
        key=tuple(getattr(row,k) for k in EKEYS);other=(row.Target,row.Horizon_s,'frozen30','raw',row.Model,row.Budget,row.Window_s,row.Subset)
        before=indexed.loc[other];g=groups[key];np.testing.assert_array_equal(g.Eligible_Events,groups[other].Eligible_Events)
        context={k:getattr(row,k) for k in EKEYS};v={}
        for metric in ['Detected','Eligible_Events','Missed_Events','Sensitivity','False_Alarm_Starts_Per_Hour','Alarm_Active_Fraction','Lead_Median_s']:v['Before_'+metric]=before[metric];v['After_'+metric]=getattr(row,metric)
        v['All_Exposed_Records_Within_Budget']=bool((g.Exposure_s.eq(0)|g.False_Alarm_Starts_Per_Hour.le(row.Budget+1e-12)).all())
        v['All_Exposed_Records_Within_Active_Cap']=bool((g.Exposure_s.eq(0)|g.Alarm_Active_Fraction.le(.05+1e-12)).all());comparisons.append(context|v)
        for metric in ['Sensitivity','False_Alarm_Starts_Per_Hour','Alarm_Active_Fraction']:paired.append(context|dict(Metric=metric,Difference=getattr(row,metric)-before[metric],**interval(boots[key][metric]-boots[other][metric]),Replicates=len(weights)))
    pd.DataFrame(comparisons).to_csv(output/'before_after_comparison.csv',index=False);pd.DataFrame(paired).to_csv(output/'paired_recording_bootstrap.csv',index=False)
    (output/'aggregation_verified.json').write_text(json.dumps(dict(status='passed',conditions=len(summary),bootstrap_repetitions=len(weights),memoized_distinct_contributions=len(memo),original_policy_parity_rows=len(base)),indent=2))


def timing(output,protocol):
    from .experiment import inputs,datasets
    from .core import fit_model
    from .reporting import benchmark
    import joblib
    path=output/'timing_only_models.joblib';models=[]
    if path.exists():
        receipt=json.loads((output/'timing_models_receipt.json').read_text())
        if sha(path)!=receipt['sha256']:raise ValueError('Timing objects changed')
        models=joblib.load(path)
    else:
        numeric,_=inputs();saved=pd.read_csv(output/'outer_scores_unlabeled.csv.gz',float_precision='round_trip');parity=[]
        for target in protocol['targets']:
            variants=datasets(target,numeric)
            for spec in protocol['settings']:
                h,fs=spec['Horizon_s'],spec['FeatureSet'];data=variants[h];test=int(data.Record.max());train=data[data.Record.ne(test)];held=data[data.Record.eq(test)];cols=protocol['features'][fs]
                for name in protocol['model_configs']:
                    model=fit_model(train,cols,name);ref=model.score(train.loc[train.Label.eq(0),cols].to_numpy(float));q=saved[saved.Target.eq(target)&saved.Horizon_s.eq(h)&saved.FeatureSet.eq(fs)&saved.Model.eq(name)&saved.Record.eq(test)].sort_values('sample_id');score=model.score(held[cols].to_numpy(float));np.testing.assert_allclose(score,q.raw,rtol=1e-9,atol=1e-10)
                    context=dict(Target=target,Horizon_s=h,FeatureSet=fs,Model=name,Outer_Test_Record=test);models.append((context,model,ref,train[cols].iloc[:1].to_numpy(float)));parity.append(context|dict(Max_Difference=float(np.max(np.abs(score-q.raw)))))
        joblib.dump(models,path);pd.DataFrame(parity).to_csv(output/'timing_model_score_parity.csv',index=False)
        (output/'timing_models_receipt.json').write_text(json.dumps(dict(sha256=sha(path),count=len(models),purpose='Timing-only outer-training refits, no scientific scores/thresholds replaced'),indent=2))
    benchmark(output,protocol,models)


def prepare(source,output):
    output=Path(output).resolve();source=Path(source).resolve()
    if output==source or source in output.parents:raise ValueError('Output must not be inside the saved input directory')
    marker=output/'cloned_inputs.json'
    if marker.exists():
        for name,w in json.loads(marker.read_text()).items():
            if sha(output/name)!=w or sha(source/name)!=w:raise ValueError('Saved input changed: '+name)
        return
    files={}
    for p in source.rglob('*'):
        if p.is_file():
            name=str(p.relative_to(source));dest=output/name
            if dest.exists():raise ValueError('Do not overwrite '+name)
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);files[name]=sha(p)
    (output/'cloned_inputs.json').write_text(json.dumps(files,indent=2))


def event_panel(output):
    """Numeric x coordinates avoid matplotlib's all-masked categorical failure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);s=pd.read_csv(output/'event_summary.csv')
    q=s[s.Target.eq('new_onset')&s.Subset.eq('all')&s.Budget.eq(5)&s.Window_s.eq(s.Horizon_s)&~s.FeatureSet.eq('fixed_baseline')].copy()
    labels=q.Model+'/'+q.FeatureSet+'/'+q.ScoreSpace+'/H'+q.Horizon_s.astype(str);x=np.arange(len(q));leads=q.Lead_Median_s.to_numpy(float);valid=np.isfinite(leads)
    fig,axes=plt.subplots(2,1,figsize=(15,10),layout='constrained')
    axes[0].bar(x,q.Detected);axes[0].set(ylabel='Detected eligible events',title='New onset, research budget5; matching W equals label H')
    if valid.any():axes[1].scatter(x[valid],leads[valid])
    else:axes[1].text(.5,.5,'No matched detections: lead times are undefined',transform=axes[1].transAxes,ha='center')
    axes[1].set(ylabel='Detected-only median lead (s)',title='Missing leads are omitted, never replaced with zero')
    for ax in axes:ax.set_xticks(x,labels,rotation=90,fontsize=6)
    path=output/'figures/event_counts_and_leads.png'
    if path.exists():raise ValueError('Event panel already exists; preserve it')
    fig.savefig(path,dpi=150);plt.close(fig)


def finish_figures(output):
    from .reporting import figures
    expected=[f'{t}_{name}.png' for t in ['future_low','new_onset'] for name in ['alarm_tradeoffs','precision_recall']]
    directory=output/'figures'
    if directory.exists():
        if not all((directory/n).exists() for n in expected):raise ValueError('Unexpected incomplete figure set; inspect before continuing')
        event_panel(output)
    else:
        from matplotlib.units import ConversionError
        try:figures(output)
        except ConversionError:
            s=pd.read_csv(output/'event_summary.csv');q=s[s.Target.eq('new_onset')&s.Subset.eq('all')&s.Budget.eq(5)&s.Window_s.eq(s.Horizon_s)&~s.FeatureSet.eq('fixed_baseline')]
            if not q.Lead_Median_s.isna().all() or not all((directory/n).exists() for n in expected):raise
            event_panel(output)

def run(output,source,stage):
    output=Path(output).resolve();prepare(source,output);protocol=json.loads((output/'protocol.json').read_text())
    for name,w in protocol['implementation'].items():
        if sha(Path(__file__).parent/name)!=w:raise ValueError('Frozen scientific source changed')
    with threadpool_limits(limits=1):
        if stage=='aggregate':
            if (output/'aggregation_verified.json').exists():raise ValueError('Aggregation already complete; reuse saved results')
            aggregate(output,protocol)
        elif stage=='timing':
            if (output/'latency.csv').exists():raise ValueError('Timing already measured; reuse saved results')
            timing(output,protocol)
        elif stage=='figures':
            finish_figures(output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,default=ROOT/'results/ml_improvement/evaluation01');p.add_argument('--stage',choices=['aggregate','timing','figures'],required=True);a=p.parse_args();run(a.output,a.source,a.stage)
