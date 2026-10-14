"""Fixed-policy local-30 evaluation: saved LORO decisions, no fitting or selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import config
from .evaluation import metrics
from .reliability_evaluation import (digest, validate_predictions, current_normal, links,
    episodes, eligible_events, match_events, raw_events, baseline_predictions, off_event_alarms)

RECORDS = tuple(range(1,31))
WINDOWS = (5,10,15,30)
MODELS = tuple(config.MODEL_CONFIGS) + ('Always_Alert','Never_Alert','Periodic_10s_1s','Pulse_Rate_GT100')
PAIRS = (('ELM','Logistic'),('ELM','Always_Alert'),('Logistic','Always_Alert'),
         ('ELM','Periodic_10s_1s'),('ELM','Pulse_Rate_GT100'))
DEFAULT_OUTPUT = config.ROOT/'results/local_30_fixed_policy_v2'


def divide(a,b):
    a,b=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
    return np.divide(a,b,out=np.full(a.shape,np.nan),where=b!=0)


def verify_splits(splits, records):
    universe=set(map(int,records))
    if sorted(s['test_record'] for s in splits)!=sorted(universe):raise ValueError('Outer folds do not cover eligible records exactly once')
    for s in splits:
        train=set(s['outer_train_records']);test=s['test_record']
        if train!=universe-{test}:raise ValueError('Outer split isolation/coverage failure')
        seen=[]
        for inner in s['inner']:
            tr=set(inner['train_records']);val=set(inner['validation_records'])
            if tr&val or tr|val!=train:raise ValueError('Inner split isolation/coverage failure')
            seen.extend(inner['validation_records'])
        if sorted(seen)!=sorted(train):raise ValueError('Inner validation not an exact record partition')


def verify_threshold_fold(training, oof, held, test_record, recall=.92):
    """Recompute training-only cutoff. Held labels are deliberately inaccessible."""
    if test_record in set(training.Record) or test_record in set(oof.Record):raise ValueError('Outer test record in threshold training')
    if oof.sample_id.duplicated().any() or set(oof.sample_id)!=set(training.sample_id):raise ValueError('OOF row coverage failure')
    expected=training.set_index('sample_id').loc[oof.sample_id]
    if not np.array_equal(expected.Record,oof.Record) or not np.array_equal(expected.Label,oof.Label):raise ValueError('OOF labels/record provenance mismatch')
    if not np.isfinite(oof.Score).all():raise ValueError('Invalid OOF scores')
    positives=np.sort(oof.loc[oof.Label.eq(1),'Score'].to_numpy())[::-1]
    if not len(positives):raise ValueError('No OOF positives')
    cutoff=float(positives[int(np.ceil(recall*len(positives)))-1])
    if oof.Threshold.nunique()!=1 or not np.isclose(oof.Threshold.iloc[0],cutoff,rtol=1e-12,atol=1e-12):raise ValueError('Saved threshold differs from training-only cutoff')
    if not held.Record.eq(test_record).all() or not np.isclose(held.Threshold,cutoff,rtol=1e-12,atol=1e-12).all():raise ValueError('Held-out threshold transfer failure')
    if not np.array_equal(held.Prediction,(held.Score>=held.Threshold).astype(int)):raise ValueError('Held-out decision mismatch')
    return cutoff


def audit_selection(directory):
    m=json.loads((directory/'run_manifest.json').read_text())
    data=pd.read_csv(directory/'causal_dataset.csv');held=pd.read_csv(directory/'per_record_predictions.csv')
    verify_splits(m['recording_splits'],data.Record.unique())
    oof=pd.read_csv(directory/'validation_oof_predictions.csv.gz')
    if oof.duplicated(['Model','Outer_Test_Record','sample_id']).any():raise ValueError('Duplicated OOF rows')
    got=set();rows=[]
    for (model,test),g in oof.groupby(['Model','Outer_Test_Record']):
        h=held[(held.Model==model)&held.Record.eq(test)]
        cutoff=verify_threshold_fold(data[data.Record.ne(test)],g,h,test)
        rows.append({'Model':model,'Outer_Test_Record':test,'Training_OOF_Rows':len(g),'Threshold':cutoff})
        got.add((model,int(test)))
    if got!={(model,int(r)) for model in config.MODEL_CONFIGS for r in data.Record.unique()}:raise ValueError('Missing outer/model threshold fold')
    return rows


def bootstrap_weights(repetitions=2000, seed=20261014, records=RECORDS):
    """Same unstratified recording draws for every model/target/window."""
    if repetitions<2 or len(set(records))!=len(records) or not len(records):raise ValueError('Invalid bootstrap dimensions')
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(records),size=(repetitions,len(records)))
    return np.array([np.bincount(d,minlength=len(records)) for d in draws])


def lead_summary(leads):
    if not len(leads):return {k:np.nan for k in ['Lead_Min_s','Lead_Q25_s','Lead_Median_s','Lead_Q75_s','Lead_Max_s']}
    q=np.quantile(leads,[0,.25,.5,.75,1])
    return dict(zip(['Lead_Min_s','Lead_Q25_s','Lead_Median_s','Lead_Q75_s','Lead_Max_s'],map(float,q)))


def bootstrap_metrics(recording, leads, weights, records=RECORDS):
    """Sum whole-record contributions; rematching across recording copies is forbidden."""
    g=recording.set_index('Record').reindex(records)
    if g.isna().any().any():raise ValueError('Missing recording contributions')
    def total(c):return weights@g[c].to_numpy(float)
    result={'Sensitivity':divide(total('Detected'),total('Eligible_Events')),
            'False_Alarm_Starts_Per_Hour':divide(3600*total('Off_Event_Alarm_Starts'),total('Exposure_s')),
            'Alarm_Active_Fraction':divide(total('Alarm_Active_s'),total('Exposure_s')),
            'Pulse_Alert_Fraction':divide(total('Alert_Timestamps'),total('Samples'))}
    # Linear quantiles of the pooled detected-event leads, repeating full record clusters.
    arr=np.full(len(weights),np.nan);index={r:i for i,r in enumerate(records)}
    vals=leads.Lead_Time_s.to_numpy(float);cols=np.array([index[r] for r in leads.Record],int)
    for i,w in enumerate(weights):
        sample=np.repeat(vals,w[cols])
        if len(sample):arr[i]=np.median(sample)
    result['Lead_Median_s']=arr
    return result


def interval(values):
    v=np.asarray(values,float);v=v[np.isfinite(v)]
    lo,hi=np.quantile(v,[.025,.975]) if len(v) else (np.nan,np.nan)
    return {'Lower_95':lo,'Upper_95':hi,'Defined_Replicates':len(v)}


def snapshot_existing(output):
    """Fingerprint every existing result/raw/archive/doc plus active predictor source."""
    files=[p for base in ['data','results','archive','docs','src'] for p in (config.ROOT/base).rglob('*')
           if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and output not in p.parents]
    return {str(p.relative_to(config.ROOT)):digest(p) for p in files}


def verify_output(output, preservation=True):
    output=Path(output);m=json.loads((output/'manifest.json').read_text())
    for p,want in m['inputs'].items():
        if digest(config.ROOT/p)!=want:raise ValueError('Changed input: '+p)
    for p,want in m['outputs'].items():
        if digest(output/p)!=want:raise ValueError('Changed output: '+p)
    if preservation:
        old=json.loads((output/'preservation.json').read_text())
        for p,want in old.items():
            if digest(config.ROOT/p)!=want:raise ValueError('Protected file changed: '+p)
    per=pd.read_csv(output/'event_per_recording.csv');summ=pd.read_csv(output/'event_summary.csv')
    for row in summ.itertuples():
        q=per[(per.Target==row.Target)&(per.Model==row.Model)&(per.Window_s==row.Window_s)&(per.Subset==row.Subset)]
        for c in ['Eligible_Events','Detected','Missed_Events','Off_Event_Alarm_Starts','Exposure_s','Alarm_Active_s']:
            if not np.isclose(q[c].sum(),getattr(row,c)):raise ValueError('Aggregation mismatch '+c)
    return {'status':'passed','recordings':30,'protected_files':len(json.loads((output/'preservation.json').read_text())),
            'event_recording_rows':len(per),'summary_rows':len(summ),'bootstrap_repetitions':m['bootstrap_repetitions']}


def run(output=DEFAULT_OUTPUT, repetitions=2000, seed=20261014):
    output=Path(output).resolve()
    if output.exists():raise ValueError('Choose a new output directory; existing results are protected')
    # Freeze preservation before any result is created. This pass never calls a model fit.
    old=snapshot_existing(output);output.mkdir(parents=True)
    (output/'preservation.json').write_text(json.dumps(old,indent=2))
    audited=config.ROOT/'results/reliability/final';receipt=json.loads((audited/'manifest.json').read_text())
    inputs={str((audited/'manifest.json').relative_to(config.ROOT)):digest(audited/'manifest.json')}
    for name,want in receipt['inputs'].items():
        if digest(config.ROOT/name)!=want:raise ValueError('Stale audited input '+name)
        inputs[name]=want
    for name,want in receipt['outputs'].items():
        p=audited/name
        if digest(p)!=want:raise ValueError('Stale audited output '+name)
        inputs[str(p.relative_to(config.ROOT))]=want
    if digest(config.ROOT/'src/reliability_evaluation.py')!=receipt['code_sha256']:raise ValueError('Audited metric implementation changed')
    numeric={};inventory=[]
    signals=sorted(config.DATA_DIR.glob('bidmc_*_Signals.csv'));nums=sorted(config.DATA_DIR.glob('bidmc_*_Numerics.csv'))
    if {int(p.stem.split('_')[1]) for p in signals}!={*RECORDS} or {int(p.stem.split('_')[1]) for p in nums}!={*RECORDS}:raise ValueError('Exactly the existing 30 recording pairs are required')
    for p in signals+nums:
        header=list(pd.read_csv(p,nrows=0).columns.str.strip())
        patient=[h for h in header if any(s in h.lower() for s in ['patient','subject','person'])]
        inventory.append({'Path':str(p.relative_to(config.ROOT)),'Columns':' | '.join(header),'Patient_ID_Columns':' | '.join(patient)})
        inputs[str(p.relative_to(config.ROOT))]=digest(p)
        if p in nums:
            n=pd.read_csv(p);n.columns=n.columns.str.strip();numeric[int(p.stem.split('_')[1])]=(n['Time [s]'].to_numpy(),n.SpO2.to_numpy())
    pd.DataFrame(inventory).to_csv(output/'source_inventory.csv',index=False)
    events=raw_events(numeric);saved=pd.read_csv(audited/'window_event_matches.csv');saved=saved[saved.Method=='episode_start_one_to_one']
    records=[];event_rows=[];timestamp=[];thresholds=[];lead_parts=[]
    for target in config.TARGETS:
        directory=config.ROOT/'results/final'/target
        print('Checking selection provenance:',target,flush=True)
        checks=audit_selection(directory)
        thresholds.extend(dict(Target=target,**r) for r in checks)
        for name in ['validation_oof_predictions.csv.gz','run_manifest.json','per_record_predictions.csv','causal_dataset.csv']:
            p=directory/name;inputs[str(p.relative_to(config.ROOT))]=digest(p)
        data=pd.read_csv(directory/'causal_dataset.csv');pred=pd.read_csv(directory/'per_record_predictions.csv');validate_predictions(pred)
        base=baseline_predictions(pred[pred.Model==pred.Model.iloc[0]],data);base['Score']=base.Prediction.astype(float)
        pred=pd.concat([pred,base],ignore_index=True)
        edges={};normal_parts={}
        template=pred[pred.Model==pred.Model.iloc[0]]
        for r,g in template.groupby('Record'):
            g=g.sort_values('time_s');g=g[current_normal(g,numeric)].reset_index(drop=True)
            edges[r]=links(g,numeric);normal_parts[r]=g[['sample_id','time_s']]
        for model,g in pred.groupby('Model'):
            print('Evaluating',target,model,flush=True)
            normal=[];alarms=[];contrib={}
            for r in RECORDS:
                full=g[g.Record.eq(r)].sort_values('time_s').reset_index(drop=True)
                norm=full[full.sample_id.isin(normal_parts[r].sample_id)].reset_index(drop=True) if r in normal_parts else full.iloc[:0].copy()
                normal.append(norm);edge=edges.get(r,np.array([],bool));t=norm.time_s.to_numpy();p=norm.Prediction.to_numpy()
                eps=episodes(norm,edge)
                alarms.extend(a|{'Record':r} for a in eps)
                contrib[r]={'Samples':len(norm),'Alert_Timestamps':int(p.sum()),'Exposure_s':float(np.diff(t)[edge].sum()),
                            'Alarm_Active_s':float((np.diff(t)[edge]*p[:-1][edge]).sum()),'Alarm_Starts':len(eps)}
                for population,part in [('full_target',full),('current_normal',norm)]:
                    timestamp.append({'Target':target,'Model':model,'Record':r,'Population':population,
                                      **metrics(part.Label,part.Score,part.Prediction)})
            normal=pd.concat(normal,ignore_index=True)
            for window in WINDOWS:
                eligible=eligible_events(events,normal,numeric,window)
                matches,used=match_events(eligible,normal,alarms,window,'episode_start_one_to_one')
                prior=saved[(saved.Target==target)&(saved.Model==model)&saved.Window_s.eq(window)]
                columns=['Record','Event_ID','Evaluation_Status','Detected','First_Alert_s','Lead_Time_s','Alarm_ID']
                left=matches[columns].reset_index(drop=True).copy();right=prior[columns].reset_index(drop=True).copy()
                left['Alarm_ID']=pd.to_numeric(left.Alarm_ID);right['Alarm_ID']=pd.to_numeric(right.Alarm_ID)
                pd.testing.assert_frame_equal(left,right,check_dtype=False)
                matches['Target']=target;matches['Model']=model;event_rows.append(matches)
                for subset in ['all','91_92']:
                    selected=matches if subset=='all' else matches[matches.Has_91_92]
                    valid=selected[selected.Evaluation_Status.eq('eligible')]
                    detected_leads=valid.loc[valid.Detected.eq(True),['Record','Lead_Time_s']]
                    lead_parts.append(detected_leads.assign(Target=target,Model=model,Window_s=window,Subset=subset))
                    for r in RECORDS:
                        v=valid[valid.Record.eq(r)];leads=v.loc[v.Detected.eq(True),['Record','Lead_Time_s']]
                        records.append({'Target':target,'Model':model,'Window_s':window,'Subset':subset,'Record':r,
                                        'Eligible_Events':len(v),'Detected':int(v.Detected.eq(True).sum()),'Missed_Events':int(v.Detected.eq(False).sum()),
                                        'Excluded_Events':int(selected.Record.eq(r).sum())-len(v),**contrib[r],**lead_summary(leads.Lead_Time_s.to_numpy()),
                                        'Off_Event_Alarm_Starts':off_event_alarms([a for a in alarms if a['Record']==r],events,window),
                                        'Unmatched_Alarm_Starts':sum(a['Record']==r and (r,a['Alarm_ID']) not in used for a in alarms)})
    per=pd.DataFrame(records)
    per['Sensitivity']=divide(per.Detected,per.Eligible_Events);per['False_Alarm_Starts_Per_Hour']=divide(3600*per.Off_Event_Alarm_Starts,per.Exposure_s)
    per['Alarm_Active_Fraction']=divide(per.Alarm_Active_s,per.Exposure_s);per['Pulse_Alert_Fraction']=divide(per.Alert_Timestamps,per.Samples)
    # NaN rates are expected for zero-opportunity records; preserve their zero counts.
    per.to_csv(output/'event_per_recording.csv',index=False)
    match=pd.concat(event_rows,ignore_index=True);match.to_csv(output/'event_matches.csv',index=False)
    pd.DataFrame(timestamp).to_csv(output/'timestamp_per_recording.csv',index=False)
    pd.DataFrame(thresholds).to_csv(output/'threshold_selection_audit.csv',index=False)
    leads=pd.concat(lead_parts,ignore_index=True);leads.to_csv(output/'detected_event_leads.csv',index=False)
    weights=bootstrap_weights(repetitions,seed);pd.DataFrame(weights,columns=[f'Record_{r:02d}' for r in RECORDS]).to_csv(output/'bootstrap_recording_weights.csv.gz',index_label='Replicate',compression='gzip')
    summary=[];cis=[];boot={}
    counts=['Eligible_Events','Detected','Missed_Events','Excluded_Events','Samples','Alert_Timestamps','Exposure_s','Alarm_Active_s','Alarm_Starts','Off_Event_Alarm_Starts','Unmatched_Alarm_Starts']
    for key,g in per.groupby(['Target','Model','Window_s','Subset'],sort=True):
        target,model,window,subset=key
        ll=leads[(leads.Target==target)&(leads.Model==model)&leads.Window_s.eq(window)&(leads.Subset==subset)]
        totals={c:g[c].sum() for c in counts}
        totals.update(Sensitivity=float(divide(totals['Detected'],totals['Eligible_Events'])),
                      False_Alarm_Starts_Per_Hour=float(divide(3600*totals['Off_Event_Alarm_Starts'],totals['Exposure_s'])),
                      Alarm_Active_Fraction=float(divide(totals['Alarm_Active_s'],totals['Exposure_s'])),
                      Pulse_Alert_Fraction=float(divide(totals['Alert_Timestamps'],totals['Samples'])))
        summary.append(dict(zip(['Target','Model','Window_s','Subset'],key))|totals|lead_summary(ll.Lead_Time_s.to_numpy())|
                       {'Recordings':30,'Nonzero_Exposure_Recordings':int(g.Exposure_s.gt(0).sum()),'Eligible_Event_Recordings':int(g.Eligible_Events.gt(0).sum())})
        boot[key]=bootstrap_metrics(g[counts+['Record']],ll,weights)
        for metric,values in boot[key].items():
            cis.append(dict(Target=target,Model=model,Window_s=window,Subset=subset,Metric=metric,**interval(values),Replicates=repetitions))
    summary=pd.DataFrame(summary);summary.to_csv(output/'event_summary.csv',index=False)
    pd.DataFrame(cis).to_csv(output/'recording_bootstrap_intervals.csv',index=False)
    paired=[]
    for target in config.TARGETS:
        for window in WINDOWS:
            for subset in ['all','91_92']:
                for a,b in PAIRS:
                    for metric,values in boot[(target,a,window,subset)].items():
                        other=boot[(target,b,window,subset)][metric]
                        point=summary.set_index(['Target','Model','Window_s','Subset'])
                        delta=point.loc[(target,a,window,subset),metric]-point.loc[(target,b,window,subset),metric]
                        paired.append(dict(Target=target,Window_s=window,Subset=subset,Model_A=a,Model_B=b,Metric=metric,Difference_A_Minus_B=delta,**interval(values-other),Replicates=repetitions))
    pd.DataFrame(paired).to_csv(output/'paired_recording_bootstrap.csv',index=False)
    # Full timestamp pooled metrics are recalculated from the same saved decisions.
    pooled=[]
    for target in config.TARGETS:
        d=config.ROOT/'results/final'/target
        p=pd.read_csv(d/'per_record_predictions.csv');data=pd.read_csv(d/'causal_dataset.csv')
        b=baseline_predictions(p[p.Model==p.Model.iloc[0]],data);b['Score']=b.Prediction.astype(float);p=pd.concat([p,b])
        for model,g in p.groupby('Model'):
            pooled.append(dict(Target=target,Model=model,Population='full_target',**metrics(g.Label,g.Score,g.Prediction)))
    pd.DataFrame(pooled).to_csv(output/'timestamp_pooled.csv',index=False)
    policy={'outer':'leave-one-recording-out (saved predictions); 30 source records, 26 onset-eligible folds',
            'threshold_objective':'training-record grouped OOF timestamp recall >=0.92; no alarm-budget optimization',
            'alarm_burden_selection_performed':False,'models_refitted':False,'thresholds_changed':False,
            'bootstrap_repetitions':repetitions,'bootstrap_seed':seed,'bootstrap_unit':'whole recording; all 30, unstratified, paired weights; conditional on fitted CV predictions',
            'false_alarm_reference':'all observed qualifying onsets, including excluded events; severe subsets retain all-event alarm burden',
            'windows_s':WINDOWS,'patient_identity':'no verified IDs or mapping found; recording-separated, not proven patient-independent',
            'inputs':inputs,'code_hashes':{str(p.relative_to(config.ROOT)):digest(p) for p in [Path(__file__),config.ROOT/'src/reliability_evaluation.py',config.ROOT/'src/evaluation.py']}}
    policy['outputs']={p.name:digest(p) for p in output.iterdir() if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(policy,indent=2))
    print(json.dumps(verify_output(output),indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--bootstrap-repetitions',type=int,default=2000);p.add_argument('--seed',type=int,default=20261014)
    p.add_argument('--verify',action='store_true');a=p.parse_args()
    try:
        if a.verify:print(json.dumps(verify_output(a.output),indent=2))
        else:run(a.output,a.bootstrap_repetitions,a.seed)
    except (ValueError,FileNotFoundError) as e:p.exit(2,str(e)+'\n')

if __name__=='__main__':main()
