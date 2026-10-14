"""Frozen nested alarm-budget sensitivity experiment; threshold-only, no model refits."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from . import config
from .evaluation import metrics
from .local_30_evaluation import (RECORDS, verify_splits, bootstrap_weights, bootstrap_metrics,
    interval, divide, lead_summary)
from .reliability_evaluation import (digest, raw_events, current_normal, links, episodes,
    eligible_events, match_events, off_event_alarms)

ROOT = config.ROOT/'results/alarm_budget_experiment'


def validate_training(data, oof, allowed, outer_test):
    """Require the actual fold-specific OOF array, never full-data/demo OOF scores."""
    allowed=set(allowed)
    if outer_test in allowed or not set(data.Record)<=allowed or not set(oof.Record)<=allowed:
        raise ValueError('Outer-test leakage into selection')
    if not oof.Outer_Test_Record.eq(outer_test).all():raise ValueError('Wrong outer-fit provenance')
    if oof.sample_id.duplicated().any() or set(oof.sample_id)!=set(data.sample_id):raise ValueError('Incomplete/duplicated OOF training rows')
    aligned=data.set_index('sample_id').loc[oof.sample_id]
    for c in ['Record','time_s','Label']:
        if not np.array_equal(aligned[c].to_numpy(),oof[c].to_numpy()):raise ValueError('Training provenance mismatch: '+c)
    if not np.isfinite(oof.Score).all():raise ValueError('Invalid training scores')
    pos=np.sort(oof.loc[oof.Label.eq(1),'Score'].to_numpy())[::-1]
    historical=float(pos[int(np.ceil(.92*len(pos)))-1])
    if not np.isclose(oof.Threshold,historical,rtol=1e-12,atol=1e-12).all():raise ValueError('Historical OOF threshold provenance failure')
    return historical


def topology(frame, events, numeric, allowed):
    """Only scoped recordings enter raw eligibility/exposure and event calculations."""
    allowed=set(allowed)
    if not set(frame.Record)<=allowed or not set(events.Record)<=allowed or not set(numeric)<=allowed:
        raise ValueError('Out-of-partition raw/event/row data supplied')
    parts=[];prev=[];duration=[]
    for record,g in frame.groupby('Record',sort=True):
        g=g.sort_values('time_s');g=g[current_normal(g,numeric)].reset_index(drop=True)
        e=links(g,numeric);parts.append(g)
        prev.extend(np.r_[False,e] if len(g) else [])
        duration.extend(np.r_[np.diff(g.time_s.to_numpy())*e,0.] if len(g) else [])
    normal=pd.concat(parts,ignore_index=True) if parts else frame.iloc[:0].copy()
    if len(normal) and (normal.duplicated(['Record','time_s']).any()):raise ValueError('Duplicate scoped timestamps')
    return normal,np.array(prev,bool),np.array(duration,float)


def evaluate_scoped_events(events, frame, numeric, alarms, window):
    """Retain schemas on zero-event records without changing the matching rule."""
    if events.empty:
        columns=list(events.columns)+['Window_s','Evaluation_Status','Opportunities','Available_Lead_Time_s',
            'Method','Detected','First_Alert_s','Lead_Time_s','Alarm_ID']
        return pd.DataFrame(columns=columns),set()
    eligible=eligible_events(events,frame,numeric,window)
    return match_events(eligible,frame,alarms,window,'episode_start_one_to_one')


def detection_count(starts_r, starts_t, eligible, window):
    """Array implementation of audited chronological earliest unused start matching."""
    used=set();detected=0
    for row in eligible.sort_values(['Record','Onset_s']).itertuples():
        if row.Evaluation_Status!='eligible':continue
        candidates=np.flatnonzero((starts_r==row.Record)&(starts_t>=row.Onset_s-window)&(starts_t<row.Onset_s))
        for candidate in candidates:
            if int(candidate) not in used:
                used.add(int(candidate));detected+=1;break
    return detected


def candidate_statistics(normal, previous_link, duration, events, eligible, quantiles, window=30.):
    scores=normal.Score.to_numpy(float);r=normal.Record.to_numpy();t=normal.time_s.to_numpy()
    if not len(scores) or not np.isfinite(scores).all() or duration.sum()<=0:raise ValueError('No finite/evaluable selection stream')
    thresholds=np.unique(np.r_[np.quantile(scores,quantiles),np.nextafter(scores.max(),np.inf)])
    false_at_start=np.ones(len(normal),bool)
    for record in np.unique(r):
        idx=np.flatnonzero(r==record);onsets=events.loc[events.Record.eq(record)&events.Qualifying_Onset,'Onset_s'].to_numpy()
        if len(onsets):
            j=np.searchsorted(onsets,t[idx],side='right');safe=np.minimum(j,len(onsets)-1)
            false_at_start[idx]=(j>=len(onsets))|(onsets[safe]>t[idx]+window)
    hours=float(duration.sum()/3600);rows=[]
    for threshold in thresholds:
        p=scores>=threshold
        start=p & ~(np.r_[False,p[:-1]] & previous_link)
        rows.append({'Threshold':float(threshold),'Threshold_hex':float(threshold).hex(),
                     'Detected':detection_count(r[start],t[start],eligible,window),
                     'Eligible_Events':int(eligible.Evaluation_Status.eq('eligible').sum()),
                     'Off_Event_Alarm_Starts':int((start&false_at_start).sum()),'Alarm_Starts':int(start.sum()),
                     'Exposure_Hours':hours,'False_Alarm_Starts_Per_Hour':float((start&false_at_start).sum()/hours),
                     'Alarm_Active_s':float(duration@p),'Alarm_Active_Fraction':float((duration@p)/duration.sum()),
                     'No_Training_Alerts':bool(not p.any())})
    return pd.DataFrame(rows)


def choose_candidate(candidates, budget, active_cap):
    feasible=candidates[(candidates.False_Alarm_Starts_Per_Hour<=budget+1e-12)&(candidates.Alarm_Active_Fraction<=active_cap+1e-12)]
    if feasible.empty:raise ValueError('No feasible training candidate; do not relax constraints')
    return feasible.sort_values(['Detected','Off_Event_Alarm_Starts','Alarm_Active_s','Threshold'],ascending=[False,True,True,False],kind='stable').iloc[0].to_dict()


def select_partition(data, oof, events, numeric, allowed, outer_test, protocol):
    """Public selector receives no outer scores/outcomes; fails closed on leakage."""
    historical=validate_training(data,oof,allowed,outer_test)
    joined=data.drop(columns=['Score'],errors='ignore').merge(oof[['sample_id','Score']],on='sample_id',validate='one_to_one')
    normal,previous,duration=topology(joined,events,numeric,allowed)
    eligible=eligible_events(events,normal,numeric,protocol['selection_window_s'])
    candidates=candidate_statistics(normal,previous,duration,events,eligible,protocol['candidate_quantiles'],protocol['selection_window_s'])
    selected=[]
    for budget in protocol['budgets_per_hour']:
        winner=choose_candidate(candidates,budget,protocol['active_fraction_cap'])
        selected.append(dict(Budget=budget,**winner,Training_Rows=len(data),Current_Normal_Training_Rows=len(normal),
                             Historical_Recall_Threshold=historical,Allowed_Records=','.join(map(str,sorted(allowed)))))
    return pd.DataFrame(selected),candidates


def verify_original_inputs(directory, manifest):
    if digest(directory/'causal_dataset.csv')!=manifest['dataset_sha256']:raise ValueError('Dataset fingerprint changed')
    for path,want in manifest['source_hashes'].items():
        if digest(path)!=want:raise ValueError('Raw source changed: '+path)
    # These modules control fitted scores. Reporting code is deliberately not refitted.
    for name in ['config.py','dataset.py','features.py','preprocessing.py','labeling.py','model.py','evaluation.py','provenance.py']:
        path=config.ROOT/'src'/name
        if digest(path)!=manifest['code_hashes'][str(path)]:raise ValueError('Fitting/generation source changed: '+name)


def run(output, selection_source=None):
    output=Path(output).resolve()
    if ROOT.resolve() not in output.parents:raise ValueError('New results must be under results/alarm_budget_experiment/')
    if output.exists():raise ValueError('Existing result directory is protected')
    protocol=json.loads((ROOT/'protocol.json').read_text());protocol_hash=digest(ROOT/'protocol.json')
    output.mkdir(parents=True)
    numeric={}
    for record in RECORDS:
        n=pd.read_csv(config.DATA_DIR/f'bidmc_{record:02d}_Numerics.csv');n.columns=n.columns.str.strip()
        numeric[record]=(n['Time [s]'].to_numpy(),n.SpO2.to_numpy())
    if len(list(config.DATA_DIR.glob('bidmc_*_Signals.csv')))!=30 or len(list(config.DATA_DIR.glob('bidmc_*_Numerics.csv')))!=30:raise ValueError('Local-30 scope mismatch')
    events=raw_events(numeric);events.to_csv(output/'raw_events.csv',index=False)
    selected=[];candidates=[];splits=[];inputs={};historical_cutoffs={}
    frozen=None;frozen_grid=None
    if selection_source is not None:
        selection_source=Path(selection_source).resolve()
        if ROOT.resolve() not in selection_source.parents:raise ValueError('Selection source must be inside experiment root')
        marker=json.loads((selection_source/'selection_complete.json').read_text())
        if marker['protocol_sha256']!=protocol_hash or marker['selected_thresholds_sha256']!=digest(selection_source/'selected_thresholds.csv'):
            raise ValueError('Frozen selection receipt mismatch')
        frozen=pd.read_csv(selection_source/'selected_thresholds.csv')
        frozen_grid=pd.read_csv(selection_source/'training_candidate_metrics.csv')
        for name in ['selected_thresholds.csv','training_candidate_metrics.csv','selection_complete.json']:
            path=selection_source/name;inputs[str(path.relative_to(config.ROOT))]=digest(path)
    # Phase 1: no held-out prediction values are read until ALL selections finish.
    for target in protocol['targets']:
        directory=config.ROOT/'results/final'/target;m=json.loads((directory/'run_manifest.json').read_text())
        verify_original_inputs(directory,m)
        data=pd.read_csv(directory/'causal_dataset.csv');verify_splits(m['recording_splits'],data.Record.unique())
        oof=pd.read_csv(directory/'validation_oof_predictions.csv.gz')
        expected={(model,s['test_record']) for model in protocol['models'] for s in m['recording_splits']}
        actual={(model,int(test)) for (model,test),g in oof.groupby(['Model','Outer_Test_Record'])}
        if actual!=expected:raise ValueError('Missing or unexpected outer/model OOF groups')
        splitmap={s['test_record']:s for s in m['recording_splits']}
        for (model,test),g in oof.groupby(['Model','Outer_Test_Record'],sort=True):
            allowed=set(splitmap[int(test)]['outer_train_records'])
            train=data[data.Record.isin(allowed)]
            ev=events[events.Record.isin(allowed)]
            num={r:numeric[r] for r in allowed}
            if frozen is None:
                chosen,grid=select_partition(train,g,ev,num,allowed,int(test),protocol)
            else:
                historical=validate_training(train,g,allowed,int(test))
                chosen=frozen[(frozen.Target==target)&frozen.Model.eq(model)&frozen.Outer_Test_Record.eq(test)].copy()
                grid=frozen_grid[(frozen_grid.Target==target)&frozen_grid.Model.eq(model)&frozen_grid.Outer_Test_Record.eq(test)].copy()
                if sorted(chosen.Budget)!=sorted(protocol['budgets_per_hour']):raise ValueError('Frozen budget scope mismatch')
                for row in chosen.itertuples():
                    if row.Allowed_Records!=','.join(map(str,sorted(allowed))) or not np.isclose(row.Historical_Recall_Threshold,historical):raise ValueError('Frozen provenance mismatch')
                    winner=choose_candidate(grid,row.Budget,protocol['active_fraction_cap'])
                    if winner['Threshold_hex']!=row.Threshold_hex:raise ValueError('Frozen winner differs from prespecified objective')
            for frame in [chosen,grid]:frame['Target']=target;frame['Model']=model;frame['Outer_Test_Record']=int(test)
            selected.append(chosen);candidates.append(grid)
            historical_cutoffs[(target,model,int(test))]=chosen.Historical_Recall_Threshold.iloc[0]
            print('SELECT',target,model,int(test),chosen[['Budget','Threshold','Detected','False_Alarm_Starts_Per_Hour','Alarm_Active_Fraction']].to_dict('records'),flush=True)
        splits.append({'Target':target,'recording_splits':m['recording_splits']})
        for name in ['run_manifest.json','causal_dataset.csv','validation_oof_predictions.csv.gz','per_record_predictions.csv']:
            path=directory/name;inputs[str(path.relative_to(config.ROOT))]=digest(path)
    selection=pd.concat(selected,ignore_index=True)
    if frozen is None:
        selection.to_csv(output/'selected_thresholds.csv',index=False)
        pd.concat(candidates,ignore_index=True).to_csv(output/'training_candidate_metrics.csv',index=False)
    else:
        if len(selection)!=len(frozen):raise ValueError('Frozen selections contain unverified extra rows')
        for name in ['selected_thresholds.csv','training_candidate_metrics.csv']:
            shutil.copyfile(selection_source/name,output/name)
    (output/'splits.json').write_text(json.dumps(splits,indent=2))
    selected_hash=digest(output/'selected_thresholds.csv')
    (output/'selection_complete.json').write_text(json.dumps({'protocol_sha256':protocol_hash,'selected_thresholds_sha256':selected_hash,
        'phase':'ALL selections completed before reading outer held-out prediction values','rows':len(selection)},indent=2))
    per=[];event_rows=[];alarm_rows=[];prediction_parts=[];lead_parts=[];threshold_status=[]
    for target in protocol['targets']:
        p=pd.read_csv(config.ROOT/'results/final'/target/'per_record_predictions.csv')
        for model in protocol['models']:
            for record,h in p[p.Model.eq(model)].groupby('Record'):
                want=historical_cutoffs[(target,model,int(record))]
                if not np.isclose(h.Threshold,want,rtol=1e-12,atol=1e-12).all():raise ValueError('Saved held-out threshold differs from OOF provenance')
        template=p[p.Model.eq(protocol['models'][0])]
        for model in protocol['models']+protocol['baselines']:
            model_rows=p[p.Model.eq(model)] if model in protocol['models'] else template
            for budget in protocol['budgets_per_hour']:
                for record in RECORDS:
                    h=model_rows[model_rows.Record.eq(record)].copy()
                    num={record:numeric[record]};ev=events[events.Record.eq(record)]
                    h,previous,duration=topology(h,ev,num,{record})
                    if model in protocol['models'] and len(h):
                        row=selection[(selection.Target==target)&selection.Model.eq(model)&selection.Outer_Test_Record.eq(record)&selection.Budget.eq(budget)]
                        if len(row)!=1:raise ValueError('No unique training-selected policy')
                        threshold=float.fromhex(row.Threshold_hex.iloc[0]);h['Prediction']=(h.Score>=threshold).astype(int)
                        origin='training_OOF';h['Threshold']=threshold
                    elif model in protocol['models']:
                        h['Prediction']=np.zeros(len(h),int);threshold=np.nan;origin='empty_target_record';h['Threshold']=threshold
                    else:
                        h['Prediction']=int(model=='Always_Alert');h['Score']=float(model=='Always_Alert')
                        threshold=.5;h['Threshold']=threshold;origin='fixed_baseline_not_budget_selected'
                    h['Model']=model;h['Target']=target;h['Budget']=budget
                    if len(h):prediction_parts.append(h)
                    eps=episodes(h,previous[1:]);alarms=[a|{'Record':record} for a in eps]
                    exposure=float(duration.sum());active=float(duration@h.Prediction.to_numpy());pulse=int(h.Prediction.sum())
                    tm=metrics(h.Label,h.Score,h.Prediction)
                    for w in protocol['evaluation_windows_s']:
                        matches,used=evaluate_scoped_events(ev,h,num,alarms,w)
                        matches['Target']=target;matches['Model']=model;matches['Budget']=budget;event_rows.append(matches)
                        valid=matches[matches.Evaluation_Status.eq('eligible')]
                        for subset in ['all','91_92']:
                            v=valid if subset=='all' else valid[valid.Has_91_92.eq(True)]
                            ll=v.loc[v.Detected.eq(True),['Record','Lead_Time_s']]
                            lead_parts.append(ll.assign(Target=target,Model=model,Budget=budget,Window_s=w,Subset=subset))
                            off=off_event_alarms(alarms,ev,w)
                            per.append({'Target':target,'Model':model,'Budget':budget,'Window_s':w,'Subset':subset,'Record':record,
                                'Policy_Origin':origin,'Threshold':threshold,'Eligible_Events':len(v),'Detected':int(v.Detected.eq(True).sum()),
                                'Missed_Events':int(v.Detected.eq(False).sum()),'Exposure_s':exposure,'Alarm_Active_s':active,
                                'Samples':len(h),'Alert_Timestamps':pulse,'Off_Event_Alarm_Starts':off,'Alarm_Starts':len(alarms),
                                'Unmatched_Alarm_Starts':len(alarms)-len(used),'TP':tm['TP'],'FP':tm['FP'],'FN':tm['FN'],'TN':tm['TN'],
                                **lead_summary(ll.Lead_Time_s.to_numpy())})
                        for alarm in alarms:
                            alarm_rows.append({k:v for k,v in alarm.items() if k!='Positive_Times'}|{'Target':target,'Model':model,'Budget':budget,'Window_s':w,'Matched_Once':(record,alarm['Alarm_ID']) in used})
                    rate=off_event_alarms(alarms,ev,protocol['selection_window_s'])/(exposure/3600) if exposure else np.nan
                    threshold_status.append(dict(Target=target,Model=model,Budget=budget,Record=record,Policy_Origin=origin,
                        Outer_False_Alarm_Starts_Per_Hour=rate,Outer_Active_Fraction=active/exposure if exposure else np.nan,
                        Outer_Budget_Violated=bool(rate>budget+1e-12) if exposure else None,
                        Outer_Active_Cap_Violated=bool(active/exposure>protocol['active_fraction_cap']+1e-12) if exposure else None))
    per=pd.DataFrame(per);per['Sensitivity']=divide(per.Detected,per.Eligible_Events)
    per['False_Alarm_Starts_Per_Hour']=divide(3600*per.Off_Event_Alarm_Starts,per.Exposure_s)
    per['Alarm_Active_Fraction']=divide(per.Alarm_Active_s,per.Exposure_s);per['Pulse_Alert_Fraction']=divide(per.Alert_Timestamps,per.Samples)
    per['Timestamp_Precision']=divide(per.TP,per.TP+per.FP);per['Timestamp_Recall']=divide(per.TP,per.TP+per.FN)
    per.to_csv(output/'per_recording.csv',index=False)
    pd.concat(event_rows,ignore_index=True).to_csv(output/'event_matches.csv',index=False)
    pd.DataFrame(alarm_rows).to_csv(output/'alarm_starts.csv',index=False)
    pd.concat(prediction_parts,ignore_index=True).to_csv(output/'outer_predictions.csv.gz',index=False,compression='gzip')
    pd.DataFrame(threshold_status).to_csv(output/'outer_constraint_status.csv',index=False)
    leads=pd.concat(lead_parts,ignore_index=True);leads.to_csv(output/'lead_times.csv',index=False)
    weights=bootstrap_weights(protocol['bootstrap_repetitions'],protocol['bootstrap_seed'])
    pd.DataFrame(weights,columns=[f'Record_{r:02d}' for r in RECORDS]).to_csv(output/'bootstrap_weights.csv.gz',index_label='Replicate',compression='gzip')
    summary=[];ci=[];boot={}
    counts=['Detected','Eligible_Events','Missed_Events','Exposure_s','Alarm_Active_s','Samples','Alert_Timestamps','Off_Event_Alarm_Starts','Alarm_Starts','Unmatched_Alarm_Starts','TP','FP','FN','TN']
    keys=['Target','Model','Budget','Window_s','Subset']
    for key,g in per.groupby(keys):
        target,model,budget,w,subset=key
        ll=leads[(leads.Target==target)&leads.Model.eq(model)&leads.Budget.eq(budget)&leads.Window_s.eq(w)&leads.Subset.eq(subset)]
        totals={c:g[c].sum() for c in counts}
        totals.update(Sensitivity=float(divide(totals['Detected'],totals['Eligible_Events'])),
            False_Alarm_Starts_Per_Hour=float(divide(3600*totals['Off_Event_Alarm_Starts'],totals['Exposure_s'])),
            Alarm_Active_Fraction=float(divide(totals['Alarm_Active_s'],totals['Exposure_s'])),
            Pulse_Alert_Fraction=float(divide(totals['Alert_Timestamps'],totals['Samples'])),
            Timestamp_Precision=float(divide(totals['TP'],totals['TP']+totals['FP'])),
            Timestamp_Recall=float(divide(totals['TP'],totals['TP']+totals['FN'])))
        summary.append(dict(zip(keys,key))|totals|lead_summary(ll.Lead_Time_s.to_numpy())|{'Budget_Selected_Model':model in protocol['models']})
        boot[key]=bootstrap_metrics(g[counts+['Record']],ll,weights)
        for metric,values in boot[key].items():ci.append(dict(zip(keys,key))|dict(Metric=metric,**interval(values),Replicates=len(weights)))
    summary=pd.DataFrame(summary);summary.to_csv(output/'summary.csv',index=False);pd.DataFrame(ci).to_csv(output/'bootstrap_intervals.csv',index=False)
    pairs=[];point=summary.set_index(keys)
    for target in protocol['targets']:
        for budget in protocol['budgets_per_hour']:
            for w in protocol['evaluation_windows_s']:
                for subset in ['all','91_92']:
                    a=(target,'ELM',budget,w,subset);b=(target,'Logistic',budget,w,subset)
                    for metric,x in boot[a].items():pairs.append(dict(Target=target,Budget=budget,Window_s=w,Subset=subset,Model_A='ELM',Model_B='Logistic',Metric=metric,
                        Difference_A_Minus_B=point.loc[a,metric]-point.loc[b,metric],**interval(x-boot[b][metric]),Replicates=len(weights)))
    pd.DataFrame(pairs).to_csv(output/'paired_bootstrap.csv',index=False)
    if digest(output/'selected_thresholds.csv')!=selected_hash or digest(ROOT/'protocol.json')!=protocol_hash:raise ValueError('Selection/protocol changed during outer evaluation')
    receipt={'protocol_sha256':protocol_hash,'inputs':inputs,'code_hashes':{str(p.relative_to(config.ROOT)):digest(p) for p in [Path(__file__),config.ROOT/'src/reliability_evaluation.py',config.ROOT/'src/local_30_evaluation.py']},
        'models_refitted':False,'threshold_only_nested_score_reuse':True,'rows_selected':len(selection),'patient_independence':'unverified',
        'outputs':{p.name:digest(p) for p in output.iterdir() if p.is_file()}}
    (output/'manifest.json').write_text(json.dumps(receipt,indent=2))
    print('Completed:',output,flush=True)


def verify(output):
    output=Path(output);m=json.loads((output/'manifest.json').read_text())
    if digest(ROOT/'protocol.json')!=m['protocol_sha256']:raise ValueError('Protocol changed')
    for name,want in m['inputs'].items():
        if digest(config.ROOT/name)!=want:raise ValueError('Input changed: '+name)
    for name,want in m['outputs'].items():
        if digest(output/name)!=want:raise ValueError('Output changed: '+name)
    for name,want in json.loads((ROOT/'preservation.json').read_text()).items():
        if digest(config.ROOT/name)!=want:raise ValueError('Historical artifact changed: '+name)
    selected=pd.read_csv(output/'selected_thresholds.csv');protocol=json.loads((ROOT/'protocol.json').read_text())
    if not (selected.False_Alarm_Starts_Per_Hour<=selected.Budget+1e-12).all() or not selected.Alarm_Active_Fraction.le(protocol['active_fraction_cap']+1e-12).all():raise ValueError('Training constraints failed')
    events=pd.read_csv(output/'event_matches.csv');det=events[events.Detected.eq(True)]
    if det.duplicated(['Target','Model','Budget','Window_s','Record','Alarm_ID']).any():raise ValueError('Duplicate event credit')
    if not ((det.Lead_Time_s>0)&(det.Lead_Time_s<=det.Window_s)).all():raise ValueError('Invalid matched lead')
    per=pd.read_csv(output/'per_recording.csv');summ=pd.read_csv(output/'summary.csv')
    for key,g in per.groupby(['Target','Model','Budget','Window_s','Subset']):
        if sorted(g.Record)!=list(RECORDS):raise ValueError('Incomplete recording roster')
        q=summ.set_index(['Target','Model','Budget','Window_s','Subset']).loc[key]
        for c in ['Detected','Missed_Events','Eligible_Events','Off_Event_Alarm_Starts','Exposure_s','Alarm_Active_s']:
            if not np.isclose(g[c].sum(),q[c]):raise ValueError('Aggregation mismatch '+c)
    return {'status':'passed','selected_policies':len(selected),'recording_rows':len(per),'event_rows':len(events),'summary_rows':len(summ),
            'protected_files':len(json.loads((ROOT/'preservation.json').read_text()))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'run02');p.add_argument('--verify',action='store_true');p.add_argument('--selection-source',type=Path);a=p.parse_args()
    if a.verify:print(json.dumps(verify(a.output),indent=2))
    else:run(a.output,a.selection_source)

if __name__=='__main__':main()
