"""Independent saved-prediction audit; no model fitting or threshold selection.

A window is [onset-W,onset). Primary matching uses new alarm-episode starts,
chronological earliest-event assignment, at most one credit per episode and event.
The reusable-timestamp comparison explicitly permits repeated event credit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import config

WINDOWS = (5, 10, 15, 30)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_window(nt, s, start, end, gap=1.01):
    """Observed numeric coverage, including both boundary gaps; no interpolation."""
    nt, s = np.asarray(nt, float), np.asarray(s, float)
    if len(nt) != len(s) or not np.isfinite(nt).all() or (np.diff(nt) <= 0).any():
        raise ValueError('Numeric times must be finite, ordered, unique and aligned')
    if not len(nt) or start > end or start < nt[0] or end > nt[-1]:
        return False
    mask = (nt >= start) & (nt <= end)
    v = s[mask]
    return bool(len(v) and np.isfinite(v).all() and ((v > 0) & (v <= 100)).all()
                and np.max(np.diff(np.r_[start, nt[mask], end]), initial=0) <= gap)


def raw_events(numeric):
    """Independent loop implementation of the observed >94 -> <=94 episode rule."""
    rows = []
    for record, (nt, s) in sorted(numeric.items()):
        i = 0
        while i < len(nt):
            if not np.isfinite(s[i]) or not 0 < s[i] <= 94:
                i += 1; continue
            first = i; i += 1
            while i < len(nt) and np.isfinite(s[i]) and 0 < s[i] <= 94 and nt[i]-nt[i-1] <= 1.01:
                i += 1
            qualifies = first > 0 and np.isfinite(s[first-1]) and 94 < s[first-1] <= 100 and nt[first]-nt[first-1] <= 1.01
            rows.append({'Record': record, 'Event_ID': first, 'Onset_s': nt[first], 'End_s': nt[i-1],
                         'Qualifying_Onset': qualifies, 'Minimum_SpO2': min(s[first:i]),
                         'Has_91_92': bool(any(x in (91, 92) for x in s[first:i]))})
    return pd.DataFrame(rows)


def validate_predictions(pred):
    needed = {'Model', 'Record', 'sample_id', 'time_s', 'Prediction', 'Label'}
    if needed-set(pred):raise ValueError('Missing prediction columns')
    if not pred.Prediction.isin([0,1]).all() or not pred.Label.isin([0,1]).all():raise ValueError('Nonbinary decisions/labels')
    reference = None
    for _, g in pred.groupby('Model'):
        if g.sample_id.duplicated().any():raise ValueError('Duplicate sample IDs')
        rows = g.sort_values('sample_id')[['sample_id','Record','time_s','Label']].reset_index(drop=True)
        if reference is None:reference = rows
        else:pd.testing.assert_frame_equal(rows, reference)
        for _, r in g.groupby('Record'):
            t = r.sort_values('time_s').time_s.to_numpy()
            if not np.isfinite(t).all() or (np.diff(t) <= 0).any():raise ValueError('Duplicate/invalid pulse timestamps')


def current_normal(g, numeric):
    nt, s = numeric[int(g.Record.iloc[0])]
    t = g.time_s.to_numpy();j = np.searchsorted(nt,t,side='right')-1
    safe = np.maximum(j,0)
    return (j>=0) & (t-nt[safe]<=1.01) & np.isfinite(s[safe]) & (s[safe]>94) & (s[safe]<=100)


def links(g, numeric=None):
    """Edges define observed exposure. Numeric checks forbid hidden low/unknown bridges."""
    t = g.time_s.to_numpy();edge = np.diff(t)<=2
    if len(t)<2:return edge
    if numeric is not None:
        nt, s = numeric[int(g.Record.iloc[0])]
        for i in np.flatnonzero(edge):
            a,b = t[i:i+2];j = np.searchsorted(nt,a,side='right')-1
            interior = s[(nt>a)&(nt<=b)]
            edge[i] = (j>=0 and a-nt[j]<=1.01 and np.isfinite(s[j]) and 94<s[j]<=100
                       and np.isfinite(interior).all() and ((interior>94)&(interior<=100)).all()
                       and valid_window(nt,s,a,b+30))
    return edge


def episodes(g, edge=None):
    """Emit one alarm per run; no resetting or clipping starts at event-window edges."""
    g = g.sort_values('time_s').reset_index(drop=True)
    if edge is None:edge = links(g)
    result = [];i = 0
    while i<len(g):
        if g.Prediction.iloc[i] == 0:i+=1;continue
        start = i;i+=1
        while i<len(g) and edge[i-1] and g.Prediction.iloc[i]==1:i+=1
        part = g.iloc[start:i]
        result.append({'Alarm_ID':len(result),'Start_s':float(part.time_s.iloc[0]),'End_s':float(part.time_s.iloc[-1]),
                       'Positive_Times':part.time_s.to_numpy(), 'Legacy_False':not part.Label.eq(1).any()})
    return result


def burden(g, numeric=None):
    g = g.sort_values('time_s').reset_index(drop=True);edge = links(g,numeric)
    dt = np.diff(g.time_s.to_numpy());p = g.Prediction.to_numpy()
    exposure = float(dt[edge].sum());eps = episodes(g,edge)
    fp = int(((g.Label==0)&(g.Prediction==1)).sum())
    false = sum(e['Legacy_False'] for e in eps)
    return {'Samples':len(g),'FP_Timestamps':fp,'Alert_Timestamps':int(p.sum()),'Alert_Fraction':float(p.mean()) if len(g) else np.nan,
            'Alert_Episodes':len(eps),'False_Episodes_Target_Run':false,'Exposure_s':exposure,
            'Alert_Time_s':float((dt[edge]*p[:-1][edge]).sum()),'False_Episodes_Per_Hour':false/(exposure/3600) if exposure else np.nan}, eps


def eligible_events(events, g, numeric, window):
    rows = []
    for event in events.to_dict('records'):
        onset = event['Onset_s'];record = event['Record'];nt,s = numeric[record]
        q = g[(g.Record==record)&(g.time_s>=onset-window)&(g.time_s<onset)]
        status = ('left_censored_or_unknown_prior' if not event['Qualifying_Onset'] else
                  'incomplete_raw_lead_in' if not valid_window(nt,s,onset-window,onset) else
                  'no_eligible_prediction_opportunity' if q.empty else 'eligible')
        rows.append({**event,'Window_s':window,'Evaluation_Status':status,'Opportunities':len(q),
                     'Available_Lead_Time_s':onset-q.time_s.min() if len(q) else np.nan})
    return pd.DataFrame(rows)


def off_event_alarms(alarm_list, events, window):
    """Alarm starts without a qualifying onset in (start,start+W].

    Uses all observed qualifying onsets, including events excluded from sensitivity.
    Redundant alerts near a real onset are not mislabeled as off-event false alarms.
    """
    qualifying = events[events.Qualifying_Onset]
    return sum(not any(e.Record == a['Record'] and a['Start_s'] < e.Onset_s <= a['Start_s'] + window
                       for e in qualifying.itertuples()) for a in alarm_list)


def match_events(eligible, g, alarm_list, window, method):
    """Assign chronological episodes to earliest eligible event, never twice."""
    used = set();rows = []
    for event in eligible.sort_values(['Record','Onset_s']).to_dict('records'):
        r = event['Record'];onset = event['Onset_s'];lo = onset-window
        result = {**event,'Method':method,'Detected':np.nan,'First_Alert_s':np.nan,'Lead_Time_s':np.nan,'Alarm_ID':None}
        if event['Evaluation_Status']=='eligible':
            if method=='episode_start_one_to_one':
                candidates = [a for a in alarm_list if a['Record']==r and lo<=a['Start_s']<onset and (r,a['Alarm_ID']) not in used]
                selected = min(candidates,key=lambda a:a['Start_s']) if candidates else None
                if selected:
                    used.add((r,selected['Alarm_ID']));result['Alarm_ID']=selected['Alarm_ID'];result['First_Alert_s']=selected['Start_s']
            elif method=='any_positive_timestamp_reusable':
                q = g[(g.Record==r)&(g.time_s>=lo)&(g.time_s<onset)&g.Prediction.eq(1)]
                if len(q):result['First_Alert_s']=q.time_s.min()
            else:raise ValueError('Unknown matching method')
            result['Detected'] = bool(np.isfinite(result['First_Alert_s']))
            if result['Detected']:result['Lead_Time_s']=onset-result['First_Alert_s']
        rows.append(result)
    return pd.DataFrame(rows),used


def independent_historical(pred, events, numeric, target, old_dense=False):
    """Recompute historical calculations without calling the production evaluator."""
    rows = []
    for model,g in pred.groupby('Model'):
        for e in events.to_dict('records'):
            nt,s = numeric[e['Record']];onset = e['Onset_s'];prior = (nt>=onset-30)&(nt<=onset)
            # Historical archived dense implementation excluded onset from missingness check.
            pv = s[prior if not old_dense else prior & (nt<onset)]
            status = 'eligible'
            if not e['Qualifying_Onset']:status='left_censored_or_unknown_prior'
            elif onset-30<nt[0]:status='insufficient_lead_in' if old_dense else 'incomplete_raw_lead_in'
            elif not np.isfinite(pv).all():status='missing_lead_in' if old_dense else 'incomplete_raw_lead_in'
            elif not old_dense and ((pv<=0)|(pv>100)).any():status='incomplete_raw_lead_in'
            elif not old_dense and np.diff(nt[prior]).max(initial=0)>1.01:status='incomplete_raw_lead_in'
            q = g[(g.Record==e['Record'])&(g.time_s>=onset-30)&(g.time_s<onset)]
            if not old_dense and target=='future_low':q=q[q.spo2>94]
            if old_dense:
                t=q.time_s.to_numpy();covered=len(t)>0 and t.min()<=onset-28 and t.max()>=onset-2 and np.diff(t).max(initial=0)<=2
                if status=='eligible' and not covered:status='incomplete_evaluable_pulse_coverage'
            elif status=='eligible' and q.empty:status='no_eligible_prediction_opportunity'
            q=q[q.Prediction==1];first=q.time_s.min() if status=='eligible' and len(q) else np.nan
            # Recover true prior episode start for the selected positive timestamp.
            containing = [a for a in episodes(g[g.Record==e['Record']]) if np.isfinite(first) and a['Start_s']<=first<=a['End_s']]
            start=containing[0]['Start_s'] if containing else np.nan
            rows.append({**e,'Model':model,'Evaluation_Status':status,'Detected':bool(len(q)) if status=='eligible' else np.nan,
                         'First_Positive_s':first,'Lead_Time_s':onset-first,'Actual_Episode_Start_s':start,
                         'Episode_Already_Active_At_Window_Start':bool(start<onset-30) if np.isfinite(start) else False})
    return pd.DataFrame(rows)


def baseline_predictions(template, dataset):
    result = []
    for name in ['Always_Alert','Never_Alert','Periodic_10s_1s','Pulse_Rate_GT100']:
        g=template.copy();g['Model']=name
        if name=='Always_Alert':p=np.ones(len(g),int)
        elif name=='Never_Alert':p=np.zeros(len(g),int)
        elif name=='Periodic_10s_1s':p=(np.mod(g.time_s,10)<1).astype(int)
        else:p=(dataset.set_index('sample_id').loc[g.sample_id,'Pulse_Rate_BPM'].to_numpy()>100).astype(int)
        g['Prediction']=p;result.append(g)
    return pd.concat(result,ignore_index=True)


def run(output, inputs):
    output=Path(output)
    if output.exists():raise ValueError('Use a new output directory; audit outputs are never overwritten')
    output.mkdir(parents=True)
    numeric={};sources={}
    for path in sorted(config.DATA_DIR.glob('bidmc_*_Numerics.csv')):
        n=pd.read_csv(path);n.columns=n.columns.str.strip();numeric[int(path.stem.split('_')[1])]=(n['Time [s]'].to_numpy(),n.SpO2.to_numpy());sources[str(path.relative_to(config.ROOT))]=digest(path)
    events=raw_events(numeric);events.to_csv(output/'independent_raw_events.csv',index=False)
    summaries=[];event_parts=[];alarm_parts=[];burdens=[];checks=[];historical=[]
    for directory in inputs:
        directory=Path(directory);manifest=json.loads((directory/'run_manifest.json').read_text());target=manifest['target_definition']['name']
        print('Auditing '+target,flush=True)
        expected=manifest['target_definition']
        if expected['horizon_s']!=30 or expected['threshold_spo2']!=94 or expected['maximum_numeric_gap_s']!=1.01:raise ValueError('Audit protocol requires saved 30s/94%/1.01s contract')
        pred=pd.read_csv(directory/'per_record_predictions.csv');data=pd.read_csv(directory/'causal_dataset.csv');validate_predictions(pred)
        for name in ['run_manifest.json','per_record_predictions.csv','causal_dataset.csv','event_metrics.csv','model_comparison.csv','ground_truth_events.csv']:
            sources[str((directory/name).relative_to(config.ROOT))]=digest(directory/name)
        saved_events=pd.read_csv(directory/'ground_truth_events.csv')
        cols=['Record','Event_ID','Onset_s','End_s','Qualifying_Onset','Minimum_SpO2','Has_91_92']
        pd.testing.assert_frame_equal(events[cols].reset_index(drop=True),saved_events[cols].reset_index(drop=True),check_dtype=False)
        hist=independent_historical(pred,events,numeric,target);saved=pd.read_csv(directory/'event_metrics.csv')
        joined=hist.merge(saved,on=['Record','Event_ID','Model'],suffixes=('_independent','_saved'),validate='one_to_one')
        for col in ['Evaluation_Status','Detected','Lead_Time_s']:
            a,b=joined[col+'_independent'],joined[col+'_saved']
            if col=='Lead_Time_s':equal=np.isclose(a,b,equal_nan=True)
            else:equal=a.fillna('undefined').eq(b.fillna('undefined'))
            checks.append({'Target':target,'Calculation':'historical_event_'+col,'Rows':len(a),'Discrepancies':int((~equal).sum())})
        hist['Target']=target;historical.append(hist)
        template=pred[pred.Model==pred.Model.iloc[0]]
        pred=pd.concat([pred,baseline_predictions(template,data)],ignore_index=True)
        for model,g in pred.groupby('Model'):
            print(target+' '+model,flush=True)
            full=[];normal=[];norm_parts=[];all_alarms=[]
            for record,part in g.groupby('Record'):
                part=part.sort_values('time_s').reset_index(drop=True)
                stats,_=burden(part);full.append(stats)
                norm=part[current_normal(part,numeric)].reset_index(drop=True);norm_parts.append(norm)
                stats,eps=burden(norm,numeric);normal.append(stats)
                for alarm in eps:all_alarms.append({**alarm,'Record':record})
            def pooled(parts,population):
                total={k:sum(x[k] for x in parts) for k in ['Samples','FP_Timestamps','Alert_Timestamps','Alert_Episodes','False_Episodes_Target_Run','Exposure_s','Alert_Time_s']}
                total.update(Target=target,Model=model,Population=population,Alert_Fraction=total['Alert_Timestamps']/total['Samples'] if total['Samples'] else np.nan,
                             Time_Weighted_Alert_Fraction=total['Alert_Time_s']/total['Exposure_s'] if total['Exposure_s'] else np.nan,
                             False_Episodes_Per_Hour=total['False_Episodes_Target_Run']/(total['Exposure_s']/3600) if total['Exposure_s'] else np.nan)
                burdens.append(total);return total
            f=pooled(full,'full_target_historical_edges');n=pooled(normal,'current_normal_strict_edges')
            if model in config.MODEL_CONFIGS:
                c=pd.read_csv(directory/'model_comparison.csv').set_index('Model').loc[model]
                for field,old in [('FP_Timestamps','FP'),('Alert_Fraction','Alert_Fraction'),('False_Episodes_Target_Run','False_Alert_Episodes'),('Exposure_s','Exposure_Hours')]:
                    expected=c[old]*(3600 if field=='Exposure_s' else 1)
                    checks.append({'Target':target,'Calculation':model+'/'+field,'Rows':f['Samples'],'Discrepancies':int(not np.isclose(f[field],expected))})
            normal_g=pd.concat(norm_parts,ignore_index=True)
            for window in WINDOWS:
                eligible=eligible_events(events,normal_g,numeric,window)
                off_event=off_event_alarms(all_alarms,events,window)
                for method in ['episode_start_one_to_one','any_positive_timestamp_reusable']:
                    matches,used=match_events(eligible,normal_g,all_alarms,window,method);matches['Model']=model;matches['Target']=target;event_parts.append(matches)
                    valid=matches[matches.Evaluation_Status=='eligible']
                    if method=='episode_start_one_to_one':false=len(all_alarms)-len(used)
                    else:
                        false=0
                        for alarm in all_alarms:
                            ev=valid[valid.Record==alarm['Record']]
                            overlap=any(((alarm['Positive_Times']>=e.Onset_s-window)&(alarm['Positive_Times']<e.Onset_s)).any() for e in ev.itertuples())
                            false+=int(not overlap)
                    for subset in ['all','91_92']:
                        v=valid if subset=='all' else valid[valid.Has_91_92];lead=v.Lead_Time_s.dropna()
                        summaries.append({'Target':target,'Model':model,'Window_s':window,'Method':method,'Subset':subset,'Eligible_Events':len(v),'Detected':int(v.Detected.eq(True).sum()),
                                          'Sensitivity':v.Detected.eq(True).mean() if len(v) else np.nan,'Excluded_Events':len(events if subset=='all' else events[events.Has_91_92])-len(v),
                                          'Unmatched_Alarm_Episodes_All_Events':false,'Exposure_Hours':n['Exposure_s']/3600,'Off_Event_Alarm_Episodes':off_event,'Off_Event_False_Alarms_Per_Hour':off_event/(n['Exposure_s']/3600) if n['Exposure_s'] else np.nan,
                                          'Unmatched_Alarms_Per_Hour_All_Events':false/(n['Exposure_s']/3600) if n['Exposure_s'] else np.nan,
                                          'Lead_Min_s':lead.min(),'Lead_Q25_s':lead.quantile(.25),'Lead_Median_s':lead.median(),'Lead_Q75_s':lead.quantile(.75),'Lead_Max_s':lead.max(),
                                          'Alert_Fraction':n['Alert_Fraction'],'Time_Weighted_Alert_Fraction':n['Time_Weighted_Alert_Fraction']})
                    for alarm in all_alarms:
                        alarm_parts.append({k:v for k,v in alarm.items() if k!='Positive_Times'}|{'Target':target,'Model':model,'Window_s':window,'Method':method,'Matched_Once':(alarm['Record'],alarm['Alarm_ID']) in used if method=='episode_start_one_to_one' else np.nan})
    # Original 13-event calculation, including its unrestricted current-low pulse windows.
    olddir=config.OUTPUT_DIR;old=pd.read_csv(olddir/'per_record_predictions.csv');legacy=independent_historical(old,events,numeric,'future_low',old_dense=True)
    saved=pd.read_csv(olddir/'event_metrics.csv');saved['Event_ID']=saved.Event_ID.str.split('_').str[-1].astype(int)
    j=legacy.merge(saved,on=['Record','Event_ID','Model'],suffixes=('_independent','_saved'),validate='one_to_one')
    for col in ['Evaluation_Status','Detected','Lead_Time_s']:
        a,b=j[col+'_independent'],j[col+'_saved'];eq=np.isclose(a,b,equal_nan=True) if col=='Lead_Time_s' else a.fillna('undefined').eq(b.fillna('undefined'))
        checks.append({'Target':'original_validated','Calculation':col,'Rows':len(a),'Discrepancies':int((~eq).sum())})
    legacy.to_csv(output/'original_lead_time_recomputation.csv',index=False)
    sources[str((olddir/'per_record_predictions.csv').relative_to(config.ROOT))]=digest(olddir/'per_record_predictions.csv')
    for name,parts in [('historical_event_recomputation',historical),('window_event_matches',event_parts)]:pd.concat(parts,ignore_index=True).to_csv(output/(name+'.csv'),index=False)
    for name,rows in [('window_summary',summaries),('alarm_episodes',alarm_parts),('timestamp_burden',burdens),('independent_checks',checks)]:pd.DataFrame(rows).to_csv(output/(name+'.csv'),index=False)
    receipt={'inputs':sources,'code_sha256':digest(__file__),'windows_s':WINDOWS,'false_alarm_definition':'new episode starts with no observed qualifying onset in (start,start+W]; unmatched counts additionally include redundant alarms; excluded events are not called false',
             'matching':'chronological earliest eligible event; each new alarm episode and event used at most once','baselines':'always, never, deterministic 1s per 10s at phase zero, pulse rate >100; no fitted parameters','configuration':{'label_horizon_s':30,'threshold_spo2':94,'numeric_gap_s':1.01,'alarm_gap_s':2},'patient_identity':'unverified; recording-level CV only','models_refitted':False,'thresholds_changed':False}
    receipt['outputs']={p.name:digest(p) for p in output.glob('*.csv')};(output/'manifest.json').write_text(json.dumps(receipt,indent=2))
    print(pd.DataFrame(checks).to_string(index=False));print('Saved reliability audit:',output)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inputs',nargs='+',type=Path,default=[config.ROOT/'results/final/future_low',config.ROOT/'results/final/new_onset'])
    args=parser.parse_args();run(args.output,args.inputs)

if __name__=='__main__':main()
