"""Causal features, matched data and fixed event semantics; historical code unchanged."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from src import config
from src.preprocessing import canonical_waveform_times, extract_pulses
from src.features import add_features
from src.labeling import future_labels
from src.reliability_evaluation import raw_events, links, episodes, off_event_alarms
from src.alarm_budget_experiment import evaluate_scoped_events

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HISTORY = ['SpO2_Current','SpO2_Mean_10s','SpO2_Mean_30s','SpO2_Trend_10s','SpO2_Trend_30s','SpO2_Std_30s','SpO2_Age_s']
FEATURE_SETS = {'SpO2_history':HISTORY,'PPG':list(config.FEATURES),'Combined':list(config.FEATURES)+HISTORY}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def history_features(times, nt, saturation):
    """Each feature reads numeric observations at/before the prediction timestamp."""
    times,nt,s=np.asarray(times,float),np.asarray(nt,float),np.asarray(saturation,float)
    if len(nt)!=len(s) or not np.isfinite(nt).all() or (np.diff(nt)<=0).any() or not np.isfinite(times).all():
        raise ValueError('Finite ordered numeric clock and aligned saturation required')
    rows=[]
    for t in times:
        j=np.searchsorted(nt,t,side='right')
        current=s[j-1] if j and t-nt[j-1]<=1.01 and np.isfinite(s[j-1]) and 0<s[j-1]<=100 else np.nan
        stats=[]
        for window in [10.,30.]:
            lo=np.searchsorted(nt,t-window,side='left');v=s[lo:j];clock=nt[lo:j]
            ok=np.isfinite(v)&(v>0)&(v<=100);v=v[ok];clock=clock[ok]
            if len(v):
                x=clock-clock.mean();den=x@x
                stats.append((float(v.mean()),float(x@(v-v.mean())/den) if den>0 else 0.,float(v.std(ddof=1)) if len(v)>1 else 0.))
            else:stats.append((np.nan,np.nan,np.nan))
        rows.append([current,stats[0][0],stats[1][0],stats[0][1],stats[1][1],stats[1][2],t-nt[j-1] if np.isfinite(current) else np.nan])
    return pd.DataFrame(rows,columns=HISTORY)

def minimum_labels(times, nt, saturation):
    """Same maintained target, independently cross-checked by direct masks."""
    y,minimum,reason=future_labels(times,nt,saturation,horizon=30,threshold=94,max_gap=1.01)
    for t,label,m in zip(times,y,minimum):
        if np.isfinite(label):
            observed=np.asarray(saturation)[(np.asarray(nt)>t)&(np.asarray(nt)<=t+30)]
            if m!=observed.min() or int(label)!=int(observed.min()<=94):raise AssertionError('Independent future-min label mismatch')
    return y,minimum,reason

def load_numeric():
    numeric={}
    for record in range(1,31):
        n=pd.read_csv(ROOT/f'data/bidmc_{record:02d}_Numerics.csv');n.columns=n.columns.str.strip()
        numeric[record]=(n['Time [s]'].to_numpy(float),n.SpO2.to_numpy(float))
    return numeric

def build_data(out):
    """Rebuild only pulse/features/labels, not old scientific fits or output files."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    numeric=load_numeric();events=raw_events(numeric);inventory=[];parts=[];pulses=[]
    for record,(nt,s) in numeric.items():
        path=ROOT/f'data/bidmc_{record:02d}_Signals.csv'
        w=pd.read_csv(path);w.columns=w.columns.str.strip()
        clock=canonical_waveform_times(w['Time [s]'])
        pulse,quality=extract_pulses(clock,w.PLETH.to_numpy(float))
        d=pd.DataFrame(pulse);d['Record']=record;d['pulse_row']=np.arange(len(d));d=add_features(d)
        hist=history_features(d.time_s,nt,s);d=pd.concat([d.reset_index(drop=True),hist],axis=1)
        y,m,reason=minimum_labels(d.time_s.to_numpy(),nt,s);d['Label']=y;d['FutureMin']=m;d['Target_Status']=reason
        pulses.append(d)
        retained=d[d.Label.notna()&d.SpO2_Current.notna()].copy();retained.Label=retained.Label.astype(int);parts.append(retained)
        inventory.append(dict(Record=record,Waveform_Rows=len(w),Numeric_Rows=len(nt),Duration_s=float(clock[-1]-clock[0]),Waveform_fs_Hz=125.,Numeric_Median_dt_s=float(np.median(np.diff(nt))),Missing_PPG=int(w.PLETH.isna().sum()),Missing_SpO2=int((~np.isfinite(s)).sum()),Pulse_Rows=len(d),Original_Valid_Target_Rows=int(d.Label.notna().sum()),Common_Rows=len(retained),Common_Positive_Rows=int(retained.Label.sum()),Common_Current_Normal_Rows=int(retained.SpO2_Current.gt(94).sum()),Excluded_Unknown_Current=int((d.Label.notna()&d.SpO2_Current.isna()).sum()),Subject_ID_Available=False,**quality))
        print('PREPARED',record,len(d),len(retained),flush=True)
    data=pd.concat(parts,ignore_index=True);data['sample_id']=np.arange(len(data))
    pd.concat(pulses,ignore_index=True).to_csv(out/'all_pulses.csv.gz',index=False,compression='gzip')
    data.to_csv(out/'matched_dataset.csv.gz',index=False,compression='gzip');pd.DataFrame(inventory).to_csv(out/'data_inventory.csv',index=False)
    events=enrich_events(events,numeric,pd.concat(pulses,ignore_index=True),data)
    events.to_csv(out/'raw_event_audit.csv',index=False)
    # Compare regenerated original eligible pulse clock/labels to preserved held-out reference.
    original=pd.read_csv(ROOT/'results/final/future_low/per_record_predictions.csv')
    original=original[original.Model.eq('ELM')].sort_values(['Record','pulse_row']).reset_index(drop=True)
    regen=pd.concat(pulses,ignore_index=True);regen=regen[regen.Label.notna()].sort_values(['Record','pulse_row']).reset_index(drop=True)
    for c in ['Record','pulse_row','time_s','Label']:
        np.testing.assert_allclose(regen[c],original[c],rtol=0,atol=1e-10,err_msg='Historical target/clock parity: '+c)
    audit=dict(original_future_low_rows=len(regen),original_positives=int(regen.Label.sum()),common_rows=len(data),common_positives=int(data.Label.sum()),normal_rows=int(data.SpO2_Current.gt(94).sum()),raw_events=len(events),qualifying_onsets=int(events.Qualifying_Onset.sum()),sustained_qualifying_onsets=int((events.Qualifying_Onset&events.Sustained3).sum()),historical_label_clock_parity='passed',subject_independence='unverified')
    (out/'data_checks.json').write_text(json.dumps(audit,indent=2))
    return data,numeric,events

def enrich_events(events,numeric,pulses,data):
    rows=[]
    for event in events.to_dict('records'):
        r,o,end=int(event['Record']),event['Onset_s'],event['End_s'];nt,s=numeric[r]
        low=s[(nt>=o)&(nt<=end)];pre=(nt>=o-30)&(nt<o)
        pulse=pulses[pulses.Record.eq(r)&pulses.time_s.ge(o-30)&pulses.time_s.lt(o)]
        common=data[data.Record.eq(r)&data.time_s.ge(o-30)&data.time_s.lt(o)]
        normal=common[common.SpO2_Current.gt(94)]
        base=event|dict(Low_Readings=len(low),Observed_Low_Span_s=float(end-o),Nominal_Low_Support_s=float(len(low)),Isolated_Reading=len(low)==1,Sustained3=bool(len(low)>=3 and end-o>=2),Pre_Event_SpO2_Trajectory_JSON=json.dumps([[float(t),float(v) if np.isfinite(v) else None] for t,v in zip(nt[pre],s[pre])]),Eligible_Common_Windows=len(common),Current_Normal_Opportunities=len(normal),Detected_Pulses_Pre30s=len(pulse),Pulse_Rate_Median=float(pulse.Pulse_Rate_BPM.median()) if len(pulse) else np.nan,Pulse_Interval_CV=float(pulse.Pulse_Interval_s.std()/pulse.Pulse_Interval_s.mean()) if len(pulse)>1 else np.nan,PPG_Amplitude_Median=float(pulse.PPG_Amplitude.median()) if len(pulse) else np.nan,Validated_SQI_Available=False,Beat_Reference_Annotations_Available=False)
        match,_=evaluate_scoped_events(pd.DataFrame([event]),normal,numeric,[],30.)
        base['Original_Event_Eligibility']=match.Evaluation_Status.iloc[0];rows.append(base)
    return pd.DataFrame(rows)

def event_definition(events,name):
    if name=='original':return events.copy()
    if name=='sustained3':return events[events.Sustained3].copy()
    raise ValueError('Unknown event definition')

def topology(data,numeric):
    """Call only with one recording; preserve the established retrospective exposure."""
    d=data.sort_values('time_s');d=d[d.SpO2_Current.gt(94)].reset_index(drop=True)
    edge=links(d,numeric) if len(d) else np.array([],bool)
    duration=np.r_[np.diff(d.time_s.to_numpy())*edge,0.] if len(d) else np.array([],float)
    previous=np.r_[False,edge] if len(d) else np.array([],bool)
    return d,previous,duration

def evaluate_record(frame,events,numeric,definition='original'):
    r=int(frame.Record.iloc[0]);normal,previous,duration=topology(frame,numeric)
    selected=event_definition(events[events.Record.eq(r)],definition)
    alarms=[a|{'Record':r} for a in episodes(normal,previous[1:])]
    matches,used=evaluate_scoped_events(selected,normal,numeric,alarms,30.)
    eligible=matches[matches.Evaluation_Status.eq('eligible')]
    leads=eligible.loc[eligible.Detected.eq(True),'Lead_Time_s'].dropna()
    exposure=float(duration.sum());active=float(duration@normal.Prediction.to_numpy())
    result=dict(Record=r,Definition=definition,Eligible_Events=len(eligible),Detected=int(eligible.Detected.eq(True).sum()),Missed_Events=int(eligible.Detected.eq(False).sum()),Exposure_s=exposure,Alarm_Active_s=active,Off_Event_Alarm_Starts=off_event_alarms(alarms,selected,30.),Alarm_Starts=len(alarms),Unmatched_Alarm_Starts=len(alarms)-len(used),Samples=len(normal),Alert_Timestamps=int(normal.Prediction.sum()),Lead_Median_s=float(leads.median()) if len(leads) else np.nan)
    return result,matches

def fixed_forecasts(data):
    current=data.SpO2_Current.to_numpy(float)
    trend=np.minimum(current,current+30*data.SpO2_Trend_10s.to_numpy(float))
    return {'Persistence':current,'Trend10':np.clip(trend,0,100)}
