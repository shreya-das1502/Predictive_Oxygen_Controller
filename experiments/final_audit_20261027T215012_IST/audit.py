"""Reproduce a sealed experiment without writing into it or fitting new models."""
import hashlib
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from .engine import (divide,interval,valid_coverage,target_at,independent_history,raw_events,
                     topology,alarms,match_event_starts,evaluate_record,check_split,
                     check_aligned,check_oof,cutoff_from_grid,event_bootstrap,lead_bootstrap,
                     paired_common_leads)

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'experiments/incremental_value_20261027'
SOURCE=OLD/'outputs'
HISTORY=['SpO2_Current','SpO2_Mean_10s','SpO2_Mean_30s','SpO2_Trend_10s','SpO2_Trend_30s','SpO2_Std_30s','SpO2_Age_s']
PPG=['PPG_Amplitude','Pulse_Interval_s','Pulse_Rate_BPM','Amplitude_Slope','Rate_Mean_5','Rate_Std_10','Rate_Change_5','Amplitude_Mean_5','Amplitude_Std_10','Interval_Std_10']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value

def write_json(name,value):(HERE/name).write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')

def read(name):return pd.read_csv(SOURCE/name,float_precision='round_trip')

def verify_preservation():
    old=json.loads((HERE/'preservation_before.json').read_text())
    changed=[name for name,want in old.items() if sha(ROOT/name)!=want]
    if changed:raise AssertionError('Historical files changed: '+str(changed))
    return len(old)

def load_inputs():
    complete=json.loads((OLD/'COMPLETION.json').read_text())
    for name,want in complete['files'].items():
        if sha(OLD/name)!=want:raise AssertionError('Sealed original experiment changed: '+name)
    protocol=json.loads((SOURCE/'protocol.json').read_text())
    for name,want in protocol['input_hashes'].items():
        if sha(ROOT/name)!=want:raise AssertionError('Original scientific input changed: '+name)
    data=read('matched_dataset.csv.gz');pred=read('classification_predictions.csv.gz');forecast=read('forecast_predictions.csv.gz');pulse=read('all_pulses.csv.gz')
    assert sorted(data.Record.unique())==list(range(1,31)) and len(data)==20150
    numeric={};waveforms={};inventory=[]
    for record in range(1,31):
        n=pd.read_csv(ROOT/f'data/bidmc_{record:02d}_Numerics.csv');n.columns=n.columns.str.strip()
        nt=n['Time [s]'].to_numpy(float);s=n.SpO2.to_numpy(float);assert np.isfinite(nt).all() and (np.diff(nt)>0).all();numeric[record]=(nt,s)
        w=pd.read_csv(ROOT/f'data/bidmc_{record:02d}_Signals.csv');w.columns=w.columns.str.strip()
        clock=w['Time [s]'].iloc[0]+np.arange(len(w))/125
        residual=float(np.abs(w['Time [s]'].to_numpy()-clock).max());assert residual<=.005001
        waveforms[record]=(clock,w.PLETH.to_numpy(float))
        inventory.append(dict(Record=record,Source_Duration_s=float(clock[-1]-clock[0]),Waveform_Rows=len(w),Numeric_Rows=len(n),Waveform_fs_Hz=125,Maximum_Export_Clock_Residual_s=residual,Missing_PPG=int(w.PLETH.isna().sum()),Missing_SpO2=int(n.SpO2.isna().sum()),Patient_Linkage='unverified'))
    inventory=pd.DataFrame(inventory);inventory.to_csv(HERE/'recording_inventory.csv',index=False)
    for row in data.itertuples():
        nt,s=numeric[row.Record];label,minimum=target_at(row.time_s,nt,s)
        assert row.Label==label and row.FutureMin==minimum
        actual=independent_history(row.time_s,nt,s);stored=np.array([getattr(row,c) for c in HISTORY])
        np.testing.assert_allclose(actual,stored,rtol=1e-10,atol=1e-10,equal_nan=True)
        assert row.peak_time_s<=row.time_s and np.isclose(row.time_s-row.peak_time_s,.008,atol=1e-10)
    ev=raw_events(numeric);ev.to_csv(HERE/'raw_event_inventory.csv',index=False)
    original=read('raw_event_audit.csv').set_index(['Record','Event_ID'])
    for e in ev.itertuples():
        want=original.loc[(e.Record,e.Event_ID)]
        for c in ['Onset_s','End_s','Minimum_SpO2','Low_Readings','Qualifying_Onset','Sustained3']:assert getattr(e,c)==want[c]
    print('AUDIT: raw labels, all seven history features, confirmation times and event inventory agree',flush=True)
    return data,pred,forecast,pulse,numeric,waveforms,inventory,ev,protocol

def check_training(data,numeric,events,protocol):
    ledger=json.loads((SOURCE/'fold_ledger.json').read_text());assert len(ledger)==270
    cached={r:topology(data[data.Record.eq(r)],*numeric[r]) for r in range(1,31)}
    signatures={};rows=[];maximum_transform_error=0.
    for meta in ledger:
        check_split(meta);held=meta['Outer_Test_Record'];features=meta['Features'];train=data[data.Record.ne(held)]
        signature=[(f['train_records'],f['validation_records']) for f in meta['Inner']]
        if held in signatures:assert signatures[held]==signature
        else:signatures[held]=signature
        ck=SOURCE/'classification_folds'/f"{meta['Modality']}__{meta['Model']}__{held:02d}"
        oof=pd.read_csv(ck/'oof.csv.gz',float_precision='round_trip');check_oof(oof,data,held)
        score_sha=hashlib.sha256(np.asarray(oof.Score,dtype='<f8').tobytes()).hexdigest();assert score_sha==meta['OOF_Scores_SHA256']
        grid=pd.read_csv(ck/'training_candidates.csv',float_precision='round_trip');winner=cutoff_from_grid(grid)
        assert winner.Threshold_hex==meta['Selected']['Threshold_hex'];theta=float.fromhex(winner.Threshold_hex)
        # Verify outer transform statistics directly from the permitted train matrix.
        model=joblib.load(ck/'outer_model.joblib')  # trusted local artifacts, sealed above
        x=train[features].to_numpy(float);median=np.nanmedian(x,axis=0);median=np.where(np.isfinite(median),median,0)
        np.testing.assert_allclose(model.imputer.statistics_,median,rtol=1e-10,atol=1e-10)
        imputed=np.where(np.isnan(x),median,x);mean=imputed.mean(axis=0);variance=imputed.var(axis=0)
        np.testing.assert_allclose(model.scaler.mean_,mean,rtol=1e-9,atol=1e-9);np.testing.assert_allclose(model.scaler.var_,variance,rtol=1e-9,atol=1e-9)
        maximum_transform_error=max(maximum_transform_error,float(np.abs(model.scaler.mean_-mean).max()))
        # Independently score the selected training policy using cached topology.
        scores=oof.set_index('sample_id').Score;counts=dict(Detected=0,Eligible_Events=0,Off_Event_Alarm_Starts=0,Alarm_Active_s=0.,Exposure_s=0.)
        for record in meta['Train_Records']:
            normal,previous,duration=cached[record];t=normal.time_s.to_numpy();p=scores.reindex(normal.sample_id).to_numpy()>=theta
            starts,_=alarms(t,p,previous);ev=events[events.Record.eq(record)];nt,s=numeric[record]
            matched,_=match_event_starts(ev,t,starts,nt,s,'original');eligible=matched[matched.Evaluation_Status.eq('eligible')]
            truth=ev[ev.Qualifying_Onset]
            counts['Detected']+=int(eligible.Detected.eq(True).sum());counts['Eligible_Events']+=len(eligible)
            counts['Off_Event_Alarm_Starts']+=sum(not (truth.Onset_s.gt(st)&truth.Onset_s.le(st+30)).any() for st in starts)
            counts['Alarm_Active_s']+=float(duration@p);counts['Exposure_s']+=float(duration.sum())
        for c,v in counts.items():
            expected=winner.Exposure_Hours*3600 if c=='Exposure_s' else winner[c]
            assert np.isclose(v,expected,rtol=1e-10,atol=1e-8),(meta['Modality'],meta['Model'],held,c,v,expected)
        assert 3600*counts['Off_Event_Alarm_Starts']/counts['Exposure_s']<=5+1e-12
        assert counts['Alarm_Active_s']/counts['Exposure_s']<=.05+1e-12
        rows.append(dict(Modality=meta['Modality'],Model=meta['Model'],Held_Out_Record=held,Threshold=theta,OOF_Rows=len(oof),Outer_Record_Excluded=True,Matched_Inner_Splits=True,Training_Winner_Reproduced=True,Outer_Transforms_Training_Only=True,**counts))
    pd.DataFrame(rows).to_csv(HERE/'threshold_and_split_audit.csv',index=False)
    print('AUDIT: all270 grouped cutoffs, selected training-policy counts and outer transform statistics agree',flush=True)
    return dict(Folds=len(rows),Maximum_Outer_Scaler_Mean_Error=maximum_transform_error,Inner_Transform_Limit='Inner objects not retained; source and causal/partition tests audited, no full inner refit performed')

def safe_slope(clock,values):
    ok=np.isfinite(clock)&np.isfinite(values);clock=np.asarray(clock)[ok];values=np.asarray(values)[ok]
    if len(values)<2:return np.nan
    x=clock-clock.mean();d=x@x
    return float(x@(values-values.mean())/d) if d>0 else np.nan

def event_context(events,data,pulses,waveforms,numeric):
    rows=[]
    for e in events.to_dict('records'):
        record=e['Record'];onset=e['Onset_s'];clock,w=waveforms[record];mask=(clock>=onset-30)&(clock<onset);segment=w[mask]
        pre=pulses[pulses.Record.eq(record)&pulses.time_s.ge(onset-30)&pulses.time_s.lt(onset)]
        opportunities=data[data.Record.eq(record)&data.time_s.ge(onset-30)&data.time_s.lt(onset)&data.SpO2_Current.gt(94)]
        last=opportunities.sort_values('time_s').iloc[-1] if len(opportunities) else None
        q=e|dict(Cause='unknown: no annotated beat/SQI or physiological-reference ground truth',Validated_SQI_Available=False,PPG_Beat_Success_Rate=np.nan,Pre30_Waveform_Samples=len(segment),Pre30_PPG_Finite_Fraction=float(np.isfinite(segment).mean()) if len(segment) else np.nan,Pre30_Raw_PPG_SD=float(np.nanstd(segment)) if len(segment) else np.nan,Pre30_Raw_PPG_Range=float(np.nanmax(segment)-np.nanmin(segment)) if len(segment) else np.nan,Pre30_Flat_Pair_Fraction=float((np.abs(np.diff(segment))<1e-12).mean()) if len(segment)>1 else np.nan,Pre30_Detected_Pulses=len(pre),Pre30_Amplitude_Median=pre.PPG_Amplitude.median(),Pre30_Interval_CV=pre.Pulse_Interval_s.std()/pre.Pulse_Interval_s.mean() if len(pre)>1 else np.nan,Last_Eligible_Prediction_s=last.time_s if last is not None else np.nan,Last_Current_SpO2=last.SpO2_Current if last is not None else np.nan,Last_SpO2_Trend10=last.SpO2_Trend_10s if last is not None else np.nan,Last_SpO2_Trend30=last.SpO2_Trend_30s if last is not None else np.nan)
        for feature in PPG:q['Last_'+feature]=last[feature] if last is not None else np.nan
        for window in [5,15,30]:
            d=pre[pre.time_s.ge(onset-window)]
            for feature in ['PPG_Amplitude','Pulse_Rate_BPM','Pulse_Interval_s']:
                q[f'{feature}_Trend_Pre{window}s']=safe_slope(d.time_s.to_numpy(),d[feature].to_numpy())
        nt,s=numeric[record];q['Pre_Event_SpO2_JSON']=json.dumps([[float(t),float(v) if np.isfinite(v) else None] for t,v in zip(nt[(nt>=onset-30)&(nt<onset)],s[(nt>=onset-30)&(nt<onset)])])
        rows.append(q)
    return pd.DataFrame(rows)

def reproduce_classification(data,pred,events,numeric,inventory,context):
    per=[];event_rows=[];failure=[];timestamp=[];lead_coverage=[]
    old_per=read('event_per_recording.csv').set_index(['Modality','Model','Definition','Record'])
    for (modality,model),condition in pred.groupby(['Modality','Model']):
        check_aligned(data,condition,['Record','time_s','Label','FutureMin','SpO2_Current'])
        np.testing.assert_array_equal(condition.Prediction,(condition.Score>=condition.Threshold).astype(int))
        for population,g in [('full_target',condition),('current_normal',condition[condition.SpO2_Current.gt(94)])]:
            for record in range(1,31):
                h=g[g.Record.eq(record)];y=h.Label.to_numpy();p=h.Prediction.to_numpy();tp=int(((y==1)&(p==1)).sum());fp=int(((y==0)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum());tn=int(((y==0)&(p==0)).sum())
                timestamp.append(dict(Modality=modality,Model=model,Population=population,Record=record,N=len(h),TP=tp,FP=fp,FN=fn,TN=tn,Recall=float(divide(tp,tp+fn)),Precision=float(divide(tp,tp+fp)),Specificity=float(divide(tn,tn+fp)),AP=average_precision_score(y,h.Score) if len(np.unique(y))==2 else np.nan,Brier=brier_score_loss(y,h.Score) if model in ['Logistic','GaussianNB'] and len(h) else np.nan))
        for record,g in condition.groupby('Record'):
            nt,s=numeric[record];ev=events[events.Record.eq(record)]
            for definition in ['original','sustained3']:
                counts,matched,normal,previous,duration,starts,flags=evaluate_record(g,ev,nt,s,definition)
                want=old_per.loc[(modality,model,definition,int(record))]
                for c in ['Detected','Eligible_Events','Missed_Events','Exposure_s','Alarm_Active_s','Off_Event_Alarm_Starts','Alarm_Starts','Unmatched_Alarm_Starts','Samples','Alert_Timestamps','Lead_Median_s']:
                    assert np.isclose(counts[c],want[c],equal_nan=True),(modality,model,definition,record,c,counts[c],want[c])
                counts.update(Modality=modality,Model=model,Source_Duration_s=float(inventory.set_index('Record').loc[record,'Source_Duration_s']),Policy=str(g.Policy.iloc[0]))
                counts['Source_Duration_Normalized_Starts_Per_Hour']=float(divide(3600*counts['Off_Event_Alarm_Starts'],counts['Source_Duration_s']))
                counts['Meets_Record_Research_Budget']=bool(counts['False_Alarm_Starts_Per_Hour']<=5+1e-12) if counts['Exposure_s']>0 else np.nan
                counts['Meets_Record_Active_Cap']=bool(counts['Alarm_Active_Fraction']<=.05+1e-12) if counts['Exposure_s']>0 else np.nan
                per.append(counts)
                if matched.empty:continue
                matched=matched.assign(Modality=modality,Model=model,Policy=str(g.Policy.iloc[0]));event_rows.append(matched)
                for m in matched.to_dict('records'):
                    o=m['Onset_s'];q=normal[normal.time_s.ge(o-30)&normal.time_s.lt(o)];margin=q.Score-q.Threshold
                    if m['Evaluation_Status']!='eligible':reason='excluded: '+m['Evaluation_Status']
                    elif m['Detected']:reason='matched actual start'
                    elif not q.Prediction.eq(1).any():reason='threshold not crossed in eligible warning window'
                    elif m['Candidate_Starts']==0:reason='positive decisions without a new start in window; alarm already active'
                    else:reason='no unused start after earlier-event one-to-one assignment'
                    current=m|dict(Warning_Outcome_Reason=reason,Threshold=float(g.Threshold.iloc[0]),Threshold_Crossed_In_Window=bool(q.Prediction.eq(1).any()),Positive_Windows=int(q.Prediction.sum()),Maximum_Score_Minus_Threshold=float(margin.max()) if len(q) else np.nan,Last_Score=float(q.Score.iloc[-1]) if len(q) else np.nan,Score_Trace_JSON=json.dumps([[float(r.time_s),float(r.Score),float(r.Threshold),int(r.Prediction)] for r in q.itertuples()]))
                    failure.append(current)
    per=pd.DataFrame(per);matched=pd.concat(event_rows,ignore_index=True);failure=pd.DataFrame(failure)
    extra=[c for c in context if c not in failure or c in ['Record','Event_ID']]
    failure=failure.merge(context[extra],on=['Record','Event_ID'],how='left',validate='many_to_one')
    failure.to_csv(HERE/'event_failure_analysis.csv',index=False);per.to_csv(HERE/'classification_per_recording.csv',index=False);matched.to_csv(HERE/'reproduced_event_matches.csv',index=False);pd.DataFrame(timestamp).to_csv(HERE/'timestamp_per_recording.csv',index=False)
    summary=[]
    for key,g in per.groupby(['Modality','Model','Definition']):
        q={c:float(g[c].sum()) for c in ['Detected','Eligible_Events','Missed_Events','Exposure_s','Source_Duration_s','Alarm_Active_s','Off_Event_Alarm_Starts','Alarm_Starts']}
        for c in ['Detected','Eligible_Events','Missed_Events','Off_Event_Alarm_Starts','Alarm_Starts']:q[c]=int(q[c])
        leads=matched[matched.Modality.eq(key[0])&matched.Model.eq(key[1])&matched.Definition.eq(key[2])&matched.Detected.eq(True)].Lead_Time_s
        q.update(Modality=key[0],Model=key[1],Definition=key[2],Sensitivity=float(divide(q['Detected'],q['Eligible_Events'])),False_Alarm_Starts_Per_Hour=float(divide(3600*q['Off_Event_Alarm_Starts'],q['Exposure_s'])),Source_Duration_Normalized_Starts_Per_Hour=float(divide(3600*q['Off_Event_Alarm_Starts'],q['Source_Duration_s'])),Alarm_Active_Fraction=float(divide(q['Alarm_Active_s'],q['Exposure_s'])),Lead_Median_s=leads.median(),Lead_Min_s=leads.min(),Lead_Max_s=leads.max(),Record_Start_Budget_Violations=int(((g.Exposure_s>0)&(g.False_Alarm_Starts_Per_Hour>5+1e-12)).sum()),Record_Activity_Violations=int(((g.Exposure_s>0)&(g.Alarm_Active_Fraction>.05+1e-12)).sum()),Policy=g.Policy.iloc[0])
        q['Meets_Pooled_Research_Constraints']=bool(q['False_Alarm_Starts_Per_Hour']<=5+1e-12 and q['Alarm_Active_Fraction']<=.05+1e-12)
        q['Meets_All_Exposed_Record_Constraints']=q['Record_Start_Budget_Violations']==0 and q['Record_Activity_Violations']==0
        summary.append(q)
        for lead in [0,5,10,15,20,30]:lead_coverage.append(dict(Modality=key[0],Model=key[1],Definition=key[2],Minimum_Observed_Lead_s=lead,Matched_Events_Meeting_Lead=int(leads.ge(lead).sum()),Eligible_Events=q['Eligible_Events'],Fraction_Of_All_Eligible=float(divide(leads.ge(lead).sum(),q['Eligible_Events']))))
    summary=pd.DataFrame(summary);summary.to_csv(HERE/'reproduced_event_summary.csv',index=False);pd.DataFrame(lead_coverage).to_csv(HERE/'lead_coverage_diagnostic.csv',index=False)
    print('AUDIT: all22 event conditions and660 per-record event rows reproduced',flush=True)
    return per,matched,failure,summary

def reproduce_forecasts(data,forecast):
    rows=[]
    for model,g in forecast.groupby('Model'):
        check_aligned(data,g,['Record','time_s','FutureMin','SpO2_Current'])
        np.testing.assert_allclose(g.Forecast,np.clip(g.Raw_Forecast,0,100),rtol=0,atol=0)
        for population,part in [('full_target',g),('current_normal',g[g.SpO2_Current.gt(94)])]:
            for record in range(1,31):
                h=part[part.Record.eq(record)];error=h.Forecast-h.FutureMin
                rows.append(dict(Model=model,Population=population,Record=record,N=len(h),Absolute_Error_Sum=float(np.abs(error).sum()),Squared_Error_Sum=float((error**2).sum()),MAE=float(np.abs(error).mean()) if len(h) else np.nan,RMSE=float(np.sqrt((error**2).mean())) if len(h) else np.nan,Units='SpO2 percentage points'))
    per=pd.DataFrame(rows);old=read('forecast_per_recording.csv').set_index(['Model','Population','Record'])
    for r in per.itertuples():
        want=old.loc[(r.Model,r.Population,r.Record)]
        for c in ['N','Absolute_Error_Sum','Squared_Error_Sum','MAE','RMSE']:assert np.isclose(getattr(r,c),want[c],equal_nan=True)
    summary=[]
    for (model,population),g in per.groupby(['Model','Population']):
        n=int(g.N.sum());summary.append(dict(Model=model,Population=population,N=n,MAE=g.Absolute_Error_Sum.sum()/n,RMSE=np.sqrt(g.Squared_Error_Sum.sum()/n)))
    summary=pd.DataFrame(summary);per.to_csv(HERE/'forecast_per_recording.csv',index=False);summary.to_csv(HERE/'reproduced_forecast_summary.csv',index=False)
    return per,summary

def paired_analysis(per,matched,fper,summary,fsum):
    weights=read('bootstrap_recording_weights.csv.gz').to_numpy()
    seeded=np.random.default_rng(20261027).integers(0,30,size=(2000,30));expected=np.array([np.bincount(d,minlength=30) for d in seeded]);np.testing.assert_array_equal(weights,expected)
    pd.DataFrame(weights,columns=[f'Record_{r:02d}' for r in range(1,31)]).to_csv(HERE/'bootstrap_recording_weights.csv.gz',index=False,compression='gzip')
    intervals=[];boots={};pairs=[];distributions=[];common_leads=[];influence=[]
    event_metrics=['Sensitivity','False_Alarm_Starts_Per_Hour','Alarm_Active_Fraction','Source_Duration_Normalized_Starts_Per_Hour']
    for key,g in per.groupby(['Modality','Model','Definition']):
        boot=event_bootstrap(g,weights);boots[key]=boot
        q=summary[summary.Modality.eq(key[0])&summary.Model.eq(key[1])&summary.Definition.eq(key[2])].iloc[0]
        for metric,v in boot.items():intervals.append(dict(Analysis='events',Modality=key[0],Model=key[1],Definition=key[2],Metric=metric,Estimate=q[metric],**interval(v)))
        leads=matched[matched.Modality.eq(key[0])&matched.Model.eq(key[1])&matched.Definition.eq(key[2])&matched.Detected.eq(True)][['Record','Event_ID','Lead_Time_s']]
        intervals.append(dict(Analysis='events',Modality=key[0],Model=key[1],Definition=key[2],Metric='Lead_Median_s_detected_only',Estimate=q.Lead_Median_s,**interval(lead_bootstrap(leads,weights))))
    for model in ['Logistic','GaussianNB','ELM']:
        for modality in ['Combined','PPG']:
            for definition in ['original','sustained3']:
                a=per[per.Modality.eq(modality)&per.Model.eq(model)&per.Definition.eq(definition)].set_index('Record').sort_index()
                b=per[per.Modality.eq('SpO2_history')&per.Model.eq(model)&per.Definition.eq(definition)].set_index('Record').sort_index()
                np.testing.assert_array_equal(a.Eligible_Events,b.Eligible_Events);np.testing.assert_allclose(a.Exposure_s,b.Exposure_s)
                aggregate_a=summary[summary.Modality.eq(modality)&summary.Model.eq(model)&summary.Definition.eq(definition)].iloc[0]
                aggregate_b=summary[summary.Modality.eq('SpO2_history')&summary.Model.eq(model)&summary.Definition.eq(definition)].iloc[0]
                for metric in event_metrics:
                    pairs.append(dict(Analysis='events',Model=model,Contrast=modality+' minus SpO2_history',Definition=definition,Population='current_normal',Metric=metric,Difference=aggregate_a[metric]-aggregate_b[metric],**interval(boots[(modality,model,definition)][metric]-boots[('SpO2_history',model,definition)][metric])))
                    for record in range(1,31):distributions.append(dict(Analysis='events',Model=model,Contrast=modality+' minus SpO2_history',Definition=definition,Population='current_normal',Record=record,Metric=metric,Difference=a.loc[record,metric]-b.loc[record,metric],Eligible_Events=int(a.loc[record,'Eligible_Events']),Exposure_s=a.loc[record,'Exposure_s']))
                ma=matched[matched.Modality.eq(modality)&matched.Model.eq(model)&matched.Definition.eq(definition)];mb=matched[matched.Modality.eq('SpO2_history')&matched.Model.eq(model)&matched.Definition.eq(definition)]
                leads=paired_common_leads(ma,mb);estimate=float(leads.Lead_Time_s.median()) if len(leads) else np.nan
                pairs.append(dict(Analysis='events',Model=model,Contrast=modality+' minus SpO2_history',Definition=definition,Population='common_detected_events_only',Metric='Paired_Lead_Median_Difference_s',Difference=estimate,Common_Events=len(leads),**interval(lead_bootstrap(leads,weights))))
                if len(leads):common_leads.append(leads.assign(Model=model,Contrast=modality+' minus SpO2_history',Definition=definition))
                for record in range(1,31):
                    local=leads[leads.Record.eq(record)]
                    distributions.append(dict(Analysis='events',Model=model,Contrast=modality+' minus SpO2_history',Definition=definition,Population='common_detected_events_only',Record=record,Metric='Paired_Lead_Median_Difference_s',Difference=float(local.Lead_Time_s.median()) if len(local) else np.nan,Common_Events=len(local)))
    forecast_boots={}
    for (model,population),g in fper.groupby(['Model','Population']):
        g=g.sort_values('Record');n=weights@g.N.to_numpy();ae=weights@g.Absolute_Error_Sum.to_numpy();se=weights@g.Squared_Error_Sum.to_numpy()
        boot=dict(MAE=divide(ae,n),RMSE=np.sqrt(divide(se,n)));forecast_boots[(model,population)]=boot
        q=fsum[fsum.Model.eq(model)&fsum.Population.eq(population)].iloc[0]
        for metric,v in boot.items():intervals.append(dict(Analysis='forecast',Model=model,Population=population,Metric=metric,Estimate=q[metric],**interval(v)))
    for model in ['Ridge_Combined','Ridge_PPG']:
        for reference in ['Ridge_SpO2_history','Persistence','Trend10']:
            for population in ['full_target','current_normal']:
                a=fper[fper.Model.eq(model)&fper.Population.eq(population)].set_index('Record').sort_index();b=fper[fper.Model.eq(reference)&fper.Population.eq(population)].set_index('Record').sort_index();np.testing.assert_array_equal(a.N,b.N)
                qa=fsum[fsum.Model.eq(model)&fsum.Population.eq(population)].iloc[0];qb=fsum[fsum.Model.eq(reference)&fsum.Population.eq(population)].iloc[0]
                for metric in ['MAE','RMSE']:
                    pairs.append(dict(Analysis='forecast',Model=model,Contrast=model+' minus '+reference,Population=population,Metric=metric,Difference=qa[metric]-qb[metric],**interval(forecast_boots[(model,population)][metric]-forecast_boots[(reference,population)][metric])))
                    for record in range(1,31):distributions.append(dict(Analysis='forecast',Model=model,Contrast=model+' minus '+reference,Population=population,Record=record,Metric=metric,Difference=a.loc[record,metric]-b.loc[record,metric],N=int(a.loc[record,'N'])))
                if model=='Ridge_Combined' and reference=='Ridge_SpO2_history':
                    for record in range(1,31):
                        remaining_n=int(a.N.sum()-a.loc[record,'N']);ae=(a.Absolute_Error_Sum-b.Absolute_Error_Sum)
                        leave=float(divide(ae.sum()-ae.loc[record],remaining_n))
                        influence.append(dict(Population=population,Removed_Record=record,Record_N=int(a.loc[record,'N']),Record_MAE_Difference=a.loc[record,'MAE']-b.loc[record,'MAE'],Signed_Contribution_To_Pooled_MAE_Difference=ae.loc[record]/a.N.sum(),Leave_One_Record_Out_MAE_Difference=leave,Full_Pooled_MAE_Difference=qa.MAE-qb.MAE,Influence_Removing_Record=leave-(qa.MAE-qb.MAE),No_Model_Refit=True))
    pairs=pd.DataFrame(pairs);intervals=pd.DataFrame(intervals);distributions=pd.DataFrame(distributions);influence=pd.DataFrame(influence)
    # Reproduce the previously reported intervals/differences, not just new ones.
    old_pairs=read('paired_modality_differences.csv')
    for row in old_pairs.itertuples():
        q=pairs[pairs.Analysis.eq('events')&pairs.Model.eq(row.Model)&pairs.Contrast.eq(row.Contrast)&pairs.Definition.eq(row.Definition)&pairs.Metric.eq(row.Metric)].iloc[0]
        for c in ['Difference','Lower_95','Upper_95','Defined_Replicates']:assert np.isclose(q[c],getattr(row,c),equal_nan=True)
    old_forecast=read('paired_forecast_differences.csv')
    for row in old_forecast.itertuples():
        q=pairs[pairs.Analysis.eq('forecast')&pairs.Contrast.eq(row.Contrast)&pairs.Population.eq(row.Population)&pairs.Metric.eq(row.Metric)].iloc[0]
        for c in ['Difference','Lower_95','Upper_95','Defined_Replicates']:assert np.isclose(q[c],getattr(row,c),equal_nan=True)
    pairs.to_csv(HERE/'paired_model_comparisons.csv',index=False);intervals.to_csv(HERE/'uncertainty_intervals.csv',index=False);distributions.to_csv(HERE/'recording_paired_differences.csv',index=False);influence.to_csv(HERE/'forecast_influence.csv',index=False)
    if common_leads:pd.concat(common_leads,ignore_index=True).to_csv(HERE/'paired_common_event_leads.csv',index=False)
    pd.concat([per.assign(Analysis='classification',Population='current_normal'),fper.assign(Analysis='forecast')],ignore_index=True).to_csv(HERE/'per_recording_metrics.csv',index=False)
    print('AUDIT: existing paired intervals agree; common-event lead and leave-record-out analyses added',flush=True)
    return pairs,intervals,distributions,influence

def figures(summary,per,matched,pairs,distributions,influence,failure):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    colors={'SpO2_history':'#2563eb','PPG':'#7c3aed','Combined':'#0f766e'}
    def save(fig,name):
        for ext in ['png','pdf']:fig.savefig(HERE/'figures'/(name+'.'+ext),dpi=170,bbox_inches='tight')
        plt.close(fig)
    g=summary[summary.Model.eq('Logistic')&summary.Definition.eq('original')].set_index('Modality').loc[list(colors)]
    fig,ax=plt.subplots(1,3,figsize=(12,4.2),layout='constrained')
    for axis,col,title in zip(ax,['Sensitivity','False_Alarm_Starts_Per_Hour','Alarm_Active_Fraction'],['Eligible-event sensitivity','Starts / evaluable hour','Alarm-active fraction']):
        axis.bar(range(3),g[col],color=list(colors.values()));axis.set_xticks(range(3),['SpO2 history','PPG','Combined'],rotation=20);axis.set(title=title)
        if col=='Sensitivity':
            axis.set_ylim(0,1.08)
            for i,r in enumerate(g.itertuples()):axis.text(i,r.Sensitivity+.03,f'{r.Detected}/{r.Eligible_Events}',ha='center')
        elif col=='False_Alarm_Starts_Per_Hour':axis.axhline(5,color='red',linestyle='--')
        else:axis.axhline(.05,color='red',linestyle='--')
    fig.suptitle('Reproduced logistic comparison: pooled constraints are not per-record guarantees');save(fig,'reproduced_alarm_comparison')
    fig,axes=plt.subplots(2,1,figsize=(12,7),layout='constrained')
    for axis,metric,label in zip(axes,['False_Alarm_Starts_Per_Hour','Alarm_Active_Fraction'],['Off-event starts / evaluable hour','Active fraction']):
        for modality,color in colors.items():
            d=per[per.Model.eq('Logistic')&per.Modality.eq(modality)&per.Definition.eq('original')]
            axis.plot(d.Record,d[metric],marker='o',label=modality,color=color)
        axis.axhline(5 if metric.startswith('False') else .05,color='red',linestyle='--');axis.set(ylabel=label,xlabel='Recording ID');axis.legend(fontsize=8)
    fig.suptitle('Per-record burden: undefined rates at zero exposure remain missing');save(fig,'per_record_alarm_constraints')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for axis,population in zip(axes,['full_target','current_normal']):
        d=influence[influence.Population.eq(population)];axis.bar(d.Removed_Record,d.Record_MAE_Difference,color=np.where(d.Record_MAE_Difference<0,'#0f766e','#be123c'));axis.axhline(0,color='black',linewidth=.7);axis.set(title=population,xlabel='Recording ID',ylabel='Combined minus history MAE (percentage points)')
    fig.suptitle('Paired forecast effects: negative favors adding PPG');save(fig,'forecast_recording_differences')
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for population,color in [('full_target','#2563eb'),('current_normal','#0f766e')]:
        d=influence[influence.Population.eq(population)];ax.plot(d.Removed_Record,d.Leave_One_Record_Out_MAE_Difference,marker='o',label=population,color=color)
    ax.axhline(0,color='black',linewidth=.7);ax.set(xlabel='Removed recording (contributions only; no refit)',ylabel='Pooled MAE difference',title='Leave-one-recording influence is not independent validation');ax.legend();save(fig,'forecast_leave_record_out_influence')
    coverage=pd.read_csv(HERE/'lead_coverage_diagnostic.csv');fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    for modality,color in colors.items():
        d=coverage[coverage.Model.eq('Logistic')&coverage.Modality.eq(modality)&coverage.Definition.eq('original')];ax.step(d.Minimum_Observed_Lead_s,d.Fraction_Of_All_Eligible,where='post',label=modality,color=color)
    ax.set(xlabel='Diagnostic minimum matched lead (s)',ylabel='Fraction of all14 eligible onsets',ylim=(0,1),title='Original matches only: misses stay in denominator, not assigned zero lead');ax.legend();save(fig,'lead_coverage_diagnostic')
    d=failure[failure.Model.eq('Logistic')&failure.Definition.eq('original')&failure.Evaluation_Status.eq('eligible')].copy();d['Event']=d.Record.astype(str)+':'+d.Event_ID.astype(str)
    m=d.pivot(index='Modality',columns='Event',values='Detected').reindex(list(colors)).astype(float)
    fig,ax=plt.subplots(figsize=(12,3.3),layout='constrained');ax.imshow(m.to_numpy(),aspect='auto',vmin=0,vmax=1,cmap=ListedColormap(['#fee2e2','#0f766e']));ax.set_xticks(range(len(m.columns)),m.columns,rotation=45,ha='right');ax.set_yticks(range(3),m.index);ax.set(title='Logistic event audit: red missed, green detected (record:event)');save(fig,'event_detection_audit')

def main():
    verify_preservation();data,pred,forecast,pulses,numeric,waveforms,inventory,events,protocol=load_inputs()
    training=check_training(data,numeric,events,protocol)
    context=event_context(events,data,pulses,waveforms,numeric)
    per,matched,failure,summary=reproduce_classification(data,pred,events,numeric,inventory,context)
    fper,fsum=reproduce_forecasts(data,forecast)
    pairs,intervals,distributions,influence=paired_analysis(per,matched,fper,summary,fsum)
    reason=failure[failure.Evaluation_Status.eq('eligible')].groupby(['Modality','Model','Definition','Warning_Outcome_Reason']).size().rename('Events').reset_index();reason.to_csv(HERE/'failure_reason_counts.csv',index=False)
    # Descriptive feature associations only; no p-values, causal claims or model changes.
    association=[]
    for key,g in failure[failure.Evaluation_Status.eq('eligible')].groupby(['Modality','Model','Definition','Detected']):
        for feature in ['Last_Current_SpO2','Last_SpO2_Trend10','Pre30_Amplitude_Median','Pre30_Interval_CV','Pre30_Detected_Pulses','Observed_Low_Span_s']:
            association.append(dict(Modality=key[0],Model=key[1],Definition=key[2],Detected=key[3],Feature=feature,N=int(g[feature].notna().sum()),Median=g[feature].median(),Interpretation='descriptive association only; no established cause'))
    pd.DataFrame(association).to_csv(HERE/'failure_associations_descriptive.csv',index=False)
    figures(summary,per,matched,pairs,distributions,influence,failure)
    old_audit=read('event_audit_long.csv');metadata_missing=int(old_audit.Policy.isna().sum())
    assert metadata_missing==11 and failure.Policy.notna().all()
    counts=summary[summary.Model.eq('Logistic')][['Modality','Definition','Detected','Eligible_Events','False_Alarm_Starts_Per_Hour','Source_Duration_Normalized_Starts_Per_Hour','Alarm_Active_Fraction','Lead_Median_s','Meets_Pooled_Research_Constraints','Meets_All_Exposed_Record_Constraints']]
    for definition,expected in [('original',{'SpO2_history':9,'PPG':0,'Combined':10}),('sustained3',{'SpO2_history':9,'PPG':0,'Combined':9})]:
        g=counts[counts.Definition.eq(definition)].set_index('Modality');assert g.Detected.to_dict()==expected
        assert g.Eligible_Events.eq(14 if definition=='original' else 13).all()
    influence_summary=[]
    for population,g in influence.groupby('Population'):
        d=g.Record_MAE_Difference.dropna();influence_summary.append(dict(Population=population,Improved_Records=int((d<0).sum()),Worse_Records=int((d>0).sum()),Tied_Records=int((d==0).sum()),Median_Record_MAE_Difference=float(d.median()),Leave_One_Record_Out_Min=g.Leave_One_Record_Out_MAE_Difference.min(),Leave_One_Record_Out_Max=g.Leave_One_Record_Out_MAE_Difference.max(),Leave_One_Out_Positive_Differences=int((g.Leave_One_Record_Out_MAE_Difference>0).sum())))
    report=dict(Status='independently_reproduced_with_documented_limits',Historical_Files_Unchanged=verify_preservation(),Source_Experiment_Files_Checked=len(json.loads((OLD/'COMPLETION.json').read_text())['files']),Matched_Rows=len(data),Recordings=30,Unique_Patients='unknown',Eligible_Original_Events=14,Eligible_Sustained_Events=13,Source_Duration_s=inventory.Source_Duration_s.sum(),Evaluable_Exposure_s=float(summary.Exposure_s.iloc[0]),Classifier_Conditions=pred.groupby(['Modality','Model']).ngroups,Reproduced_Event_Conditions=len(summary),Outer_Training_Audit=training,Logistic_Results=counts.to_dict('records'),Forecast_Results=fsum.to_dict('records'),Paired_Forecast= pairs[pairs.Analysis.eq('forecast')&pairs.Contrast.eq('Ridge_Combined minus Ridge_SpO2_history')].to_dict('records'),Forecast_Influence=influence_summary,Policy_Metadata_Omission_Rows=metadata_missing,Metric_Changing_Defect_Found=False,New_Model_Fits=0,Clinical_Validation=False,Source_Hour_Normalization_Interpretation='Observed evaluable starts / source hours; not a full-monitoring-time false-alarm estimate',Bootstrap_Limit='Whole-record paired draws conditional on saved overlapping CV fits; no patient or refit uncertainty',All_Analysis_Source_SHA256={p.name:sha(p) for p in HERE.glob('*.py')})
    write_json('reproduced_metrics.json',report)
    write_json('analysis_receipt.json',dict(Files={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file() and p.name not in ['analysis_receipt.json','preservation_before.json'] and 'logs' not in p.parts},History_Preserved=report['Historical_Files_Unchanged'],Command=' '.join(sys.argv)))
    print(json.dumps(clean({k:report[k] for k in ['Status','Historical_Files_Unchanged','Matched_Rows','Reproduced_Event_Conditions','Metric_Changing_Defect_Found','New_Model_Fits']}),indent=2),flush=True)

if __name__=='__main__':main()
