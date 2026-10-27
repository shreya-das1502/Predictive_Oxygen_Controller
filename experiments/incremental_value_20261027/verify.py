"""Read-only, independent numerical verification of completed new outputs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from .study import HERE,ROOT,sha

def coverage(nt,s,a,b):
    mask=(nt>=a)&(nt<=b);v=s[mask];t=nt[mask]
    return bool(a>=nt[0] and b<=nt[-1] and len(v) and np.isfinite(v).all() and ((v>0)&(v<=100)).all() and np.diff(np.r_[a,t,b]).max(initial=0)<=1.01)

def stream_counts(frame,events,numeric):
    """Separate implementation of current-normal edges, starts and event matching."""
    starts=[];exposure=active=0.;opportunities={}
    for record,g in frame.groupby('Record'):
        nt,s=numeric[record];g=g.sort_values('time_s');g=g[g.SpO2_Current.gt(94)];t=g.time_s.to_numpy();p=g.Prediction.to_numpy(bool)
        edges=[]
        for a,b in zip(t[:-1],t[1:]):
            j=np.searchsorted(nt,a,side='right')-1;v=s[(nt>a)&(nt<=b)]
            edges.append(bool(b-a<=2 and j>=0 and a-nt[j]<=1.01 and np.isfinite(s[j]) and 94<s[j]<=100 and np.isfinite(v).all() and ((v>94)&(v<=100)).all() and coverage(nt,s,a,b+30)))
        edge=np.asarray(edges,bool);dt=np.diff(t)*edge
        exposure+=dt.sum();active+=(dt*p[:-1]).sum()
        previous=np.r_[False,edge] if len(t) else np.array([],bool)
        flags=p&~(np.r_[False,p[:-1]]&previous) if len(t) else np.array([],bool)
        starts.extend((int(record),float(v)) for v in t[flags]);opportunities[int(record)]=t
    results={}
    for definition in ['original','sustained3']:
        ev=events if definition=='original' else events[events.Sustained3]
        eligible=[];used=set();hits=[]
        for e in ev.sort_values(['Record','Onset_s']).itertuples():
            nt,s=numeric[e.Record];t=opportunities.get(e.Record,np.array([]));o=e.Onset_s
            if not(e.Qualifying_Onset and coverage(nt,s,o-30,o) and ((t>=o-30)&(t<o)).any()):continue
            eligible.append((e.Record,e.Event_ID))
            for i,(r,start) in enumerate(starts):
                if r==e.Record and o-30<=start<o and i not in used:
                    used.add(i);hits.append((e.Record,e.Event_ID,start,o-start));break
        false=sum(not (ev.Record.eq(r)&ev.Qualifying_Onset&ev.Onset_s.gt(start)&ev.Onset_s.le(start+30)).any() for r,start in starts)
        results[definition]=dict(Eligible_Events=len(eligible),Detected=len(hits),Exposure_s=float(exposure),Alarm_Active_s=float(active),Alarm_Starts=len(starts),Off_Event_Alarm_Starts=int(false),Matches=hits)
    return results

def main():
    out=HERE/'outputs';report={}
    for name in ['classification_fit_receipt.json','forecast_fit_receipt.json','prepared_receipt.json']:
        for file,w in json.loads((out/name).read_text()).items():assert sha(out/file)==w,file
    receipt=json.loads((out/'analysis_receipt.json').read_text())
    for n,w in receipt['tables'].items():assert sha(out/n)==w,n
    for n,w in receipt['figures'].items():assert sha(out/'figures'/n)==w,n
    if sha(HERE/'analysis.py')!=receipt['analysis_source_sha256']:raise AssertionError('Analysis source changed')
    protocol=json.loads((out/'protocol.json').read_text())
    for n,w in protocol['input_hashes'].items():assert sha(ROOT/n)==w,n
    baseline=json.loads((HERE/'audit/historical_files_before.json').read_text())
    changed=[n for n,w in baseline.items() if sha(ROOT/n)!=w]
    assert not changed,changed;report['historical_files_unchanged']=len(baseline)
    pred=pd.read_csv(out/'classification_predictions.csv.gz',float_precision='round_trip');data=pd.read_csv(out/'matched_dataset.csv.gz',float_precision='round_trip');events=pd.read_csv(out/'raw_event_audit.csv')
    numeric={}
    for r in range(1,31):
        d=pd.read_csv(ROOT/f'data/bidmc_{r:02d}_Numerics.csv');d.columns=d.columns.str.strip();numeric[r]=(d['Time [s]'].to_numpy(float),d.SpO2.to_numpy(float))
    for row in data.itertuples():
        nt,s=numeric[row.Record];mask=(nt>row.time_s)&(nt<=row.time_s+30);m=s[mask].min();j=np.searchsorted(nt,row.time_s,side='right')-1
        assert np.isfinite(m) and m==row.FutureMin and int(m<=94)==row.Label
        assert row.SpO2_Current==s[j] and row.time_s-nt[j]<=1.01
    report['common_labels_and_latest_measurements']='passed'
    np.testing.assert_array_equal(pred.Prediction,(pred.Score>=pred.Threshold).astype(int))
    ledger=json.loads((out/'fold_ledger.json').read_text());assert len(ledger)==270
    for m in ledger:
        test=m['Outer_Test_Record'];allowed=set(m['Train_Records']);seen=[]
        assert allowed==set(range(1,31))-{test}
        for fold in m['Inner']:
            tr=set(fold['train_records']);va=set(fold['validation_records']);assert not tr&va and tr|va==allowed;seen.extend(va)
        assert sorted(seen)==sorted(allowed)
        assert m['Threshold_Selected_Before_Heldout_Scoring']
        ck=out/'classification_folds'/f"{m['Modality']}__{m['Model']}__{test:02d}"
        for name,w in json.loads((ck/'complete.json').read_text())['files'].items():assert sha(ck/name)==w
        grid=pd.read_csv(ck/'training_candidates.csv',float_precision='round_trip')
        feasible=grid[grid.False_Alarm_Starts_Per_Hour.le(5+1e-12)&grid.Alarm_Active_Fraction.le(.05+1e-12)]
        selected=feasible.sort_values(['Detected','Off_Event_Alarm_Starts','Alarm_Active_s','Threshold'],ascending=[False,True,True,False],kind='stable').iloc[0]
        assert selected.Threshold_hex==m['Selected']['Threshold_hex']
    report['record_grouped_fits_and_training_cutoffs']=len(ledger)
    es=pd.read_csv(out/'event_summary.csv');per=pd.read_csv(out/'event_per_recording.csv');ts=pd.read_csv(out/'timestamp_summary.csv');independent=[]
    for (modality,model),g in pred.groupby(['Modality','Model']):
        assert len(g)==len(data) and set(g.sample_id)==set(data.sample_id) and not g.sample_id.duplicated().any()
        aligned=data.set_index('sample_id').loc[g.sample_id]
        for column in ['Record','time_s','Label','FutureMin','SpO2_Current']:
            np.testing.assert_array_equal(g[column].to_numpy(),aligned[column].to_numpy())
        for population,h in [('full_target',g),('current_normal',g[g.SpO2_Current.gt(94)])]:
            want=ts[ts.Modality.eq(modality)&ts.Model.eq(model)&ts.Population.eq(population)].iloc[0]
            for name,a,b in [('TP',1,1),('FP',0,1),('FN',1,0),('TN',0,0)]:assert ((h.Label==a)&(h.Prediction==b)).sum()==want[name]
            assert np.isclose(average_precision_score(h.Label,h.Score),want.PR_AUC_AP)
        counts=stream_counts(g,events,numeric)
        for definition,v in counts.items():
            row=es[es.Modality.eq(modality)&es.Model.eq(model)&es.Definition.eq(definition)].iloc[0]
            for c in ['Eligible_Events','Detected','Exposure_s','Alarm_Active_s','Alarm_Starts','Off_Event_Alarm_Starts']:assert np.isclose(v[c],row[c]),(modality,model,definition,c,v[c],row[c])
            independent.append(dict(Modality=modality,Model=model,Definition=definition,**v))
    report['independently_reconstructed_event_conditions']=len(independent)
    (out/'independent_event_verification.json').write_text(json.dumps(independent,indent=2))
    weights=pd.read_csv(out/'bootstrap_recording_weights.csv.gz').to_numpy();assert weights.shape==(2000,30) and (weights.sum(axis=1)==30).all()
    cis=pd.read_csv(out/'event_bootstrap_intervals.csv')
    for key,g in per.groupby(['Modality','Model','Definition']):
        g=g.sort_values('Record');n=weights@g.Detected.to_numpy();d=weights@g.Eligible_Events.to_numpy();valid=d>0;v=n[valid]/d[valid]
        q=cis[cis.Modality.eq(key[0])&cis.Model.eq(key[1])&cis.Definition.eq(key[2])&cis.Metric.eq('Sensitivity')].iloc[0]
        assert np.allclose(np.quantile(v,[.025,.975]),[q.Lower_95,q.Upper_95]) and len(v)==q.Defined_Replicates
    report['recording_bootstrap_sensitivity_intervals']='passed'
    f=pd.read_csv(out/'forecast_predictions.csv.gz',float_precision='round_trip');summary=pd.read_csv(out/'forecast_summary.csv')
    for model,g in f.groupby('Model'):
        assert len(g)==len(data) and set(g.sample_id)==set(data.sample_id)
        aligned=data.set_index('sample_id').loc[g.sample_id]
        for column in ['Record','time_s','FutureMin','SpO2_Current']:
            np.testing.assert_array_equal(g[column].to_numpy(),aligned[column].to_numpy())
        assert g.Forecast.between(0,100).all()
        for population,h in [('full_target',g),('current_normal',g[g.SpO2_Current.gt(94)])]:
            e=h.Forecast-h.FutureMin;q=summary[summary.Model.eq(model)&summary.Population.eq(population)].iloc[0]
            assert np.isclose(np.abs(e).mean(),q.MAE) and np.isclose(np.sqrt((e**2).mean()),q.RMSE)
    report['forecast_errors_and_matched_rows']='passed'
    for row in json.loads((out/'regression_ledger.json').read_text()):
        assert set(row['Train_Records'])==set(range(1,31))-{row['Outer_Test_Record']}
        assert sha(out/'regression_models'/row['Model_File'])==row['Model_SHA256']
    fp=pd.read_csv(out/'forecast_per_recording.csv');fc=pd.read_csv(out/'forecast_bootstrap_intervals.csv')
    for (model,population),g in fp.groupby(['Model','Population']):
        g=g.sort_values('Record');n=weights@g.N.to_numpy();ae=weights@g.Absolute_Error_Sum.to_numpy();se=weights@g.Squared_Error_Sum.to_numpy();valid=n>0
        for metric,values in [('MAE',ae[valid]/n[valid]),('RMSE',np.sqrt(se[valid]/n[valid]))]:
            row=fc[fc.Model.eq(model)&fc.Population.eq(population)&fc.Metric.eq(metric)].iloc[0]
            assert np.allclose(np.quantile(values,[.025,.975]),[row.Lower_95,row.Upper_95])
    report['regression_training_separation_and_forecast_intervals']='passed'
    report.update(status='passed',limits=['No patient linkage or independent cohort','Conditional recording bootstrap, no refitting','Unvalidated clinical event and alarm tolerances','No clinical benefit or model winner certified'])
    (out/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
