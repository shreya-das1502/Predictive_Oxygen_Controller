"""Coordinator sampling, pure-PPG review, then hash-gated automated associations."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
import numpy as np
import pandas as pd
from scipy.signal import sosfilt
from src.preprocessing import WaveformState,PulseDetector,canonical_waveform_times
from .quality import candidate_indicators

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PRIVATE=HERE/'coordinator_only';PACKET=HERE/'reviewer_packet'
OLD=ROOT/'experiments/incremental_value_20261027'
AUDIT=ROOT/'experiments/final_audit_20261027T215012_IST'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def cleaned(x):
    if isinstance(x,dict):return {k:cleaned(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [cleaned(v) for v in x]
    if isinstance(x,np.ndarray):return cleaned(x.tolist())
    if isinstance(x,np.generic):return cleaned(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def dump(path,x):Path(path).write_text(json.dumps(cleaned(x),indent=2,allow_nan=False)+'\n')

def instrument_source(clock,raw):
    """Observe accepted peaks without modifying the historical detector state/rules."""
    state=WaveformState(1/np.median(np.diff(clock)));filtered=sosfilt(state.sos,raw);detector=PulseDetector();accepted=[];emitted=[]
    for t,value in zip(clock,filtered):
        previous=detector.last_peak;row=detector.update(float(t),float(value))
        if detector.last_peak!=previous:
            accepted.append(dict(PeakTime=detector.last_peak,ConfirmationTime=float(t),Amplitude=detector.last_amplitude,PreviousAcceptedInterval=detector.last_peak-previous if previous is not None else np.nan,Emitted=row is not None,NonEmissionReason='' if row is not None else 'initial_unpaired' if previous is None else 'interval_above2s'))
        if row is not None:emitted.append(row)
    return filtered,pd.DataFrame(accepted),pd.DataFrame(emitted)

def sample_mapping(protocol):
    rng=np.random.default_rng(protocol['sampling_seed']);rows=[]
    for r in range(1,31):
        for quarter in range(4):
            start=int(rng.integers(quarter*15000,(quarter+1)*15000-2500+1))
            rows.append(dict(SourceRecord=r,StartSample=start,EndSample=start+2500,SamplingOrigin='time_stratified',SourceQuarter=quarter+1,LinkedEvent=np.nan))
    columns=['Record','Event_ID','Onset_s','Qualifying_Onset','Evaluation_Status','Definition','Modality','Model']
    ev=pd.read_csv(AUDIT/'event_failure_analysis.csv',usecols=columns)
    ev=ev[ev.Modality.eq('Combined')&ev.Model.eq('Logistic')&ev.Definition.eq('original')&ev.Evaluation_Status.eq('eligible')]
    assert len(ev)==14
    for e in ev.itertuples():
        end=int(round(e.Onset_s*125))-1
        rows.append(dict(SourceRecord=int(e.Record),StartSample=end-2500,EndSample=end,SamplingOrigin='pre_event_enrichment',SourceQuarter=np.nan,LinkedEvent=int(e.Event_ID)))
    mapping=pd.DataFrame(rows);ids=np.random.default_rng(protocol['identifier_seed'])
    records=ids.permutation(np.arange(1,31));record_names={r:f'R{int(records[r-1]):03d}' for r in range(1,31)}
    mapping['AnonRecordID']=mapping.SourceRecord.map(record_names)
    mapping['WindowID']=['W'+str(int(i)) for i in ids.choice(9000000,len(mapping),replace=False)+1000000]
    mapping=mapping.iloc[ids.permutation(len(mapping))].reset_index(drop=True);mapping['ReviewOrder']=np.arange(1,len(mapping)+1)
    assert mapping.WindowID.is_unique and (mapping.StartSample>=0).all() and (mapping.EndSample<=60000).all()
    return mapping

def select_reference_subset(summary,mapping):
    selected=[]
    d=summary.merge(mapping[['WindowID','StartSample','EndSample']],on='WindowID',validate='one_to_one')
    for _,g in d.groupby('AnonRecordID'):
        ordered=g.sort_values(['Candidate_Flag_Count','WindowID']);low=ordered.iloc[0];selected.append(low.WindowID)
        other=ordered[(ordered.EndSample<low.StartSample)|(ordered.StartSample>low.EndSample)]
        assert len(other),'Prespecified nonoverlapping subset unavailable'
        selected.append(other.iloc[-1].WindowID)
    assert len(selected)==60 and len(set(selected))==60
    return set(selected)

def panel(window,raw,filtered,peaks,emitted,title,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=np.arange(len(raw))/125;fig,ax=plt.subplots(3,1,figsize=(11,7),layout='constrained',sharex=False)
    ax[0].plot(t,raw,lw=.7,color='#334155');ax[0].set(ylabel='Raw PPG',xlim=(0,20))
    ax[1].plot(t,filtered,lw=.7,color='#0f766e')
    if len(peaks):ax[1].scatter(peaks,np.interp(peaks,t,filtered),s=18,color='#be123c',label='All accepted maxima')
    if len(emitted):ax[1].scatter(emitted,np.interp(emitted,t,filtered),s=40,facecolors='none',edgecolors='#2563eb',label='Emitted feature pulses')
    ax[1].set(ylabel='Causal filtered PPG',xlabel='Relative time (s)',xlim=(0,20));ax[1].legend(fontsize=8)
    if len(peaks)>1:ax[2].plot(peaks[1:],np.diff(peaks),marker='o',ms=3)
    ax[2].axhline(.3,color='gray',ls='--');ax[2].axhline(2,color='gray',ls='--');ax[2].set(ylabel='Accepted-peak interval(s)',xlabel='Relative time (s)',xlim=(0,20))
    fig.suptitle(f'{window} | {title}\nCandidate indicators only; no reference-annotated quality or accuracy',fontsize=11)
    for ext in ['png','pdf']:fig.savefig(Path(str(path)+'.'+ext),dpi=145,bbox_inches='tight')
    plt.close(fig)

def packet_scan():
    forbidden={'SourceRecord','StartSample','EndSample','SamplingOrigin','SourceQuarter','LinkedEvent','Event_ID','Onset_s','SpO2','Prediction','Score','Threshold','Model','Detected_Warning','Missed_Warning'}
    for p in PACKET.rglob('*.csv'):
        cols=set(pd.read_csv(p,nrows=0).columns)
        if cols&forbidden:raise AssertionError('Reviewer metadata leakage: '+str((p,cols&forbidden)))
    js=(PACKET/'review_data.js').read_text()
    for token in ['SourceRecord','Event_ID','Onset_s','SpO2','SamplingOrigin','LinkedEvent','Prediction','Threshold']:
        assert token not in js,token

def build():
    protocol=json.loads((HERE/'protocol.json').read_text())
    if (PRIVATE/'automated_review_freeze.json').exists():raise FileExistsError('Review already frozen; use separate run, never overwrite frozen review')
    PACKET.mkdir();(PACKET/'waveforms').mkdir();(PACKET/'figures').mkdir()
    mapping=sample_mapping(protocol);mapping.to_csv(PRIVATE/'source_window_mapping.csv',index=False)
    mapping_sha=sha(PRIVATE/'source_window_mapping.csv');summary=[];detrows=[];js=[];parity=[];allaccepted=[];plot_data={}
    cached=pd.read_csv(OLD/'outputs/all_pulses.csv.gz',float_precision='round_trip')
    for r in range(1,31):
        w=pd.read_csv(ROOT/f'data/bidmc_{r:02d}_Signals.csv');w.columns=w.columns.str.strip();clock=canonical_waveform_times(w['Time [s]']);raw=w.PLETH.to_numpy(float)
        filtered,accepted,emitted=instrument_source(clock,raw);want=cached[cached.Record.eq(r)].reset_index(drop=True)
        assert len(emitted)==len(want)
        cols=['time_s','peak_time_s','PPG_Amplitude','Pulse_Interval_s','Pulse_Rate_BPM','Amplitude_Slope']
        for c in cols:np.testing.assert_allclose(emitted[c],want[c],rtol=1e-9,atol=1e-9,err_msg=f'Record{r} historical pulse parity: '+c)
        parity.append(dict(SourceRecord=r,Accepted_Peaks=len(accepted),Emitted_Feature_Pulses=len(emitted),Cached_Pulses=len(want),Rejected_Long_Intervals=int(accepted.NonEmissionReason.eq('interval_above2s').sum()),Historical_Emitted_Parity='passed'))
        allaccepted.append(accepted.assign(SourceRecord=r))
        for m in mapping[mapping.SourceRecord.eq(r)].itertuples():
            start=m.StartSample;end=m.EndSample;t0=clock[start];t1=clock[end];segment=raw[start:end+1];f=filtered[start:end+1]
            det=accepted[accepted.PeakTime.ge(t0)&accepted.ConfirmationTime.le(t1)].copy();peak=det.PeakTime.to_numpy()-t0;em=det.loc[det.Emitted,'PeakTime'].to_numpy()-t0
            indicators=candidate_indicators(segment,f,peak);indicators.update(WindowID=m.WindowID,AnonRecordID=m.AnonRecordID,ReviewOrder=m.ReviewOrder,Duration_s=float(t1-t0),fs_Hz=125,Emitted_Feature_Pulses=int(det.Emitted.sum()),NonEmitted_Long_Intervals=int(det.NonEmissionReason.eq('interval_above2s').sum()),ReviewMode='automated_candidate_indicators_only')
            summary.append(indicators)
            pd.DataFrame(dict(RelativeTime_s=np.arange(len(segment))/125,RawPPG=segment,FilteredPPG=f)).to_csv(PACKET/'waveforms'/f'{m.WindowID}.csv',index=False)
            for a in det.itertuples():detrows.append(dict(WindowID=m.WindowID,AnonRecordID=m.AnonRecordID,PeakRelative_s=a.PeakTime-t0,ConfirmationRelative_s=a.ConfirmationTime-t0,Amplitude=a.Amplitude,Interval_From_Previous_Accepted_s=a.PreviousAcceptedInterval,EmittedFeaturePulse=a.Emitted,NonEmissionReason=a.NonEmissionReason))
            js.append(dict(id=m.WindowID,record=m.AnonRecordID,order=m.ReviewOrder,duration=20,fs=125,raw=segment.tolist(),filtered=f.tolist(),detected=peak.tolist(),emitted=em.tolist()))
            plot_data[m.WindowID]=(segment,f,peak,em)
        print('PPG-ONLY REVIEW',r,'accepted',len(accepted),'emitted',len(emitted),flush=True)
    summary=pd.DataFrame(summary).sort_values('ReviewOrder');selected=select_reference_subset(summary,mapping)
    manifest=mapping[['WindowID','AnonRecordID','ReviewOrder']].copy();manifest['Duration_s']=20.;manifest['fs_Hz']=125;manifest['ReferenceSubset']=manifest.WindowID.isin(selected)
    manifest.to_csv(HERE/'window_review_manifest.csv',index=False);manifest.to_csv(PACKET/'window_review_manifest.csv',index=False)
    summary.to_csv(HERE/'window_quality_summary.csv',index=False);summary.to_csv(PACKET/'window_quality_summary.csv',index=False)
    pd.DataFrame(detrows).to_csv(PACKET/'detected_pulses.csv',index=False);pd.DataFrame(parity).to_csv(PRIVATE/'detector_parity.csv',index=False);pd.concat(allaccepted).to_csv(PRIVATE/'full_record_accepted_peaks.csv.gz',index=False,compression='gzip')
    packet_id='PQ'+sha(HERE/'window_review_manifest.csv')[:12]
    for entry in js:entry['reference_subset']=entry['id'] in selected
    js.sort(key=lambda w:w['order']);(PACKET/'review_data.js').write_text('const PACKET_ID='+json.dumps(packet_id)+';\nconst WINDOWS='+json.dumps(cleaned(js),allow_nan=False,separators=(',',':'))+';\n')
    dump(PACKET/'packet_metadata.json',dict(packet_id=packet_id,window_count=len(manifest),recording_count=30,manual_priority_windows=len(selected),matching_tolerance_s=.08,reference_landmark='filtered pulse maximum corroborated against raw waveform',ground_truth_available=False))
    for row in manifest.itertuples():
        raw,f,peaks,em=plot_data[row.WindowID];panel(row.WindowID,raw,f,peaks,em,'detector inspection; no quality label',PACKET/'figures'/row.WindowID)
    ranked=summary.sort_values(['Candidate_Flag_Count','WindowID']).reset_index(drop=True)
    (HERE/'figures').mkdir()
    representatives=[]
    for label,index in [('low_candidate_flag_burden',0),('mixed_candidate_indicators',len(ranked)//2),('high_candidate_flag_burden',len(ranked)-1)]:
        row=ranked.iloc[index];raw,f,peaks,em=plot_data[row.WindowID];panel(row.WindowID,raw,f,peaks,em,label,HERE/'figures'/label);representatives.append(dict(Label=label,WindowID=row.WindowID,Candidate_Flag_Count=int(row.Candidate_Flag_Count),Validated_Quality_Category=False))
    dump(HERE/'representative_windows.json',representatives)
    for name in ['ANNOTATION_GUIDE.md','review.html','review_app.js']:
        shutil.copy2(HERE/name,PACKET/name)
    dump(PACKET/'annotation_template.json',dict(packet_id=packet_id,reviewer_id='',finalized=False,windows=[]))
    packet_scan()
    # Outcome-free calculations and sampling/annotation plan sealed before outcome join.
    freeze=dict(Mode='automated_PPG_candidate_indicators_not_human_reference',PacketID=packet_id,Manual_Reviewed_Windows=0,Window_Count=len(manifest),Manual_Priority_Windows=len(selected),Source_Mapping_SHA256=mapping_sha,Protocol_SHA256=sha(HERE/'protocol.json'),Source_SHA256={n:sha(HERE/n) for n in ['build.py','quality.py','reference.py']},Frozen_Artifacts={str(p.relative_to(HERE)):sha(p) for p in PACKET.rglob('*') if p.is_file()},Root_Manifest_SHA256=sha(HERE/'window_review_manifest.csv'),Root_Summary_SHA256=sha(HERE/'window_quality_summary.csv'),No_Event_Metadata_In_Reviewer_Packet=True)
    dump(PRIVATE/'automated_review_freeze.json',freeze)
    with zipfile.ZipFile(HERE/'reviewer_packet.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(PACKET.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(PACKET)))
    print('FROZEN automated PPG review:',len(manifest),'windows;',len(selected),'prioritized for future manual reference',flush=True)

def unblind_automated():
    freeze=json.loads((PRIVATE/'automated_review_freeze.json').read_text())
    for name,want in freeze['Frozen_Artifacts'].items():assert sha(HERE/name)==want,name
    assert sha(PRIVATE/'source_window_mapping.csv')==freeze['Source_Mapping_SHA256']
    assert sha(HERE/'window_quality_summary.csv')==freeze['Root_Summary_SHA256']
    summary=pd.read_csv(HERE/'window_quality_summary.csv');mapping=pd.read_csv(PRIVATE/'source_window_mapping.csv')
    events=pd.read_csv(AUDIT/'event_failure_analysis.csv');events=events[events.Model.eq('Logistic')&events.Definition.eq('original')]
    linked=summary.merge(mapping,on=['WindowID','AnonRecordID','ReviewOrder'],validate='one_to_one');linked.to_csv(PRIVATE/'unblinded_windows.csv',index=False)
    pairs=[]
    for w in linked.itertuples():
        a=w.StartSample/125;b=w.EndSample/125
        for e in events[events.Record.eq(w.SourceRecord)&events.Evaluation_Status.eq('eligible')].itertuples():
            overlap=max(0.,min(b,e.Onset_s)-max(a,e.Onset_s-30))
            if overlap>0:pairs.append(dict(WindowID=w.WindowID,AnonRecordID=w.AnonRecordID,SourceRecord=w.SourceRecord,Event_ID=e.Event_ID,Modality=e.Modality,WarningDetected=e.Detected,PreEvent_Overlap_s=overlap,Candidate_Flag_Count=w.Candidate_Flag_Count,Long_Gap_Candidates=w.Intervals_Above_2s,Flat_Sample_Fraction=w.Flat_Sample_Fraction,SamplingOrigin=w.SamplingOrigin,Reference_Validated=False))
    associations=pd.DataFrame(pairs);associations.to_csv(PRIVATE/'event_window_associations.csv',index=False)
    description=[]
    for (modality,detected),g in associations.groupby(['Modality','WarningDetected']):
        description.append(dict(Modality=modality,WarningDetected=bool(detected),WindowEvent_Pairs=len(g),Unique_Windows=g.WindowID.nunique(),Unique_Events=len(g[['SourceRecord','Event_ID']].drop_duplicates()),Unique_Recordings=g.SourceRecord.nunique(),Median_Candidate_Flag_Count=g.Candidate_Flag_Count.median(),Median_Long_Gap_Candidates=g.Long_Gap_Candidates.median(),Median_Flat_Sample_Fraction=g.Flat_Sample_Fraction.median(),Interpretation='automated descriptive proxy association; clustered/overlapping; no cause or precision/recall'))
    pd.DataFrame(description).to_csv(PRIVATE/'automated_association_summary.csv',index=False)
    dump(HERE/'reference_availability.json',dict(ReviewMode='automated',Processed_Windows=len(summary),Processed_Recordings=30,Human_Annotated_Windows=0,Human_Reviewers=0,Detector_Precision=None,Detector_Recall=None,Matched_Reference_Pulses=None,Verified_Feature_Corrections=0,Feature_Impact_Available=False,Manual_Pulse_Detection_Metrics_CSV_Created=False,Manual_Feature_Impact_CSV_Created=False,Automated_Freeze_Verified_Before_Metadata_Join=True,Blinded_Human_Validation=False))
    print('Automated-only unblinding after freeze; human reference metrics remain unavailable',flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['build','unblind'],required=True);args=parser.parse_args()
    build() if args.stage=='build' else unblind_automated()
