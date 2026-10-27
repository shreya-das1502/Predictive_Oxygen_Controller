"""Post-freeze automated candidate associations; no human accuracy or causal claim."""
import json
import numpy as np
import pandas as pd
from .build import HERE,PRIVATE,AUDIT,sha,dump

def main():
    freeze=json.loads((PRIVATE/'automated_review_freeze.json').read_text())
    assert sha(HERE/'window_quality_summary.csv')==freeze['Root_Summary_SHA256']
    assert sha(PRIVATE/'source_window_mapping.csv')==freeze['Source_Mapping_SHA256']
    plan=dict(Mode='automated exploratory after PPG freeze; no blinded human validation',Case='Exact14 enriched pre20s windows, one per eligible onset regardless of warning',Control='Uniform time-stratified windows from the SAME recording with zero overlap of any eligible pre30s interval',Metric='Case candidate flag count minus median count of available same-record controls; long-gap/flat/jump indicators also retained',Inference='Descriptive only, repeated events clustered by recording; no p-value or causal interpretation',Source_SHA256=sha(__file__))
    plan_path=PRIVATE/'association_analysis_plan.json'
    if plan_path.exists():raise FileExistsError('Association plan already frozen')
    dump(plan_path,plan)
    mapping=pd.read_csv(PRIVATE/'source_window_mapping.csv');quality=pd.read_csv(HERE/'window_quality_summary.csv');data=mapping.merge(quality,on=['WindowID','AnonRecordID','ReviewOrder'],validate='one_to_one')
    event=pd.read_csv(AUDIT/'event_failure_analysis.csv');event=event[event.Model.eq('Logistic')&event.Definition.eq('original')&event.Evaluation_Status.eq('eligible')]
    times=event[['Record','Event_ID','Onset_s']].drop_duplicates();rows=[]
    for w in data[data.SamplingOrigin.eq('pre_event_enrichment')].itertuples():
        controls=data[data.SourceRecord.eq(w.SourceRecord)&data.SamplingOrigin.eq('time_stratified')].copy()
        good=[]
        for c in controls.itertuples():
            a,b=c.StartSample/125,c.EndSample/125
            good.append(not any(max(a,e.Onset_s-30)<min(b,e.Onset_s) for e in times[times.Record.eq(w.SourceRecord)].itertuples()))
        controls=controls[np.asarray(good,bool)]
        median=controls.Candidate_Flag_Count.median()
        for e in event[event.Record.eq(w.SourceRecord)&event.Event_ID.eq(w.LinkedEvent)].itertuples():
            rows.append(dict(WindowID=w.WindowID,AnonRecordID=w.AnonRecordID,SourceRecord=w.SourceRecord,Event_ID=int(w.LinkedEvent),Modality=e.Modality,WarningDetected=e.Detected,Case_Candidate_Flag_Count=w.Candidate_Flag_Count,Same_Record_Control_Windows=len(controls),Control_Median_Candidate_Flag_Count=median,Case_Minus_Control_Median_Flags=w.Candidate_Flag_Count-median,Flat_Sample_Fraction=w.Flat_Sample_Fraction,Abrupt_Difference_Candidates=w.Abrupt_Difference_Candidates,Interval_CV=w.Interval_CV,Validated_Detector_Error=False))
    result=pd.DataFrame(rows);assert len(result)==42
    result.to_csv(PRIVATE/'recording_matched_candidate_comparison.csv',index=False)
    summary=[]
    for key,g in result.groupby(['Modality','WarningDetected']):
        summary.append(dict(Modality=key[0],WarningDetected=bool(key[1]),Events=len(g),Recordings=g.SourceRecord.nunique(),Median_Case_Flags=g.Case_Candidate_Flag_Count.median(),Median_Case_Minus_WithinRecord_Control=g.Case_Minus_Control_Median_Flags.median(),Median_Control_Count=g.Same_Record_Control_Windows.median(),Interpretation='candidate association only; no validated quality/errors/causality'))
    pd.DataFrame(summary).to_csv(PRIVATE/'recording_matched_association_summary.csv',index=False)
    result.groupby(['Modality','WarningDetected','SourceRecord'])[['Case_Candidate_Flag_Count','Case_Minus_Control_Median_Flags']].median().to_csv(PRIVATE/'within_record_candidate_effects.csv')
    print('Post-freeze automated comparison:',len(result),'event/model rows;14 distinct enriched windows; same-record controls')

if __name__=='__main__':main()
