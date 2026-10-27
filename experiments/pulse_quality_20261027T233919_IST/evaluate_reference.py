"""Future real-reference evaluation. No annotation input means no accuracy output."""
import argparse
import itertools
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from .build import HERE,PACKET,sha,dump
from .reference import validate_annotations,scored_pulses,match_pulses

def ratio(a,b):return float(a/b) if b else np.nan

def strict_schema(payload):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}',payload.get('reviewer_id','')):raise ValueError('Reviewer code must be1–40 letters, digits, underscore or hyphen')
    if set(payload)-{'packet_id','reviewer_id','finalized','windows'}:raise ValueError('Unexpected annotation metadata; blinded schema only')
    allowed={'window_id','quality','scored_regions','pulses','notes','reviewed_no_reference_pulses','detector_overlay_ever_shown'}
    for row in payload.get('windows',[]):
        if set(row)-allowed:raise ValueError('Unexpected window metadata')
        for p in row.get('pulses',[]):
            if set(p)-{'time_s','certainty'}:raise ValueError('Unexpected pulse metadata')

def validated_reference_rows(payload,manifest,packet_id):
    strict_schema(payload);rows=validate_annotations(payload,manifest,packet_id)
    for row in rows:
        if row['quality']!='uninterpretable':
            _,reference=scored_pulses([],row)
            if not len(reference) and row.get('reviewed_no_reference_pulses') is not True:
                raise ValueError('No clear reference inside exhaustive scoring regions; do not treat out-of-region or ambiguous marks as ground truth')
    return rows

def evaluate(paths,out):
    out=Path(out).resolve();parent=(HERE/'manual_runs').resolve()
    if parent not in out.parents or out.exists():raise ValueError('Use a NEW directory below this review/manual_runs')
    metadata=json.loads((PACKET/'packet_metadata.json').read_text());manifest=pd.read_csv(PACKET/'window_review_manifest.csv')
    accepted=pd.read_csv(PACKET/'detected_pulses.csv');inputs=[];reviewers=set()
    for path in paths:
        payload=json.loads(Path(path).read_text())
        rows=validated_reference_rows(payload,manifest,metadata['packet_id'])
        if payload['reviewer_id'] in reviewers:raise ValueError('Separate revisions from independent reviewers; duplicate reviewer code')
        reviewers.add(payload['reviewer_id']);inputs.append((payload,rows,sha(path)))
    if not inputs:raise ValueError('Real finalized reference annotations required')
    out.mkdir(parents=True)
    # Freeze references and plan BEFORE metrics or any additional metadata join.
    dump(out/'reference_freeze.json',dict(PacketID=metadata['packet_id'],References=[{'Payload':p,'SHA256':w} for p,_,w in inputs],MatchingTolerance_s=.08,MatchingSource_SHA256=sha(HERE/'reference.py'),EvaluationSource_SHA256=sha(__file__),Scoring='Exhaustive regions only; ambiguous pulse bands±80ms excluded; compare peak times',Event_Metadata_Joined=False))
    per=[];errors=[];labels=[]
    for payload,rows,_ in inputs:
        reviewer=payload['reviewer_id']
        for row in rows:
            ident=row['window_id'];record=manifest.set_index('WindowID').loc[ident,'AnonRecordID']
            detections=accepted.loc[accepted.WindowID.eq(ident),'PeakRelative_s'].to_numpy(float)
            labels.append(dict(ReviewerID=reviewer,WindowID=ident,AnonRecordID=record,Quality=row['quality'],DetectorOverlayEverShown=row.get('detector_overlay_ever_shown',False)))
            if row['quality']=='uninterpretable':
                per.append(dict(ReviewerID=reviewer,WindowID=ident,AnonRecordID=record,Quality=row['quality'],Scorable=False,Matched=0,ReferencePulses=0,Detections=0,MissedReferences=0,ExtraDetections=0,Precision=np.nan,Recall=np.nan));continue
            d,r=scored_pulses(detections,row);pairs=match_pulses(d,r,.08);m=len(pairs)
            per.append(dict(ReviewerID=reviewer,WindowID=ident,AnonRecordID=record,Quality=row['quality'],Scorable=True,Matched=m,ReferencePulses=len(r),Detections=len(d),MissedReferences=len(r)-m,ExtraDetections=len(d)-m,Precision=ratio(m,len(d)),Recall=ratio(m,len(r))))
            for i,j,error in pairs:errors.append(dict(ReviewerID=reviewer,WindowID=ident,AnonRecordID=record,DetectionTime_s=float(d[i]),ReferenceTime_s=float(r[j]),TimingError_s=error))
    per=pd.DataFrame(per);errors=pd.DataFrame(errors,columns=['ReviewerID','WindowID','AnonRecordID','DetectionTime_s','ReferenceTime_s','TimingError_s']);labels=pd.DataFrame(labels)
    per.to_csv(out/'pulse_detection_metrics.csv',index=False);errors.to_csv(out/'matched_timing_errors.csv',index=False);labels.to_csv(out/'human_quality_labels.csv',index=False)
    summaries=[];intervals=[];rng=np.random.default_rng(2026102703)
    for reviewer,g in per.groupby('ReviewerID'):
        recording=g.groupby('AnonRecordID')[['Matched','ReferencePulses','Detections','MissedReferences','ExtraDetections']].sum().sort_index();recording.to_csv(out/f'{reviewer}_recording_counts.csv')
        n=len(recording);draws=rng.integers(0,n,size=(2000,n));weights=np.array([np.bincount(x,minlength=n) for x in draws]);totals={c:weights@recording[c].to_numpy() for c in recording}
        m=g.Matched.sum();ref=g.ReferencePulses.sum();det=g.Detections.sum();err=errors[errors.ReviewerID.eq(reviewer)]
        summaries.append(dict(ReviewerID=reviewer,ReviewedWindows=len(g),ReviewedRecordings=n,UninterpretableWindows=int((~g.Scorable).sum()),Matched=int(m),MissedReferences=int(ref-m),ExtraDetections=int(det-m),ReferencePulses=int(ref),Detections=int(det),Precision=ratio(m,det),Recall=ratio(m,ref),MedianSignedTimingError_s=err.TimingError_s.median(),MedianAbsoluteTimingError_s=err.TimingError_s.abs().median(),ReviewMode='manual_reference_not_clinical_ground_truth'))
        for metric,den in [('Precision','Detections'),('Recall','ReferencePulses')]:
            valid=totals[den]>0;v=totals['Matched'][valid]/totals[den][valid];q=np.quantile(v,[.025,.975]) if len(v) else [np.nan,np.nan]
            intervals.append(dict(ReviewerID=reviewer,Metric=metric,Lower95=q[0],Upper95=q[1],DefinedReplicates=len(v),ResamplingUnit='pseudonymous recording',Repetitions=2000))
    pd.DataFrame(summaries).to_csv(out/'detector_summary.csv',index=False);pd.DataFrame(intervals).to_csv(out/'recording_bootstrap_intervals.csv',index=False)
    agreements=[]
    if len(inputs)>1:
        from sklearn.metrics import cohen_kappa_score
        for (pa,ra,_),(pb,rb,_) in itertools.combinations(inputs,2):
            a={r['window_id']:r for r in ra};b={r['window_id']:r for r in rb};common=sorted(set(a)&set(b));qa=[a[w]['quality'] for w in common];qb=[b[w]['quality'] for w in common]
            assessable=len(set(qa+qb))>1 and len(common)>0
            agreements.append(dict(ReviewerA=pa['reviewer_id'],ReviewerB=pb['reviewer_id'],CommonWindows=len(common),QualityAgreement=ratio(sum(x==y for x,y in zip(qa,qb)),len(common)),QualityKappa=float(cohen_kappa_score(qa,qb)) if assessable else np.nan,Interpretation='quality-label agreement only; independent reference concordance still needs common-region landmark review'))
        pd.DataFrame(agreements).to_csv(out/'inter_reviewer_quality_agreement.csv',index=False)
    dump(out/'reference_evaluation_status.json',dict(ReferenceReviewers=len(reviewers),InterRaterQualityAssessed=bool(agreements),EventMetadataJoined=False,FeatureImpactAvailable=False,FeatureImpactLimit='No boundary-aware manually verified feature correction has been approved/replayed; no fabricated corrected features',NoModelRefits=True,NoHistoricalSubstitution=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--annotations',type=Path,action='append',required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();evaluate(args.annotations,args.out)
