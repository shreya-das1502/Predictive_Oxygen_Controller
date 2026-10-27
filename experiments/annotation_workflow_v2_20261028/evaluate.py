"""Validate actual human files; evaluate only after validation and hash freeze."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .validation import load_json,validate_payload,TOLERANCE
from .matching import score_window

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'experiments/pulse_quality_20261027T233919_IST/reviewer_packet'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value

def dump(path,value):Path(path).write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')

def context():
    sources=load_json(HERE/'SOURCE_ARTIFACTS.json')
    for name,w in sources['source_artifact_sha256'].items():
        if sha(ROOT/name)!=w:raise ValueError('Original packet artifact changed: '+name)
    manifest=pd.read_csv(OLD/'window_review_manifest.csv');metadata=load_json(OLD/'packet_metadata.json');detected=pd.read_csv(OLD/'detected_pulses.csv',float_precision='round_trip')
    if not manifest.WindowID.is_unique or len(manifest)!=134 or int(manifest.ReferenceSubset.sum())!=60:raise ValueError('Packet manifest integrity failure')
    for ident,g in detected.groupby('WindowID'):
        if ident not in set(manifest.WindowID) or not np.isfinite(g.PeakRelative_s).all() or (np.diff(g.PeakRelative_s)<=0).any():raise ValueError('Invalid source detections')
    return manifest,metadata,detected,sources

def validate_files(paths,manifest,packet_id,require_priority=True):
    validated=[];reviewers=set()
    for path in paths:
        path=Path(path)
        if not path.is_file():raise ValueError('Annotation file does not exist: '+str(path))
        payload=load_json(path);rows,coverage=validate_payload(payload,manifest,packet_id,require_priority)
        reviewer=payload['reviewer_id']
        if reviewer in reviewers:raise ValueError('Duplicate reviewer_id across files; keep revisions separate')
        reviewers.add(reviewer);validated.append(dict(path=str(path),sha256=sha(path),payload=payload,rows=rows,coverage=coverage))
    if not validated:raise ValueError('Actual finalized human annotation files are required')
    return validated

def ratio(a,b):return float(a/b) if b else np.nan

def interval(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)];q=np.quantile(x,[.025,.975]) if len(x) else [np.nan,np.nan]
    return dict(Lower95=float(q[0]),Upper95=float(q[1]),DefinedReplicates=len(x),Repetitions=2000,ResamplingUnit='pseudonymous recording')

def write_evaluation(validated,manifest,metadata,detected,out,source_checks):
    """Internal engine; tests use explicitly synthetic packets in temporary folders."""
    out=Path(out)
    if out.exists():raise ValueError('Refusing to overwrite an existing evaluation directory')
    out.mkdir(parents=True)
    dump(out/'reference_freeze.json',dict(PacketID=metadata['packet_id'],WorkflowRevision=2,MatchingTolerance_s=TOLERANCE,Definitions='Filtered peak times; maximum-cardinality/minimum-absolute-error one-to-one matching within each scored connected region; no matching across excluded gaps',Annotations=[dict(File=r['path'],SHA256=r['sha256'],Payload=r['payload'],Coverage=r['coverage']) for r in validated],Code_SHA256={p.name:sha(p) for p in [HERE/'validation.py',HERE/'matching.py',HERE/'evaluate.py']},SourceArtifactChecks=source_checks,EventMetadataJoined=False,ClinicalValidation=False))
    indexed=manifest.set_index('WindowID');per=[];timing=[]
    for data in validated:
        reviewer=data['payload']['reviewer_id']
        for row in data['rows']:
            ident=row['window_id'];record=indexed.loc[ident,'AnonRecordID'];d=detected.loc[detected.WindowID.eq(ident),'PeakRelative_s'].to_numpy(float)
            counts,pairs=score_window(d,row)
            counts.update(ReviewerID=reviewer,WindowID=ident,AnonRecordID=record,Quality=row['quality'],WindowDuration_s=float(indexed.loc[ident,'Duration_s']),PriorityWindow=bool(indexed.loc[ident,'ReferenceSubset']),DetectorOverlayEverShown=row['detector_overlay_ever_shown'])
            counts['Precision']=ratio(counts['Matched'],counts['Detections']) if counts['Scorable'] else np.nan
            counts['Recall']=ratio(counts['Matched'],counts['ReferencePulses']) if counts['Scorable'] else np.nan
            if counts['Scorable']:
                assert counts['Matched']+counts['ExtraDetections']==counts['Detections']
                assert counts['Matched']+counts['MissedReferences']==counts['ReferencePulses']
            per.append(counts)
            for pair in pairs:timing.append(dict(ReviewerID=reviewer,WindowID=ident,AnonRecordID=record,**pair))
    per=pd.DataFrame(per);timing=pd.DataFrame(timing,columns=['ReviewerID','WindowID','AnonRecordID','DetectionTime_s','ReferenceTime_s','TimingError_s'])
    per.to_csv(out/'pulse_detection_metrics.csv',index=False);timing.to_csv(out/'matched_timing_errors.csv',index=False)
    summaries=[];recordings=[];cis=[]
    counts_cols=['Matched','Detections','ReferencePulses','ExtraDetections','MissedReferences']
    for reviewer,g in per.groupby('ReviewerID'):
        table=g.groupby('AnonRecordID')[counts_cols+['ScoredDuration_s','WindowDuration_s']].sum(min_count=1).sort_index()
        table['ReviewedWindows']=g.groupby('AnonRecordID').size();table['ScorableWindows']=g.groupby('AnonRecordID').Scorable.sum();table['UninterpretableWindows']=table.ReviewedWindows-table.ScorableWindows
        table['Precision']=[ratio(a,b) for a,b in zip(table.Matched,table.Detections)];table['Recall']=[ratio(a,b) for a,b in zip(table.Matched,table.ReferencePulses)]
        recordings.append(table.reset_index().assign(ReviewerID=reviewer))
        scored=g[g.Scorable];m=int(scored.Matched.sum());nd=int(scored.Detections.sum());nr=int(scored.ReferencePulses.sum());e=timing[timing.ReviewerID.eq(reviewer)]
        annotation=next(r for r in validated if r['payload']['reviewer_id']==reviewer)
        summaries.append(dict(ReviewerID=reviewer,**annotation['coverage'],ScorableWindows=len(scored),UninterpretableWindows=int((~g.Scorable).sum()),Matched=m,TimingErrorDenominator_Matches=m,ExtraDetections=nd-m,MissedReferences=nr-m,PrecisionDenominator_Detections=nd,RecallDenominator_References=nr,Precision=ratio(m,nd),Recall=ratio(m,nr),MedianSignedTimingError_s=e.TimingError_s.median(),MeanSignedTimingError_s=e.TimingError_s.mean(),MedianAbsoluteTimingError_s=e.TimingError_s.abs().median(),MeanAbsoluteTimingError_s=e.TimingError_s.abs().mean(),P95AbsoluteTimingError_s=e.TimingError_s.abs().quantile(.95),ScoredDuration_s=float(g.ScoredDuration_s.sum()),ReviewedWindowDuration_s=float(g.WindowDuration_s.sum()),AccuracyAssessable=bool(len(scored) and (nd or nr)),SamplingDesign='Enriched/flag-stratified window review, conditional on annotated regions; not population accuracy or independent patients'))
        # Reset the specified seed per reviewer so file ordering cannot change intervals.
        records=table.index.tolist();n=len(records);rng=np.random.default_rng(2026102703);draw=rng.integers(0,n,size=(2000,n));weights=np.array([np.bincount(r,minlength=n) for r in draw])
        totals={c:weights@table[c].fillna(0).to_numpy() for c in counts_cols}
        for name,denom in [('Precision','Detections'),('Recall','ReferencePulses')]:
            good=totals[denom]>0;values=totals['Matched'][good]/totals[denom][good];cis.append(dict(ReviewerID=reviewer,Metric=name,**interval(values),RecordingClusters=n))
        timing_boot={k:np.full(2000,np.nan) for k in ['MedianSignedTimingError_s','MedianAbsoluteTimingError_s','MeanAbsoluteTimingError_s']}
        if len(e):
            columns=np.array([records.index(r) for r in e.AnonRecordID]);values=e.TimingError_s.to_numpy()
            for i,w in enumerate(weights):
                sample=np.repeat(values,w[columns])
                if len(sample):
                    timing_boot['MedianSignedTimingError_s'][i]=np.median(sample);timing_boot['MedianAbsoluteTimingError_s'][i]=np.median(np.abs(sample));timing_boot['MeanAbsoluteTimingError_s'][i]=np.mean(np.abs(sample))
        for name,values in timing_boot.items():cis.append(dict(ReviewerID=reviewer,Metric=name,**interval(values),RecordingClusters=n))
    pd.DataFrame(summaries).to_csv(out/'detector_summary.csv',index=False);pd.concat(recordings,ignore_index=True).to_csv(out/'per_recording_metrics.csv',index=False);pd.DataFrame(cis).to_csv(out/'recording_bootstrap_intervals.csv',index=False)
    dump(out/'evaluation_status.json',dict(Status='reference_evaluation_completed',ReferenceReviewers=len(validated),PacketID=metadata['packet_id'],EventMetadataJoined=False,ScientificModelsRefitted=False,ClinicalValidityEstablished=False,Uncertainty='2000 whole-recording draws; windows within recordings stay together; conditional on enriched/flag-stratified reviewed regions; few clusters may give unstable intervals',NoUndefinedMetricReplacedByZero=True))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--annotations',type=Path,action='append',required=True);parser.add_argument('--validate-only',action='store_true');parser.add_argument('--allow-partial',action='store_true',help='Explicit exploratory partial review; never label as completion of the60-window priority set');parser.add_argument('--out',type=Path);args=parser.parse_args()
    try:
        manifest,metadata,detected,sources=context();validated=validate_files(args.annotations,manifest,metadata['packet_id'],not args.allow_partial)
        if args.validate_only:
            print(json.dumps(dict(Status='valid_final_annotations',MetricsCalculated=False,Reviewers=[dict(ReviewerID=r['payload']['reviewer_id'],**r['coverage']) for r in validated]),indent=2));return
        if args.out is None:raise ValueError('--out is required for evaluation; use --validate-only to check without outputs')
        parent=HERE/'manual_runs'
        if parent.is_symlink():raise ValueError('manual_runs must not be a symlink')
        out=args.out.resolve()
        if parent.resolve() not in out.parents:raise ValueError('--out must be a NEW directory below '+str(parent))
        write_evaluation(validated,manifest,metadata,detected,out,sources);print('Reference evaluation saved:',out)
    except (ValueError,OSError) as e:parser.exit(2,'Annotation evaluation rejected: '+str(e)+'\n')

if __name__=='__main__':main()
