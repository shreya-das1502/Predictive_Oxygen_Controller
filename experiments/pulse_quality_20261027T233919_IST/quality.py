"""Pure PPG candidate indicators; no event, saturation or classifier input."""
import numpy as np

def constant_runs(x):
    x=np.asarray(x,float)
    if not len(x):return []
    cuts=np.r_[0,np.flatnonzero(np.diff(x)!=0)+1,len(x)]
    return [(int(a),int(b),float(x[a])) for a,b in zip(cuts[:-1],cuts[1:])]

def candidate_indicators(raw,filtered,peak_times,fs=125):
    raw=np.asarray(raw,float);filtered=np.asarray(filtered,float);peaks=np.asarray(peak_times,float)
    finite=raw[np.isfinite(raw)];runs=constant_runs(raw);interval=np.diff(peaks)
    minimum=float(np.min(finite)) if len(finite) else np.nan;maximum=float(np.max(finite)) if len(finite) else np.nan
    flat=[(a,b) for a,b,v in runs if b-a>=int(np.ceil(.1*fs))]
    plateaus=[(a,b) for a,b,v in runs if b-a>=3 and v in (minimum,maximum)]
    diff=np.abs(np.diff(raw));diff=diff[np.isfinite(diff)];med=float(np.median(diff)) if len(diff) else np.nan
    mad=float(np.median(np.abs(diff-med))) if len(diff) else np.nan
    jump_threshold=med+10*1.4826*mad if mad>0 else np.nan
    jumps=int((diff>jump_threshold).sum()) if np.isfinite(jump_threshold) else np.nan
    row=dict(Samples=len(raw),Missing_Raw=int((~np.isfinite(raw)).sum()),Accepted_Peaks=len(peaks),Internal_Intervals=len(interval),Intervals_Below_030s=int((interval<.3-1e-10).sum()),Intervals_Above_2s=int((interval>2+1e-10).sum()),Interval_Median_s=float(np.median(interval)) if len(interval) else np.nan,Interval_CV=float(interval.std(ddof=1)/interval.mean()) if len(interval)>1 else np.nan,Longest_Internal_Interval_s=float(interval.max()) if len(interval) else np.nan,Start_Edge_Gap_s=float(peaks[0]) if len(peaks) else np.nan,End_Edge_Gap_s=float(20-peaks[-1]) if len(peaks) else np.nan,Raw_Range=maximum-minimum,Filtered_Range=float(np.ptp(filtered)) if len(filtered) else np.nan,Flat_Runs_100ms=len(flat),Flat_Sample_Fraction=sum(b-a for a,b in flat)/len(raw) if len(raw) else np.nan,Extreme_Plateau_Runs=len(plateaus),Extreme_Plateau_Sample_Fraction=sum(b-a for a,b in plateaus)/len(raw) if len(raw) else np.nan,Abrupt_Difference_Candidates=jumps,Abrupt_Difference_Threshold=jump_threshold,Jump_Assessment='candidate_relative_outlier' if np.isfinite(jump_threshold) else 'unavailable_zero_or_missing_MAD')
    flags={'no_peaks':len(peaks)==0,'long_internal_gap':row['Intervals_Above_2s']>0,'too_close':row['Intervals_Below_030s']>0,'flat_run':bool(flat),'extreme_plateau':bool(plateaus),'abrupt_difference':bool(jumps>0) if np.isfinite(jumps) else False,'variation_below_detector_amplitude':bool(row['Filtered_Range']<.03),'missing_raw':row['Missing_Raw']>0}
    row.update({'Flag_'+name:bool(value) for name,value in flags.items()});row['Candidate_Flag_Count']=sum(flags.values())
    row['Validated_SQI']=False
    return row
