"""Original maximum-count/minimum-error80ms matcher, with region isolation."""
import numpy as np
from .validation import TOLERANCE,eligible

def match_pulses(detections,references,tolerance=.08):
    d=np.asarray(detections,float);r=np.asarray(references,float)
    if tolerance<=0 or not np.isfinite(d).all() or not np.isfinite(r).all() or (np.diff(d)<=0).any() or (np.diff(r)<=0).any():raise ValueError('Finite strictly ordered pulse times and positive tolerance required')
    n,m=len(d),len(r);count=np.zeros((n+1,m+1),int);cost=np.zeros((n+1,m+1));action=np.zeros((n+1,m+1),int)
    for i in range(1,n+1):
        for j in range(1,m+1):
            options=[(count[i-1,j],cost[i-1,j],1),(count[i,j-1],cost[i,j-1],2)]
            delta=abs(d[i-1]-r[j-1])
            if delta<=tolerance+1e-12:options.append((count[i-1,j-1]+1,cost[i-1,j-1]+delta,3))
            best=min(options,key=lambda q:(-q[0],q[1],-q[2]));count[i,j],cost[i,j],action[i,j]=best
    pairs=[];i,j=n,m
    while i>0 and j>0:
        a=action[i,j]
        if a==3:pairs.append((i-1,j-1,float(d[i-1]-r[j-1])));i-=1;j-=1
        elif a==1:i-=1
        else:j-=1
    return list(reversed(pairs))

def segments(row):
    parts=[tuple(r) for r in row['scored_regions']]
    for pulse in row['pulses']:
        if pulse['certainty']!='ambiguous':continue
        lo,hi=pulse['time_s']-TOLERANCE,pulse['time_s']+TOLERANCE;new=[]
        for a,b in parts:
            if hi<a or lo>b:new.append((a,b));continue
            if a<lo:new.append((a,min(b,lo)))
            if hi<b:new.append((max(a,hi),b))
        parts=new
    return parts

def score_window(detections,row):
    if row['quality']=='uninterpretable':return dict(Scorable=False,Matched=None,Detections=None,ReferencePulses=None,MissedReferences=None,ExtraDetections=None,ScoredDuration_s=0.),[]
    det=np.asarray(detections,float);refs=np.array([p['time_s'] for p in row['pulses'] if p['certainty']=='clear']);pairs=[];nd=nr=0
    parts=segments(row)
    for a,b in parts:
        d=np.array([t for t in det if a<=t<=b and eligible(t,row)]);r=np.array([t for t in refs if a<=t<=b and eligible(t,row)])
        nd+=len(d);nr+=len(r)
        for i,j,error in match_pulses(d,r,TOLERANCE):pairs.append(dict(DetectionTime_s=float(d[i]),ReferenceTime_s=float(r[j]),TimingError_s=error))
    m=len(pairs)
    return dict(Scorable=True,Matched=m,Detections=nd,ReferencePulses=nr,MissedReferences=nr-m,ExtraDetections=nd-m,ScoredDuration_s=sum(b-a for a,b in parts),AmbiguousLandmarks=sum(p['certainty']=='ambiguous' for p in row['pulses']),Excluded_Detections=len(det)-nd,Excluded_ClearReferences=len(refs)-nr),pairs
