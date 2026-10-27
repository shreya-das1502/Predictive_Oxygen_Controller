"""Reference matching and strict annotation validation; requires real final input."""
import numpy as np

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

def validate_annotations(payload,manifest,packet_id):
    if payload.get('packet_id')!=packet_id or payload.get('finalized') is not True:raise ValueError('Matching packet and explicit finalization required')
    reviewer=payload.get('reviewer_id','').strip()
    if not reviewer:raise ValueError('Nonidentifying reviewer code required')
    allowed=set(manifest.WindowID);seen=set();validated=[]
    for row in payload.get('windows',[]):
        ident=row.get('window_id');quality=row.get('quality')
        if ident not in allowed or ident in seen:raise ValueError('Unknown/duplicate window')
        seen.add(ident)
        if quality not in ['adequately_interpretable','partially_interpretable','uninterpretable']:raise ValueError('Only actually reviewed quality labels may be finalized')
        regions=row.get('scored_regions',[]);previous=-1
        for start,end in regions:
            if not(0<=start<end<=20) or start<previous:raise ValueError('Finite ordered nonoverlapping scoring regions required')
            previous=end
        if quality=='uninterpretable' and regions:raise ValueError('Uninterpretable window cannot supply a ground-truth scoring region')
        if quality!='uninterpretable' and not regions:raise ValueError('Explicit exhaustive regions required')
        pulses=row.get('pulses',[]);times=[]
        for p in pulses:
            if not np.isfinite(p['time_s']) or not 0<=p['time_s']<=20 or p.get('certainty') not in ['clear','ambiguous']:raise ValueError('Invalid reference pulse')
            times.append(p['time_s'])
        if len(times)!=len(set(times)):raise ValueError('Duplicate reference times')
        clear=[p['time_s'] for p in pulses if p['certainty']=='clear']
        if regions and not clear and not row.get('reviewed_no_reference_pulses',False):raise ValueError('Empty reference requires explicit reviewed-no-pulse confirmation; never treat blank input as ground truth')
        validated.append(dict(row,reviewer_id=reviewer,pulses=sorted(pulses,key=lambda p:p['time_s'])))
    if not validated:raise ValueError('No reviewed windows; no accuracy can be evaluated')
    return validated

def scored_pulses(detections,row,tolerance=.08):
    regions=row['scored_regions'];ambiguous=np.array([p['time_s'] for p in row['pulses'] if p['certainty']=='ambiguous'])
    def eligible(t):return any(a<=t<=b for a,b in regions) and not np.any(np.abs(ambiguous-t)<=tolerance)
    det=np.array(sorted(t for t in detections if eligible(t)))
    ref=np.array(sorted(p['time_s'] for p in row['pulses'] if p['certainty']=='clear' and eligible(p['time_s'])))
    return det,ref
