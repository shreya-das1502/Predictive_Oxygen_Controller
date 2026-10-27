"""Independent observation masks, alarm matching, metrics and paired estimands."""
import numpy as np
import pandas as pd

def divide(a,b):
    a,b=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
    value=np.full(a.shape,np.nan)
    return np.divide(a,b,out=value,where=b!=0)

def interval(values):
    v=np.asarray(values,float);v=v[np.isfinite(v)]
    q=np.quantile(v,[.025,.975]) if len(v) else [np.nan,np.nan]
    return dict(Lower_95=float(q[0]),Upper_95=float(q[1]),Defined_Replicates=len(v))

def valid_coverage(nt,s,a,b):
    if not len(nt) or a>b or a<nt[0] or b>nt[-1]:return False
    mask=(nt>=a)&(nt<=b);v=s[mask]
    return bool(len(v) and np.isfinite(v).all() and ((v>0)&(v<=100)).all() and np.diff(np.r_[a,nt[mask],b]).max(initial=0)<=1.01)

def target_at(t,nt,s):
    mask=(nt>t)&(nt<=t+30);v=s[mask]
    if not len(nt) or t<nt[0] or t+30>nt[-1] or not len(v):return np.nan,np.nan
    if not np.isfinite(v).all() or ((v<=0)|(v>100)).any() or np.diff(np.r_[t,nt[mask],t+30]).max(initial=0)>1.01:return np.nan,np.nan
    minimum=float(v.min());return int(minimum<=94),minimum

def independent_history(t,nt,s):
    before=np.flatnonzero(nt<=t)
    current=np.nan;age=np.nan
    if len(before):
        i=before[-1]
        if t-nt[i]<=1.01 and np.isfinite(s[i]) and 0<s[i]<=100:current=float(s[i]);age=float(t-nt[i])
    stats=[]
    for window in [10,30]:
        use=(nt>=t-window)&(nt<=t)&np.isfinite(s)&(s>0)&(s<=100);v=s[use];clock=nt[use]
        if len(v):
            centered=clock-clock.mean();denom=np.sum(centered**2)
            trend=float(np.sum(centered*(v-v.mean()))/denom) if denom>0 else 0.
            stats.append((v.mean(),trend,v.std(ddof=1) if len(v)>1 else 0.))
        else:stats.append((np.nan,np.nan,np.nan))
    return np.array([current,stats[0][0],stats[1][0],stats[0][1],stats[1][1],stats[1][2],age])

def raw_events(numeric):
    rows=[]
    for record,(nt,s) in sorted(numeric.items()):
        i=0
        while i<len(s):
            if not(np.isfinite(s[i]) and 0<s[i]<=94):i+=1;continue
            first=i;i+=1
            while i<len(s) and np.isfinite(s[i]) and 0<s[i]<=94 and nt[i]-nt[i-1]<=1.01:i+=1
            good=first>0 and np.isfinite(s[first-1]) and 94<s[first-1]<=100 and nt[first]-nt[first-1]<=1.01
            rows.append(dict(Record=int(record),Event_ID=first,Onset_s=float(nt[first]),End_s=float(nt[i-1]),Minimum_SpO2=float(s[first:i].min()),Low_Readings=i-first,Qualifying_Onset=bool(good),Sustained3=bool(i-first>=3 and nt[i-1]-nt[first]>=2),Isolated_Reading=i-first==1,Observed_Low_Span_s=float(nt[i-1]-nt[first])))
    return pd.DataFrame(rows)

def topology(frame,nt,s):
    """Independent recreation of established retrospective current-normal exposure."""
    g=frame.sort_values('time_s');g=g[g.SpO2_Current.gt(94)].reset_index(drop=True);t=g.time_s.to_numpy()
    edges=[]
    for a,b in zip(t[:-1],t[1:]):
        prior=np.flatnonzero(nt<=a);inside=s[(nt>a)&(nt<=b)]
        edges.append(bool(b-a<=2 and len(prior) and a-nt[prior[-1]]<=1.01 and np.isfinite(s[prior[-1]]) and 94<s[prior[-1]]<=100 and np.isfinite(inside).all() and ((inside>94)&(inside<=100)).all() and valid_coverage(nt,s,a,b+30)))
    edges=np.asarray(edges,bool)
    duration=np.r_[np.diff(t)*edges,0.] if len(t) else np.array([],float)
    previous=np.r_[False,edges] if len(t) else np.array([],bool)
    return g,previous,duration

def alarms(times,decisions,previous):
    times=np.asarray(times,float);p=np.asarray(decisions,bool);previous=np.asarray(previous,bool)
    if not(len(times)==len(p)==len(previous)):raise ValueError('Alarm arrays must align')
    starts=p&~(np.r_[False,p[:-1]]&previous) if len(p) else np.array([],bool)
    return times[starts],starts

def match_event_starts(events,times,starts,nt,s,definition):
    """One recording, chronological earliest unused actual start, half-open W30."""
    used=set();rows=[]
    for e in events.sort_values('Onset_s').to_dict('records'):
        lo=e['Onset_s']-30;o=e['Onset_s'];q=(times>=lo)&(times<o)
        status=('not_sustained_under_sensitivity_definition' if definition=='sustained3' and not e['Sustained3'] else
                'left_censored_or_unknown_prior' if not e['Qualifying_Onset'] else
                'incomplete_raw_lead_in' if not valid_coverage(nt,s,lo,o) else
                'no_eligible_prediction_opportunity' if not q.any() else 'eligible')
        selected=None;candidate=[i for i,st in enumerate(starts) if lo<=st<o]
        if status=='eligible':
            selected=next((i for i in candidate if i not in used),None)
            if selected is not None:used.add(selected)
        rows.append(e|dict(Definition=definition,Evaluation_Status=status,Detected=bool(selected is not None) if status=='eligible' else np.nan,First_Alert_s=float(starts[selected]) if selected is not None else np.nan,Lead_Time_s=float(o-starts[selected]) if selected is not None else np.nan,Alarm_Index=selected,Opportunities=int(q.sum()),Candidate_Starts=len(candidate),Candidates_Used_By_Earlier_Events=len([i for i in candidate if i in used and i!=selected])))
    columns=list(events.columns)+['Definition','Evaluation_Status','Detected','First_Alert_s','Lead_Time_s','Alarm_Index','Opportunities','Candidate_Starts','Candidates_Used_By_Earlier_Events']
    return pd.DataFrame(rows,columns=columns),used

def evaluate_record(frame,events,nt,s,definition):
    g,previous,duration=topology(frame,nt,s);t=g.time_s.to_numpy();p=g.Prediction.to_numpy(bool)
    starts,flags=alarms(t,p,previous);matched,used=match_event_starts(events,t,starts,nt,s,definition)
    truth=events[events.Qualifying_Onset & (events.Sustained3 if definition=='sustained3' else True)]
    false=sum(not ((truth.Onset_s>st)&(truth.Onset_s<=st+30)).any() for st in starts)
    eligible=matched[matched.Evaluation_Status.eq('eligible')];leads=eligible.loc[eligible.Detected.eq(True),'Lead_Time_s'].dropna()
    exp=float(duration.sum());active=float(duration@p)
    counts=dict(Record=int(frame.Record.iloc[0]),Definition=definition,Eligible_Events=len(eligible),Detected=int(eligible.Detected.eq(True).sum()),Missed_Events=int(eligible.Detected.eq(False).sum()),Exposure_s=exp,Alarm_Active_s=active,Off_Event_Alarm_Starts=int(false),Alarm_Starts=len(starts),Unmatched_Alarm_Starts=len(starts)-len(used),Samples=len(g),Alert_Timestamps=int(p.sum()),Sensitivity=float(divide(eligible.Detected.eq(True).sum(),len(eligible))),False_Alarm_Starts_Per_Hour=float(divide(3600*false,exp)),Alarm_Active_Fraction=float(divide(active,exp)),Lead_Median_s=float(leads.median()) if len(leads) else np.nan)
    return counts,matched,g,previous,duration,starts,flags

def check_split(metadata,universe=range(1,31)):
    test=metadata['Outer_Test_Record'];train=set(metadata['Train_Records'])
    if train!=set(universe)-{test}:raise ValueError('Outer recording separation/coverage failure')
    seen=[]
    for fold in metadata['Inner']:
        a=set(fold['train_records']);b=set(fold['validation_records'])
        if a&b or a|b!=train:raise ValueError('Inner recording separation/coverage failure')
        seen.extend(b)
    if sorted(seen)!=sorted(train):raise ValueError('Each training recording needs one OOF partition')

def check_aligned(reference,other,columns):
    if reference.sample_id.duplicated().any() or other.sample_id.duplicated().any() or set(reference.sample_id)!=set(other.sample_id):raise ValueError('Unmatched sample identities')
    a=reference.set_index('sample_id');b=other.set_index('sample_id').loc[a.index]
    for c in columns:
        if not np.array_equal(a[c].to_numpy(),b[c].to_numpy(),equal_nan=True):raise ValueError('Unmatched '+c)

def check_oof(oof,data,test_record):
    if oof.Record.eq(test_record).any():raise ValueError('Held-out record in OOF training data')
    expected=data[data.Record.ne(test_record)]
    check_aligned(expected,oof,['Record','time_s','Label'])

def cutoff_from_grid(grid,budget=5,cap=.05):
    feasible=grid[grid.False_Alarm_Starts_Per_Hour.le(budget+1e-12)&grid.Alarm_Active_Fraction.le(cap+1e-12)]
    if feasible.empty:raise ValueError('No feasible training cutoff; do not relax constraints')
    return feasible.sort_values(['Detected','Off_Event_Alarm_Starts','Alarm_Active_s','Threshold'],ascending=[False,True,True,False],kind='stable').iloc[0]

def event_bootstrap(per,weights):
    g=per.set_index('Record').reindex(range(1,31));total=lambda c:weights@g[c].to_numpy(float)
    return dict(Sensitivity=divide(total('Detected'),total('Eligible_Events')),False_Alarm_Starts_Per_Hour=divide(3600*total('Off_Event_Alarm_Starts'),total('Exposure_s')),Source_Duration_Normalized_Starts_Per_Hour=divide(3600*total('Off_Event_Alarm_Starts'),total('Source_Duration_s')),Alarm_Active_Fraction=divide(total('Alarm_Active_s'),total('Exposure_s')))

def lead_bootstrap(leads,weights):
    values=np.asarray(leads.Lead_Time_s,float);records=np.asarray(leads.Record,int)
    result=np.full(len(weights),np.nan)
    for i,w in enumerate(weights):
        sample=np.repeat(values,w[records-1])
        if len(sample):result[i]=np.median(sample)
    return result

def paired_common_leads(a,b):
    a=a[a.Detected.eq(True)][['Record','Event_ID','Lead_Time_s']];b=b[b.Detected.eq(True)][['Record','Event_ID','Lead_Time_s']]
    d=a.merge(b,on=['Record','Event_ID'],validate='one_to_one',suffixes=('_A','_B'))
    d['Lead_Time_s']=d.Lead_Time_s_A-d.Lead_Time_s_B
    return d
