"""Strict final-annotation contract. No inference that an empty file is a review."""
import json
import math
import re

TOLERANCE=.08
QUALITIES={'adequately_interpretable','partially_interpretable','uninterpretable'}

def load_json(path):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out:raise ValueError('Duplicate JSON key: '+k)
            out[k]=v
        return out
    def constant(value):raise ValueError('Non-finite JSON constant: '+value)
    try:
        with open(path,encoding='utf-8') as f:return json.load(f,object_pairs_hook=pairs,parse_constant=constant)
    except (json.JSONDecodeError,UnicodeDecodeError) as e:raise ValueError('Malformed UTF-8 JSON: '+str(e)) from e

def keys(obj,required,label):
    if not isinstance(obj,dict):raise ValueError(label+' must be an object')
    missing=set(required)-set(obj);unknown=set(obj)-set(required)
    if missing or unknown:raise ValueError(f'{label}: missing={sorted(missing)}, unknown={sorted(unknown)}')

def number(x,label):
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):raise ValueError(label+' must be a finite number, not boolean/string')
    return float(x)

def eligible(t,row):
    return any(a<=t<=b for a,b in row['scored_regions']) and not any(abs(t-p['time_s'])<=TOLERANCE+1e-12 for p in row['pulses'] if p['certainty']=='ambiguous')

def validate_payload(payload,manifest,packet_id,require_priority=True):
    keys(payload,['packet_id','reviewer_id','finalized','windows'],'annotation')
    if payload['packet_id']!=packet_id:raise ValueError('Wrong packet_id')
    if not isinstance(payload['reviewer_id'],str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,40}',payload['reviewer_id']):raise ValueError('reviewer_id must be1–40 letters/digits/underscore/hyphen')
    if payload['finalized'] is not True:raise ValueError('Draft or unfinalized annotations cannot be evaluated')
    if not isinstance(payload['windows'],list) or not payload['windows']:raise ValueError('windows must contain actual reviewed windows')
    indexed=manifest.set_index('WindowID');allowed=set(indexed.index);seen=set();rows=[]
    for row in payload['windows']:
        keys(row,['window_id','quality','scored_regions','pulses','notes','reviewed_no_reference_pulses','detector_overlay_ever_shown'],'window')
        ident=row['window_id']
        if not isinstance(ident,str) or ident not in allowed:raise ValueError('Unknown window_id: '+str(ident))
        if ident in seen:raise ValueError('Duplicate window_id: '+ident)
        seen.add(ident);duration=float(indexed.loc[ident,'Duration_s'])
        if not isinstance(row['quality'],str) or row['quality'] not in QUALITIES:raise ValueError(ident+': invalid/unreviewed quality label')
        if not isinstance(row['notes'],str):raise ValueError(ident+': notes must be a string')
        for name in ['reviewed_no_reference_pulses','detector_overlay_ever_shown']:
            if type(row[name]) is not bool:raise ValueError(ident+': '+name+' must be boolean')
        if not isinstance(row['scored_regions'],list) or not isinstance(row['pulses'],list):raise ValueError(ident+': regions/pulses must be arrays')
        previous=-1.
        for region in row['scored_regions']:
            if not isinstance(region,list) or len(region)!=2:raise ValueError(ident+': each region requires exactly[start,end]')
            a,b=[number(v,ident+' region') for v in region]
            if not 0<=a<b<=duration or a<=previous:raise ValueError(ident+': regions must be ordered, separated, nonoverlapping and inside the window; merge touching regions')
            previous=b
        previous=-1.
        for pulse in row['pulses']:
            keys(pulse,['time_s','certainty'],ident+' pulse');t=number(pulse['time_s'],ident+' pulse time')
            if not 0<=t<=duration:raise ValueError(ident+': pulse time outside relevant window')
            if t<=previous:raise ValueError(ident+': pulse times must be strictly increasing; duplicates/unsorted values rejected')
            previous=t
            if not isinstance(pulse['certainty'],str) or pulse['certainty'] not in {'clear','ambiguous'}:raise ValueError(ident+': invalid certainty')
        if row['quality']!='adequately_interpretable' or any(p['certainty']=='ambiguous' for p in row['pulses']):
            if not row['notes'].strip():raise ValueError(ident+': partial/uninterpretable/ambiguous review requires notes')
        clear=[p for p in row['pulses'] if p['certainty']=='clear'];usable=[p for p in clear if eligible(p['time_s'],row)]
        if row['quality']=='uninterpretable':
            if row['scored_regions'] or clear or row['reviewed_no_reference_pulses']:raise ValueError(ident+': uninterpretable means unknown reference; no scored regions, clear pulses or confirmed-empty reference')
        else:
            if not row['scored_regions']:raise ValueError(ident+': exhaustive scoring regions required')
            if usable and row['reviewed_no_reference_pulses']:raise ValueError(ident+': confirmed-empty reference contradicts usable clear pulses')
            if not usable and not row['reviewed_no_reference_pulses']:raise ValueError(ident+': no clear reference in scored regions; empty/outside/ambiguous marks are not ground truth')
        rows.append(row)
    priority=set(manifest.loc[manifest.ReferenceSubset,'WindowID']);missing=priority-seen
    if require_priority and missing:raise ValueError(f'Incomplete priority review: {len(priority)-len(missing)}/{len(priority)} present; missing {sorted(missing)[:8]}')
    return rows,dict(reviewed_windows=len(rows),priority_reviewed=len(priority&seen),priority_required=len(priority),priority_complete=not missing,reviewed_recordings=int(indexed.loc[list(seen),'AnonRecordID'].nunique()))
