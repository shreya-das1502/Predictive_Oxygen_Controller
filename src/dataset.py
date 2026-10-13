"""Raw loading, source-independent label audit and explicit stage accounting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import DATA_DIR, INPUT_DATASET, OUTPUT_DIR, BASE_FEATURES, FEATURES, HORIZON, SPO2_THRESHOLD, MAX_NUMERIC_GAP
from .labeling import future_labels
from .preprocessing import extract_pulses, canonical_waveform_times
from .features import add_features


def load_csv(path, required):
    if not path.exists():
        raise FileNotFoundError(f'Missing required source: {path}; supply paired BIDMC Signals/Numerics CSVs in data/')
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if set(required)-set(df):
        raise ValueError(f'{path}: missing columns {sorted(set(required)-set(df))}')
    for c in required:
        df[c] = pd.to_numeric(df[c], errors='raise')
    return df


def validate_dataset(df):
    required = {'Record', 'time_s', 'Label', *BASE_FEATURES}
    if required-set(df):
        raise ValueError(f'Missing columns: {sorted(required-set(df))}')
    if df.Record.isna().any() or not df.Label.isin([0,1]).all():
        raise ValueError('Recording IDs must be present and labels must be binary without missing values')
    for _, g in df.groupby('Record'):
        if not np.isfinite(g.time_s).all() or (np.diff(g.time_s)<=0).any():
            raise ValueError('Invalid or duplicate/unordered timestamps within recording')
    if np.isinf(df[BASE_FEATURES].to_numpy(float)).any():
        raise ValueError('Infinite feature values; missing NaN features use training-only median imputation')
    return {'rows':len(df), 'recordings':df.Record.nunique(), 'prevalence':float(df.Label.mean()) if len(df) else None,
            'missing':df.isna().sum().to_dict(), 'duplicates':int(df.duplicated().sum()),
            'zero_variance':[c for c in BASE_FEATURES if df[c].nunique(dropna=True)<=1],
            'unexpected_ranges':{'negative_amplitude':int((df.PPG_Amplitude<0).sum()),
                                 'pulse_interval_outside_0_3_to_2':int((df.Pulse_Interval_s.notna() & ~df.Pulse_Interval_s.between(.30,2)).sum()),
                                 'pulse_rate_outside_30_to_200':int((df.Pulse_Rate_BPM.notna() & ~df.Pulse_Rate_BPM.between(30,200.01)).sum())}}


def independent_label_audit(legacy, data_dir=DATA_DIR):
    """Direct boolean time masks independent of the searchsorted label implementation."""
    rows=[]
    for record,g in legacy.groupby('Record'):
        n=load_csv(data_dir/f'bidmc_{int(record):02d}_Numerics.csv',['Time [s]','SpO2'])
        nt=n['Time [s]'].to_numpy(float);s=n.SpO2.to_numpy(float)
        if not np.isfinite(nt).all() or (np.diff(nt)<=0).any():
            raise ValueError('Raw numerics must have valid ordered unique timestamps')
        for idx,row in g.iterrows():
            t=float(row.time_s);v=s[(nt>t)&(nt<=t+HORIZON)];w=nt[(nt>t)&(nt<=t+HORIZON)]
            reason='verified'
            if t+HORIZON>nt[-1] or t<nt[0] or not len(v):reason='incomplete_future_horizon'
            elif np.max(np.diff(np.r_[t,w,t+HORIZON]))>MAX_NUMERIC_GAP:reason='future_measurement_gap'
            elif not np.isfinite(v).all() or ((v<=0)|(v>100)).any():reason='invalid_future_spo2'
            y=int(np.min(v)<=SPO2_THRESHOLD) if reason=='verified' else np.nan
            mismatch=reason!='verified' or y!=row.Label
            rows.append({'Record':int(record),'source_row':idx,'time_s':t,'Existing_Label':row.Label,'Recomputed_Label':y,
                         'Mismatch':mismatch,'Reason':reason if reason!='verified' else ('label_disagreement' if mismatch else 'verified')})
    return pd.DataFrame(rows)


def build_dataset(output=OUTPUT_DIR, data_dir=DATA_DIR, legacy_path=INPUT_DATASET):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    parts=[];audit=[];raw_quality=[];stages=[];sources={}
    pairs=sorted(data_dir.glob('bidmc_*_Signals.csv'))
    if not pairs:raise FileNotFoundError(f'No BIDMC waveform CSVs in {data_dir}')
    for path in pairs:
        r=int(path.stem.split('_')[1]);npath=data_dir/f'bidmc_{r:02d}_Numerics.csv'
        sig=load_csv(path,['Time [s]','PLETH']);num=load_csv(npath,['Time [s]','SpO2'])
        for name,frame in [('waveform',sig),('numerics',num)]:
            nums=frame.select_dtypes(include='number')
            raw_quality.append({'Record':r,'Source':name,'Rows':len(frame),'Missing_JSON':json.dumps(frame.isna().sum().to_dict()),
                                'Nonfinite_JSON':json.dumps({c:int((~np.isfinite(nums[c])).sum()) for c in nums}),
                                'Duplicate_Rows':int(frame.duplicated().sum()),'Duplicate_Timestamps':int(frame['Time [s]'].duplicated().sum()),
                                'Start_s':float(frame['Time [s]'].min()),'End_s':float(frame['Time [s]'].max()),
                                'SpO2_Out_Of_Range':int(((frame.SpO2<=0)|(frame.SpO2>100)).sum()) if name=='numerics' else 0})
        clock=canonical_waveform_times(sig['Time [s]'])
        pulses,counts=extract_pulses(clock,sig.PLETH)
        if not pulses:
            audit.append({'Record':r,'Rows':0,'Positives':0,'Negatives':0,'Empty':True,**counts});continue
        d=pd.DataFrame(pulses);d['Record']=r
        d['pulse_row']=np.arange(len(d))
        labels,minimum,reasons=future_labels(d.time_s,num['Time [s]'],num.SpO2)
        # Independent brute-force check of all regenerated valid labels.
        nt=num['Time [s]'].to_numpy();s=num.SpO2.to_numpy()
        for t,y in zip(d.time_s,labels):
            if np.isfinite(y):
                expected=int(np.min(s[(nt>t)&(nt<=t+HORIZON)])<=SPO2_THRESHOLD)
                if expected!=y:raise AssertionError('Independent label verification failed')
        d['Label']=labels;d['Future_Min_SpO2_30s']=minimum;d['Target_Status']=reasons
        # Current SpO2 is diagnostic only; backward join preserves original time coordinate.
        d=pd.merge_asof(d,num[['Time [s]','SpO2']].rename(columns={'Time [s]':'time_s','SpO2':'spo2'}).astype({'time_s':float}),on='time_s',direction='backward',tolerance=1.01)
        # Compute features before label exclusions to retain all available causal history.
        d=add_features(d)
        d.to_csv(output/f'raw_pulses_{r:02d}.csv',index=False)
        valid=d[d.Label.notna()].copy();valid.Label=valid.Label.astype(int)
        validate_dataset(valid)
        parts.append(valid)
        statuses=pd.Series(reasons).value_counts().to_dict()
        duration=float(sig['Time [s]'].iloc[-1]-sig['Time [s]'].iloc[0])
        audit.append({'Record':r,'Raw_Waveform_Rows':len(sig),'Raw_Numeric_Rows':len(num),'Pulse_Rows':len(d),'Rows':len(valid),
                      'Positives':int(valid.Label.sum()),'Negatives':int((valid.Label==0).sum()),'Prevalence':float(valid.Label.mean()) if len(valid) else np.nan,
                      'Duration_s':duration,'Eligible_Duration_s':float(valid.time_s.max()-valid.time_s.min()) if len(valid)>1 else 0,
                      'No_Positives':not valid.Label.eq(1).any(),'All_Positive':len(valid)>0 and valid.Label.eq(1).all(),
                      'Too_Short_For_Horizon':duration<HORIZON,'Empty':valid.empty,'Missing_Base':int(valid[BASE_FEATURES].isna().sum().sum()),
                      'Duplicate_Timestamps':int(valid.time_s.duplicated().sum()),'Unavailable_Targets':int(d.Label.isna().sum()),
                      **counts,**statuses})
        stages += [{'Record':r,'Stage':'waveform_to_pulses','Input_Rows':len(sig),'Output_Rows':len(d),'Rows_Removed':len(sig)-len(d),'Positive_Rows_Removed':np.nan,'Reason':'representation change; waveform samples have no timestamp labels'},
                   {'Record':r,'Stage':'rolling_features','Input_Rows':len(d),'Output_Rows':len(d),'Rows_Removed':0,'Positive_Rows_Removed':0,'Reason':'partial windows; singleton std and unavailable diff=0'},
                   {'Record':r,'Stage':'complete_future_target','Input_Rows':len(d),'Output_Rows':len(valid),'Rows_Removed':len(d)-len(valid),'Positive_Rows_Removed':np.nan,'Reason':json.dumps(statuses)+'; excluded targets unknown, never assumed negative'}]
        for f in [path,npath]:sources[str(f.relative_to(data_dir.parent))]=hashlib.sha256(f.read_bytes()).hexdigest()
        print(f'Raw {r:02d}: {len(d)} pulses, {len(valid)} verified targets, {int(valid.Label.sum())} positive',flush=True)
    if not parts:raise ValueError('No recordings produce pulse rows')
    result=pd.concat(parts,ignore_index=True);result['sample_id']=np.arange(len(result))
    if result.empty:raise ValueError('No complete future targets remain; check numeric coverage and waveform pulse quality')
    validate_dataset(result);result.to_csv(output/'causal_dataset.csv',index=False)
    pd.DataFrame(audit).to_csv(output/'dataset_audit.csv',index=False)
    pd.DataFrame(raw_quality).to_csv(output/'raw_data_quality.csv',index=False)
    pd.DataFrame(stages).to_csv(output/'preprocessing_stages.csv',index=False)
    quality=validate_dataset(result)
    quality['all_feature_missing']=result[FEATURES].isna().sum().to_dict()
    quality['all_feature_nonfinite']={c:int((~np.isfinite(result[c])).sum()) for c in FEATURES}
    quality['feature_zero_variance']=[c for c in FEATURES if result[c].nunique(dropna=True)<=1]
    (output/'dataset_summary.json').write_text(json.dumps(quality,indent=2,default=int))
    if legacy_path.exists():
        legacy=pd.read_csv(legacy_path); independent_label_audit(legacy,data_dir).to_csv(output/'label_audit.csv',index=False)
        sources[str(legacy_path)]=hashlib.sha256(legacy_path.read_bytes()).hexdigest()
    (output/'source_hashes.json').write_text(json.dumps(sources,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUTPUT_DIR);a=parser.parse_args()
    d=build_dataset(a.output);print(json.dumps(validate_dataset(d),indent=2,default=int))
if __name__=='__main__':main()
