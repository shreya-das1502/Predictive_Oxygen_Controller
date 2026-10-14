"""Offline causal raw-waveform replay with a persisted demonstration model."""
import argparse
from pathlib import Path
import pandas as pd
from .config import OUTPUT_DIR,DATA_DIR,HARDENED_DIR,TARGETS
from .controller import Controller
from .dataset import load_csv
from .preprocessing import canonical_waveform_times


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,default=None);p.add_argument('--record',type=int,default=12);p.add_argument('--output',type=Path,default=None);p.add_argument('--target',choices=TARGETS,default='future_low');a=p.parse_args()
    a.model=a.model or HARDENED_DIR/a.target/'models/Logistic.joblib'
    a.output=a.output or HARDENED_DIR/a.target/'demo_replay.csv'
    if a.output.resolve()==OUTPUT_DIR.resolve() or OUTPUT_DIR.resolve() in a.output.resolve().parents:
        p.error('Historical results/validated is protected; choose a new replay output')
    raw=load_csv(DATA_DIR/f'bidmc_{a.record:02d}_Signals.csv',['Time [s]','PLETH'])
    numeric=load_csv(DATA_DIR/f'bidmc_{a.record:02d}_Numerics.csv',['Time [s]','SpO2'])
    import numpy as np
    nt=numeric['Time [s]'].to_numpy(float);sv=numeric.SpO2.to_numpy(float)
    clock=canonical_waveform_times(raw['Time [s]']);controller=Controller.load(a.model);rows=[]
    for t,v in zip(clock,raw.PLETH):
        j=np.searchsorted(nt,t,side='right')-1
        current=float(sv[j]) if j>=0 else np.nan;st=float(nt[j]) if j>=0 else np.nan
        result=controller.update_waveform(float(t),float(v),current,st)
        if result is not None:rows.append(result)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(a.output,index=False);print(f'{len(rows)} causal predictions saved to {a.output}. Full-data model replay is a demonstration, not held-out evaluation.')
if __name__=='__main__':main()
