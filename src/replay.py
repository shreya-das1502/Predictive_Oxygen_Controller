"""Offline causal raw-waveform replay with a persisted demonstration model."""
import argparse
from pathlib import Path
import pandas as pd
from .config import OUTPUT_DIR,DATA_DIR
from .controller import Controller
from .dataset import load_csv
from .preprocessing import canonical_waveform_times


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,default=OUTPUT_DIR/'models/Logistic.joblib');p.add_argument('--record',type=int,default=12);p.add_argument('--output',type=Path,default=OUTPUT_DIR/'demo_replay.csv');a=p.parse_args()
    raw=load_csv(DATA_DIR/f'bidmc_{a.record:02d}_Signals.csv',['Time [s]','PLETH'])
    clock=canonical_waveform_times(raw['Time [s]']);controller=Controller.load(a.model);rows=[]
    for t,v in zip(clock,raw.PLETH):
        result=controller.update_waveform(float(t),float(v))
        if result is not None:rows.append(result)
    pd.DataFrame(rows).to_csv(a.output,index=False);print(f'{len(rows)} causal predictions saved to {a.output}. Full-data model replay is a demonstration, not held-out evaluation.')
if __name__=='__main__':main()
