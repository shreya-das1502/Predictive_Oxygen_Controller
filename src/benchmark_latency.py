"""Warm single-sample and waveform-to-alert timings; no training in benchmarks."""
import argparse
import hashlib
from datetime import datetime, timezone, timedelta
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits, threadpool_info
from .config import OUTPUT_DIR, FEATURES, BASE_FEATURES, TIMING
from .model import FittedModel
from .features import FeatureState
from .controller import Controller
from .preprocessing import canonical_waveform_times


def timed(function,repetitions=TIMING['repetitions'],warmup=TIMING['warmup']):
    for _ in range(warmup):function()
    timings=np.empty(repetitions)
    for i in range(repetitions):
        start=time.perf_counter_ns();function();timings[i]=(time.perf_counter_ns()-start)/1e6
    return summarize(timings)


def summarize(timings):
    return {'Median_ms':float(np.median(timings)),'P95_ms':float(np.percentile(timings,95)),
            'P99_ms':float(np.percentile(timings,99)) if len(timings)>=1000 else np.nan,
            'P99_Reliable':len(timings)>=1000,'Mean_ms':float(np.mean(timings)),
            'Max_ms':float(np.max(timings)),'Throughput_per_s':float(1000/np.mean(timings)),
            'Repetitions':len(timings)}


def environment():
    import numpy,pandas,scipy,sklearn,matplotlib,joblib,threadpoolctl
    cpu='unknown'
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):cpu=line.split(':',1)[1].strip();break
    except OSError:pass
    return {'python':platform.python_version(),'platform':platform.platform(),'machine':platform.machine(),'cpu':cpu,
            'logical_cpus':os.cpu_count(),'dependencies':{m.__name__:m.__version__ for m in [numpy,pandas,scipy,sklearn,matplotlib,joblib,threadpoolctl]},
            'threadpools':threadpool_info(),'methodology':TIMING,
            'measured_at':datetime.now(timezone(timedelta(hours=5,minutes=30))).isoformat(),
            'code_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((OUTPUT_DIR.parents[1]/'src').glob('*.py'))}}


def benchmark_all(output=OUTPUT_DIR):
    output=Path(output);data=pd.read_csv(output/'causal_dataset.csv');row=data.iloc[0].to_dict();x=data[FEATURES].iloc[:1].to_numpy(float)
    rows=[]
    with threadpool_limits(limits=1):
        for path in sorted((output/'models').glob('*.joblib')):
            start=time.perf_counter_ns();model=FittedModel.load(path);load_ms=(time.perf_counter_ns()-start)/1e6
            state=FeatureState();controller=Controller(model);step=[0]
            def pulse():
                step[0]+=1;return controller.predict({**row,'time_s':step[0]*row['Pulse_Interval_s']})
            z=model.scaler.transform(model.imputer.transform(x));score=model.score(x)[0]
            est=lambda: model.estimator.score(z) if model.name=='ELM' else model.estimator.predict_proba(z)[:,1]
            calls={'feature_calculation':lambda:state.update(row),'model_only':est,'transforms_and_model':lambda:model.score(x),
                   'thresholding':lambda:bool(score>=model.threshold),'pulse_to_alert':pulse}
            for component,call in calls.items():rows.append({'Model':model.name,'Component':component,'Model_Load_ms':load_ms,**timed(call)})
            # Replay raw waveform samples once; warm-up first 100 samples, classify pulse-producing samples separately.
            raw=pd.read_csv('data/bidmc_01_Signals.csv');raw.columns=raw.columns.str.strip()
            replay=Controller(model);regular=[];pulse_times=[]
            for i,(t,v) in enumerate(zip(canonical_waveform_times(raw['Time [s]'])[:20001],raw.PLETH.iloc[:20001])):
                start=time.perf_counter_ns();result=replay.update_waveform(float(t),float(v));dt=(time.perf_counter_ns()-start)/1e6
                if i>=100:
                    regular.append(dt)
                    if result is not None:pulse_times.append(dt)
            rows.append({'Model':model.name,'Component':'waveform_sample_to_alert','Model_Load_ms':load_ms,**summarize(regular)})
            rows.append({'Model':model.name,'Component':'waveform_pulse_producing_sample','Model_Load_ms':load_ms,**summarize(pulse_times)})
    result=pd.DataFrame(rows);result.to_csv(output/'latency_benchmark.csv',index=False)
    with threadpool_limits(limits=1):
        (output/'benchmark_environment.json').write_text(json.dumps(environment(),indent=2))
    if (output/'model_comparison.csv').exists():
        comp=pd.read_csv(output/'model_comparison.csv').drop(columns=['Median_ms','P95_ms','P99_ms'],errors='ignore')
        comp=comp.merge(result[result.Component=='pulse_to_alert'][['Model','Median_ms','P95_ms','P99_ms']],on='Model')
        comp.to_csv(output/'model_comparison.csv',index=False)
    if (output/'run_manifest.json').exists():
        manifest=json.loads((output/'run_manifest.json').read_text());manifest['benchmark_completed']=True
        manifest['benchmark_environment_file']='benchmark_environment.json'
        (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUTPUT_DIR);a=parser.parse_args()
    print(benchmark_all(a.output).to_string(index=False))
if __name__=='__main__':main()
