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
from .config import OUTPUT_DIR, FEATURES, BASE_FEATURES, TIMING, DATA_DIR, ROOT, BENCHMARK_RECORD, BENCHMARK_WAVEFORM_SAMPLES, P99_MIN_REPETITIONS, HARDENED_DIR, TARGETS
from .model import FittedModel
from .features import FeatureState
from .controller import Controller
from .preprocessing import canonical_waveform_times


def timed(function,repetitions=TIMING['repetitions'],warmup=TIMING['warmup'], samples=None):
    for _ in range(warmup):function()
    timings=np.empty(repetitions)
    for i in range(repetitions):
        start=time.perf_counter_ns();function();timings[i]=(time.perf_counter_ns()-start)/1e6
    if samples is not None:samples.extend(timings.tolist())
    return summarize(timings)


def summarize(timings):
    return {'Median_ms':float(np.median(timings)),'P95_ms':float(np.percentile(timings,95)),
            'P99_ms':float(np.percentile(timings,99)) if len(timings)>=P99_MIN_REPETITIONS else np.nan,
            'P99_Reliable':len(timings)>=P99_MIN_REPETITIONS,'Mean_ms':float(np.mean(timings)),
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
            'code_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').glob('*.py'))}}


def benchmark_all(output=None):
    output=Path(output) if output is not None else HARDENED_DIR/'future_low'
    if output.resolve()==OUTPUT_DIR.resolve() or OUTPUT_DIR.resolve() in output.resolve().parents:
        raise ValueError('Historical results/validated is protected; benchmark a new target run')
    if not list((output/'models').glob('*.joblib')):
        raise ValueError(f'No fitted models in {output}; run target experiments first')
    data=pd.read_csv(output/'causal_dataset.csv');row=data.iloc[0].to_dict();x=data[FEATURES].iloc[:1].to_numpy(float)
    rows=[];raw_timings=[]
    with threadpool_limits(limits=1):
        for path in sorted((output/'models').glob('*.joblib')):
            start=time.perf_counter_ns();model=FittedModel.load(path);load_ms=(time.perf_counter_ns()-start)/1e6
            def measure(component,call,**kwargs):
                samples=[]
                stats=timed(call,samples=samples,**kwargs)
                raw_timings.extend({'Model':model.name,'Component':component,'Iteration':i,'Latency_ms':v} for i,v in enumerate(samples))
                return stats
            load_stats=measure('model_loading',lambda:FittedModel.load(path),repetitions=TIMING['load_repetitions'],warmup=TIMING['load_warmup'])
            rows.append({'Model':model.name,'Component':'model_loading','Model_Load_ms':load_ms,**load_stats})
            state=FeatureState();controller=Controller(model);step=[0]
            def pulse():
                step[0]+=1
                t=step[0]*row['Pulse_Interval_s']
                return controller.predict({**row,'time_s':t,'spo2_time_s':t})
            z=model.scaler.transform(model.imputer.transform(x));score=model.score(x)[0]
            est=lambda: model.estimator.score(z) if model.name=='ELM' else model.estimator.predict_proba(z)[:,1]
            calls={'feature_calculation':lambda:state.update(row),'model_only':est,'transforms_and_model':lambda:model.score(x),
                   'thresholding':lambda:bool(score>=model.threshold),'pulse_to_alert':pulse}
            for component,call in calls.items():rows.append({'Model':model.name,'Component':component,'Model_Load_ms':load_ms,**measure(component,call)})
            # Replay raw waveform samples once; warm-up first 100 samples, classify pulse-producing samples separately.
            raw=pd.read_csv(DATA_DIR/f'bidmc_{BENCHMARK_RECORD:02d}_Signals.csv');raw.columns=raw.columns.str.strip()
            numeric=pd.read_csv(DATA_DIR/f'bidmc_{BENCHMARK_RECORD:02d}_Numerics.csv');numeric.columns=numeric.columns.str.strip()
            nt=numeric['Time [s]'].to_numpy(float);sv=numeric.SpO2.to_numpy(float)
            replay=Controller(model);regular=[];pulse_times=[]
            for i,(t,v) in enumerate(zip(canonical_waveform_times(raw['Time [s]'])[:BENCHMARK_WAVEFORM_SAMPLES],raw.PLETH.iloc[:BENCHMARK_WAVEFORM_SAMPLES])):
                j=np.searchsorted(nt,t,side='right')-1
                current=float(sv[j]) if j>=0 else np.nan;current_time=float(nt[j]) if j>=0 else np.nan
                start=time.perf_counter_ns();result=replay.update_waveform(float(t),float(v),current,current_time);dt=(time.perf_counter_ns()-start)/1e6
                if i>=TIMING['warmup']:
                    regular.append(dt)
                    if result is not None:pulse_times.append(dt)
            for component,samples in [('waveform_sample_to_alert',regular),('waveform_pulse_producing_sample',pulse_times)]:
                raw_timings.extend({'Model':model.name,'Component':component,'Iteration':i,'Latency_ms':v} for i,v in enumerate(samples))
            rows.append({'Model':model.name,'Component':'waveform_sample_to_alert','Model_Load_ms':load_ms,**summarize(regular)})
            rows.append({'Model':model.name,'Component':'waveform_pulse_producing_sample','Model_Load_ms':load_ms,**summarize(pulse_times)})
    pd.DataFrame(raw_timings).to_csv(output/'latency_samples.csv.gz',index=False,compression='gzip')
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
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--target',choices=TARGETS,default='future_low');a=parser.parse_args()
    print(benchmark_all(a.output if a.output else HARDENED_DIR/a.target).to_string(index=False))
if __name__=='__main__':main()
