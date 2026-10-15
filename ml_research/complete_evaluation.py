"""Complete only evaluation from persisted fits, with per-condition checkpoints."""
import argparse
import json
import shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .baseline import ROOT,sha,verify_snapshot
from .experiment import inputs,datasets
from .core import fit_model


def finish(output,protocol):
    from .reporting import figures
    from .summarize import write_report
    from src.benchmark_latency import environment
    verify_snapshot(json.loads((output/'preservation.json').read_text()))
    figures(output)
    global_report=ROOT/'docs/ML_IMPROVEMENT_REPORT.md';existed=global_report.exists()
    write_report(output,protocol)
    local=output/'ML_IMPROVEMENT_REPORT.md';text=local.read_text()
    text=text.replace('fold ledgers include exact permitted recordings, inner splits and OOF/reference hashes.',
        'fold ledgers include exact permitted recordings and inner splits; lost fit-time hashes are explicitly unavailable for recovered groups, as explained below.')
    a=text.index('Run in the pinned repository environment;');b=text.index('Outputs include original baseline',a)
    text=text[:a]+"""The original frozen fitting run at `results/ml_improvement/run01` was interrupted after 1,074 of 1,120 model/fold groups. `recovery01` stopped on a metadata-column collision before scientific fitting/evaluation and remains preserved. `recovery02` completed the remaining 46 groups, froze all 6,720 training-only cutoffs, then was interrupted during outcome evaluation. All these files are preserved. This final evaluation reads the completed scores/cutoffs and runs **no scientific model refitting or new selection**.

```bash
MPLCONFIGDIR=/tmp/oxygen-ml-mpl OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m ml_research.complete_evaluation --source results/ml_improvement/recovery02 --output results/ml_improvement/evaluation01
```

For another independent evaluation reproduction, choose a NEW sibling output directory. `--resume` resumes verified complete condition checkpoints in the same evaluation directory after an interruption. Scientific formulas match the frozen evaluator; repeated identical metric inputs/recording decision streams are memoized and condition outputs checkpointed. This is a computation/recovery change, not a different evaluation policy. Original 30-second raw-score/event parity and the final read-only verifier are still required.

To reproduce fitting from source, use a fresh directory with `python -m ml_research.baseline --output results/ml_improvement/reproduction01` then `python -m ml_research.experiment --output results/ml_improvement/reproduction01`, using the same environment variables. No current output path is overwritten. All declared comparisons are exploratory, with no outer-result-selected winner.

**Recovery provenance limitation:** the 1,074 reused groups have saved unlabeled scores/training candidate grids, but their lost in-memory OOF/reference hashes are unavailable. Reconstructed deterministic split membership is not a recovered execution trace. No missing hash was invented. The remaining 46 groups have their actual fit hashes. Source receipt hashes, row identity checks, unchanged frozen implementation, all 765 protected files, and the independent 6,720-cutoff audit provide additional evidence without removing this limitation. Timing-only outer-training refits restore lost objects and are saved separately; they do not replace scientific predictions or thresholds.

The [verification record](ML_IMPROVEMENT_VERIFICATION.md) records actual tests/checks and exact commands. No raw data, production cache, source model, original report or historical result is modified.

"""+text[b:]
    local.write_text(text)
    if not existed:global_report.write_text(text)
    manifest=dict(environment=environment(),status='complete',protocol_sha256=sha(output/'protocol.json'),baseline_document_sha256=sha(ROOT/'docs/ML_IMPROVEMENT_BASELINE.md'),sources={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')},original_artifacts_unchanged=True,outputs={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    from .verify import verify
    (output/'verification.json').write_text(json.dumps(verify(output),indent=2))
    print('COMPLETE',output,flush=True)


def run(source,output,resume=False):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.parent!=source.parent:raise ValueError('Use an isolated sibling evaluation directory')
    protocol=json.loads((source/'protocol.json').read_text());marker=json.loads((source/'selection_complete.json').read_text())
    for file,key in [('selected_thresholds.csv','thresholds_sha256'),('protocol.json','protocol_sha256'),('outer_scores_unlabeled.csv.gz','outer_unlabeled_sha256')]:
        if sha(source/file)!=marker[key]:raise ValueError('Source selection changed')
    for name,w in protocol['implementation'].items():
        if sha(Path(__file__).parent/name)!=w:raise ValueError('Frozen source changed')
    verify_snapshot(json.loads((source/'preservation.json').read_text()))
    if not resume:
        if output.exists():raise ValueError('Output already exists; use a new directory or --resume')
        output.mkdir()
        names=['protocol.json','selected_thresholds.csv','outer_scores_unlabeled.csv.gz','selection_complete.json','preservation.json','fold_ledger.json','training_feature_analysis.csv','training_candidates.csv.gz','dataset_counts.csv','recovery.json','training_policy_summary.csv','raw_events.csv','ML_IMPROVEMENT_BASELINE.md','ML_IMPROVEMENT_VERIFICATION.md']
        names+= [p.name for p in source.glob('baseline_*')]+[p.name for p in source.glob('labels_*')]
        for name in names:shutil.copy2(source/name,output/name)
        (output/'continuation.json').write_text(json.dumps(dict(source=str(source),source_files={n:sha(source/n) for n in names},scientific_refits=0,threshold_changes=0),indent=2))
    else:
        c=json.loads((output/'continuation.json').read_text())
        if c['source']!=str(source):raise ValueError('Wrong resume source')
        for n,w in c['source_files'].items():
            if sha(output/n)!=w or sha(source/n)!=w:raise ValueError('Continuation source changed')
    numeric,events=inputs()
    with threadpool_limits(limits=1):
        from .checkpoint_evaluation import evaluate
        if not (output/'event_summary.csv').exists():evaluate(output,protocol,numeric,events)
        if not (output/'latency.csv').exists():
            from .reporting import benchmark
            models=[]
            for target in protocol['targets']:
                variants=datasets(target,numeric)
                for spec in protocol['settings']:
                    h,fs=spec['Horizon_s'],spec['FeatureSet'];data=variants[h];test=int(data.Record.max());train=data[data.Record.ne(test)];cols=protocol['features'][fs]
                    for name in protocol['model_configs']:
                        model=fit_model(train,cols,name);ref=model.score(train.loc[train.Label.eq(0),cols].to_numpy(float))
                        models.append((dict(Target=target,Horizon_s=h,FeatureSet=fs,Model=name,Outer_Test_Record=test),model,ref,train[cols].iloc[:1].to_numpy(float)))
            joblib.dump(models,output/'timing_only_models.joblib')
            benchmark(output,protocol,models)
        finish(output,protocol)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.source,a.output,a.resume)
