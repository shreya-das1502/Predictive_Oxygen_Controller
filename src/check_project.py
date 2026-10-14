"""Read-only environment, dataset, documentation and saved-run quality checks."""
import argparse
import importlib.metadata
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from . import config
from .evaluation import metrics
from .provenance import fingerprint, cache_identity, cache_matches


def verify_environment():
    versions={}
    for line in (config.ROOT/'requirements.txt').read_text().splitlines():
        name,expected=line.split('==');actual=importlib.metadata.version(name)
        if actual!=expected:raise ValueError(f'{name}: expected {expected}, installed {actual}')
        versions[name]=actual
    return versions


def audit_data():
    records=[]
    for signal in sorted(config.DATA_DIR.glob('bidmc_*_Signals.csv')):
        numeric=signal.with_name(signal.name.replace('_Signals','_Numerics'))
        n=pd.read_csv(numeric);n.columns=n.columns.str.strip()
        records.append({'Record':int(signal.stem.split('_')[1]),'Numeric_Rows':len(n),'Missing_SpO2':int(n.SpO2.isna().sum()),
                        'Observed_Low_Readings':int(n.SpO2.le(config.SPO2_THRESHOLD).sum())})
    if not records:raise FileNotFoundError('No raw BIDMC pairs under data/')
    return records


def verify_links():
    errors=[];checked=0
    for file in [config.ROOT/'README.md',*sorted((config.ROOT/'docs').glob('*.md')),*sorted((config.ROOT/'archive').glob('*.md')),*sorted((config.ROOT/'results').rglob('*.md'))]:
        for link in re.findall(r'\]\(([^)]+)\)',file.read_text()):
            target=link.strip('<>').split('#')[0]
            if not target or re.match(r'[a-zA-Z]+://',target) or target.startswith('mailto:'):continue
            target=re.sub(r':\d+$','',target);path=Path(target)
            path=path if path.is_absolute() else file.parent/path
            checked+=1
            if not path.exists():errors.append(f'{file.relative_to(config.ROOT)} -> {link}')
    if errors:raise ValueError('Broken local links: '+str(errors))
    return checked


def verify_run(output):
    output=Path(output);manifest=json.loads((output/'run_manifest.json').read_text())
    data=pd.read_csv(output/'causal_dataset.csv');pred=pd.read_csv(output/'per_record_predictions.csv')
    comp=pd.read_csv(output/'model_comparison.csv').set_index('Model')
    if fingerprint(output/'causal_dataset.csv')!=manifest['dataset_sha256']:raise ValueError('Dataset mismatch')
    receipt=json.loads((output/'dataset_cache.json').read_text())
    if not cache_matches(receipt,cache_identity(manifest['target_definition']['name']),output):raise ValueError('Stale/corrupted prepared cache')
    for model,g in pred.groupby('Model'):
        if g.sample_id.duplicated().any() or set(g.sample_id)!=set(data.sample_id):raise ValueError('Evaluation row mismatch')
        labels=data.set_index('sample_id').loc[g.sample_id].Label
        np.testing.assert_array_equal(labels,g.Label)
        got=metrics(g.Label,g.Score,g.Prediction)
        for key in ['TP','FP','TN','FN','Recall','Precision','Specificity','F1','PR_AUC_AP']:
            if not np.isclose(comp.loc[model,key],got[key],equal_nan=True):raise ValueError(f'Metric mismatch: {model}/{key}')
        metadata=json.loads((output/'models'/f'{model}.json').read_text())
        if metadata['features']!=config.FEATURES or metadata['target']!=manifest['target_definition']:raise ValueError('Model metadata mismatch')
    for split in manifest['recording_splits']:
        if split['test_record'] in split['outer_train_records']:raise ValueError('Outer leakage')
        for inner in split['inner']:
            if set(inner['train_records'])&set(inner['validation_records']) or split['test_record'] in inner['train_records']+inner['validation_records']:raise ValueError('Inner leakage')
    figures=output/'figures';fm=json.loads((figures/'figure_manifest.json').read_text())
    if fm['run_manifest_sha256']!=fingerprint(output/'run_manifest.json'):raise ValueError('Figures reference another run manifest')
    for name,digest in fm['sources'].items():
        if fingerprint(output/name)!=digest:raise ValueError(f'Changed figure source: {name}')
    for name,digest in fm['files'].items():
        if fingerprint(figures/name)!=digest:raise ValueError(f'Changed figure: {name}')
    chart=pd.read_csv(figures/'model_metrics.csv')
    for row in chart.itertuples():
        if not np.isclose(row.Value,comp.loc[row.Model,row.Metric],equal_nan=True):raise ValueError('Chart values mismatch')
    lat=pd.read_csv(output/'latency_benchmark.csv').set_index(['Model','Component'])
    samples=pd.read_csv(output/'latency_samples.csv.gz')
    for (model,component),part in samples.groupby(['Model','Component']):
        q=lat.loc[(model,component)]
        if len(part)!=q.Repetitions or not np.isclose(part.Latency_ms.median(),q.Median_ms) or not np.isclose(part.Latency_ms.quantile(.95),q.P95_ms):raise ValueError('Latency sample mismatch')
    return {'target':manifest['target_definition']['name'],'rows':len(data),'models':len(comp),'figures_png':len(list(figures.glob('*.png'))),'cache':'verified','figure_sources':'verified'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',action='store_true');parser.add_argument('--links',action='store_true');parser.add_argument('--verify-run',type=Path)
    args=parser.parse_args();result={'dependencies':verify_environment()}
    if args.audit:result['raw_recordings']=audit_data()
    if args.links:result['local_links_checked']=verify_links()
    if args.verify_run:result['quality_gate']=verify_run(args.verify_run)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
