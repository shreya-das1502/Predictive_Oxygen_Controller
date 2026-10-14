"""Content-verified dataset cache: dependencies and generated artifacts must match."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from . import config
from .dataset import build_dataset, validate_dataset

GENERATION_MODULES = ('config.py', 'dataset.py', 'features.py', 'labeling.py', 'preprocessing.py', 'provenance.py')


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_identity(target, data_dir=config.DATA_DIR, code_dir=config.ROOT / 'src', legacy_path=config.INPUT_DATASET):
    sources = sorted(Path(data_dir).glob('*.csv'))
    if not sources:
        raise FileNotFoundError(f'No raw CSV sources in {data_dir}')
    return {'cache_schema': 2, 'target_definition': config.target_definition(target),
            'feature_order': list(config.FEATURES),
            'source_hashes': {p.name: fingerprint(p) for p in sources},
            'code_hashes': {name: fingerprint(Path(code_dir) / name) for name in GENERATION_MODULES},
            'legacy_audit_input': fingerprint(legacy_path) if Path(legacy_path).exists() else None}


def cache_matches(receipt, identity, output):
    if receipt.get('identity') != identity or not receipt.get('artifacts'):
        return False
    return all((Path(output) / name).is_file() and fingerprint(Path(output) / name) == digest
               for name, digest in receipt['artifacts'].items())


def load_prepared(output, target, rebuild=False):
    output = Path(output)
    identity = cache_identity(target)
    receipt_path = output / 'dataset_cache.json'
    try:
        receipt = json.loads(receipt_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        receipt = {}
    if not rebuild and cache_matches(receipt, identity, output):
        data = pd.read_csv(output / 'causal_dataset.csv')
        validate_dataset(data)
        return data, receipt, True
    data = build_dataset(output, target=target)
    names = ['causal_dataset.csv', 'dataset_audit.csv', 'ground_truth_events.csv', 'target_definition.json',
             'raw_data_quality.csv', 'preprocessing_stages.csv', 'dataset_summary.json', 'source_hashes.json']
    if (output / 'label_audit.csv').exists():
        names.append('label_audit.csv')
    names += [p.name for p in sorted(output.glob('raw_pulses_*.csv'))]
    receipt = {'identity': identity, 'artifacts': {name: fingerprint(output / name) for name in names}}
    temporary = receipt_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(receipt, indent=2))
    temporary.replace(receipt_path)
    return data, receipt, False
