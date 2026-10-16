"""Authenticate completion outputs and narrowly authorized presentation changes.

This supplements, and never modifies, frozen historical preservation contracts.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
AUTHORIZED = {
    'README.md', 'dashboard/app.py', 'docs/DASHBOARD_GUIDE.md',
    'docs/FINAL_REPORT.md', 'docs/PROJECT_REVIEW_EVIDENCE.md',
    'docs/FINAL_READINESS_AUDIT.md', 'docs/FINAL_READINESS_CHECKLIST.md',
    'docs/ML_COMPLETION_TRACKER.md',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def baseline_check():
    baseline = json.loads((OUT / 'protected_baseline.json').read_text())
    changes = {}
    for name, before in baseline.items():
        current = sha(ROOT / name)
        if current == before:
            continue
        if name not in AUTHORIZED:
            raise ValueError('Unauthorized baseline change: ' + name)
        backup = OUT / 'preserved_documents' / (name + '.original')
        if sha(backup) != before:
            raise ValueError('Original backup mismatch: ' + name)
        changes[name] = dict(before=before, after=current,
                             backup=str(backup.relative_to(ROOT)))
    if set(changes) != AUTHORIZED:
        raise ValueError('Authorized-change inventory differs from actual changes')
    sources = json.loads((OUT / 'ml_source_baseline.json').read_text())
    for name, expected in sources.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen source changed: ' + name)
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for name, expected in manifest['outputs'].items():
        if sha(OUT / name) != expected:
            raise ValueError('Initial sealed output changed: ' + name)
    return dict(baseline_files=len(baseline),
                unchanged_baseline_files=len(baseline) - len(changes),
                authorized_changes=changes, frozen_sources=len(sources),
                initial_sealed_outputs=len(manifest['outputs']))


def inventory():
    return {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*'))
            if p.is_file() and p.name != 'final_completion_receipt.json'}


def documents():
    names = ['docs/DATA_PROVENANCE.md', 'docs/ML_IMPROVEMENT_REPORT.md',
             'docs/ML_COMPLETION_VERIFICATION.md', 'ml_research/finalize_saved.py',
             'tests/test_ml_finalization.py']
    return {name: sha(ROOT / name) for name in names}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    receipt = OUT / 'final_completion_receipt.json'
    state = baseline_check()
    if args.seal:
        if receipt.exists():
            raise FileExistsError('Refusing to replace completion receipt')
        data = dict(status='verified_with_documented_historical_contract_failures',
                    baseline=state, outputs=inventory(), documents=documents(),
                    historical_contracts_modified=False, clinical_validation=False)
        receipt.write_text(json.dumps(data, indent=2) + '\n')
    saved = json.loads(receipt.read_text())
    if saved['baseline'] != state or saved['outputs'] != inventory() or saved['documents'] != documents():
        raise ValueError('Completion receipt mismatch')
    print(json.dumps(dict(status=saved['status'], baseline_files=state['baseline_files'],
                          unchanged_baseline_files=state['unchanged_baseline_files'],
                          authorized_changes=len(state['authorized_changes']),
                          authenticated_outputs=len(saved['outputs']),
                          historical_contracts_modified=False), indent=2))
