"""Restore exact historical source into a NEW directory; never overwrite work."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--with-data', action='store_true', help='Copy local raw data and historical results; may consume hundreds of MB')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must not exist; choose a new directory')
    base = Path(__file__).resolve().parent
    manifest = json.loads((base / 'legacy_sources.json').read_text())
    with zipfile.ZipFile(base / 'legacy_sources.zip') as archive:
        content = {name: archive.read(name) for name in manifest}
    for name, blob in content.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or hashlib.sha256(blob).hexdigest() != manifest[name]:
            raise ValueError('Unsafe path or changed archived source')
    args.output.mkdir(parents=True)
    for name, blob in content.items():
        dest = args.output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
    for name in ['features.py', 'labeling.py', 'benchmark_latency.py']:
        shutil.copy2(args.output / 'archive/legacy_src' / name, args.output / 'src' / name)
    if args.with_data:
        shutil.copytree(base.parent / 'data', args.output / 'data')
        shutil.copytree(base.parent / 'results', args.output / 'results', ignore=shutil.ignore_patterns('final', 'hardened'))
        output_manifest=json.loads((base/'legacy_results.json').read_text())
        with zipfile.ZipFile(base/'legacy_results.zip') as z:
            for name,digest in output_manifest.items():
                blob=z.read(name)
                if Path(name).is_absolute() or '..' in Path(name).parts or hashlib.sha256(blob).hexdigest()!=digest:
                    raise ValueError('Unsafe or changed historical artifact')
                dest=args.output/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(blob)
    print(f'Restored exact legacy source into {args.output}. Historical algorithms retain their original limitations.')

if __name__ == '__main__':
    main()
