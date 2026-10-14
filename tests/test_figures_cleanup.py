import ast
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from src import config
from src.benchmark_latency import timed,summarize
from src.figures import plotted_metrics
from src.check_project import verify_environment,verify_links,verify_run


class CleanupAndFigureTests(unittest.TestCase):
    def test_archive_preserves_exact_source_and_restores_safely(self):
        manifest=json.loads((config.ROOT/'archive/legacy_sources.json').read_text())
        with zipfile.ZipFile(config.ROOT/'archive/legacy_sources.zip') as archive:
            for path,digest in manifest.items():self.assertEqual(hashlib.sha256(archive.read(path)).hexdigest(),digest)
        output_manifest=json.loads((config.ROOT/'archive/legacy_results.json').read_text())
        with zipfile.ZipFile(config.ROOT/'archive/legacy_results.zip') as archive:
            for path,digest in output_manifest.items():self.assertEqual(hashlib.sha256(archive.read(path)).hexdigest(),digest)
        with tempfile.TemporaryDirectory() as temp:
            dest=Path(temp)/'restored'
            subprocess.run([sys.executable,'archive/restore_legacy.py','--output',str(dest)],check=True,capture_output=True)
            for path,digest in manifest.items():
                # Three original scripts intentionally restored into src as well as original archive paths.
                self.assertEqual(hashlib.sha256((dest/path).read_bytes()).hexdigest(),digest)
            self.assertNotEqual(subprocess.run([sys.executable,'archive/restore_legacy.py','--output',str(dest)],capture_output=True).returncode,0)
    def test_active_relative_imports_resolve_after_cleanup(self):
        for path in (config.ROOT/'src').glob('*.py'):
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom) and node.level and node.module:
                    self.assertTrue((config.ROOT/'src'/node.module.split('.')[0]).with_suffix('.py').exists(),f'{path}: {node.module}')
    def test_plotted_metric_values_are_exact_csv_values(self):
        c=pd.read_csv(config.HARDENED_DIR/'new_onset/model_comparison.csv');values=plotted_metrics(c)
        for row in values.itertuples():self.assertEqual(row.Value,c.set_index('Model').loc[row.Model,row.Metric])
    def test_saved_raw_timings_reproduce_summary(self):
        raw=[];result=timed(lambda:sum(range(4)),warmup=2,repetitions=25,samples=raw)
        self.assertEqual(len(raw),25);self.assertEqual(result['Median_ms'],np.median(raw));self.assertFalse(result['P99_Reliable'])
        self.assertTrue(np.isnan(result['P99_ms']))
    def test_pinned_dependencies_installed(self):self.assertEqual(len(verify_environment()),7)

@unittest.skipUnless((config.ROOT/'results/final/new_onset/figures/figure_manifest.json').exists(),'Generate final figures for final-artifact checks')
class FinalArtifactTests(unittest.TestCase):
    def test_both_runs_figures_metrics_and_cache_agree(self):
        for target in config.TARGETS:self.assertGreaterEqual(verify_run(config.ROOT/'results/final'/target)['figures_png'],20)
    def test_documentation_local_links_exist(self):self.assertGreater(verify_links(),10)
