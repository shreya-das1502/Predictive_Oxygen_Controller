"""Integration checks of actual deliverables, skipped before the first full run."""
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import OUTPUT_DIR,FEATURES,MODEL_CONFIGS
from src.controller import Controller
from src.model import FittedModel
from src.evaluation import metrics
from src.features import add_features
from src.labeling import future_labels
from src.dataset import load_csv
from src.config import DATA_DIR

@unittest.skipUnless((OUTPUT_DIR/'per_record_predictions.csv').exists(),'Run python -m src.run_experiments first for artifact integration checks')
class SavedResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=pd.read_csv(OUTPUT_DIR/'causal_dataset.csv');cls.pred=pd.read_csv(OUTPUT_DIR/'per_record_predictions.csv')
    def test_every_model_same_rows_once(self):
        for name in MODEL_CONFIGS:
            g=self.pred[self.pred.Model==name]
            self.assertEqual(len(g),len(self.data));self.assertFalse(g.sample_id.duplicated().any());self.assertEqual(set(g.sample_id),set(self.data.sample_id))
            np.testing.assert_array_equal(g.sort_values('sample_id').Label,self.data.sort_values('sample_id').Label)
    def test_saved_pooled_counts_recompute(self):
        pooled=pd.read_csv(OUTPUT_DIR/'pooled_metrics.csv').set_index('Model')
        for name,g in self.pred.groupby('Model'):
            m=metrics(g.Label,g.Score,g.Prediction)
            for c in ['TP','FP','TN','FN']:self.assertEqual(pooled.loc[name,c],m[c])
    def test_all_recording_and_inner_split_provenance(self):
        splits=json.loads((OUTPUT_DIR/'splits.json').read_text());self.assertEqual(len(splits),self.data.Record.nunique())
        for s in splits:
            self.assertNotIn(s['test_record'],s['outer_train_records']);seen=[]
            for inner in s['inner']:
                self.assertFalse(set(inner['train_records'])&set(inner['validation_records']))
                self.assertNotIn(s['test_record'],inner['train_records']+inner['validation_records']);seen+=inner['validation_records']
            self.assertEqual(sorted(seen),sorted(s['outer_train_records']))
    def test_no_initial_feature_rows_or_positives_lost(self):
        stages=pd.read_csv(OUTPUT_DIR/'preprocessing_stages.csv');f=stages[stages.Stage=='rolling_features']
        self.assertTrue(f.Rows_Removed.eq(0).all());self.assertTrue(f.Positive_Rows_Removed.eq(0).all())
    def test_saved_all_model_replay_parity(self):
        raw=pd.read_csv(OUTPUT_DIR/'raw_pulses_01.csv').iloc[:120]
        for name in MODEL_CONFIGS:
            m=FittedModel.load(OUTPUT_DIR/'models'/f'{name}.joblib');c=Controller(m)
            want=m.score(raw[FEATURES].to_numpy())
            got=np.array([c.predict(row)['score'] for row in raw.to_dict('records')])
            np.testing.assert_allclose(want,got,rtol=1e-10,atol=1e-10)
    def test_validation_targets_separate_from_test(self):
        val=pd.read_csv(OUTPUT_DIR/'validation_metrics.csv');self.assertTrue(val.Recall.ge(.92).all())
        selected=pd.read_csv(OUTPUT_DIR/'threshold_analysis.csv');self.assertEqual(len(selected),len(val))
        self.assertTrue(selected.Selected.eq(True).all());self.assertTrue(selected.Recall.ge(.92).all())
    def test_source_files_unchanged(self):
        import hashlib
        hashes=json.loads((OUTPUT_DIR/'run_manifest.json').read_text())['source_hashes']
        for path,want in hashes.items():self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),want)

    def test_current_feature_code_matches_all_saved_pulse_history(self):
        for r in self.data.Record.unique():
            raw=pd.read_csv(OUTPUT_DIR/f'raw_pulses_{r:02d}.csv')
            recomputed=add_features(raw)
            np.testing.assert_allclose(recomputed[FEATURES],raw[FEATURES],rtol=1e-10,atol=1e-10,equal_nan=True)
            valid=self.data[self.data.Record==r]
            want=recomputed.set_index('pulse_row').loc[valid.pulse_row,FEATURES].to_numpy()
            np.testing.assert_allclose(valid[FEATURES],want,rtol=1e-10,atol=1e-10,equal_nan=True)
    def test_current_label_code_matches_all_saved_targets(self):
        for r in self.data.Record.unique():
            raw=pd.read_csv(OUTPUT_DIR/f'raw_pulses_{r:02d}.csv')
            numeric=load_csv(DATA_DIR/f'bidmc_{r:02d}_Numerics.csv',['Time [s]','SpO2'])
            label,minimum,_=future_labels(raw.time_s,numeric['Time [s]'],numeric.SpO2)
            np.testing.assert_allclose(label,raw.Label,equal_nan=True)
            np.testing.assert_allclose(minimum,raw.Future_Min_SpO2_30s,equal_nan=True)
