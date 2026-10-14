"""Target, provenance, eligibility and grouped evaluation regression tests."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from src import config
from src.labeling import future_labels, onset_labels, desaturation_events
from src.provenance import cache_identity, cache_matches, fingerprint, GENERATION_MODULES, load_prepared
from src.model import FittedModel
from src.controller import Controller
from src.evaluation import grouped_oof
from src.run_experiments import run
from src.target_reporting import evaluate_events
from test_pipeline import frame, sample
from src.features import add_features


class OnsetTests(unittest.TestCase):
    def setUp(self):
        self.nt=np.arange(101,dtype=float);self.s=np.full(101,98.);self.s[30:34]=92
    def test_future_low_preserved_and_ongoing_low_ineligible(self):
        old,_,_=future_labels([31],self.nt,self.s)
        new,_,reason=onset_labels([31],self.nt,self.s)
        self.assertEqual(old[0],1);self.assertTrue(np.isnan(new[0]));self.assertEqual(reason[0],'current_spo2_not_above_threshold')
    def test_horizon_includes_right_endpoint_and_excludes_current(self):
        self.assertEqual(onset_labels([0],self.nt,self.s)[0][0],1)
        self.assertEqual(onset_labels([-.01],self.nt,self.s)[2][0],'invalid_current_spo2')
        self.assertEqual(onset_labels([34],self.nt,self.s)[0][0],0)
        s=self.s.copy();s[:]=98;s[31]=94
        self.assertEqual(onset_labels([0],self.nt,s)[0][0],0)
    def test_left_censored_low_not_an_onset(self):
        self.s[:4]=92
        events=desaturation_events(self.nt,self.s)
        self.assertFalse(events[0]['Qualifying_Onset']);self.assertTrue(events[1]['Qualifying_Onset'])
    def test_missing_prior_and_time_gap_do_not_invent_crossing(self):
        s=self.s.copy();s[29]=np.nan
        self.assertFalse(desaturation_events(self.nt,s)[0]['Qualifying_Onset'])
        nt=np.delete(self.nt,29);s=np.delete(self.s,29)
        self.assertFalse(desaturation_events(nt,s)[0]['Qualifying_Onset'])
    def test_missing_current_or_future_is_unknown(self):
        s=self.s.copy();s[0]=np.nan
        self.assertEqual(onset_labels([0],self.nt,s)[2][0],'invalid_current_spo2')
        s=self.s.copy();s[12]=np.nan
        self.assertTrue(np.isnan(onset_labels([0],self.nt,s)[0][0]))
        self.assertTrue(np.isnan(onset_labels([90],self.nt,self.s)[0][0]))
    def test_strict_current_age_and_record_boundary(self):
        self.assertTrue(np.isnan(onset_labels([1.5],[0,3,4,50],[98,98,98,98])[0][0]))
        self.assertEqual(desaturation_events([0,1],[92,92])[0]['Qualifying_Onset'],False)
        self.assertEqual(desaturation_events([0,1],[98,92])[0]['Qualifying_Onset'],True)
    def test_empty_and_invalid_inputs(self):
        self.assertEqual(desaturation_events([],[]),[])
        self.assertEqual(len(onset_labels([],[],[])[0]),0)
        with self.assertRaises(ValueError):onset_labels([0],[0,0],[98,94])
    def test_91_92_is_observed_episode_subset(self):
        e=desaturation_events(self.nt,self.s)[0];self.assertTrue(e['Has_91_92'])
        s=self.s.copy();s[30:34]=94;self.assertFalse(desaturation_events(self.nt,s)[0]['Has_91_92'])
    def test_independent_expected_crossings(self):
        for t in np.arange(0,70,.25):
            label,current,_=onset_labels([t],self.nt,self.s)
            j=np.flatnonzero(self.nt<=t)[-1]
            if self.s[j]>94:
                expected=any(self.s[k]<=94 and self.s[k-1]>94 and t<self.nt[k]<=t+30 for k in range(1,len(self.nt)))
                self.assertEqual(label[0],expected)
            else:self.assertTrue(np.isnan(label[0]))


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.data=self.root/'data';self.code=self.root/'src';self.data.mkdir();self.code.mkdir()
        (self.data/'source.csv').write_text('a\n1\n')
        for name in GENERATION_MODULES:(self.code/name).write_text('# source\n')
    def identity(self,target='future_low'):
        return cache_identity(target,self.data,self.code,self.root/'missing_legacy.csv')
    def test_every_generation_dependency_invalidates(self):
        before=self.identity()
        for name in GENERATION_MODULES:
            p=self.code/name;p.write_text('# changed\n');self.assertNotEqual(before,self.identity(),name);p.write_text('# source\n')
    def test_source_target_and_feature_order_invalidate(self):
        before=self.identity();self.assertNotEqual(before,self.identity('new_onset'))
        (self.data/'source.csv').write_text('a\n2\n');self.assertNotEqual(before,self.identity())
        with patch.object(config,'FEATURES',list(reversed(config.FEATURES))):self.assertNotEqual(before,self.identity())
    def test_corrupted_or_missing_artifact_cannot_hit_cache(self):
        p=self.root/'dataset.csv';p.write_text('a\n1\n');identity=self.identity()
        receipt={'identity':identity,'artifacts':{'dataset.csv':fingerprint(p)}}
        self.assertTrue(cache_matches(receipt,identity,self.root))
        p.write_text('a\n2\n');self.assertFalse(cache_matches(receipt,identity,self.root))
        self.assertFalse(cache_matches({},identity,self.root))
    def test_repeated_load_reuses_receipt_without_rebuilding(self):
        d=frame();output=self.root/'output';output.mkdir()
        d.to_csv(output/'causal_dataset.csv',index=False)
        identity=self.identity()
        receipt={'identity':identity,'artifacts':{'causal_dataset.csv':fingerprint(output/'causal_dataset.csv')}}
        (output/'dataset_cache.json').write_text(json.dumps(receipt))
        with patch('src.provenance.cache_identity',return_value=identity),patch('src.provenance.build_dataset',side_effect=AssertionError('unexpected rebuild')):
            _,_,reused=load_prepared(output,'future_low');self.assertTrue(reused)
    def test_historical_run_directory_protected(self):
        with self.assertRaisesRegex(ValueError,'protected'):run(config.OUTPUT_DIR,target='new_onset')


class TargetInferenceTests(unittest.TestCase):
    def setUp(self):
        f=add_features(frame());self.m=FittedModel('ELM').fit(f[config.FEATURES],f.Label)
        self.m.threshold=.5;self.m.target_definition=config.target_definition('new_onset')
    def test_eligibility_gates_low_missing_stale_future_current(self):
        for spo2,st in [(92,1),(np.nan,1),(98,-5),(98,2)]:
            c=Controller(self.m);out=c.predict({**sample(1),'spo2':spo2,'spo2_time_s':st})
            self.assertFalse(out['eligible']);self.assertFalse(out['alert']);self.assertIsNone(out['score'])
        out=Controller(self.m).predict({**sample(1),'spo2':98,'spo2_time_s':1})
        self.assertTrue(out['eligible']);self.assertTrue(np.isfinite(out['score']))
    def test_ineligible_rows_still_update_history(self):
        c=Controller(self.m)
        c.predict({**sample(1),'spo2':92,'spo2_time_s':1})
        c.predict({**sample(2),'spo2':98,'spo2_time_s':2})
        self.assertEqual(len(c.feature_state.history),2)
    def test_target_metadata_persists_with_model(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'model.joblib';self.m.save(path,{'target':self.m.target_definition})
            self.assertEqual(FittedModel.load(path).target_definition,config.target_definition('new_onset'))
    def test_every_inner_transform_only_sees_inner_training(self):
        f=add_features(frame());original=FittedModel.fit;fits=[]
        def track(model,x,y):
            result=original(model,x,y);fits.append((np.asarray(x).copy(),model.scaler.mean_.copy()));return result
        with patch.object(FittedModel,'fit',track):grouped_oof(f[config.FEATURES],f.Label,f.Record,'ELM')
        self.assertEqual(len(fits),3)
        for x,mean in fits:
            self.assertEqual(len(x),16);np.testing.assert_allclose(mean,np.mean(x,axis=0))


class EventMatchingTests(unittest.TestCase):
    def test_onset_opportunity_and_no_lookahead_at_onset(self):
        with tempfile.TemporaryDirectory() as td:
            numeric=pd.DataFrame({'Time [s]':np.arange(70),'SpO2':np.r_[np.full(40,98),[92],np.full(29,98)]})
            numeric.to_csv(Path(td)/'bidmc_01_Numerics.csv',index=False)
            event=pd.DataFrame([{'Record':1,**desaturation_events(numeric['Time [s]'],numeric.SpO2)[0]}])
            prediction=pd.DataFrame({'Model':['ELM']*3,'Record':[1]*3,'time_s':[10,39,40],'Prediction':[0,1,1],'spo2':[98,98,92]})
            with patch.object(config,'DATA_DIR',Path(td)):
                matches,summary=evaluate_events(prediction,event,'new_onset')
                self.assertTrue(matches.Detected.iloc[0]);self.assertEqual(matches.Lead_Time_s.iloc[0],1)
                prediction.loc[prediction.time_s<40,'Prediction']=0
                matches,_=evaluate_events(prediction,event,'new_onset');self.assertFalse(matches.Detected.iloc[0])

@unittest.skipUnless(all((config.HARDENED_DIR/target/'causal_dataset.csv').exists() and all((config.HARDENED_DIR/target/'models'/f'{name}.joblib').exists() for name in config.MODEL_CONFIGS) for target in config.TARGETS),'Saved hardening integration requires optional local prepared data and model binaries for both targets')
class SavedTargetTests(unittest.TestCase):
    def test_targets_have_identical_rows_across_models_and_no_split_leakage(self):
        for target in config.TARGETS:
            out=config.HARDENED_DIR/target;d=pd.read_csv(out/'causal_dataset.csv');p=pd.read_csv(out/'per_record_predictions.csv')
            for _,g in p.groupby('Model'):
                self.assertFalse(g.sample_id.duplicated().any());self.assertEqual(set(g.sample_id),set(d.sample_id))
            for split in json.loads((out/'splits.json').read_text()):
                self.assertNotIn(split['test_record'],split['outer_train_records'])
                for inner in split['inner']:
                    self.assertFalse(set(inner['train_records'])&set(inner['validation_records']))
                    self.assertNotIn(split['test_record'],inner['train_records']+inner['validation_records'])
    def test_onset_eligibility_and_expected_equivalence_on_normal_state(self):
        out=config.HARDENED_DIR/'new_onset';d=pd.read_csv(out/'causal_dataset.csv')
        self.assertTrue(d.spo2.gt(config.SPO2_THRESHOLD).all())
        np.testing.assert_array_equal(d.Label,d.Label_New_Onset)
        np.testing.assert_array_equal(d.Label_New_Onset,d.Label_Future_Low)
        for r,g in d.groupby('Record'):
            numeric=pd.read_csv(config.DATA_DIR/f'bidmc_{r:02d}_Numerics.csv');numeric.columns=numeric.columns.str.strip()
            nt=numeric['Time [s]'].to_numpy(float);s=numeric.SpO2.to_numpy(float)
            onsets=[nt[k] for k in range(1,len(nt)) if np.isfinite(s[k-1]) and 0<s[k]<=config.SPO2_THRESHOLD<s[k-1]<=100 and nt[k]-nt[k-1]<=config.MAX_NUMERIC_GAP]
            expected=[int(any(t<o<=t+config.HORIZON for o in onsets)) for t in g.time_s]
            np.testing.assert_array_equal(g.Label,expected)
    def test_future_low_predictions_unchanged(self):
        keys=['Model','Record','pulse_row']
        old=pd.read_csv(config.OUTPUT_DIR/'per_record_predictions.csv').sort_values(keys)
        new=pd.read_csv(config.HARDENED_DIR/'future_low/per_record_predictions.csv').sort_values(keys)
        self.assertEqual(len(new),len(old))
        for c in ['Label','Score','Threshold','Prediction']:np.testing.assert_allclose(old[c],new[c],rtol=1e-10,atol=1e-10)
    def test_current_receipts_hit_without_regeneration(self):
        for target in config.TARGETS:
            with patch('src.provenance.build_dataset',side_effect=AssertionError('stale cache')):
                _,_,reused=load_prepared(config.HARDENED_DIR/target,target)
                self.assertTrue(reused)
    def test_onset_artifact_replay_matches_eligible_batch_scores(self):
        out=config.HARDENED_DIR/'new_onset';raw=pd.read_csv(out/'raw_pulses_17.csv')
        num=pd.read_csv(config.DATA_DIR/'bidmc_17_Numerics.csv');num.columns=num.columns.str.strip()
        nt=num['Time [s]'].to_numpy(float);s=num.SpO2.to_numpy(float)
        for name in config.MODEL_CONFIGS:
            model=FittedModel.load(out/'models'/f'{name}.joblib');c=Controller(model);want=model.score(raw[config.FEATURES].to_numpy())
            for i,row in enumerate(raw.to_dict('records')):
                j=np.searchsorted(nt,row['time_s'],side='right')-1
                result=c.predict({**row,'spo2':float(s[j]),'spo2_time_s':float(nt[j])})
                if result['eligible']:self.assertAlmostEqual(result['score'],want[i],places=9)
                else:self.assertFalse(result['alert'])
    def test_historical_outputs_preserved(self):
        snapshots=json.loads((config.HARDENED_DIR/'historical_preservation.json').read_text())
        for path,digest in snapshots.items():self.assertEqual(fingerprint(path),digest,path)
