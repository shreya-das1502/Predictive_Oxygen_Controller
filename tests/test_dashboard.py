"""Dashboard adapters must not train, reveal future outcomes or share case state."""
import unittest
import numpy as np
import pandas as pd
from dashboard.backend import Replay, load_record, recordings, policy_rows, result_rows, verified_events, FINAL, ROOT

class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wave, cls.numeric = load_record(1)

    def test_selection_and_availability(self):
        self.assertEqual(len(recordings()),30)
        self.assertTrue(all(recordings().values()))
        with self.assertRaises(ValueError): load_record(31)

    def test_chronology_prefix_and_independent_reset(self):
        a,b = Replay(self.wave,self.numeric), Replay(self.wave,self.numeric)
        a.seek(4); b.seek(2)
        self.assertLessEqual(a.visible_wave().time_s.max(),4)
        self.assertLessEqual(a.observation()['Time [s]'],4)
        self.assertTrue(all(x['time_s']<=4 for x in a.pulses))
        a.seek(2)
        self.assertEqual(a.pulses,b.pulses)
        a.reset()
        self.assertEqual(len(a.pulses),0)
        self.assertGreater(len(b.pulses),0)
        with self.assertRaises(ValueError): a.seek(np.nan)

    def test_missing_waveform_and_stale_numeric(self):
        wave = self.wave.iloc[:500].copy(); wave.loc[100,'PLETH'] = np.nan
        replay = Replay(wave,self.numeric.iloc[:1]).seek(3)
        self.assertIn('discontinuity',replay.quality)
        self.assertIsNone(replay.observation())

    def test_no_future_numeric_or_waveform_to_controller(self):
        seen=[]
        class Spy:
            def update_waveform(self,t,value,spo2,spo2_time_s):
                seen.append((t,spo2_time_s)); return None
        r=Replay(self.wave,self.numeric,controller_factory=Spy).seek(2)
        self.assertTrue(all(t<=2 and (np.isnan(s) or s<=t) for t,s in seen))
        self.assertEqual(len(seen),r.index)

    def test_real_model_outputs_do_not_depend_on_future_samples(self):
        from dashboard.backend import demo_factory
        wave = self.wave.copy()
        numeric = self.numeric.copy()
        wave.loc[wave.time_s > 4, 'PLETH'] = 999.0
        numeric.loc[numeric['Time [s]'] > 4, 'SpO2'] = 40.0
        for target in ('new_onset', 'future_low'):
            for model in ('ELM', 'Logistic', 'GaussianNB', 'Tree'):
                with self.subTest(target=target, model=model):
                    factory = demo_factory(target, model)
                    original = Replay(self.wave, self.numeric, controller_factory=factory).seek(4)
                    changed = Replay(wave, numeric, controller_factory=factory).seek(4)
                    self.assertGreater(len(original.outputs), 0)
                    self.assertEqual(original.outputs, changed.outputs)
                    self.assertEqual(original.pulses, changed.pulses)

    def test_predictions_and_verification_separated(self):
        saved=pd.DataFrame({'time_s':[1.,3.], 'Score':[.1,.9], 'Threshold':[.5,.5], 'Prediction':[0,1]})
        r=Replay(self.wave,self.numeric,saved).seek(2)
        self.assertEqual(len(r.visible_predictions()),1)
        self.assertFalse(r.current_prediction().alert)
        table=pd.DataFrame([dict(Record=1, Model='ELM',Target='new_onset',Budget=5,Window_s=30,Onset_s=10,Evaluation_Status='eligible',Detected=True,First_Alert_s=1,Lead_Time_s=9)])
        self.assertTrue(verified_events(table,1,'ELM','new_onset',5,39).empty)
        self.assertEqual(verified_events(table,1,'ELM','new_onset',5,40).Lead_Time_s.iloc[0],9)

    def test_current_prediction_expires(self):
        saved=pd.DataFrame({'time_s':[1.], 'Score':[.1], 'Threshold':[.5], 'Prediction':[0]})
        self.assertIsNone(Replay(self.wave,self.numeric,saved).seek(4).current_prediction())

    def test_actual_denominators_and_policy_separation(self):
        summary=pd.read_csv(FINAL/'summary.csv')
        for w,n in [(5,15),(10,15),(15,15),(30,14)]:
            for target in ['new_onset','future_low']:
                for budget in [1,5,10]:
                    rows=result_rows(summary,target,budget,w)
                    learned=rows[rows.Model.isin(['ELM','Logistic','GaussianNB','Tree'])]
                    self.assertEqual(len(learned),4)
                    self.assertTrue((learned.Eligible_Events==n).all())
                    self.assertTrue((learned.Missed_Events==n).all())
                    self.assertTrue((learned.Detected==0).all())
                    self.assertTrue(learned.Lead_Median_s.isna().all())
                    np.testing.assert_allclose(learned.False_Alarm_Starts_Per_Hour,learned.Off_Event_Alarm_Starts/(learned.Exposure_s/3600))

    def test_future_label_columns_removed(self):
        rows=pd.DataFrame([dict(Record=1, Model='ELM',Target='new_onset',Budget=5,time_s=1,Score=.2,Threshold=.5,Prediction=0,Label=1,Future_Min_SpO2_30s=92)])
        self.assertEqual(list(policy_rows(rows,1,'ELM','new_onset',5)),['time_s','Score','Threshold','Prediction'])

if __name__=='__main__': unittest.main()
