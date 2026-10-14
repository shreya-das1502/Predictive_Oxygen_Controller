"""Fixed-policy, recording-cluster uncertainty and selection isolation tests."""
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from src.local_30_evaluation import (DEFAULT_OUTPUT, bootstrap_weights, bootstrap_metrics,
    verify_splits, verify_threshold_fold, interval, divide, verify_output)
from src.reliability_evaluation import match_events, links


def split_fixture():
    return [{'test_record':test,'outer_train_records':[r for r in [1,2,3] if r!=test],
             'inner':[{'train_records':[r for r in [1,2,3] if r not in [test,val]],'validation_records':[val]}
                      for val in [1,2,3] if val!=test]} for test in [1,2,3]]


def threshold_fixture():
    training=pd.DataFrame({'sample_id':[0,1,2,3],'Record':[1,1,2,2],'Label':[1,0,1,1]})
    oof=training.assign(Score=[.9,.8,.7,.1],Threshold=.1)
    held=pd.DataFrame({'Record':[3],'Score':[.2],'Threshold':[.1],'Prediction':[1],'Label':[0]})
    return training,oof,held


class Local30UnitTests(unittest.TestCase):
    def test_exact_recording_split_partition(self):verify_splits(split_fixture(),[1,2,3])
    def test_outer_record_cannot_enter_training(self):
        s=split_fixture();s[0]['outer_train_records'].append(1)
        with self.assertRaises(ValueError):verify_splits(s,[1,2,3])
    def test_inner_overlap_and_missing_validation_rejected(self):
        s=split_fixture();s[0]['inner'][0]['train_records'].append(2)
        with self.assertRaises(ValueError):verify_splits(s,[1,2,3])
        s=split_fixture();s[0]['inner'].pop()
        with self.assertRaises(ValueError):verify_splits(s,[1,2,3])
    def test_threshold_does_not_access_held_out_labels(self):
        training,oof,held=threshold_fixture();first=verify_threshold_fold(training,oof,held,3)
        held.Label=1;self.assertEqual(first,verify_threshold_fold(training,oof,held,3))
        self.assertEqual(first,verify_threshold_fold(training,oof,held.drop(columns='Label'),3))
    def test_threshold_cannot_contain_outer_test_rows(self):
        training,oof,held=threshold_fixture();oof.loc[0,'Record']=3
        with self.assertRaises(ValueError):verify_threshold_fold(training,oof,held,3)
    def test_missing_oof_rows_and_wrong_training_labels_rejected(self):
        training,oof,held=threshold_fixture()
        with self.assertRaises(ValueError):verify_threshold_fold(training,oof.iloc[:-1],held,3)
        oof.loc[0,'Label']=0
        with self.assertRaises(ValueError):verify_threshold_fold(training,oof,held,3)
    def test_test_informed_threshold_not_accepted(self):
        training,oof,held=threshold_fixture();oof.Threshold=.2;held.Threshold=.2
        with self.assertRaises(ValueError):verify_threshold_fold(training,oof,held,3)
    def test_record_bootstrap_keeps_all_records_and_is_paired_reproducible(self):
        a=bootstrap_weights(100,42);b=bootstrap_weights(100,42)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a.sum(axis=1),np.full(100,30))
        self.assertEqual(a.shape,(100,30));self.assertTrue((a.sum(axis=0)>0).all())
        with self.assertRaises(ValueError):bootstrap_weights(1)
    def test_zero_event_recording_keeps_exposure_and_undefined_sensitivity(self):
        records=pd.DataFrame({'Record':[1,2],'Detected':[1,0],'Eligible_Events':[1,0],
                              'Off_Event_Alarm_Starts':[2,0],'Exposure_s':[100,100],
                              'Alarm_Active_s':[20,0],'Samples':[10,10],'Alert_Timestamps':[2,0]})
        leads=pd.DataFrame({'Record':[1],'Lead_Time_s':[5.]})
        result=bootstrap_metrics(records,leads,np.array([[2,0],[0,2]]),records=(1,2))
        self.assertEqual(result['Sensitivity'][0],1);self.assertTrue(np.isnan(result['Sensitivity'][1]))
        np.testing.assert_allclose(result['False_Alarm_Starts_Per_Hour'],[72,0])
        np.testing.assert_allclose(result['Alarm_Active_Fraction'],[.2,0])
        self.assertEqual(result['Lead_Median_s'][0],5);self.assertTrue(np.isnan(result['Lead_Median_s'][1]))
    def test_undefined_intervals_report_valid_replicate_count(self):
        self.assertEqual(interval([1,np.nan,2])['Defined_Replicates'],2)
        self.assertTrue(np.isnan(interval([np.nan])['Lower_95']))
        self.assertTrue(np.isnan(divide(0,0)))
    def test_one_alarm_cannot_match_twice_and_onset_alert_is_late(self):
        e=pd.DataFrame({'Record':[1,1],'Onset_s':[10.,12.],'Event_ID':[1,2],'Evaluation_Status':['eligible']*2})
        alarm=[{'Record':1,'Alarm_ID':0,'Start_s':5.}]
        matched,_=match_events(e,pd.DataFrame(),alarm,10,'episode_start_one_to_one')
        self.assertEqual(matched.Detected.tolist(),[True,False]);self.assertEqual(matched.Lead_Time_s.iloc[0],5)
        matched,_=match_events(e.iloc[:1],pd.DataFrame(),[{'Record':1,'Alarm_ID':0,'Start_s':10.}],10,'episode_start_one_to_one')
        self.assertFalse(matched.Detected.iloc[0])
    def test_low_between_normal_endpoints_is_not_evaluable_exposure(self):
        g=pd.DataFrame({'Record':[1,1],'time_s':[1.5,3.1]});nt=np.arange(40.);s=np.full(40,98.)
        self.assertTrue(links(g,{1:(nt,s)})[0]);s[2]=94
        self.assertFalse(links(g,{1:(nt,s)})[0])


@unittest.skipUnless((DEFAULT_OUTPUT/'manifest.json').exists(),'Generate the fixed-policy local-30 evaluation')
class Local30ArtifactTests(unittest.TestCase):
    def test_all_recordings_are_retained_and_counts_match_audit(self):
        per=pd.read_csv(DEFAULT_OUTPUT/'event_per_recording.csv');summary=pd.read_csv(DEFAULT_OUTPUT/'event_summary.csv')
        for _,g in per.groupby(['Target','Model','Window_s','Subset']):self.assertEqual(sorted(g.Record),list(range(1,31)))
        old=pd.read_csv('results/reliability/final/window_summary.csv')
        for r in summary.itertuples():
            v=old[(old.Target==r.Target)&(old.Model==r.Model)&old.Window_s.eq(r.Window_s)&(old.Subset==r.Subset)&(old.Method=='episode_start_one_to_one')].iloc[0]
            self.assertEqual(r.Eligible_Events,v.Eligible_Events);self.assertEqual(r.Detected,v.Detected)
            self.assertAlmostEqual(r.False_Alarm_Starts_Per_Hour,v.Off_Event_False_Alarms_Per_Hour)
            np.testing.assert_allclose(r.Lead_Median_s,v.Lead_Median_s,equal_nan=True)
    def test_event_matching_and_cluster_draws_are_auditable(self):
        m=pd.read_csv(DEFAULT_OUTPUT/'event_matches.csv');d=m[m.Detected.eq(True)]
        self.assertFalse(d.duplicated(['Target','Model','Window_s','Record','Alarm_ID']).any())
        self.assertTrue((d.Lead_Time_s>0).all());self.assertTrue((d.Lead_Time_s<=d.Window_s).all())
        weights=pd.read_csv(DEFAULT_OUTPUT/'bootstrap_recording_weights.csv.gz',index_col=0)
        self.assertTrue(weights.sum(axis=1).eq(30).all());self.assertEqual(weights.shape,(2000,30))
    def test_outputs_and_protected_files_are_unchanged(self):self.assertEqual(verify_output(DEFAULT_OUTPUT)['status'],'passed')
