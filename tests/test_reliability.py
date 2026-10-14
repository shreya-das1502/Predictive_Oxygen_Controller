"""Independent reliability calculation and alarm-matching regression checks."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.reliability_evaluation import (valid_window, raw_events, links, episodes, burden,
    eligible_events, match_events, validate_predictions, off_event_alarms)
from src.target_reporting import evaluate_events


def stream(times, decisions, labels=None):
    return pd.DataFrame({'Record':1,'sample_id':range(len(times)),'time_s':times,'Prediction':decisions,
                         'Label':labels if labels is not None else np.zeros(len(times),int),'Model':'Test','spo2':98.})


def event(onset, ident=0):
    return {'Record':1,'Event_ID':ident,'Onset_s':float(onset),'Qualifying_Onset':True,'Has_91_92':False}


class ReliabilityTests(unittest.TestCase):
    def test_raw_boundary_gap_and_complete_horizon(self):
        nt=np.r_[0.,np.arange(3.,41.)];s=np.full(len(nt),98.);s[-1]=94
        self.assertFalse(valid_window(nt,s,1.,40.))
        self.assertTrue(valid_window(nt,s,3.,40.))
        self.assertFalse(valid_window(nt,s,-1,40))
        self.assertFalse(valid_window(nt,s,3,41))
        # Production regression: old internal-only gap check incorrectly accepted [1,31].
        n=pd.DataFrame({'Time [s]':nt,'SpO2':s});pred=stream([4.],[1]);e=pd.DataFrame([event(31)])
        with patch('src.target_reporting.pd.read_csv',return_value=n):
            matched,_=evaluate_events(pred,e,'new_onset')
        self.assertEqual(matched.Evaluation_Status.iloc[0],'incomplete_raw_lead_in')
    def test_missing_invalid_and_gapped_lead_in(self):
        nt=np.arange(51.);s=np.full(51,98.)
        for bad in [np.nan,0.,101.]:
            v=s.copy();v[20]=bad;self.assertFalse(valid_window(nt,v,10,40))
        self.assertFalse(valid_window(np.delete(nt,20),np.delete(s,20),10,40))
        self.assertTrue(valid_window(nt,s,10.5,40.5))
        with self.assertRaises(ValueError):valid_window([0,0],[98,98],0,1)
    def test_raw_onset_left_censoring_and_unknown_prior(self):
        s=np.array([92,98,np.nan,92,98,94,94,98,92.])
        e=raw_events({1:(np.arange(len(s),dtype=float),s)})
        self.assertEqual(e.Qualifying_Onset.tolist(),[False,False,True,True])
        self.assertEqual(e.End_s.tolist(),[0,3,6,8]);self.assertTrue(e.Has_91_92.iloc[0])
        e=raw_events({1:(np.array([0.,3.]),np.array([98.,92.]))});self.assertFalse(e.Qualifying_Onset.iloc[0])
    def test_half_open_alert_boundary_and_exact_lead(self):
        e=pd.DataFrame([event(40)]);g=stream([30.,40.],[1,1]);alarms=[dict(Record=1,Alarm_ID=0,Start_s=30.)]
        m,used=match_events(e.assign(Evaluation_Status='eligible'),g,alarms,10,'episode_start_one_to_one')
        self.assertTrue(m.Detected.iloc[0]);self.assertEqual(m.Lead_Time_s.iloc[0],10.)
        m,_=match_events(e.assign(Evaluation_Status='eligible'),g,[dict(Record=1,Alarm_ID=0,Start_s=40.)],10,'episode_start_one_to_one')
        self.assertFalse(m.Detected.iloc[0]);self.assertTrue(np.isnan(m.Lead_Time_s.iloc[0]))
    def test_shared_alarm_cannot_credit_two_events(self):
        e=pd.DataFrame([event(40,1),event(45,2)]).assign(Evaluation_Status='eligible')
        g=stream([35.],[1]);alarms=[dict(Record=1,Alarm_ID=0,Start_s=35.)]
        m,used=match_events(e,g,alarms,15,'episode_start_one_to_one')
        self.assertEqual(m.Detected.tolist(),[True,False]);self.assertEqual(len(used),1)
        m,_=match_events(e,g,alarms,15,'any_positive_timestamp_reusable');self.assertTrue(m.Detected.all())
    def test_event_cannot_take_multiple_alarms(self):
        e=pd.DataFrame([event(40)]).assign(Evaluation_Status='eligible');g=stream([31,32,33],[1,0,1])
        a=[dict(Record=1,Alarm_ID=i,Start_s=t) for i,t in enumerate([31.,33.])]
        m,used=match_events(e,g,a,10,'episode_start_one_to_one');self.assertEqual(len(used),1);self.assertEqual(m.Lead_Time_s.iloc[0],9)
    def test_existing_alarm_not_reset_at_window_start(self):
        g=stream(np.arange(0.,41.),np.ones(41,int));a=[x|{'Record':1} for x in episodes(g)]
        e=pd.DataFrame([event(40)]).assign(Evaluation_Status='eligible')
        m,_=match_events(e,g,a,10,'episode_start_one_to_one');self.assertFalse(m.Detected.iloc[0])
        m,_=match_events(e,g,a,10,'any_positive_timestamp_reusable');self.assertEqual(m.Lead_Time_s.iloc[0],10.)
    def test_independent_false_runs_and_exposure(self):
        g=stream([0.,1.,2.,3.,7.,8.],[1,1,0,1,1,1],[0,0,0,1,0,0])
        b,eps=burden(g);self.assertEqual(b['FP_Timestamps'],4);self.assertEqual(len(eps),3)
        self.assertEqual(b['False_Episodes_Target_Run'],2);self.assertEqual(b['Exposure_s'],4)
        self.assertEqual(b['Alert_Fraction'],5/6);self.assertEqual(b['Alert_Time_s'],3)
        # Exactly 2s remains linked, strictly greater splits.
        self.assertEqual(len(episodes(stream([0.,2.],[1,1]))),1)
        self.assertEqual(len(episodes(stream([0.,2.001],[1,1]))),2)
    def test_empty_singleton_and_never_alert(self):
        b,_=burden(stream([],[]),{1:(np.arange(40.),np.full(40,98.))});self.assertEqual(b['Exposure_s'],0)
        self.assertTrue(np.isnan(b['False_Episodes_Per_Hour']))
        b,_=burden(stream([1.],[1]));self.assertEqual(b['Alert_Episodes'],1);self.assertEqual(b['Exposure_s'],0)
        b,_=burden(stream([1.,2.],[0,0]));self.assertEqual(b['Alert_Episodes'],0)
    def test_hidden_low_invalid_or_gap_does_not_bridge_exposure(self):
        g=stream([1.5,3.1],[1,1]);nt=np.arange(50.);s=np.full(50,98.)
        self.assertTrue(links(g,{1:(nt,s)})[0])
        for bad in [94.,np.nan]:
            v=s.copy();v[2]=bad;self.assertFalse(links(g,{1:(nt,v)})[0])
        self.assertFalse(links(g,{1:(np.delete(nt,2),np.delete(s,2))})[0])
    def test_eligibility_identical_for_all_predictions(self):
        nt=np.arange(61.);s=np.full(61,98.);e=pd.DataFrame([event(40)])
        g=stream([35.],[0]);a=eligible_events(e,g,{1:(nt,s)},5)
        g.Prediction=1;b=eligible_events(e,g,{1:(nt,s)},5)
        pd.testing.assert_frame_equal(a,b);self.assertEqual(a.Evaluation_Status.iloc[0],'eligible')
        self.assertEqual(eligible_events(e,g,{1:(nt,s)},4).Evaluation_Status.iloc[0],'no_eligible_prediction_opportunity')
    def test_off_event_false_alarms_distinguish_redundancy(self):
        alarms=[dict(Record=1,Start_s=x) for x in [10.,31.,33.,40.,50.]]
        # Two alarms near the same true event are redundant, not both off-event.
        self.assertEqual(off_event_alarms(alarms,pd.DataFrame([event(40)]),10),3)
        self.assertEqual(off_event_alarms([dict(Record=1,Start_s=30.)],pd.DataFrame([event(40)]),10),0)
    def test_duplicate_rows_rejected(self):
        g=stream([1.,1.],[1,0])
        with self.assertRaises(ValueError):validate_predictions(g)


@unittest.skipUnless(Path('results/reliability/final/manifest.json').exists(),'Run reliability audit for artifact checks')
class ReliabilityArtifactTests(unittest.TestCase):
    def test_independent_recomputations_match_saved_results(self):
        c=pd.read_csv('results/reliability/final/independent_checks.csv');self.assertEqual(c.Discrepancies.sum(),0)
    def test_no_duplicate_primary_event_credit(self):
        m=pd.read_csv('results/reliability/final/window_event_matches.csv');m=m[(m.Method=='episode_start_one_to_one')&m.Detected.eq(True)]
        self.assertFalse(m.duplicated(['Target','Model','Window_s','Record','Alarm_ID']).any())
        self.assertTrue((m.Lead_Time_s>0).all());self.assertTrue((m.Lead_Time_s<=m.Window_s).all())
