import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.final_audit_20261027T215012_IST.engine import (divide,interval,target_at,valid_coverage,independent_history,raw_events,alarms,match_event_starts,evaluate_record,check_split,check_aligned,check_oof,cutoff_from_grid,event_bootstrap,paired_common_leads,lead_bootstrap)

HERE=Path(__file__).resolve().parents[1]

class EngineTests(unittest.TestCase):
    def test_half_open_future_target_and_current_exclusion(self):
        nt=np.arange(101.);s=np.full(101,98.);s[0]=90;s[30]=94
        self.assertEqual(target_at(0,nt,s),(1,94.))
        s[30]=98;s[31]=94;self.assertEqual(target_at(0,nt,s),(0,98.))

    def test_missing_horizon_is_not_a_negative(self):
        nt=np.arange(61.);s=np.full(61,98.);s[20]=np.nan
        self.assertTrue(np.isnan(target_at(0,nt,s)[0]));self.assertTrue(np.isnan(target_at(40,nt,s)[0]))

    def test_future_mutations_cannot_change_history(self):
        nt=np.arange(61.);s=99-.02*nt;a=independent_history(14.5,nt,s)
        s[15:]=50;np.testing.assert_array_equal(a,independent_history(14.5,nt,s))
        np.testing.assert_array_equal(a,independent_history(14.5,nt[:15],s[:15]))

    def test_current_and_history_time_alignment(self):
        nt=np.arange(21.);s=98-.1*nt;f=independent_history(10.5,nt,s)
        self.assertAlmostEqual(f[0],97.);self.assertAlmostEqual(f[3],-.1);self.assertAlmostEqual(f[6],.5)

    def test_invalid_latest_value_is_not_imputed_from_future(self):
        f=independent_history(1.2,np.arange(3.),np.array([98.,np.nan,94.]))
        self.assertTrue(np.isnan(f[0]));self.assertEqual(f[1],98.)

    def test_record_separation_guard(self):
        good=dict(Outer_Test_Record=4,Train_Records=[1,2,3],Inner=[dict(train_records=[2,3],validation_records=[1]),dict(train_records=[1,3],validation_records=[2]),dict(train_records=[1,2],validation_records=[3])])
        check_split(good,range(1,5))
        bad=good|dict(Train_Records=[1,2,4])
        with self.assertRaises(ValueError):check_split(bad,range(1,5))

    def test_overlapping_inner_fold_is_rejected(self):
        bad=dict(Outer_Test_Record=3,Train_Records=[1,2],Inner=[dict(train_records=[1,2],validation_records=[2])])
        with self.assertRaises(ValueError):check_split(bad,range(1,4))

    def test_oof_guard_excludes_evaluation_record_and_wrong_labels(self):
        data=pd.DataFrame(dict(sample_id=range(6),Record=[1,1,2,2,3,3],time_s=[1.,2.,1.,2.,1.,2.],Label=[0,1,0,1,0,1]))
        fit=data[data.Record.ne(3)].copy();check_oof(fit,data,3)
        with self.assertRaises(ValueError):check_oof(data,data,3)
        fit.loc[0,'Label']=1
        with self.assertRaises(ValueError):check_oof(fit,data,3)

    def test_matched_targets_required_for_paired_forecasts(self):
        a=pd.DataFrame(dict(sample_id=[1,2],FutureMin=[94.,98.]));b=a.copy();check_aligned(a,b,['FutureMin'])
        b.loc[0,'FutureMin']=95
        with self.assertRaises(ValueError):check_aligned(a,b,['FutureMin'])

    def test_cutoff_is_training_grid_only_and_constraints_are_joint(self):
        grid=pd.DataFrame(dict(Threshold=[.1,.5,.9],Threshold_hex=[float(v).hex() for v in [.1,.5,.9]],Detected=[10,5,3],Off_Event_Alarm_Starts=[9,2,1],Alarm_Active_s=[100.,20.,2.],False_Alarm_Starts_Per_Hour=[9.,2.,1.],Alarm_Active_Fraction=[.9,.1,.01]))
        self.assertEqual(cutoff_from_grid(grid).Threshold,.9)
        with self.assertRaises(ValueError):cutoff_from_grid(grid.iloc[:2])

    def test_boundary_and_one_to_one_matching(self):
        ev=pd.DataFrame(dict(Record=[1,1],Event_ID=[40,45],Onset_s=[40.,45.],Qualifying_Onset=[True,True],Sustained3=[True,True]))
        nt=np.arange(101.);s=np.full(101,98.)
        matched,_=match_event_starts(ev,np.array([35.]),np.array([35.]),nt,s,'original')
        self.assertEqual(matched.Detected.tolist(),[True,False])
        matched,_=match_event_starts(ev.iloc[:1],np.array([10.,40.]),np.array([10.,40.]),nt,s,'original')
        self.assertTrue(matched.Detected.iloc[0]);self.assertEqual(matched.Lead_Time_s.iloc[0],30.)
        matched,_=match_event_starts(ev.iloc[:1],np.array([39.,40.]),np.array([40.]),nt,s,'original')
        self.assertFalse(matched.Detected.iloc[0])

    def test_old_alarm_not_restarted_at_window_boundary(self):
        times=np.arange(41.);starts,_=alarms(times,np.ones(41,bool),np.r_[False,np.ones(40,bool)])
        self.assertEqual(starts.tolist(),[0.])
        ev=pd.DataFrame(dict(Record=[1],Event_ID=[40],Onset_s=[40.],Qualifying_Onset=[True],Sustained3=[True]));nt=np.arange(101.);s=np.full(101,98.)
        m,_=match_event_starts(ev,times,starts,nt,s,'original');self.assertFalse(m.Detected.iloc[0])

    def test_denominators_remain_undefined_at_zero_exposure(self):
        self.assertTrue(np.isnan(float(divide(0,0))))
        self.assertEqual(float(divide(0,10)),0.)
        self.assertEqual(float(divide(3*3600,1800)),6.)

    def test_common_event_leads_not_different_detected_populations(self):
        a=pd.DataFrame(dict(Record=[1,1],Event_ID=[10,20],Detected=[True,False],Lead_Time_s=[10.,np.nan]));b=pd.DataFrame(dict(Record=[1,1],Event_ID=[10,30],Detected=[True,True],Lead_Time_s=[6.,15.]))
        d=paired_common_leads(a,b);self.assertEqual(len(d),1);self.assertEqual(d.Lead_Time_s.iloc[0],4.)
        empty=paired_common_leads(a,b[b.Event_ID.eq(30)]);self.assertTrue(np.isnan(lead_bootstrap(empty,np.ones((2,30),int))).all())

    def test_record_bootstrap_preserves_whole_contributions(self):
        p=pd.DataFrame(dict(Record=range(1,31),Detected=[1]+[0]*29,Eligible_Events=[2,1]+[0]*28,Off_Event_Alarm_Starts=[1]+[0]*29,Exposure_s=[100.]*30,Alarm_Active_s=[1.]+[0]*29,Source_Duration_s=[480.]*30))
        w=np.zeros((2,30),int);w[0,0]=2;w[1,2]=30
        b=event_bootstrap(p,w);self.assertEqual(b['Sensitivity'][0],.5);self.assertTrue(np.isnan(b['Sensitivity'][1]));self.assertEqual(interval(b['Sensitivity'])['Defined_Replicates'],1)

    def test_zero_event_record_is_valid_and_missing_leads_stay_missing(self):
        nt=np.arange(101.);s=np.full(101,98.);ev=raw_events({1:(nt,s)})
        # Explicit empty schema also tests records with no event of either definition.
        ev=pd.DataFrame(columns=['Record','Event_ID','Onset_s','End_s','Qualifying_Onset','Sustained3'])
        frame=pd.DataFrame(dict(Record=1,time_s=[1.,2.],SpO2_Current=98.,Prediction=[0,0]))
        result,*_=evaluate_record(frame,ev,nt,s,'original');self.assertEqual(result['Detected'],0);self.assertTrue(np.isnan(result['Sensitivity']));self.assertTrue(np.isnan(result['Lead_Median_s']))

class SavedAuditTests(unittest.TestCase):
    def setUp(self):self.result=json.loads((HERE/'reproduced_metrics.json').read_text())

    def test_reported_counts_reproduced(self):
        d=pd.DataFrame(self.result['Logistic_Results'])
        for definition,want in [('original',{'SpO2_history':9,'PPG':0,'Combined':10}),('sustained3',{'SpO2_history':9,'PPG':0,'Combined':9})]:
            self.assertEqual(d[d.Definition.eq(definition)].set_index('Modality').Detected.to_dict(),want)

    def test_paired_mae_difference_matches_raw_errors(self):
        d=pd.read_csv(HERE/'reproduced_forecast_summary.csv');a=d[d.Model.eq('Ridge_Combined')&d.Population.eq('full_target')].iloc[0];b=d[d.Model.eq('Ridge_SpO2_history')&d.Population.eq('full_target')].iloc[0]
        self.assertAlmostEqual(a.MAE-b.MAE,-.005663791,places=6)
        self.assertEqual(a.N,b.N);self.assertEqual(a.N,20150)

    def test_all_recordings_preserved_and_no_refits(self):
        self.assertEqual(self.result['Recordings'],30);self.assertEqual(self.result['New_Model_Fits'],0);self.assertEqual(self.result['Historical_Files_Unchanged'],4347)

    def test_failure_table_has_explicit_policy_for_excluded_events(self):
        d=pd.read_csv(HERE/'event_failure_analysis.csv');self.assertEqual(len(d),484);self.assertFalse(d.Policy.isna().any());self.assertFalse(d.Cause.isna().any())

if __name__=='__main__':unittest.main()
