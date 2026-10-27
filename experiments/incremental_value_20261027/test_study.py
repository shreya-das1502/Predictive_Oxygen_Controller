"""Hand-checkable causal label, history and alarm invariants for the new study."""
import unittest
import numpy as np
import pandas as pd
from experiments.incremental_value_20261027.study import (HISTORY,FEATURE_SETS,history_features,minimum_labels,event_definition,evaluate_record,fixed_forecasts)
from experiments.incremental_value_20261027.run import fit_classifier
from src.reliability_evaluation import raw_events

class StudyTests(unittest.TestCase):
    def test_history_cannot_read_future_numeric_samples(self):
        nt=np.arange(61.);s=99-.02*nt
        a=history_features([10.25,14.5],nt,s)
        changed=s.copy();changed[15:]=50
        pd.testing.assert_frame_equal(a,history_features([10.25,14.5],nt,changed))
        pd.testing.assert_frame_equal(a,history_features([10.25,14.5],nt[:15],s[:15]))

    def test_history_slope_units_and_clock_age(self):
        f=history_features([10.5],np.arange(21.),98-.1*np.arange(21.)).iloc[0]
        self.assertAlmostEqual(f.SpO2_Current,97.)
        self.assertAlmostEqual(f.SpO2_Trend_10s,-.1)
        self.assertAlmostEqual(f.SpO2_Age_s,.5)

    def test_unknown_latest_numeric_does_not_silently_fill_from_future(self):
        s=np.array([98.,np.nan,94.]);f=history_features([1.2],np.arange(3.),s).iloc[0]
        self.assertTrue(np.isnan(f.SpO2_Current))
        self.assertEqual(f.SpO2_Mean_30s,98.)

    def test_stale_current_and_first_history(self):
        f=history_features([0.,3.],np.array([0.,1.]),[98.,97.])
        self.assertEqual(f.SpO2_Std_30s.iloc[0],0.)
        self.assertEqual(f.SpO2_Trend_30s.iloc[0],0.)
        self.assertTrue(np.isnan(f.SpO2_Current.iloc[1]))

    def test_primary_future_min_boundary_and_current_exclusion(self):
        nt=np.arange(101.);s=np.full(101,98.);s[0]=90;s[30]=94.
        y,m,_=minimum_labels([0.],nt,s)
        self.assertEqual(y[0],1);self.assertEqual(m[0],94)
        s[30]=98;s[31]=94
        self.assertEqual(minimum_labels([0.],nt,s)[0][0],0)

    def test_missing_future_and_incomplete_tail_are_not_negatives(self):
        nt=np.arange(61.);s=np.full(61,98.);s[20]=np.nan
        y,_,_=minimum_labels([0.,40.],nt,s)
        self.assertTrue(np.isnan(y).all())

    def test_sustained_definition_preserves_original_events(self):
        e=pd.DataFrame({'Sustained3':[False,True],'Record':[1,1],'Event_ID':[10,30]})
        self.assertEqual(len(event_definition(e,'original')),2)
        self.assertEqual(event_definition(e,'sustained3').Event_ID.tolist(),[30])
        self.assertEqual(len(e),2)

    def fixture(self,starts):
        nt=np.arange(101.);s=np.full(101,98.);s[40]=94.;s[45:48]=93.
        numeric={1:(nt,s)};events=raw_events(numeric)
        events['Sustained3']=(events.End_s-events.Onset_s)>=2
        times=np.arange(1.,40.)
        pred=np.isin(times,starts).astype(int)
        f=pd.DataFrame(dict(Record=1,time_s=times,SpO2_Current=98.,Prediction=pred,Label=0))
        return f,events,numeric

    def test_actual_start_is_not_reused_or_clipped(self):
        f,events,numeric=self.fixture([30.])
        result,matched=evaluate_record(f,events,numeric)
        self.assertEqual(result['Detected'],1)
        self.assertEqual(matched.loc[matched.Detected.eq(True),'Lead_Time_s'].tolist(),[10.])
        f.Prediction=1
        result,_=evaluate_record(f,events,numeric)
        self.assertEqual(result['Detected'],0)  # starts at1, before either30s warning window

    def test_sensitivity_changes_truth_and_false_burden_not_predictions(self):
        f,e,n=self.fixture([30.]);a,_=evaluate_record(f,e,n,'original');b,_=evaluate_record(f,e,n,'sustained3')
        self.assertEqual(a['Detected'],1);self.assertEqual(b['Detected'],1) # start30 can match sustained onset45
        self.assertEqual(a['Alarm_Starts'],b['Alarm_Starts'])
        self.assertEqual(a['Eligible_Events'],2);self.assertEqual(b['Eligible_Events'],1)

    def test_predictor_schema_excludes_labels_future_and_identifiers(self):
        forbidden={'Label','FutureMin','Future_Min_SpO2_30s','Record','time_s','sample_id','pulse_row'}
        for features in FEATURE_SETS.values():self.assertFalse(set(features)&forbidden)
        self.assertEqual(len(FEATURE_SETS['PPG']),10)
        self.assertEqual(len(FEATURE_SETS['Combined']),17)

    def test_transform_statistics_stay_training_only(self):
        x=pd.DataFrame({c:np.arange(12,dtype=float) for c in HISTORY});x['Label']=np.tile([0,1],6)
        model=fit_classifier(x,HISTORY,'Logistic');median=model.imputer.statistics_.copy();mean=model.scaler.mean_.copy()
        model.score(np.full((2,len(HISTORY)),100000.))
        np.testing.assert_array_equal(model.imputer.statistics_,median)
        np.testing.assert_array_equal(model.scaler.mean_,mean)
        self.assertEqual(median[0],5.5)

    def test_forecast_baselines_and_fixed_threshold_consistent(self):
        d=pd.DataFrame({'SpO2_Current':[98.,93.,98.],'SpO2_Trend_10s':[-.2,.1,0.]})
        forecasts=fixed_forecasts(d)
        np.testing.assert_array_equal(forecasts['Persistence'],[98.,93.,98.])
        np.testing.assert_allclose(forecasts['Trend10'],[92.,93.,98.])
        np.testing.assert_array_equal(94-forecasts['Trend10']>=0,forecasts['Trend10']<=94)

if __name__=='__main__':unittest.main()
