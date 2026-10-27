"""Software fixtures only; never patient/reference measurements."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('lead_analysis',P/'analyze.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
from src.reliability_evaluation import match_events
class LeadAnalysis(unittest.TestCase):
 def test_undefined_rates(self):self.assertTrue(np.isnan(a.rate(0,0)));self.assertEqual(a.rate(1,2),.5)
 def test_signed_delta(self):
  onset=100.;history=90.;combined=85.;self.assertEqual((onset-combined)-(onset-history),5.)
  self.assertEqual((onset-95.)-(onset-history),-5.)
 def test_one_to_one_and_boundary(self):
  ev=pd.DataFrame(dict(Record=[1,1],Event_ID=[100,110],Onset_s=[100.,110.],Evaluation_Status=['eligible']*2));alarms=[dict(Record=1,Alarm_ID=0,Start_s=80.)]
  m,used=match_events(ev,pd.DataFrame(),alarms,30,'episode_start_one_to_one');self.assertEqual(m.Detected.tolist(),[True,False]);self.assertTrue(np.isnan(m.Lead_Time_s.iloc[1]));self.assertEqual(len(used),1)
  for start,detected in [(70.,True),(100.,False),(101.,False),(69.99,False)]:
   m,_=match_events(ev.iloc[:1],pd.DataFrame(),[dict(Record=1,Alarm_ID=0,Start_s=start)],30,'episode_start_one_to_one');self.assertEqual(m.Detected.iloc[0],detected)
 def test_record_cluster_weights(self):
  w=np.zeros((3,30),int);w[0,0]=2;w[1,1]=3;w[2,2]=1
  mean,median=a.paired_bootstrap([1.,3.,10.],[1,1,2],w)
  np.testing.assert_allclose(mean[:2],[2.,10.]);np.testing.assert_allclose(median[:2],[2.,10.]);self.assertTrue(np.isnan(mean[2]))
 def test_empty_paired(self):
  mean,median=a.paired_bootstrap([],[],np.ones((2,30),int));self.assertTrue(np.isnan(mean).all());self.assertTrue(np.isnan(median).all());self.assertEqual(a.ci(mean)[2],0)
 def test_outputs(self):
  ev=pd.read_csv(P/'event_level_lead_times.csv');per=pd.read_csv(P/'recording_level_summary.csv');pair=pd.read_csv(P/'paired_lead_time_summary.csv');a.validate_tables(ev,per,pair)
  self.assertEqual(len(per),360);self.assertEqual(len(pair),6)
  summary=pd.read_csv(P/'overall_summary.csv');t=pd.read_csv(P/'timely_warning_summary.csv')
  for r in summary.itertuples():
   q=ev[ev.Model.eq(r.Model)&ev.Definition.eq(r.Definition)];prefix='History' if r.Modality=='SpO2_history' else 'Combined';self.assertEqual(r.Detected,q[prefix+'_Detected'].eq(True).sum());self.assertEqual(r.Timely10,q[prefix+'_Lead_s'].ge(10).sum());self.assertEqual(r.Eligible_Events,q[prefix+'_Detected'].notna().sum())
   for row in t[t.Model.eq(r.Model)&t.Definition.eq(r.Definition)&t.Modality.eq(r.Modality)].itertuples():self.assertEqual(row.Events,q[prefix+'_Lead_s'].ge(row.MinimumLead_s).sum())
  for n in ['warning_lead_time_comparison.png','timely_warning_comparison.png']:self.assertGreater((P/n).stat().st_size,10000)
if __name__=='__main__':unittest.main()
