"""SYNTHETIC fixtures only; never reference annotations."""
import copy,json,tempfile,unittest
from pathlib import Path
import pandas as pd
from experiments.annotation_workflow_v2_20261028.validation import validate_payload,load_json
from experiments.annotation_workflow_v2_20261028.matching import match_pulses,score_window
from experiments.annotation_workflow_v2_20261028.evaluate import write_evaluation
class Workflow(unittest.TestCase):
 def setUp(self):
  self.man=pd.DataFrame([dict(WindowID='SYNTHETIC_W',Duration_s=20,ReferenceSubset=True,AnonRecordID='SYNTHETIC_R')]);self.row=dict(window_id='SYNTHETIC_W',quality='adequately_interpretable',scored_regions=[[.08,19.92]],pulses=[dict(time_s=1.,certainty='clear')],notes='',reviewed_no_reference_pulses=False,detector_overlay_ever_shown=False);self.p=dict(packet_id='SYNTHETIC_ONLY',reviewer_id='SYNTHETIC',finalized=True,windows=[self.row])
 def valid(self):return validate_payload(self.p,self.man,'SYNTHETIC_ONLY')
 def test_contract(self):
  self.valid()
  for change in [lambda r:r.pop('notes'),lambda r:r.update(window_id='unknown'),lambda r:r.update(notes=False),lambda r:r.update(scored_regions=[[0,21]]),lambda r:r.update(scored_regions=[[0,1],[1,2]]),lambda r:r.update(quality='partially_interpretable'),lambda r:r.update(pulses=[]),lambda r:r.update(reviewed_no_reference_pulses=True)]:
   original=copy.deepcopy(self.p);change(self.row)
   with self.assertRaises(ValueError):self.valid()
   self.p=original;self.row=self.p['windows'][0]
 def test_timestamps(self):
  for t in [True,'1',-1,21,float('nan'),float('inf')]:
   self.row['pulses'][0]['time_s']=t
   with self.assertRaises(ValueError):self.valid()
 def test_duplicates_and_incomplete(self):
  self.row['pulses']*=2
  with self.assertRaises(ValueError):self.valid()
  self.row['pulses']=self.row['pulses'][:1];self.p['windows']*=2
  with self.assertRaises(ValueError):self.valid()
  self.p['windows']=[]
  with self.assertRaises(ValueError):self.valid()
 def test_json(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'synthetic.json'
   for s in ['{"a":1,"a":2}','{"a":NaN}','{']:
    p.write_text(s)
    with self.assertRaises(ValueError):load_json(p)
 def test_matching(self):
  self.assertEqual(len(match_pulses([1.08],[1.])),1);self.assertEqual(len(match_pulses([1.081],[1.])),0)
  self.assertEqual(len(match_pulses([1.,1.05],[1.025])),1)
  self.assertEqual(len(match_pulses([1.,1.1],[1.06,1.15])),2)
  self.row.update(scored_regions=[[0,.99],[1.01,2]],pulses=[dict(time_s=1.02,certainty='clear')]);c,_=score_window([.98],self.row);self.assertEqual(c['Matched'],0)
 def test_unknown(self):
  self.row.update(quality='uninterpretable',scored_regions=[],pulses=[],notes='SYNTHETIC unreadable');self.valid();c,_=score_window([1.],self.row);self.assertIsNone(c['Matched'])
 def test_synthetic_evaluation(self):
  rows,cov=self.valid();v=[dict(path='SYNTHETIC_ONLY',sha256='SYNTHETIC_ONLY',payload=self.p,rows=rows,coverage=cov)]
  with tempfile.TemporaryDirectory() as t:
   out=Path(t)/'synthetic';write_evaluation(v,self.man,dict(packet_id='SYNTHETIC_ONLY'),pd.DataFrame(dict(WindowID=['SYNTHETIC_W']*2,PeakRelative_s=[1.02,2.])),out,{})
   s=pd.read_csv(out/'detector_summary.csv').iloc[0];self.assertEqual(s.Matched,1);self.assertEqual(s.PrecisionDenominator_Detections,2);self.assertEqual(s.RecallDenominator_References,1)
   self.assertEqual(len(pd.read_csv(out/'recording_bootstrap_intervals.csv')),5)
   with self.assertRaises(ValueError):write_evaluation(v,self.man,dict(packet_id='SYNTHETIC_ONLY'),pd.DataFrame(),out,{})
if __name__=='__main__':unittest.main()
