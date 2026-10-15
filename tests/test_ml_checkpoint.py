"""Cache safety regressions; synthetic arrays are not research outcomes."""
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from ml_research.checkpoint_evaluation import MetricCache,event_key
from src.evaluation import metrics
from ml_research.baseline import sha
from src.reliability_evaluation import raw_events


class CheckpointTests(unittest.TestCase):
    def test_metric_cache_matches_original_and_separates_labels_scores_decisions(self):
        cache=MetricCache()
        for y,s,p in [([0,1,0],[.1,.8,.3],[0,1,0]),([1,1,0],[.1,.8,.3],[0,1,0]),([0,1,0],[.9,.8,.3],[0,1,0]),([0,1,0],[.1,.8,.3],[1,1,0]),([],[],[])]:
            a=cache(y,s,p);b=metrics(y,s,p)
            for key in a:np.testing.assert_equal(a[key],b[key])
            a['TP']=999
            self.assertNotEqual(cache(y,s,p)['TP'],999)
        self.assertEqual(len(cache.values),5)

    def test_event_cache_separates_record_target_window_decisions(self):
        a=event_key('new_onset',1,30,[0,1])
        self.assertEqual(a,event_key('new_onset',1,30,np.array([0.,1.])))
        for other in [event_key('future_low',1,30,[0,1]),event_key('new_onset',2,30,[0,1]),event_key('new_onset',1,15,[0,1]),event_key('new_onset',1,30,[1,0])]:self.assertNotEqual(a,other)
        with self.assertRaises(ValueError):event_key('new_onset',1,30,[0,256])

    def test_interrupted_condition_resumes_without_duplicates_and_rejects_corruption(self):
        from ml_research.checkpoint_evaluation import evaluate
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'new';output.mkdir()
            old=root/'results/final/new_onset';old.mkdir(parents=True)
            original=root/'results/alarm_budget_experiment/run02';original.mkdir(parents=True)
            data=pd.DataFrame([dict(sample_id=(r-1)*2+i,Record=r,pulse_row=i,time_s=float(i+1),Label=0,Pulse_Rate_BPM=70.) for r in [1,2] for i in range(2)])
            scored=data.assign(Model='ELM',Score=.2,Prediction=0,Threshold=1.)
            scored.to_csv(old/'per_record_predictions.csv',index=False)
            data[['sample_id','Record','time_s','pulse_row']].assign(raw=.2,Target='new_onset',Horizon_s=30,FeatureSet='base10',Model='ELM').to_csv(output/'outer_scores_unlabeled.csv.gz',index=False)
            selected=pd.DataFrame([dict(Target='new_onset',Model='ELM',Budget=1.,Outer_Test_Record=r,Threshold_hex=float(1).hex(),Horizon_s=30,FeatureSet='base10',ScoreSpace='raw') for r in [1,2]])
            selected.to_csv(output/'selected_thresholds.csv',index=False);selected.to_csv(original/'selected_thresholds.csv',index=False)
            (output/'selection_complete.json').write_text(json.dumps(dict(thresholds_sha256=sha(output/'selected_thresholds.csv'))))
            pd.DataFrame([dict(Target='new_onset',Model='ELM',Budget=1.,Window_s=5,Subset=s,Detected=0,Eligible_Events=0,Missed_Events=0,Exposure_s=2.,Off_Event_Alarm_Starts=0,Alarm_Active_s=0.) for s in ['all','91_92']]).to_csv(original/'summary.csv',index=False)
            numeric={r:(np.arange(121.),np.full(121,98.)) for r in range(1,31)};numeric[1][1][60]=94.
            events=raw_events(numeric)
            protocol=dict(targets=['new_onset'],score_spaces=['raw'],budgets=[1.],event_windows=[5],bootstrap_repetitions=2,bootstrap_seed=42)
            with patch('ml_research.checkpoint_evaluation.ROOT',root),patch('ml_research.experiment.datasets',return_value={h:data.copy() for h in [5,10,15,30]}):
                with patch('builtins.print',side_effect=RuntimeError('simulated interruption')):
                    with self.assertRaisesRegex(RuntimeError,'simulated interruption'):evaluate(output,protocol,numeric,events)
                first=next((output/'checkpoints').glob('*/complete.json'));before=sha(first)
                with patch('builtins.print'):evaluate(output,protocol,numeric,events)
                self.assertEqual(sha(first),before)
                predictions=pd.read_csv(output/'outer_predictions.csv.gz')
                self.assertEqual(len(predictions),84)
                self.assertFalse(predictions.duplicated(['Target','Horizon_s','FeatureSet','Model','ScoreSpace','Budget','sample_id']).any())
                corrupt=first.parent/'per.csv';corrupt.write_text(corrupt.read_text()+'\n')
                with self.assertRaisesRegex(ValueError,'Checkpoint changed'):evaluate(output,protocol,numeric,events)


if __name__=='__main__':unittest.main()
