"""Synthetic safety regressions are not patient performance evidence."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from src import config
from src.reliability_evaluation import raw_events
from ml_research.core import ContextState,CONTEXT,add_context,negative_rank,training_scores
from ml_research.baseline import sha,verify_snapshot
from ml_research.experiment import select,topology_cache

class ImprovementTests(unittest.TestCase):
    def raw(self):
        return pd.DataFrame([dict(Record=r,pulse_row=i,time_s=float(i+1),PPG_Amplitude=.2+i*.01,Pulse_Rate_BPM=60.,Pulse_Interval_s=1.) for r in [1,2] for i in range(40)])

    def test_context_is_prefix_causal_and_record_reset(self):
        raw=self.raw();all_features=add_context(raw)
        prefix=raw[(raw.Record==1)&raw.pulse_row.le(10)]
        pd.testing.assert_frame_equal(add_context(prefix),all_features[(all_features.Record==1)&all_features.pulse_row.le(10)].reset_index(drop=True))
        np.testing.assert_allclose(all_features[all_features.Record==1][CONTEXT],all_features[all_features.Record==2][CONTEXT])
        changed=raw.copy();changed.loc[changed.pulse_row.gt(10),'PPG_Amplitude']=999
        np.testing.assert_allclose(add_context(changed).loc[all_features.pulse_row.le(10),CONTEXT],all_features.loc[all_features.pulse_row.le(10),CONTEXT])
        state=ContextState();state.update(raw.iloc[0]);
        with self.assertRaises(ValueError):state.update(raw.iloc[0])
        self.assertFalse(any('spo2' in f.lower() or 'label' in f.lower() for f in CONTEXT))

    def test_negative_rank_ties_and_domain(self):
        np.testing.assert_allclose(negative_rank([1,2,2,4],[0,1,2,3,5]),[0,.25,.75,.75,1])
        with self.assertRaises(ValueError):negative_rank([], [1])
        with self.assertRaises(ValueError):negative_rank([np.nan],[1])

    def test_grouped_transforms_and_calibration_reference_is_fit_only(self):
        rng=np.random.default_rng(42);rows=[]
        for r in range(1,5):
            for i in range(20):rows.append(dict(Record=r,Label=i%2,sample_id=(r-1)*20+i,**dict(zip(config.FEATURES,rng.normal(size=10)))))
        train=pd.DataFrame(rows);original=copy.deepcopy(config.MODEL_CONFIGS)
        from ml_research import core
        seen=[];actual=core.fit_model
        def track(frame,features,name):
            seen.append(set(frame.Record));return actual(frame,features,name)
        with patch('ml_research.core.fit_model',side_effect=track):
            oof,model,ref,splits=training_scores(train,config.FEATURES,'Logistic',5)
        self.assertEqual(len(seen),4)
        for records,split in zip(seen,splits):
            self.assertEqual(records,set(split['train_records']))
            self.assertFalse(records&set(split['validation_records']))
        self.assertEqual(len(ref),40)
        self.assertTrue(np.isfinite(oof['negative_rank']).all())
        self.assertEqual(config.MODEL_CONFIGS,original)
        with self.assertRaises(ValueError):training_scores(train,config.FEATURES,'Logistic',1)

    def test_outer_event_poison_cannot_change_training_selection(self):
        numeric={r:(np.arange(101.),np.full(101,98.)) for r in range(1,5)}
        for r in numeric:numeric[r][1][40:43]=94
        events=raw_events(numeric)
        frame=pd.DataFrame([dict(Record=r,sample_id=r*100+i,time_s=float(i),Label=int(35<=i<40),Prediction=0) for r in range(1,5) for i in range(1,65)])
        cache=topology_cache(frame,numeric,events);train=frame[frame.Record.ne(4)].reset_index(drop=True)
        scores=np.linspace(0,1,len(train));protocol=dict(quantiles=[0,.5,1],budgets=[1,5,10],active_cap=.05)
        a,_=select(train,scores,cache,events,numeric,4,protocol,5)
        altered=events.copy();altered.loc[altered.Record.eq(4),'Onset_s']=1
        numeric[4][1][:]=20
        b,_=select(train,scores,cache,altered,numeric,4,protocol,5)
        self.assertEqual(a,b)
        with self.assertRaises(ValueError):select(frame,np.ones(len(frame)),cache,events,numeric,4,protocol,5)

    def test_preservation_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'protected';p.write_text('original');snapshot={'protected':sha(p)}
            with patch('ml_research.baseline.ROOT',root):
                self.assertEqual(verify_snapshot(snapshot),1)
                p.write_text('changed')
                with self.assertRaises(ValueError):verify_snapshot(snapshot)

if __name__=='__main__':unittest.main()
