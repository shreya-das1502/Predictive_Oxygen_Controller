import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.config import FEATURES,BASE_FEATURES,TARGET_RECALL
from src.dataset import validate_dataset
from src.labeling import future_labels
from src.features import FeatureState,add_features
from src.preprocessing import extract_pulses,WaveformState,canonical_waveform_times
from src.model import ELM,FittedModel
from src.controller import Controller
from src.evaluation import outer_splits,grouped_oof,choose_threshold,metrics,alarm_statistics
from src.benchmark_latency import timed


def sample(t=1,amp=.2,interval=.8):
    return {'time_s':t,'PPG_Amplitude':amp,'Pulse_Interval_s':interval,'Pulse_Rate_BPM':60/interval,'Amplitude_Slope':0.0}

def frame():
    rows=[]
    for r in [1,2,3]:
        for i in range(8):rows.append({'Record':r,'Label':i%2,**sample(i+1,.1*r+i*.01,.8+.01*i)})
    return pd.DataFrame(rows)

class LabelTests(unittest.TestCase):
    def test_strict_future_current_excluded(self):
        nt=np.arange(61);s=np.full(61,99.);s[0]=90
        y,_,_=future_labels([0],nt,s);self.assertEqual(y[0],0)
        s[30]=94;y,mins,_=future_labels([0],nt,s);self.assertEqual(y[0],1);self.assertEqual(mins[0],94)
        s[30]=99;s[31]=90;self.assertEqual(future_labels([0],nt,s)[0][0],0)
    def test_exact_time_horizon_irregular_rows(self):
        nt=np.arange(0,61,.5);s=np.full(len(nt),99.);s[nt==25]=94
        self.assertEqual(future_labels([0],nt,s)[0][0],1)
    def test_incomplete_tail_not_negative(self):
        y,_,reason=future_labels([40],np.arange(61),np.full(61,99))
        self.assertTrue(np.isnan(y[0]));self.assertEqual(reason[0],'incomplete_future_horizon')
    def test_missing_measurements_unknown(self):
        s=np.full(61,99.);s[10]=np.nan
        self.assertTrue(np.isnan(future_labels([0],np.arange(61),s)[0][0]))
    def test_gap_unknown(self):
        nt=np.r_[np.arange(10),np.arange(12,61)]
        self.assertTrue(np.isnan(future_labels([0],nt,np.full(len(nt),99))[0][0]))
    def test_invalid_numeric_timestamps(self):
        for times in [[0,1,1],[0,np.nan,2],[1,0,2]]:
            with self.assertRaises(ValueError):future_labels([0],times,[99]*3)
    def test_empty_sources(self):
        self.assertEqual(len(future_labels([],[],[])[0]),0)
        self.assertTrue(np.isnan(future_labels([0],[],[])[0][0]))
    def test_boundaries_independent(self):
        self.assertEqual(future_labels([0],np.arange(31),np.full(31,99))[0][0],0)
        self.assertEqual(future_labels([0],np.arange(31),np.full(31,90))[0][0],1)

class FeatureTests(unittest.TestCase):
    def test_schema_validation(self):
        with self.assertRaisesRegex(ValueError,'Missing columns'):validate_dataset(pd.DataFrame({'Label':[1]}))
        bad=frame();bad.loc[0,'Label']=2
        with self.assertRaises(ValueError):validate_dataset(bad)
    def test_rolling_no_boundary_cross(self):
        f=frame();out=add_features(f)
        for _,g in out.groupby('Record'):
            self.assertEqual(g.Rate_Std_10.iloc[0],0);self.assertEqual(g.Rate_Change_5.iloc[0],0)
            self.assertEqual(g.Rate_Mean_5.iloc[0],g.Pulse_Rate_BPM.iloc[0])
    def test_no_initial_rows_lost(self):
        f=frame();out=add_features(f);self.assertEqual(len(f),len(out));self.assertEqual(f.Label.sum(),out.Label.sum())
    def test_feature_causal_prefix_and_no_ground_truth(self):
        f=frame().iloc[:8].copy();a=add_features(f)
        b=add_features(f.iloc[:4]);np.testing.assert_allclose(a[FEATURES].iloc[:4],b[FEATURES])
        f.loc[4:,BASE_FEATURES[0]]=20;f['spo2']=1;f['Future_Min_SpO2_30s']=1;f.Label=1-f.Label
        c=add_features(f);np.testing.assert_allclose(a[FEATURES].iloc[:4],c[FEATURES].iloc[:4])
        self.assertFalse({'Label','spo2','Future_Min_SpO2_30s','Record','time_s'}&set(FEATURES))
    def test_streaming_feature_equivalence_order(self):
        f=frame().iloc[:8];a=add_features(f)[FEATURES].to_numpy();s=FeatureState()
        b=np.vstack([s.update(r) for r in f.to_dict('records')]);np.testing.assert_allclose(a,b)
        self.assertEqual(len(s.update(sample(10))),len(FEATURES))
    def test_missing_features_retained(self):
        f=frame();f.loc[0,'PPG_Amplitude']=np.nan
        self.assertEqual(len(add_features(f)),len(f));validate_dataset(f)
    def test_bad_timestamps_rejected(self):
        f=frame();f.loc[1,'time_s']=f.loc[0,'time_s']
        with self.assertRaises(ValueError):add_features(f)
    def test_empty_data(self):
        f=frame().iloc[:0];self.assertTrue(add_features(f).empty);self.assertEqual(validate_dataset(f)['rows'],0)
    def test_waveform_streaming_and_prefix(self):
        t=np.arange(2000)/125;x=.3*np.sin(2*np.pi*1.2*t)+.4
        rows,_=extract_pulses(t,x);prefix,_=extract_pulses(t[:900],x[:900])
        want=[r for r in rows if r['time_s']<=t[899]];self.assertEqual(prefix,want)
        state=WaveformState();stream=[]
        for time,value in zip(t,x):
            row=state.update(float(time),float(value))
            if row is not None:stream.append(row)
        self.assertEqual(len(stream),len(rows))
        for a,b in zip(rows,stream):np.testing.assert_allclose(list(a.values()),list(b.values()),rtol=1e-10,atol=1e-10)
        self.assertTrue(all(r['time_s']>r['peak_time_s'] for r in rows))
    def test_rounded_clock_recovers_without_deletion(self):
        t=np.arange(60001)/125;rounded=np.round(t,2);clock=canonical_waveform_times(rounded)
        np.testing.assert_allclose(clock,t);self.assertEqual(len(clock),len(rounded))
        bad=rounded.copy();bad[1000:]+=.1
        with self.assertRaises(ValueError):canonical_waveform_times(bad)
    def test_invalid_waveform_input(self):
        s=WaveformState()
        with self.assertRaises(ValueError):s.update(np.nan,.3)
        s.update(0,.3)
        with self.assertRaises(ValueError):s.update(0,.3)

class ModelTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(3);self.X=rng.normal(size=(90,len(FEATURES)));self.y=(self.X[:,0]>0).astype(int)
    def test_elm_shape_finite_reproducible(self):
        a=ELM().fit(self.X,self.y);b=ELM().fit(self.X,self.y)
        self.assertEqual(a.score(self.X).shape,(90,));self.assertTrue(np.isfinite(a.score(self.X)).all())
        np.testing.assert_allclose(a.score(self.X),b.score(self.X),atol=0,rtol=0)
    def test_weighted_ridge_solution(self):
        m=ELM(hidden_nodes=5,ridge=2,positive_weight=3).fit(self.X,self.y)
        H=np.tanh(self.X@m.weights+m.bias);w=np.where(self.y==1,3,1)
        expected=np.linalg.solve(H.T@(w[:,None]*H)+2*np.eye(5),H.T@(w*self.y))
        np.testing.assert_allclose(m.beta,expected,rtol=1e-10)
    def test_ill_conditioned_features(self):
        x=np.ones((90,10));x[:,1]=1+1e-10*self.X[:,1]
        m=ELM(ridge=.01).fit(x,self.y);self.assertTrue(np.isfinite(m.score(x)).all())
    def test_training_only_transformations(self):
        m=FittedModel('Logistic').fit(self.X,self.y);mean=m.scaler.mean_.copy();m.score(np.full((1,10),10000))
        np.testing.assert_allclose(m.scaler.mean_,mean);np.testing.assert_allclose(mean,self.X.mean(axis=0))
    def test_missing_imputation_train_only(self):
        x=self.X.copy();x[0,0]=np.nan;m=FittedModel('ELM').fit(x,self.y)
        self.assertEqual(m.imputer.statistics_[0],np.nanmedian(x[:,0]));self.assertTrue(np.isfinite(m.score(x)).all())
    def test_persist_clean_python_process(self):
        m=FittedModel('ELM').fit(self.X,self.y);m.threshold=.5
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'model.joblib';m.save(path,{'training_records':[1,2]});np.save(Path(td)/'x.npy',self.X)
            code='from src.model import FittedModel; import numpy as np,sys; np.save(sys.argv[3],FittedModel.load(sys.argv[1]).score(np.load(sys.argv[2])))'
            subprocess.run([sys.executable,'-c',code,str(path),str(Path(td)/'x.npy'),str(Path(td)/'score.npy')],check=True)
            np.testing.assert_allclose(m.score(self.X),np.load(Path(td)/'score.npy'),rtol=1e-12)

class ValidationTests(unittest.TestCase):
    def test_outer_disjoint_and_complete(self):
        groups=np.repeat(np.arange(6),10);seen=[]
        for tr,te in outer_splits(groups):self.assertFalse(set(groups[tr])&set(groups[te]));seen+=te.tolist()
        self.assertEqual(sorted(seen),list(range(len(groups))))
    def test_oof_disjoint_and_complete(self):
        f=add_features(frame());x=f[FEATURES].to_numpy();y=f.Label.to_numpy();g=f.Record.to_numpy()
        threshold,scores,table,folds=grouped_oof(x,y,g,'ELM')
        self.assertTrue(np.isfinite(scores).all());self.assertGreaterEqual((scores[y==1]>=threshold).mean(),TARGET_RECALL)
        for fold in folds:self.assertFalse(set(fold['train_records'])&set(fold['validation_records']))
    def test_outer_labels_do_not_affect_threshold(self):
        f=add_features(frame());tr,te=next(outer_splits(f.Record));train=f.iloc[tr]
        a=grouped_oof(train[FEATURES],train.Label,train.Record,'Logistic')[0]
        f.loc[te,'Label']=1-f.loc[te,'Label'];train=f.iloc[tr]
        b=grouped_oof(train[FEATURES],train.Label,train.Record,'Logistic')[0];self.assertEqual(a,b)
    def test_threshold_ties_and_prespecified_goal(self):
        y=np.array([1,1,0,0,1]);s=np.array([.8,.5,.5,.2,.1]);threshold,tab=choose_threshold(y,s)
        self.assertEqual(threshold,.1);self.assertGreaterEqual((s[y==1]>=threshold).mean(),.92)
        with self.assertRaisesRegex(ValueError,'No validation positives'):choose_threshold([0,0],[.2,.3])
    def test_metrics_undefined_not_zero(self):
        m=metrics([0,0],[.1,.2],[0,0]);self.assertTrue(np.isnan(m['Recall']));self.assertTrue(np.isnan(m['Precision']))
        m=metrics([1,1],[.1,.2],[1,1]);self.assertTrue(np.isnan(m['Specificity']))
    def test_alarm_episode_not_timestamp(self):
        g=pd.DataFrame({'time_s':[0,1,2,3,4],'Prediction':[1,1,0,1,1],'Label':[0,0,0,0,1]})
        a=alarm_statistics(g);self.assertEqual(a['Alert_Episodes'],2);self.assertEqual(a['False_Alert_Episodes'],1)

class InferenceTests(unittest.TestCase):
    def setUp(self):
        f=add_features(frame());self.model=FittedModel('ELM').fit(f[FEATURES],f.Label);self.model.threshold=.5
    def test_single_new_sample_deterministic_replay(self):
        a=Controller(self.model);b=Controller(self.model)
        for i in range(1,9):self.assertEqual(a.predict(sample(i)),b.predict(sample(i)))
    def test_explicit_invalid_inputs(self):
        c=Controller(self.model)
        with self.assertRaises(ValueError):c.predict({'time_s':1})
        with self.assertRaises(ValueError):c.predict(sample(1,interval=.1))
        with self.assertRaisesRegex(ValueError,'observed'):c.predict({'time_s':1,**{k:np.nan for k in BASE_FEATURES}})
        c.predict(sample(1))
        with self.assertRaises(ValueError):c.predict(sample(1))
    def test_missing_policy_exposed(self):
        c=Controller(self.model);m=sample(1);m['PPG_Amplitude']=np.nan
        p=c.predict(m);self.assertGreater(p['missing_features'],0);self.assertTrue(np.isfinite(p['score']))
    def test_benchmark_does_not_fit(self):
        c=Controller(self.model);i=[0]
        def predict():i[0]+=1;return c.predict(sample(i[0]))
        with patch.object(FittedModel,'fit',side_effect=AssertionError('Training in benchmark')):
            r=timed(predict,warmup=2,repetitions=20)
        self.assertEqual(r['Repetitions'],20);self.assertEqual(i[0],22);self.assertGreater(r['Median_ms'],0)

if __name__=='__main__':
    with threadpool_limits(limits=1):unittest.main()
