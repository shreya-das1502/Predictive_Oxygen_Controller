"""Nested alarm-budget policy tests: synthetic fixtures are never experiment data."""
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from src.alarm_budget_experiment import (ROOT, validate_training, topology, candidate_statistics,
    select_partition, choose_candidate, detection_count, verify, evaluate_scoped_events)
from src.reliability_evaluation import raw_events, eligible_events, episodes, match_events, links
from src.local_30_evaluation import bootstrap_weights, verify_splits


def fixture():
    numeric={1:(np.arange(121.),np.full(121,98.)),2:(np.arange(121.),np.full(121,98.))}
    numeric[1][1][60]=94.
    rows=[]
    for r in [1,2]:
        for t in range(1,90):rows.append({'sample_id':len(rows),'Record':r,'time_s':float(t),'Label':int(r==1 and 30<=t<60)})
    data=pd.DataFrame(rows);oof=data.assign(Outer_Test_Record=3,Model='ELM')
    rng=np.random.default_rng(42);oof['Score']=rng.uniform(0,.5,len(data))
    oof.loc[oof.Record.eq(1)&oof.time_s.eq(58),'Score']=1.
    pos=np.sort(oof.loc[oof.Label.eq(1),'Score'])[::-1];oof['Threshold']=pos[int(np.ceil(.92*len(pos)))-1]
    events=raw_events(numeric)
    protocol=json.loads((ROOT/'protocol.json').read_text())
    return data,oof,events,numeric,protocol


class AlarmBudgetTests(unittest.TestCase):
    def test_actual_outer_test_rows_cannot_select_budget(self):
        d,o,e,n,p=fixture();o.loc[0,'Record']=3
        with self.assertRaises(ValueError):select_partition(d,o,e,n,{1,2},3,p)
    def test_wrong_outer_fit_oof_scores_rejected(self):
        d,o,e,n,p=fixture();o.Outer_Test_Record=4
        with self.assertRaises(ValueError):select_partition(d,o,e,n,{1,2},3,p)
    def test_missing_or_duplicate_oof_rows_rejected(self):
        d,o,e,n,p=fixture()
        for bad in [o.iloc[:-1],pd.concat([o,o.iloc[:1]])]:
            with self.assertRaises(ValueError):validate_training(d,bad,{1,2},3)
    def test_test_raw_data_or_events_rejected(self):
        d,o,e,n,p=fixture();n[3]=n[2]
        with self.assertRaises(ValueError):select_partition(d,o,e,n,{1,2},3,p)
        d,o,e,n,p=fixture();e.loc[0,'Record']=3
        with self.assertRaises(ValueError):select_partition(d,o,e,n,{1,2},3,p)
    def test_selector_has_no_outer_outcomes_and_repeats_exactly(self):
        d,o,e,n,p=fixture();a,grid=select_partition(d,o,e,n,{1,2},3,p)
        b,grid2=select_partition(d,o,e,n,{1,2},3,p)
        pd.testing.assert_frame_equal(a,b);pd.testing.assert_frame_equal(grid,grid2)
        self.assertTrue((a.False_Alarm_Starts_Per_Hour<=a.Budget).all())
        self.assertTrue(a.Alarm_Active_Fraction.le(p['active_fraction_cap']).all())
    def test_grouped_inner_partition_rejects_overlap(self):
        s=[{'test_record':1,'outer_train_records':[2,3],
            'inner':[{'train_records':[2],'validation_records':[2,3]}]}]
        with self.assertRaises(ValueError):verify_splits(s,[1,2,3])
    def test_no_training_alert_candidate_is_strictly_above_max(self):
        d,o,e,n,p=fixture();norm,prev,dur=topology(d.merge(o[['sample_id','Score']]),e,n,{1,2})
        grid=candidate_statistics(norm,prev,dur,e,eligible_events(e,norm,n,30),p['candidate_quantiles'])
        largest=grid.sort_values('Threshold').iloc[-1]
        self.assertGreater(largest.Threshold,norm.Score.max());self.assertTrue(largest.No_Training_Alerts)
        self.assertEqual(float.fromhex(largest.Threshold_hex),largest.Threshold)
    def test_nonmonotonic_false_alarm_counts_evaluated_explicitly(self):
        c=pd.DataFrame({'Threshold':[0.,1.,2.],'Detected':[0,3,2],'Off_Event_Alarm_Starts':[0,10,1],
                        'False_Alarm_Starts_Per_Hour':[0.,10.,1.],'Alarm_Active_s':[0.,1.,2.],'Alarm_Active_Fraction':[0.,.01,.02]})
        self.assertEqual(choose_candidate(c,5,.05)['Threshold'],2.)
    def test_active_cap_rejects_continuous_alert_with_low_start_rate(self):
        c=pd.DataFrame({'Threshold':[0.,1.],'Detected':[10,0],'Off_Event_Alarm_Starts':[0,0],
                        'False_Alarm_Starts_Per_Hour':[0.,0.],'Alarm_Active_s':[100.,0.],'Alarm_Active_Fraction':[1.,0.]})
        self.assertEqual(choose_candidate(c,5,.05)['Threshold'],1.)
    def test_ties_prefer_less_burden_activity_then_higher_threshold(self):
        c=pd.DataFrame({'Threshold':[0.,1.,2.],'Detected':[1,1,1],'Off_Event_Alarm_Starts':[1,0,0],
                        'False_Alarm_Starts_Per_Hour':[1.,0.,0.],'Alarm_Active_s':[1.,1.,1.],'Alarm_Active_Fraction':[.01]*3})
        self.assertEqual(choose_candidate(c,5,.05)['Threshold'],2.)
        with self.assertRaises(ValueError):choose_candidate(c,-1,.05)
    def test_array_matching_exactly_matches_audited_one_to_one_rule(self):
        ev=pd.DataFrame({'Record':[1,1,2],'Event_ID':[1,2,1],'Onset_s':[10.,12.,10.],'Evaluation_Status':['eligible']*3})
        r=np.array([1,1,2]);t=np.array([5.,11.,10.])
        alarms=[dict(Record=int(rr),Start_s=float(tt),Alarm_ID=i) for i,(rr,tt) in enumerate(zip(r,t))]
        m,_=match_events(ev,pd.DataFrame(),alarms,10,'episode_start_one_to_one')
        self.assertEqual(detection_count(r,t,ev,10),m.Detected.eq(True).sum());self.assertEqual(detection_count(r,t,ev,10),2)
    def test_record_boundaries_split_positive_runs(self):
        d,o,e,n,p=fixture();o.Score=1.;merged=d.merge(o[['sample_id','Score']])
        norm,prev,dur=topology(merged,e,n,{1,2});self.assertFalse(prev[0])
        self.assertFalse(prev[np.flatnonzero(norm.Record.to_numpy()==2)[0]])
        self.assertGreater(dur.sum(),0)
    def test_missing_numeric_and_hidden_low_edges_not_exposure(self):
        frame=pd.DataFrame({'Record':[1,1],'time_s':[1.5,3.1]})
        nt=np.arange(50.);s=np.full(50,98.)
        self.assertTrue(links(frame,{1:(nt,s)})[0])
        for bad in [94.,np.nan]:
            v=s.copy();v[2]=bad;self.assertFalse(links(frame,{1:(nt,v)})[0])
    def test_zero_event_record_retains_matching_schema(self):
        d,o,e,n,p=fixture();empty=e.iloc[:0]
        matched,used=evaluate_scoped_events(empty,d[d.Record.eq(2)],{2:n[2]},[],30)
        self.assertEqual(len(matched),0);self.assertEqual(used,set())
        self.assertIn('Evaluation_Status',matched);self.assertIn('Lead_Time_s',matched)
        self.assertEqual(len(matched[matched.Detected.eq(True)]),0)
    def test_fixed_seed_paired_record_bootstrap(self):
        a=bootstrap_weights(50,20261014);b=bootstrap_weights(50,20261014)
        np.testing.assert_array_equal(a,b);self.assertTrue((a.sum(axis=1)==30).all())


@unittest.skipUnless((ROOT/'run02/manifest.json').exists(),'Execute alarm-budget experiment for artifact checks')
class AlarmBudgetArtifactTests(unittest.TestCase):
    def test_inputs_outputs_and_preservation_verified(self):self.assertEqual(verify(ROOT/'run02')['status'],'passed')
    def test_thresholds_selected_before_outer_evaluation_are_unchanged(self):
        from src.reliability_evaluation import digest
        receipt=json.loads((ROOT/'run02/selection_complete.json').read_text())
        self.assertEqual(receipt['selected_thresholds_sha256'],digest(ROOT/'run02/selected_thresholds.csv'))
        self.assertEqual(receipt['protocol_sha256'],digest(ROOT/'protocol.json'))
    def test_saved_outer_decisions_are_exact_threshold_comparisons(self):
        p=pd.read_csv(ROOT/'run02/outer_predictions.csv.gz',float_precision='round_trip')
        np.testing.assert_array_equal(p.Prediction,(p.Score>=p.Threshold).astype(int))
    def test_saved_exposure_alarm_runs_and_false_starts_recompute(self):
        from src.reliability_evaluation import off_event_alarms
        p=pd.read_csv(ROOT/'run02/outer_predictions.csv.gz',float_precision='round_trip')
        per=pd.read_csv(ROOT/'run02/per_recording.csv')
        per=per[per.Window_s.eq(30)&per.Subset.eq('all')].set_index(['Target','Model','Budget','Record'])
        numeric={}
        for r in range(1,31):
            n=pd.read_csv(Path('data')/f'bidmc_{r:02d}_Numerics.csv');n.columns=n.columns.str.strip()
            numeric[r]=(n['Time [s]'].to_numpy(),n.SpO2.to_numpy())
        ev=raw_events(numeric)
        for key,g in p.groupby(['Target','Model','Budget','Record']):
            r=int(key[-1]);g=g.sort_values('time_s').reset_index(drop=True);edge=links(g,{r:numeric[r]})
            row=per.loc[key];dt=np.diff(g.time_s.to_numpy())
            self.assertAlmostEqual(float(dt[edge].sum()),row.Exposure_s,places=8)
            self.assertAlmostEqual(float((dt[edge]*g.Prediction.to_numpy()[:-1][edge]).sum()),row.Alarm_Active_s,places=8)
            alarms=[a|{'Record':r} for a in episodes(g,edge)]
            self.assertEqual(len(alarms),row.Alarm_Starts)
            self.assertEqual(off_event_alarms(alarms,ev[ev.Record.eq(r)],30),row.Off_Event_Alarm_Starts)
    def test_train_winners_follow_fixed_objective_without_test_metrics(self):
        selected=pd.read_csv(ROOT/'run02/selected_thresholds.csv');grid=pd.read_csv(ROOT/'run02/training_candidate_metrics.csv')
        for row in selected.itertuples():
            c=grid[(grid.Target==row.Target)&grid.Model.eq(row.Model)&grid.Outer_Test_Record.eq(row.Outer_Test_Record)]
            got=choose_candidate(c,row.Budget,.05)
            self.assertEqual(got['Threshold_hex'],row.Threshold_hex)
