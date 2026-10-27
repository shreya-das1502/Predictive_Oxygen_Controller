import copy
import json
import unittest
import zipfile
import numpy as np
import pandas as pd
from experiments.pulse_quality_20261027T233919_IST.reference import match_pulses,validate_annotations,scored_pulses
from experiments.pulse_quality_20261027T233919_IST.quality import candidate_indicators
from experiments.pulse_quality_20261027T233919_IST.evaluate_reference import strict_schema,ratio,validated_reference_rows
from experiments.pulse_quality_20261027T233919_IST.build import HERE,PACKET,PRIVATE,sha,sample_mapping,packet_scan

class ReferenceTests(unittest.TestCase):
    def payload(self):return dict(packet_id='P1',reviewer_id='A',finalized=True,windows=[dict(window_id='W1',quality='adequately_interpretable',scored_regions=[[.08,19.92]],pulses=[dict(time_s=1.,certainty='clear')],notes='')])

    def test_one_to_one_count_and_signed_error(self):
        p=match_pulses([.99,1.02,2.04],[1.,2.]);self.assertEqual(len(p),2);self.assertEqual(len({i for i,j,e in p}),2);self.assertEqual(len({j for i,j,e in p}),2)
        self.assertAlmostEqual(p[0][2],-.01);self.assertAlmostEqual(p[1][2],.04)

    def test_maximum_cardinality_precedes_timing_cost(self):
        p=match_pulses([.06,.14],[0.,.08],.08);self.assertEqual(len(p),2)

    def test_exact_tolerance_and_sample_timing(self):
        self.assertEqual(len(match_pulses([1.08],[1.],.08)),1)
        self.assertEqual(len(match_pulses([1.088],[1.],.08)),0)

    def test_empty_or_duplicate_reference_not_perfect_accuracy(self):
        self.assertEqual(match_pulses([],[]),[]);self.assertTrue(np.isnan(ratio(0,0)))
        with self.assertRaises(ValueError):match_pulses([1.,1.],[1.])

    def test_draft_or_blank_annotations_are_not_ground_truth(self):
        p=self.payload();p['finalized']=False
        with self.assertRaises(ValueError):validate_annotations(p,pd.DataFrame({'WindowID':['W1']}),'P1')
        p=self.payload();p['windows'][0]['pulses']=[]
        with self.assertRaises(ValueError):validate_annotations(p,pd.DataFrame({'WindowID':['W1']}),'P1')

    def test_unknown_packet_window_or_metadata_rejected(self):
        p=self.payload();p['windows'][0]['window_id']='unknown'
        with self.assertRaises(ValueError):validate_annotations(p,pd.DataFrame({'WindowID':['W1']}),'P1')
        p=self.payload();p['windows'][0]['Event_ID']=99
        with self.assertRaises(ValueError):strict_schema(p)
        p=self.payload();p['reviewer_id']='../../escape'
        with self.assertRaises(ValueError):strict_schema(p)

    def test_partial_regions_and_ambiguity_are_not_forced(self):
        p=self.payload();r=p['windows'][0];r['quality']='partially_interpretable';r['scored_regions']=[[.1,3.]];r['pulses']=[dict(time_s=1.,certainty='clear'),dict(time_s=2.,certainty='ambiguous')]
        rows=validate_annotations(p,pd.DataFrame({'WindowID':['W1']}),'P1');d,ref=scored_pulses([1.01,2.01,4.],rows[0]);np.testing.assert_allclose(d,[1.01]);np.testing.assert_allclose(ref,[1.])

    def test_uninterpretable_has_no_scored_region(self):
        p=self.payload();p['windows'][0].update(quality='uninterpretable',scored_regions=[],pulses=[])
        rows=validate_annotations(p,pd.DataFrame({'WindowID':['W1']}),'P1');d,r=scored_pulses([1.],rows[0]);self.assertEqual(len(d),0);self.assertEqual(len(r),0)
        p['windows'][0]['scored_regions']=[[0,20]]
        with self.assertRaises(ValueError):validate_annotations(p,pd.DataFrame({'WindowID':['W1']}),'P1')

    def test_invalid_duplicate_or_overlapping_reference_handling(self):
        p=self.payload();p['windows'][0]['scored_regions']=[[1,5],[4,6]]
        with self.assertRaises(ValueError):validate_annotations(p,pd.DataFrame({'WindowID':['W1']}),'P1')

    def test_marks_outside_scoring_region_cannot_be_empty_ground_truth(self):
        p=self.payload();p['windows'][0]['scored_regions']=[[10,19]]
        with self.assertRaises(ValueError):validated_reference_rows(p,pd.DataFrame({'WindowID':['W1']}),'P1')
        p=self.payload();p['windows'][0]['pulses'].append(copy.deepcopy(p['windows'][0]['pulses'][0]))
        with self.assertRaises(ValueError):validate_annotations(p,pd.DataFrame({'WindowID':['W1']}),'P1')

class CandidateTests(unittest.TestCase):
    def test_fixed_detector_interval_flags(self):
        raw=np.sin(np.arange(2501)/125*2*np.pi);q=candidate_indicators(raw,raw,[1.,1.2,4.])
        self.assertEqual(q['Intervals_Below_030s'],1);self.assertEqual(q['Intervals_Above_2s'],1);self.assertFalse(q['Validated_SQI'])

    def test_flat_and_extreme_plateaus_are_candidates_not_clipping_truth(self):
        q=candidate_indicators(np.zeros(2501),np.zeros(2501),[])
        self.assertTrue(q['Flag_flat_run']);self.assertTrue(q['Flag_extreme_plateau']);self.assertEqual(q['Jump_Assessment'],'unavailable_zero_or_missing_MAD');self.assertFalse(q['Validated_SQI'])

    def test_feature_pure_ppg_has_no_outcome_argument(self):
        raw=np.sin(np.arange(2501)/125*2*np.pi);a=candidate_indicators(raw,raw,[1.,2.]);b=candidate_indicators(raw.copy(),raw.copy(),[1.,2.])
        self.assertEqual(a.keys(),b.keys())
        for k in a:
            if isinstance(a[k],float) and np.isnan(a[k]):self.assertTrue(np.isnan(b[k]))
            else:self.assertEqual(a[k],b[k])

class PacketTests(unittest.TestCase):
    def test_sampling_seed_and_all_recording_quarters(self):
        p=json.loads((HERE/'protocol.json').read_text());a=sample_mapping(p);b=sample_mapping(p);pd.testing.assert_frame_equal(a,b)
        random=a[a.SamplingOrigin.eq('time_stratified')];self.assertEqual(len(random),120);self.assertTrue(random.groupby('SourceRecord').size().eq(4).all());self.assertEqual(len(a),134)

    def test_reference_priority_subset_nonoverlapping(self):
        m=pd.read_csv(HERE/'window_review_manifest.csv');self.assertEqual(int(m.ReferenceSubset.sum()),60);self.assertTrue(m.groupby('AnonRecordID').ReferenceSubset.sum().eq(2).all())
        private=pd.read_csv(PRIVATE/'source_window_mapping.csv');d=m[m.ReferenceSubset].merge(private,on=['WindowID','AnonRecordID','ReviewOrder'])
        for _,g in d.groupby('AnonRecordID'):
            a,b=g.sort_values('StartSample').itertuples(index=False,name=None)
            h=g.sort_values('StartSample');self.assertLess(h.EndSample.iloc[0],h.StartSample.iloc[1])

    def test_reviewer_packet_no_outcome_fields_or_coordinator_files(self):
        packet_scan()
        with zipfile.ZipFile(HERE/'reviewer_packet.zip') as z:
            self.assertIsNone(z.testzip());names=z.namelist();self.assertFalse(any('coordinator' in n or 'mapping' in n or 'REPORT' in n for n in names))

    def test_automated_freeze_precedes_join_and_sources_match(self):
        freeze=json.loads((PRIVATE/'automated_review_freeze.json').read_text());self.assertEqual(freeze['Manual_Reviewed_Windows'],0);self.assertEqual(freeze['Window_Count'],134)
        for name,want in freeze['Frozen_Artifacts'].items():self.assertEqual(sha(HERE/name),want)
        for name,want in freeze['Source_SHA256'].items():self.assertEqual(sha(HERE/name),want)

    def test_no_unreferenced_detector_accuracy_or_feature_effect(self):
        self.assertFalse((HERE/'pulse_detection_metrics.csv').exists());self.assertFalse((HERE/'feature_impact_analysis.csv').exists())
        self.assertFalse((HERE/'manual_runs').exists())

if __name__=='__main__':unittest.main()
