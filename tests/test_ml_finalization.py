"""Synthetic bootstrap equivalence and immutable-input tests, not patient evidence."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from ml_research.finalize_saved import bootstrap_fast,prepare,COUNTS,event_panel
from src.local_30_evaluation import bootstrap_metrics,bootstrap_weights


class FinalizationTests(unittest.TestCase):
    def test_fast_recording_bootstrap_matches_original_repeated_sample_medians(self):
        g=pd.DataFrame({c:np.zeros(30) for c in COUNTS});g['Record']=np.arange(1,31)
        g['Exposure_s']=np.arange(1,31)*10.;g['Samples']=np.arange(1,31)*20
        g.loc[:2,'Eligible_Events']=[2,1,1];g.loc[:2,'Detected']=[2,0,1];g['Missed_Events']=g.Eligible_Events-g.Detected
        g.loc[0,'Alarm_Active_s']=1.;g.loc[4,'Off_Event_Alarm_Starts']=2
        w=bootstrap_weights(100,20261015);w[0]=0;w[0,29]=30
        for leads in [pd.DataFrame(dict(Record=[1,1,3],Lead_Time_s=[9.824,9.824,13.44])),pd.DataFrame(columns=['Record','Lead_Time_s'])]:
            actual=bootstrap_fast(g,leads,w);want=bootstrap_metrics(g,leads,w)
            for k in want:np.testing.assert_allclose(actual[k],want[k],rtol=0,atol=0,equal_nan=True,err_msg=k)
        with self.assertRaises(ValueError):bootstrap_fast(g.iloc[:-1],leads,w)

    def test_saved_input_cloning_preserves_source_and_rejects_mutation_or_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();(source/'scores').write_text('saved')
            output=root/'new';prepare(source,output);prepare(source,output)
            self.assertEqual((output/'scores').read_text(),'saved')
            (output/'scores').write_text('changed')
            with self.assertRaises(ValueError):prepare(source,output)
            collision=root/'collision';collision.mkdir();(collision/'scores').write_text('protected')
            with self.assertRaises(ValueError):prepare(source,collision)
            self.assertEqual((source/'scores').read_text(),'saved')
            with self.assertRaises(ValueError):prepare(source,source/'nested')

    def test_all_missing_leads_render_without_zero_imputation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'figures').mkdir()
            q=pd.DataFrame(dict(Target=['new_onset'],Subset=['all'],Budget=[5],Window_s=[30],Horizon_s=[30],FeatureSet=['base10'],Model=['ELM'],ScoreSpace=['raw'],Detected=[0],Lead_Median_s=[np.nan]))
            q.to_csv(root/'event_summary.csv',index=False);event_panel(root)
            self.assertTrue((root/'figures/event_counts_and_leads.png').is_file())
            self.assertTrue(pd.read_csv(root/'event_summary.csv').Lead_Median_s.isna().all())
            with self.assertRaises(ValueError):event_panel(root)


if __name__=='__main__':unittest.main()
