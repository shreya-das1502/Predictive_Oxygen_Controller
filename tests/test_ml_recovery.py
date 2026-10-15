"""Recovery integrity tests use synthetic tables, not patient evidence."""
import unittest
import numpy as np
import pandas as pd
from ml_research.recover import recovered_choices, validate_scores


class RecoveryTests(unittest.TestCase):
    def test_exact_threshold_and_unrelaxed_training_constraints(self):
        values=[.1,np.nextafter(.1,np.inf),.9]
        grid=pd.DataFrame(dict(Threshold=[0,0,0],Threshold_hex=[float(v).hex() for v in values],Detected=[1,1,9],Off_Event_Alarm_Starts=[0,0,50],Alarm_Active_s=[1.,1.,50.],False_Alarm_Starts_Per_Hour=[0.,0.,50.],Alarm_Active_Fraction=[.01,.01,.5]))
        got=recovered_choices(grid,dict(budgets=[1.,5.,10.],active_cap=.05))
        self.assertEqual([v['Threshold_hex'] for v in got],[float(values[1]).hex()]*3)
        with_metadata=grid.assign(Target='new_onset',Horizon_s=30,FeatureSet='base10',Model='ELM',ScoreSpace='raw',Outer_Test_Record=3)
        self.assertEqual(got,recovered_choices(with_metadata,dict(budgets=[1.,5.,10.],active_cap=.05)))
        with self.assertRaises(ValueError):recovered_choices(pd.concat([grid,grid.iloc[:1]]),dict(budgets=[1.],active_cap=.05))

    def test_recovered_scores_require_exact_row_identity(self):
        held=pd.DataFrame(dict(sample_id=[1,2],Record=[3,3],pulse_row=[4,5],time_s=[1.,2.]))
        scores=held.assign(raw=[.3,.4],negative_rank=[.5,.6])
        validate_scores(scores.iloc[::-1],held)
        with self.assertRaises(AssertionError):validate_scores(scores.assign(time_s=[1.,3.]),held)
        with self.assertRaises(ValueError):validate_scores(pd.concat([scores,scores.iloc[:1]]),held)
        with self.assertRaises(ValueError):validate_scores(scores.assign(raw=[np.nan,.4]),held)


if __name__=='__main__':unittest.main()
