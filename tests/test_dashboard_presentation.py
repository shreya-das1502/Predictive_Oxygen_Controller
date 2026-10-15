"""Readable statuses, scientific empty values, and deliberate contrast pairs."""
import unittest
from pathlib import Path
from types import SimpleNamespace
import tomllib
import numpy as np
import pandas as pd
from dashboard.presentation import COLORS, contrast, summary, display_number

class PresentationTests(unittest.TestCase):
    def engine(self, decision=None, obs=98., old=False, pulse=True):
        row = None if decision is None else pd.Series(dict(time_s=3.,score=.2,threshold=.5,eligible=True,alert=decision))
        observations = pd.DataFrame({'Time [s]':[0. if old else 3.], 'SpO2':[obs]})
        return SimpleNamespace(time=3., numeric=observations,
            pulses=[dict(time_s=3.,Pulse_Rate_BPM=80.)] if pulse else [],
            current_prediction=lambda:row, observation=lambda: None if old else observations.iloc[-1],
            visible_predictions=lambda:pd.DataFrame([] if row is None else [row]))

    def test_explicit_statuses_not_clinical_normality(self):
        self.assertEqual(summary(self.engine(True),'new_onset')['status'],'Experimental alert triggered')
        below=summary(self.engine(False),'new_onset')
        self.assertEqual(below['status'],'No experimental alert at this time')
        self.assertIn('does not establish',below['explanation'])
        self.assertEqual(summary(self.engine(None,pulse=False),'new_onset')['status'],'Insufficient valid signal')
        self.assertEqual(summary(self.engine(None,obs=np.nan),'new_onset')['status'],'Prediction unavailable')
        self.assertEqual(summary(self.engine(None,old=True),'new_onset')['status'],'Current reading is stale')

    def test_missing_values_and_targets_remain_separate(self):
        low=summary(self.engine(False,obs=92.),'new_onset')
        self.assertEqual(low['status'],'Prediction unavailable')
        self.assertEqual(low['spo2'],'92%')
        future=summary(self.engine(False,obs=92.),'future_low')
        self.assertEqual(future['status'],'No experimental alert at this time')
        missing=summary(self.engine(None,obs=np.nan,pulse=False),'new_onset')
        self.assertEqual(missing['spo2'],'Unavailable')
        self.assertEqual(missing['pulse'],'Unavailable')
        self.assertIsNone(missing['last_valid'])
        self.assertEqual(display_number(np.nan,'s'),'Undefined')
        self.assertEqual(display_number(None),'Unavailable')

    def test_solid_text_pair_contrast_and_light_theme(self):
        for fg,bg in [('ink','paper'),('ink','page'),('ink','soft'),('muted','page'),('muted','paper'),
                      ('muted','soft'),('purple','paper'),('purple','soft'),('warning_ink','warning'),
                      ('danger_ink','danger'),('info_ink','info'),('disabled_ink','disabled'),
                      ('paper','purple'),('paper','ink')]:
            with self.subTest(foreground=fg,background=bg):
                self.assertGreaterEqual(contrast(COLORS[fg],COLORS[bg]),4.5)
        theme=tomllib.loads((Path(__file__).resolve().parents[1]/'.streamlit/config.toml').read_text())['theme']
        self.assertEqual(theme['base'],'light')
        self.assertEqual(theme['textColor'],COLORS['ink'])
        self.assertEqual(theme['primaryColor'],COLORS['purple'])
        self.assertEqual(theme['backgroundColor'],COLORS['page'])

    def test_meaningful_chart_and_border_contrast(self):
        for fg in ['purple','pulse','onset','threshold','border']:
            self.assertGreaterEqual(contrast(COLORS[fg],COLORS['paper']),3.)

if __name__=='__main__': unittest.main()
