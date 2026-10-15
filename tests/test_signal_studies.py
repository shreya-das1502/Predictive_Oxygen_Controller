"""Independent checks of illustrative simulations; never load research artifacts."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from scipy import signal
from signal_studies.core import (P, PI, PID, response_metrics, settling_time,
                                 closed_loop, control_run, margins, frequency_response, loop)
from signal_studies.run import synthetic_ppg


class SignalStudyTests(unittest.TestCase):
    def test_controller_error_step(self):
        t = np.linspace(0, 2, 1001); e = np.ones_like(t)
        np.testing.assert_allclose(P(2).response(t, e), 2)
        np.testing.assert_allclose(PI(2, 3).response(t, e), 2+3*t, atol=1e-11)
        # Filtered derivative of a unit step is (Kd/tau)*exp(-t/tau).
        c = PID(2, 3, .1, .02)
        np.testing.assert_allclose(c.response(t, e), 2+3*t+5*np.exp(-t/.02), atol=1e-10)
        np.testing.assert_array_equal(c.response(t, np.zeros_like(t)), 0)

    def test_known_first_order_metrics_and_never_settles(self):
        t = np.arange(0, 10, .001); y = 1-np.exp(-t)
        result = response_metrics(t, y)
        self.assertAlmostEqual(result['settling_2pct_s'], -np.log(.02), delta=.001)
        self.assertEqual(result['overshoot_pct'], 0)
        self.assertTrue(np.isnan(settling_time(t, np.full_like(t, .1), .02)))
        self.assertTrue(np.isnan(settling_time(t, np.where(t < 9.5, 1, 0), .02)))
        self.assertEqual(settling_time(t, np.zeros_like(t), .02), 0)
        self.assertAlmostEqual(response_metrics(t, np.full_like(t, 1.2))['overshoot_pct'], 20)

    def test_known_finite_infinite_and_undefined_margins(self):
        # L=1/[s(s+1)(s+2)] has -180deg at sqrt(2), |L|=1/6.
        result, gc, pc = margins([1.], [1., 3., 2., 0.])
        self.assertAlmostEqual(result['gain_margin'], 6)
        self.assertAlmostEqual(pc[0], np.sqrt(2))
        self.assertAlmostEqual(abs(frequency_response([1], [1,3,2,0], gc[0])), 1)
        result, _, _ = margins([1.], [1., 1., 0.])
        self.assertTrue(np.isinf(result['gain_margin']))
        self.assertAlmostEqual(result['phase_margin_deg'], 51.82729237298775)
        result, gc, pc = margins([.5], [1., 1.])
        self.assertTrue(np.isnan(result['phase_margin_deg']))
        self.assertTrue(np.isinf(result['gain_margin']))
        self.assertEqual(gc, []);self.assertEqual(pc, [])

    def test_dc_rejection_and_pole_stability(self):
        for c in [P(), PI(), PID()]:
            tracking, rejection = closed_loop(c)
            self.assertTrue(np.all(np.roots(tracking[1]).real < 0))
            t,r,d,nominal,delta,y,m = control_run(c)
            expected = 1/3 if type(c) is P else 0
            self.assertAlmostEqual(m['tail_error'], expected, places=9)
            self.assertAlmostEqual(m['disturbance_tail'], .3*expected, places=9)
            np.testing.assert_allclose(y, nominal+delta)
            self.assertEqual(d[0], 0);self.assertEqual(d[-1], .3)
            # Grid refinement bounds time-domain metric discretization error.
            refined = control_run(c, dt=.001)[-1]
            self.assertAlmostEqual(m['disturbance_peak'], refined['disturbance_peak'], delta=2e-5)
            if np.isfinite(m['settling_2pct_s']):
                self.assertAlmostEqual(m['settling_2pct_s'], refined['settling_2pct_s'], delta=.0021)
        # Stability check can detect an unstable illustrative tuning.
        self.assertTrue(np.any(np.roots(closed_loop(PI(2, 1000))[0][1]).real > 0))

    def test_synthetic_and_filter_rejection(self):
        from src.preprocessing import WaveformState
        a = synthetic_ppg(); b = synthetic_ppg()
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        t, clean, dc, step = a
        self.assertTrue(np.all(dc == .4));self.assertTrue(np.all(step[t < 20] == .4))
        self.assertTrue(np.all(step[t >= 20] == 1.))
        sos = WaveformState().sos
        residual = signal.sosfilt(sos, clean+step)-signal.sosfilt(sos, clean)
        self.assertLess(np.max(np.abs(residual[t >= 30])), 1e-6)
        self.assertGreater(np.max(np.abs(residual[(t >= 20)&(t < 22)])), .4)

    def test_full_artifact_reproducibility_and_output_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1')
            for name in ['a', 'b']:
                subprocess.run([sys.executable, '-m', 'signal_studies.run', '--output', str(root/name)],
                               env=env, check=True, capture_output=True)
            files = sorted((root/'a').glob('*'))
            self.assertEqual(len(list((root/'a').glob('*.png'))), 7)
            self.assertGreater(len(files), 20)
            for p in files:
                self.assertEqual(hashlib.sha256(p.read_bytes()).digest(),
                                 hashlib.sha256((root/'b'/p.name).read_bytes()).digest(), p.name)
            bad = subprocess.run([sys.executable, '-m', 'signal_studies.run', '--output', 'results/final'],
                                 env=env, capture_output=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn(b'Output must be within', bad.stderr)


if __name__ == '__main__':
    unittest.main()
