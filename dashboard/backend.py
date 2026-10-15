"""Prefix-only replay and read-only artifact access, independent of the UI."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.controller import Controller
from src.preprocessing import WaveformState, canonical_waveform_times
from src.labeling import desaturation_events

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / 'results/alarm_budget_experiment/run02'
MODELS = {'ELM': 'ELM', 'Logistic Regression': 'Logistic', 'GaussianNB': 'GaussianNB', 'Decision Tree': 'Tree'}


def recordings(root=ROOT):
    return {r: all((root / f'data/bidmc_{r:02}_{kind}.csv').exists()
                   for kind in ('Signals', 'Numerics')) for r in range(1, 31)}


def load_record(record, root=ROOT):
    if record not in range(1, 31):
        raise ValueError('Choose a local recording from 01–30')
    wave = pd.read_csv(root / f'data/bidmc_{record:02}_Signals.csv')
    numeric = pd.read_csv(root / f'data/bidmc_{record:02}_Numerics.csv')
    wave.columns = wave.columns.str.strip()
    numeric.columns = numeric.columns.str.strip()
    wave['time_s'] = canonical_waveform_times(wave['Time [s]'])
    nt = numeric['Time [s]'].to_numpy(float)
    if not np.isfinite(nt).all() or (np.diff(nt) <= 0).any():
        raise ValueError('Numeric timestamps must be finite and strictly increasing')
    return wave, numeric


def policy_rows(predictions, record, model, target, budget):
    # Future labels and future minima are deliberately never passed to replay.
    g = predictions[(predictions.Record == record) & (predictions.Model == model)
                    & (predictions.Target == target) & (predictions.Budget == budget)]
    return g[['time_s', 'Score', 'Threshold', 'Prediction']].sort_values('time_s').reset_index(drop=True)


def result_rows(summary, target, budget, window=30):
    return summary[(summary.Target == target) & (summary.Budget == budget)
                   & (summary.Window_s == window) & (summary.Subset == 'all')].copy()


class Replay:
    """Each instance owns causal state. Backward seeks reset and replay the prefix.

    Saved-policy mode presents recorded held-out decisions, never recomputes them.
    Demo mode invokes the existing Controller with only samples available now.
    """
    def __init__(self, wave, numeric, saved=None, controller_factory=None):
        self.wave, self.numeric = wave, numeric
        self.saved = saved if saved is not None else pd.DataFrame(columns=['time_s', 'Score', 'Threshold', 'Prediction'])
        self.factory = controller_factory
        self.reset()

    def reset(self):
        self.controller = self.factory() if self.factory else None
        self.state = WaveformState()
        self.index = 0
        self.time = -np.inf
        self.pulses, self.outputs = [], []
        self.quality = 'Waiting for causal pulse history; SQI is not validated.'

    def observation(self, time=None):
        t = self.time if time is None else time
        index = np.searchsorted(self.numeric['Time [s]'].to_numpy(), t, side='right') - 1
        if index < 0:
            return None
        row = self.numeric.iloc[index]
        if t - row['Time [s]'] > 1.01:
            return None
        return row

    def seek(self, time):
        if not np.isfinite(time):
            raise ValueError('Playback time must be finite')
        if time < self.time:
            self.reset()
        times = self.wave.time_s.to_numpy()
        stop = np.searchsorted(times, time, side='right')
        for i in range(self.index, stop):
            t, value = float(times[i]), float(self.wave.PLETH.iloc[i])
            if not np.isfinite(value) or (self.state.last_time is not None and not np.isclose(t-self.state.last_time, .008, atol=1e-6)):
                self.state = WaveformState()
                self.controller = self.factory() if self.factory else None
                self.quality = 'Waveform discontinuity: causal state reset; no imputed waveform.'
                if not np.isfinite(value):
                    continue
            pulse = self.state.update(t, value)
            if pulse:
                self.pulses.append(pulse)
                if 'discontinuity' not in self.quality:
                    self.quality = 'Causal pulse confirmed; SQI is not validated.'
            if self.controller:
                obs = self.observation(t)
                out = self.controller.update_waveform(t, value,
                    spo2=float(obs.SpO2) if obs is not None else np.nan,
                    spo2_time_s=float(obs['Time [s]']) if obs is not None else np.nan)
                if out:
                    self.outputs.append(out)
        self.index, self.time = stop, float(time)
        return self

    def visible_predictions(self):
        if self.factory:
            return pd.DataFrame(self.outputs)
        g = self.saved[self.saved.time_s <= self.time].copy()
        return g.rename(columns={'Score': 'score', 'Threshold': 'threshold', 'Prediction': 'alert'}).assign(eligible=True)

    def observed_events(self):
        n = self.numeric[self.numeric['Time [s]'] <= self.time]
        return desaturation_events(n['Time [s]'], n.SpO2)

    def visible_wave(self, seconds=12):
        return self.wave.iloc[:self.index].loc[lambda x: x.time_s >= self.time-seconds]

    def current_prediction(self):
        p = self.visible_predictions()
        if p.empty or self.time - float(p.iloc[-1].time_s) > 2:
            return None
        return p.iloc[-1]


def demo_factory(target, model, root=ROOT):
    path = root / f'results/final/{target}/models/{model}.joblib'
    if not path.exists():
        raise FileNotFoundError(f'Missing trusted local demo model: {path}. No model will be fitted.')
    return lambda: Controller.load(path)


def verified_events(table, record, model, target, budget, cursor):
    # Wait until onset+30: all pre-onset prediction horizons can then be verified.
    # Return no future end, severity, minimum or other post-onset event information.
    g = table[(table.Record == record) & (table.Model == model) & (table.Target == target)
              & (table.Budget == budget) & (table.Window_s == 30) & (table.Onset_s + 30 <= cursor)]
    return g[['Onset_s', 'Evaluation_Status', 'Detected', 'First_Alert_s', 'Lead_Time_s']].copy()


def demo_alarm_starts(outputs):
    """Starts of observed demo decision runs, not audited exposure/event metrics."""
    starts = []
    previous_alert, previous_time = False, None
    for row in outputs:
        active = bool(row.get('eligible', False) and row.get('alert', False))
        if active and (not previous_alert or previous_time is None or row['time_s']-previous_time > 2):
            starts.append(row['time_s'])
        previous_alert, previous_time = active, row['time_s']
    return starts
