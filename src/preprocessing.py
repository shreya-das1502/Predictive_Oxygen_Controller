"""Causal waveform filtering and pulse confirmation; no retrospective peak search."""
import numpy as np
from scipy.signal import butter, sosfilt

class PulseDetector:
    """Local maximum confirmed by next sample; amplitude uses past trough only."""
    def __init__(self):
        self.previous = None
        self.before = None
        self.trough = np.inf
        self.last_peak = None
        self.last_amplitude = None
        self.rejected_intervals = 0
        self.candidates = 0

    def update(self, time_s, value):
        out = None
        if self.previous is not None and self.before is not None:
            pt, pv = self.previous
            if pv > self.before and pv >= value and pv - self.trough >= 0.03:
                if self.last_peak is None or pt - self.last_peak >= 0.30:
                    self.candidates += 1
                    amplitude = pv - self.trough
                    if self.last_peak is not None:
                        interval = pt - self.last_peak
                        if interval <= 2.0:
                            slope = 0.0 if self.last_amplitude is None else (amplitude-self.last_amplitude)/interval
                            out = {'time_s': time_s, 'peak_time_s': pt, 'PPG_Amplitude': amplitude,
                                   'Pulse_Interval_s': interval, 'Pulse_Rate_BPM': 60/interval,
                                   'Amplitude_Slope': slope}
                        else:
                            self.rejected_intervals += 1
                    self.last_peak, self.last_amplitude = pt, amplitude
                    self.trough = pv
        self.trough = min(self.trough, value)
        self.before = None if self.previous is None else self.previous[1]
        self.previous = (time_s, value)
        return out

class WaveformState:
    def __init__(self, fs=125.0):
        if not np.isfinite(fs) or fs <= 10:
            raise ValueError('Sampling rate must exceed 10 Hz')
        self.fs = fs
        self.sos = butter(3, [0.5, 5.0], btype='bandpass', fs=fs, output='sos')
        self.zi = np.zeros((len(self.sos), 2))
        self.detector = PulseDetector()
        self.last_time = None

    def update(self, time_s, ppg):
        if not np.isfinite(time_s) or not np.isfinite(ppg):
            raise ValueError('Waveform timestamp and PPG must be finite')
        if self.last_time is not None and not np.isclose(time_s-self.last_time, 1/self.fs, atol=1e-6, rtol=1e-4):
            raise ValueError('Waveform timestamps must increase at the configured sampling rate; reset after gaps')
        filtered, self.zi = sosfilt(self.sos, [ppg], zi=self.zi)
        self.last_time = time_s
        return self.detector.update(time_s, float(filtered[0]))


def extract_pulses(times, ppg):
    times, ppg = np.asarray(times, float), np.asarray(ppg, float)
    if len(times) != len(ppg) or len(times) < 2 or not np.isfinite(times).all() or not np.isfinite(ppg).all():
        raise ValueError('Waveform requires at least two finite aligned samples')
    dt = np.diff(times)
    if (dt <= 0).any() or not np.allclose(dt, np.median(dt), rtol=1e-4, atol=1e-6):
        raise ValueError('Duplicate, unordered or irregular waveform timestamps')
    state = WaveformState(1/np.median(dt))
    # Vectorized causal filtering is exactly the same recursion as streaming.
    filtered = sosfilt(state.sos, ppg)
    rows = []
    for t, v in zip(times, filtered):
        row = state.detector.update(float(t), float(v))
        if row is not None:
            rows.append(row)
    return rows, {'pulse_candidates': state.detector.candidates,
                  'invalid_interval_pulses': state.detector.rejected_intervals,
                  'initial_unpaired_pulses': min(state.detector.candidates, 1)}


def canonical_waveform_times(source_times, fs=125.0):
    """Recover acquisition clock from row index only if rounded export fits it.

    BIDMC CSV timestamps lose precision after 100s; samples are never deduplicated.
    Require all 125Hz row-clock residuals <= half a 0.01s export increment.
    Reject actual missing rows, reordered timestamps, and incompatible timing.
    """
    t = np.asarray(source_times, float)
    if len(t) < 2 or not np.isfinite(t).all():
        raise ValueError('Waveform timestamps require finite acquisition times')
    clock = t[0] + np.arange(len(t)) / fs
    if (np.diff(t) < 0).any() or np.max(np.abs(t-clock)) > 0.005001:
        raise ValueError('Waveform timestamps do not match the documented 125Hz rounded export clock')
    return clock
