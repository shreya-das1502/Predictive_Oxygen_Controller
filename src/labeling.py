"""Strict future labels with complete observed windows; unavailable target is NaN."""
import numpy as np
from .config import HORIZON, SPO2_THRESHOLD, MAX_NUMERIC_GAP


def future_labels(times, numeric_times, spo2, horizon=HORIZON, threshold=SPO2_THRESHOLD, max_gap=MAX_NUMERIC_GAP):
    t, nt, s = np.asarray(times,float), np.asarray(numeric_times,float), np.asarray(spo2,float)
    if horizon <= 0 or not np.isfinite(horizon):
        raise ValueError('Horizon must be positive and finite')
    if len(nt)!=len(s) or not np.isfinite(nt).all() or not np.isfinite(t).all() or (np.diff(nt)<=0).any():
        raise ValueError('Numeric timestamps must be finite, strictly ordered and unique')
    labels, mins, reasons = [], [], []
    for x in t:
        a, b = np.searchsorted(nt, [x, x+horizon], side='right')
        reason = 'verified'
        if not len(nt) or x+horizon > nt[-1] or x < nt[0] or a==b:
            reason = 'incomplete_future_horizon'
        elif np.max(np.diff(np.r_[x, nt[a:b], x+horizon])) > max_gap:
            reason = 'future_measurement_gap'
        elif not np.isfinite(s[a:b]).all() or ((s[a:b] <= 0) | (s[a:b] > 100)).any():
            reason = 'invalid_future_spo2'
        minimum = float(np.min(s[a:b])) if reason=='verified' else np.nan
        mins.append(minimum); labels.append(float(minimum<=threshold) if reason=='verified' else np.nan); reasons.append(reason)
    return np.asarray(labels), np.asarray(mins), np.asarray(reasons)
