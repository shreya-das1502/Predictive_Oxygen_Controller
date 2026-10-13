"""Shared causal feature generation for offline evaluation and streaming inference."""
from collections import deque
import numpy as np
import pandas as pd
from .config import BASE_FEATURES, FEATURES

class FeatureState:
    def __init__(self):
        self.history = deque(maxlen=10)

    def update(self, measurement):
        missing = set(BASE_FEATURES) - set(measurement)
        if missing:
            raise ValueError(f'Missing pulse features: {sorted(missing)}')
        values = np.array([measurement[c] for c in BASE_FEATURES], dtype=float)
        if not np.isfinite(values).any():
            raise ValueError('At least one observed pulse measurement is required')
        if np.isinf(values).any():
            raise ValueError('Infinite pulse features are invalid')
        amp, interval, rate, slope = values
        if (np.isfinite(amp) and amp < 0) or (np.isfinite(interval) and not 0.30 <= interval <= 2.0) or (np.isfinite(rate) and not 30 <= rate <= 200.01):
            raise ValueError('Pulse measurements outside supported ranges')
        if np.isfinite(interval) and np.isfinite(rate) and not np.isclose(rate, 60/interval, rtol=1e-4):
            raise ValueError('Pulse rate and interval are inconsistent')
        self.history.append(values)
        h = np.asarray(self.history)
        def mean(col):
            v = h[-5:, col]; v = v[np.isfinite(v)]
            return float(v.mean()) if len(v) else np.nan
        def std(col):
            v = h[:, col]; v = v[np.isfinite(v)]
            return float(v.std(ddof=1)) if len(v)>1 else 0.0
        change = rate-h[-6, 2] if len(h)>=6 else 0.0
        return np.r_[values, mean(2), std(2), change, mean(0), std(0), std(1)]


def add_features(data):
    """Require temporal ordering per group, retain every row, stable feature order."""
    required = {'Record', 'time_s', *BASE_FEATURES}
    if required-set(data):
        raise ValueError(f'Missing columns: {sorted(required-set(data))}')
    result = data.copy()
    matrix = np.empty((len(data), len(FEATURES)))
    for _, indices in data.groupby('Record', sort=False).indices.items():
        group = data.iloc[indices]
        times = group.time_s.to_numpy(float)
        if not np.isfinite(times).all() or (np.diff(times)<=0).any():
            raise ValueError('Invalid, duplicate or unordered pulse timestamps')
        state = FeatureState()
        for i, row in zip(indices, group.to_dict('records')):
            matrix[i] = state.update(row)
    result[FEATURES] = matrix
    return result
