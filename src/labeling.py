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
    return np.asarray(labels), np.asarray(mins), np.asarray(reasons, dtype=object)


def desaturation_events(numeric_times, spo2, threshold=SPO2_THRESHOLD, max_gap=MAX_NUMERIC_GAP):
    """Observed low episodes. Missing/invalid measurements and time gaps split them.

    A qualifying onset needs an immediately preceding observed normal measurement.
    Initial low episodes and lows following unknown measurements are left censored.
    """
    from .config import SEVERE_SPO2_VALUES
    nt, s = np.asarray(numeric_times, float), np.asarray(spo2, float)
    if len(nt) != len(s) or not np.isfinite(nt).all() or (np.diff(nt) <= 0).any():
        raise ValueError('Numeric timestamps must be finite, ordered and unique')
    if not len(nt):
        return []
    valid = np.isfinite(s) & (s > 0) & (s <= 100)
    low = valid & (s <= threshold)
    events = []
    for i in np.flatnonzero(low & np.r_[True, ~low[:-1] | (np.diff(nt) > max_gap)]):
        end = i + 1
        while end < len(nt) and low[end] and nt[end] - nt[end - 1] <= max_gap:
            end += 1
        qualifying = i > 0 and valid[i - 1] and s[i - 1] > threshold and nt[i] - nt[i - 1] <= max_gap
        events.append({'Event_ID': int(i), 'Onset_s': float(nt[i]), 'End_s': float(nt[end - 1]),
                       'Qualifying_Onset': bool(qualifying), 'Minimum_SpO2': float(s[i:end].min()),
                       'Has_91_92': bool(np.isin(s[i:end], SEVERE_SPO2_VALUES).any()),
                       'Status': 'qualifying' if qualifying else 'left_censored_or_unknown_prior'})
    return events


def onset_labels(times, numeric_times, spo2, horizon=HORIZON, threshold=SPO2_THRESHOLD, max_gap=MAX_NUMERIC_GAP):
    """Complete-window onset labels, with explicit current-normal eligibility."""
    times, nt, s = np.asarray(times, float), np.asarray(numeric_times, float), np.asarray(spo2, float)
    labels, _, reasons = future_labels(times, nt, s, horizon, threshold, max_gap)
    events = desaturation_events(nt, s, threshold, max_gap)
    onsets = np.asarray([e['Onset_s'] for e in events if e['Qualifying_Onset']])
    current = np.full(len(times), np.nan)
    for i, t in enumerate(times):
        j = np.searchsorted(nt, t, side='right') - 1
        if j < 0 or t - nt[j] > max_gap or not np.isfinite(s[j]) or not 0 < s[j] <= 100:
            labels[i] = np.nan
            reasons[i] = 'invalid_current_spo2'
        elif s[j] <= threshold:
            labels[i] = np.nan
            reasons[i] = 'current_spo2_not_above_threshold'
            current[i] = s[j]
        else:
            current[i] = s[j]
            if reasons[i] == 'verified':
                labels[i] = float(np.any((onsets > t) & (onsets <= t + horizon)))
    return labels, current, reasons
