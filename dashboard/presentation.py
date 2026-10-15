"""Plain-language, read-only presentation of existing replay state."""
import numpy as np

COLORS = {
    'ink': '#202A40', 'muted': '#515B70', 'page': '#F4F1FB',
    'paper': '#FFFFFF', 'soft': '#EDE8F5', 'purple': '#5B21B6',
    'border': '#79718C', 'warning': '#FFF4D6', 'warning_ink': '#664000',
    'danger': '#FCE8EE', 'danger_ink': '#8C1742',
    'info': '#EAF0FC', 'info_ink': '#243E6B',
    'disabled': '#E7E3ED', 'disabled_ink': '#555062',
    'pulse': '#9D265E', 'onset': '#875000', 'threshold': '#52617A',
}
TARGETS = {
    'new_onset': 'New oxygen drop to 94% or lower',
    'future_low': 'Any oxygen reading at 94% or lower (including ongoing lows)',
}
TARGET_HELP = ('Both models look 30 seconds ahead for a recorded SpO₂ reading ≤94%. '
               'A new-drop prediction requires a fresh current reading >94%. '
               'The other target includes oxygen that is already low. This is a study definition, not a treatment rule.')
SCORE_HELP = ('Model score and decision threshold are algorithmic values, not calibrated clinical probabilities. '
              'An experimental alert occurs when the score reaches the saved threshold. No threshold is adjusted here.')
MODES = ['Saved held-out final policy', 'Causal full-data demo model']
MODE_LABELS = {
    MODES[0]: 'Previously evaluated recording · saved final policy',
    MODES[1]: 'Experimental replay · full-data demonstration model',
}


def contrast(foreground, background):
    """WCAG relative-luminance ratio for solid sRGB colors (not a browser audit)."""
    def luminance(value):
        rgb = [int(value[i:i+2], 16)/255 for i in (1,3,5)]
        rgb = [v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4 for v in rgb]
        return sum(a*b for a,b in zip(rgb, (.2126,.7152,.0722)))
    a,b = sorted((luminance(foreground),luminance(background)))
    return (b+.05)/(a+.05)


def summary(engine, target):
    """No event matching or inference here; only label the available prefix."""
    current = engine.current_prediction()
    obs = engine.observation()
    valid = obs is not None and np.isfinite(obs.SpO2) and 0 < obs.SpO2 <= 100
    pulse = engine.pulses[-1] if engine.pulses and engine.time-engine.pulses[-1]['time_s'] <= 2 else None
    data = engine.visible_predictions()
    last_valid = None
    if not data.empty:
        decisions = data[data.eligible & data.score.notna()]
        if not decisions.empty:
            last_valid = float(decisions.iloc[-1].time_s)
    label, icon, tone = 'Prediction unavailable', '○', 'info'
    reason = 'No recent eligible model decision is available. Continue playback or check this target’s reading requirements; a missing decision is not a negative prediction.'
    if pulse is None:
        label, icon = 'Insufficient valid signal', '⊘'
        reason = 'No confirmed recent PPG pulse. Start playback or inspect missing signal; no decision can be inferred from absence.'
    if current is not None and bool(current.eligible):
        if bool(current.alert):
            label, icon, tone = 'Experimental alert triggered', '!', 'danger'
            reason = 'The last pulse score reached its saved cutoff. This is not a recommendation to give oxygen.'
        else:
            label, icon = 'No experimental alert at this time', '−'
            reason = 'The last pulse score was below its saved cutoff. This does not establish that the recording is safe.'
    elif current is not None:
        label, icon = 'Prediction unavailable', '○'
        reason = 'A new-drop prediction requires a valid current oxygen reading above 94%; low or missing readings do not meet this study rule.'
    if target == 'new_onset' and not valid:
        observed = engine.numeric[engine.numeric['Time [s]'] <= engine.time]
        stale = not observed.empty and engine.time-float(observed.iloc[-1]['Time [s]']) > 1.01
        label, icon, tone = ('Current reading is stale' if stale else 'Prediction unavailable'), '!', 'warning'
        reason = ('The latest oxygen reading is more than 1.01 seconds old.' if stale else 'A valid current oxygen reading is missing.') + ' The new-drop model cannot make a current decision.'
    if target == 'new_onset' and valid and obs.SpO2 <= 94:
        label, icon, tone = 'Prediction unavailable', '○', 'warning'
        reason = 'Current oxygen is already at or below 94%. This model evaluates new drops only when current oxygen is above 94%.'
    return dict(status=label, icon=icon, tone=tone, explanation=reason,
                spo2=f'{obs.SpO2:.0f}%' if valid else 'Unavailable',
                pulse=f'{pulse["Pulse_Rate_BPM"]:.0f} BPM' if pulse else 'Unavailable',
                last_valid=last_valid, decision=current, valid_spo2=valid)


def display_number(value, unit='', digits=2):
    if value is None or not np.isfinite(value):
        return 'Undefined' if unit == 's' else 'Unavailable'
    return f'{value:.{digits}f}' + (f' {unit}' if unit else '')
