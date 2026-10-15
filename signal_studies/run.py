"""Run isolated simulations: python -m signal_studies.run --study all."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from dataclasses import asdict
os.environ.setdefault('MPLCONFIGDIR', '/tmp/signal-studies-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.signal import sosfilt
from src.preprocessing import WaveformState
from .core import P, PI, PID, plant, loop, closed_loop, control_run, margins, frequency_response, settling_time

DEFAULT_OUTPUT = Path(__file__).resolve().parent/'outputs'
CONTROLLERS = {'P': P(), 'PI': PI(), 'PID': PID()}


def save(fig, output, name):
    fig.suptitle('SIMULATED · ILLUSTRATIVE · '+name.replace('_', ' '))
    fig.tight_layout()
    fig.savefig(output/f'{name}.png', dpi=150)
    plt.close(fig)


def synthetic_ppg(fs=125., duration=40., step_time=20.):
    """Deterministic periodic AC surrogate; offsets in normalized linear units."""
    t = np.arange(round(fs*duration))/fs
    phase = 2*np.pi*1.2*t
    clean = .3*np.sin(phase)+.09*np.sin(2*phase-.5)+.03*np.sin(3*phase-1)
    return t, clean, np.full_like(t, .4), .4+.6*(t >= step_time)


def waveform_study(output):
    t, clean, constant, step = synthetic_ppg()
    sos = WaveformState(fs=125).sos
    filtered_clean = sosfilt(sos, clean)
    rows = []
    for name, ambient in [('constant', constant), ('step', step)]:
        disturbed = clean+ambient
        recovered = sosfilt(sos, disturbed)  # Same coefficients and zero state as maintained pipeline.
        residual = recovered-filtered_clean
        event = 0. if name == 'constant' else 20.
        post = t >= event
        for label, start, end in [('steady_pre', 10., 20.), ('early_post', 20., 22.), ('steady_post', 30., 40.)]:
            mask = (t >= start)&(t < end)
            rows.append({'scenario': name, 'window': label, 'start_s': start, 'end_s_exclusive': end,
                         'raw_rmse': float(np.sqrt(np.mean((disturbed[mask]-clean[mask])**2))),
                         'reconstruction_rmse_unaligned': float(np.sqrt(np.mean((recovered[mask]-clean[mask])**2))),
                         'disturbance_only_rmse': float(np.sqrt(np.mean(residual[mask]**2))),
                         'residual_baseline_mean': float(np.mean(residual[mask])),
                         'residual_peak_abs': float(np.max(np.abs(residual[mask])))})
        amplitude = .4 if name == 'constant' else .6
        recovery = settling_time(t[post], residual[post], .02*amplitude)
        for row in rows:
            if row['scenario'] == name: row['recovery_2pct_s'] = recovery
        pd.DataFrame({'time_s': t, 'clean': clean, 'ambient': ambient, 'disturbed': disturbed,
                      'filtered': recovered, 'filtered_clean': filtered_clean,
                      'disturbance_only_residual': residual}).to_csv(output/f'waveform_{name}.csv', index=False)
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        axes[0].plot(t, clean, label='Clean AC PPG surrogate');axes[0].plot(t, disturbed, label='Disturbed', alpha=.65)
        axes[1].plot(t, clean, label='Clean target', alpha=.6);axes[1].plot(t, recovered, label='Causal filtered output')
        axes[2].plot(t, residual, label='Filtered disturbed − filtered clean')
        for ax in axes:
            ax.axvline(event, color='black', ls='--', label=f'Offset applied at {event:g} s')
            ax.set_ylabel('Normalized amplitude');ax.legend(loc='upper right');ax.grid(alpha=.3)
        axes[-1].set_xlabel('Time (s)')
        save(fig, output, f'waveform_{name}')
    pd.DataFrame(rows).to_csv(output/'waveform_metrics.csv', index=False)


def control_study(output):
    rows = []
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for name, controller in CONTROLLERS.items():
        t, r, d, nominal, delta, y, metrics = control_run(controller)
        rows.append({'controller': name, **asdict(controller), **metrics})
        pd.DataFrame({'time_s': t, 'reference': r, 'input_offset': d,
                      'undisturbed_output': nominal, 'disturbance_effect': delta,
                      'output': y}).to_csv(output/f'control_{name}.csv', index=False)
        axes[0].plot(t, y, label=name);axes[1].plot(t, nominal, label=name);axes[2].plot(t, delta, label=name)
    axes[0].plot(t, r, 'k--', label='Reference = 1')
    axes[0].set_ylabel('Combined output');axes[1].set_ylabel('Reference-only output');axes[2].set_ylabel('Offset-induced output')
    for ax in axes:
        ax.axvline(10, color='black', ls=':');ax.legend();ax.grid(alpha=.3)
    axes[-1].set_xlabel('Time (s); +0.3 input offset at 10 s')
    save(fig, output, 'control_comparison')
    pd.DataFrame(rows).to_csv(output/'control_metrics.csv', index=False)


def bode_study(output):
    rows, crossrows, polerows = [], [], []
    w = np.logspace(-3, 4, 5000)
    comparison, ca = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for name, controller in CONTROLLERS.items():
        num, den = loop(controller)
        result, gc, pc = margins(num, den)
        poles = np.roots(closed_loop(controller)[0][1])
        rows.append({'controller': name, **result, 'closed_loop_stable': bool(np.all(poles.real < 0)),
                     'largest_pole_real_part': float(np.max(poles.real)),
                     'open_loop_num': json.dumps(num.tolist()), 'open_loop_den': json.dumps(den.tolist())})
        for p in poles: polerows.append({'controller': name, 'real': p.real, 'imaginary': p.imag})
        for kind, frequencies in [('gain', gc), ('phase', pc)]:
            for frequency in frequencies: crossrows.append({'controller': name, 'kind': kind, 'frequency_rad_s': frequency})
        h = frequency_response(num, den, w)
        gain = 20*np.log10(abs(h));phase = np.degrees(np.unwrap(np.angle(h)))
        pd.DataFrame({'frequency_rad_s': w, 'gain_db': gain, 'phase_deg': phase}).to_csv(output/f'bode_{name}.csv', index=False)
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for axset in [axes, ca]:
            axset[0].semilogx(w, gain, label=name);axset[1].semilogx(w, phase, label=name)
        for frequency in gc:
            axes[0].plot(frequency, 0, 'ko');axes[1].axvline(frequency, color='gray', ls=':')
        for frequency in pc:
            axes[1].plot(frequency, -180, 'ks');axes[0].axvline(frequency, color='gray', ls=':')
        decorate_bode(axes)
        axes[0].set_title(f"{name}: GM={result['gain_margin_db']:.3g} dB; PM={result['phase_margin_deg']:.3g}°")
        save(fig, output, f'bode_{name}')
    decorate_bode(ca);save(comparison, output, 'bode_comparison')
    pd.DataFrame(rows).to_csv(output/'stability_margins.csv', index=False)
    pd.DataFrame(crossrows, columns=['controller', 'kind', 'frequency_rad_s']).to_csv(output/'crossovers.csv', index=False)
    pd.DataFrame(polerows).to_csv(output/'closed_loop_poles.csv', index=False)


def decorate_bode(axes):
    axes[0].axhline(0, color='black', ls='--');axes[1].axhline(-180, color='black', ls='--')
    axes[0].set_ylabel('Open-loop gain (dB)');axes[1].set_ylabel('Open-loop phase (deg)');axes[1].set_xlabel('Angular frequency (rad/s)')
    for ax in axes: ax.legend();ax.grid(which='both', alpha=.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=['waveform', 'control', 'bode', 'all'], default='all')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    root = Path(__file__).resolve().parents[1]
    # Prevent a mistaken output argument from overwriting protected research evidence.
    allowed = (root/'signal_studies').resolve()
    import tempfile
    temporary = Path(tempfile.gettempdir()).resolve()
    if not (output.is_relative_to(allowed) or output.is_relative_to(temporary)):
        parser.error('Output must be within signal_studies/ or the system temporary directory')
    output.mkdir(parents=True, exist_ok=True)
    for name, function in [('waveform', waveform_study), ('control', control_study), ('bode', bode_study)]:
        if args.study in [name, 'all']:
            function(output)
            metadata = {'study': name, 'python': platform.python_version(), 'numpy': np.__version__,
                        'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__,
                        'controllers': {k: asdict(v) for k,v in CONTROLLERS.items()},
                        'plant_num': plant()[0].tolist(), 'plant_den': plant()[1].tolist(),
                        'waveform': {'fs_hz': 125, 'duration_s': 40, 'fundamental_hz': 1.2,
                                     'harmonic_amplitudes': [.3, .09, .03], 'harmonic_phase_rad': [0, -.5, -1],
                                     'initial_offset': .4, 'step_increment': .6, 'step_time_s': 20,
                                     'filter': 'existing WaveformState SOS; zero initial state'},
                        'control': {'model': 'continuous-time; no sampling delay', 'output_grid_s': .002,
                                    'duration_s': 20, 'reference': 1, 'input_step': .3, 'step_time_s': 10,
                                    'initial_state': 'zero', 'settling_fraction': .02, 'minimum_hold_s': 1},
                        'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in [Path(__file__), Path(__file__).with_name('core.py'), root/'src/preprocessing.py']}}
            (output/f'{name}_settings.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Simulated illustrative outputs: {output}')


if __name__ == '__main__':
    main()
