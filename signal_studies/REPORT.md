# Simulated, illustrative signal-conditioning studies

These studies are separate from the recorded BIDMC ML research pipeline. They neither use patient data nor control oxygen. The signal-conditioning plant is an illustrative model, not an experimentally identified device. Results do not establish clinical validity, sensor validation, or deployable controller tuning. No dashboard integration is added; the outputs remain explicitly separate from recordings and held-out evaluation.

## Reproduction and files

From the repository root, use the existing pinned `requirements.txt` environment:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -m signal_studies.run --study waveform
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -m signal_studies.run --study control
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -m signal_studies.run --study bode
# Equivalent complete run:
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -m signal_studies.run --study all
# Optional isolated destination (must be under signal_studies/ or system temporary directory):
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -m signal_studies.run --study all --output /tmp/oxygen-signal-studies
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -m unittest tests.test_signal_studies -v
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/signal-studies-matplotlib python -m unittest discover -s tests -v
```

Default output: `signal_studies/outputs/`. Commands replace only their own named study outputs. `core.py` contains controllers, transfer functions, metrics and crossover calculations; `run.py` contains waveform generation, study runners, plots and CLI; `__init__.py` declares the isolated package. `tests/test_signal_studies.py` checks analytical controller responses, known margins, known settling behavior, steady-state limits, pole stability, time-grid convergence, deterministic artifact regeneration and output-path protection. No new dependencies are required.

Each study writes a settings JSON with parameters, numerical-library versions and source hashes. These are new simulation receipts, unrelated to existing research integrity hashes. Plots are PNG; tables are CSV. Empty numeric cells mean undefined/not established, `inf` means mathematically infinite. Unused controller gains are empty. Floating-point residuals near 1e-14 represent numerical roundoff, not measured accuracy.

## Study 1: PPG surrogate with additive illumination coupling

A deterministic 40 s waveform at 125 Hz is

`p(t) = 0.30 sin(phi) + 0.09 sin(2 phi − 0.5) + 0.03 sin(3 phi − 1)`, `phi = 2 pi × 1.2 t`.

This zero-mean three-harmonic AC surrogate has a 72 beats/min fundamental. It approximates periodic pulse morphology without claiming physiological fidelity. No RNG or seed is needed. Two separate experiments use `x=p+a`: (a) constant `a=0.4` applied at startup and (b) `a=0.4+0.6 H(t−20)`, an explicitly time-varying illumination offset. The assumed linear, unit coupling maps illumination to normalized electrical offset. The second test includes an abrupt illumination change; the constant-only experiment is a DC rejection baseline, not a complete ambient-light simulation.

Both use the existing `src.preprocessing.WaveformState().sos` coefficients: forward-only third-order 0.5–5 Hz Butterworth bandpass, zero initial state, applied with `scipy.signal.sosfilt`. No filter retuning or alternative is needed for this specific offset-rejection question. Its phase/amplitude distortion is explicitly retained and measured. The clean reference is also filtered in an independent zero-state run to isolate disturbance effects. This matched reference is available only because the input is synthetic; it is not an estimator available to a real sensor.

For each interval `[start,end)`, the exported metrics are:

- Raw RMSE: `sqrt(mean((x−p)^2))`.
- Unaligned reconstruction RMSE: `sqrt(mean((filtered(x)−p)^2))`. No delay alignment or retrospective fitting hides phase distortion.
- Disturbance-only RMSE: `sqrt(mean(q^2))`, where `q=filtered(x)−filtered(p)`.
- Residual baseline: `mean(q)`; residual peak: `max(abs(q))`. Reporting both avoids cancellation of oscillating residuals in the mean.
- Recovery: first sample after the last `abs(q)>0.02×offset_increment`, with at least one second remaining in the observation window. Measured from startup for the constant test and from 20 s for the step test. The threshold is 0.008 or 0.012 normalized units respectively. This is finite-record evidence of staying inside the band, not an infinite-time proof.

Measured results from the supplied command:

| Scenario/window | Raw RMSE | Unaligned reconstruction RMSE | Disturbance-only RMSE | Residual baseline mean |
|---|---:|---:|---:|---:|
| Constant, 10–20 s | 0.4000 | 0.095914 | 6.80e-8 | 1.01e-8 |
| Step, 20–22 s | 1.0000 | 0.163537 | 0.130842 | 0.003303 |
| Step, 30–40 s | 1.0000 | 0.095914 | 1.02e-7 | 1.51e-8 |

Recovery is **2.816 s** for both offset applications (linearity and identical fractional thresholds explain the equality). The step transient peaks at **0.416750** in the first two seconds. Offset rejection recovers; the unaligned waveform still has about 0.096 RMSE from filtering itself. This is not perfect clean-waveform reconstruction.

Outputs: `waveform_constant.png`, `waveform_step.png`, their sample-level CSVs, `waveform_metrics.csv`, and `waveform_settings.json`. The plots show clean, disturbed and filtered traces, the matched residual, and the offset application time.

## Study 2: illustrative conditioning-amplifier feedback

Consider two cascaded first-order stages with unity DC gain:

`0.2 dx1/dt + x1 = u + d`, `0.05 dy/dt + y = x1`.

Here `u` is a signed electronic compensation drive, `y` is a normalized conditioning output level, and `d` is an additive electrical offset representing assumed ambient coupling at the amplifier input. Thus

`G(s)=Y(s)/(U(s)+D(s))=1/[(0.2s+1)(0.05s+1)]=1/(0.01s²+0.25s+1)`.

The two stable lags (200 ms and 50 ms) are a plausible abstract bandwidth-limited conditioning chain, **chosen for illustration**, not measured hardware constants. Negative unity feedback uses `e=r−y`. This experiment regulates a scalar conditioning level; it is not a model of an oxygen actuator or a validated pulse-preserving optical cancellation loop. It is not cascaded with Study 1's bandpass filter. Study 1 and Study 2 address different, explicitly defined questions.

| Controller | Transfer function | Gains |
|---|---|---|
| P | `C=Kp` | Kp=2 |
| PI | `C=Kp+Ki/s` | Kp=2, Ki=8 s⁻¹ |
| PID | `C=Kp+Ki/s+Kd s/(1+tau_d s)` | Kp=2, Ki=8 s⁻¹, Kd=0.08 s, tau_d=0.01 s |

The PID derivative is proper and filtered, and acts on error including the reference step. Gains are fixed illustrative settings; proportional gain is shared and integral gain is shared where present. This is a controlled structural comparison, not independently optimized equal-bandwidth/equal-effort tuning or a claim of controller superiority. There is no actuator limit, saturation, anti-windup, noise, quantization, dead time or digital computation delay.

All controller/plant states start at zero. Reference steps from 0 to 1 at t=0; the same input disturbance `d=0.3 H(t−10)` is applied to each. Simulation lasts 20 s with a 0.002 s output grid. Controllers and plant are **continuous-time**, not discretized implementations. `scipy.signal.lsim(..., interp=False)` uses held input steps and state-space propagation; the output grid does not introduce a sampled feedback delay. See the [SciPy lsim documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lsim.html). A test halves the output grid to 0.001 s and checks metric convergence.

By superposition, `T=GC/(1+GC)` maps the reference to output and `H_d=G/(1+GC)` maps the input offset to output. Both responses are simulated from zero state; their sum is the combined response. This preserves identical initial conditions and separates reference tracking from disturbance rejection.

Reference metrics use samples before 10 s: settling requires error to the **requested reference** within ±0.02 and at least a one-second final hold; overshoot is `100 max(0,max(y)−1)`. Steady-state error is approximated by `1−mean(y)` over the final second of that segment and independently checked against analytical DC limits. Separate columns report settling/overshoot relative to each controller's own analytical equilibrium, to avoid concealing P's tracking error.

Disturbance metrics use the difference from the undisturbed response: peak absolute deviation, signed final-one-second mean, integral absolute deviation on [10,20] s, and recovery inside ±0.006 (2% of the input disturbance) measured from 10 s. Tail rejection is `100(1−abs(mean(delta_y)) / 0.3)`, relative to the open-loop unity-DC-gain disturbance. Post-step total reference error is also exported. A missing settling time means the criterion was not established during the observation window.

| Controller | Reference settling (s) | Overshoot vs reference (%) | Reference tail error | Disturbance peak | Disturbance recovery (s) | Disturbance tail |
|---|---:|---:|---:|---:|---:|---:|
| P | Not attained | 0 | 0.333333 | 0.103780 | Not attained | 0.100000 |
| PI | 0.572 | approximately 0 | approximately 0 | 0.089701 | 0.964 | approximately 0 |
| PID | 0.502 | 0.004413 | approximately 0 | 0.079892 | 0.930 | approximately 0 |

P settles to its own equilibrium 2/3 in 0.342 s, with 3.7804% overshoot relative to that equilibrium, while never satisfying the reference band. Analytical P errors are 1/3 before the disturbance and 0.233333 after it. PI/PID have zero DC tracking error and zero DC input-disturbance transmission under this ideal model. Integrated disturbance errors are 0.991667, 0.037500 and 0.037500 amplitude-seconds for P, PI and PID; the corresponding tail rejection is 66.6667%, approximately 100%, approximately 100%.

Outputs: `control_comparison.png`, `control_P.csv`, `control_PI.csv`, `control_PID.csv`, `control_metrics.csv`, `control_settings.json`.

## Study 3: open-loop Bode plots and margins

The negative-feedback loop transfer is `L(s)=C(s)G(s)`. Explicitly:

- P: `2/(0.01s²+0.25s+1)`.
- PI: `(2s+8)/(0.01s³+0.25s²+s)`.
- PID: `(0.1s²+2.08s+8)/(0.0001s⁴+0.0125s³+0.26s²+s)`.

Bode plot frequencies cover 1e-3 to 1e4 rad/s (not Hz). Margins are calculated independently of the plotting grid: positive real roots of `|N(jw)|²−|D(jw)|²=0` give unity-gain crossovers, and roots of `Im(N(jw) conj(D(jw)))=0` with negative real loop response give −180-degree crossovers. Continuous phase is calculated from the individual numerator/denominator factors. This routine supports the real, proper, minimum-phase loop family used here; it is not presented as a general delay-system margin library. All finite positive crossovers are exported. If several occur, the summary chooses the phase margin nearest zero and the gain margin nearest unity in logarithmic distance.

Phase margin is `180°+arg(L(jw_gc))`; gain margin is `1/abs(L(jw_pc))`, also reported as `20 log10(GM)`. No unity-gain crossover gives undefined phase margin/frequency. No finite negative-real-axis crossover gives infinite gain margin with undefined phase crossover. CSV empty cells and `inf` preserve these distinctions. Tests cover a system with finite GM=6 at sqrt(2) rad/s, a known phase margin, and absent crossovers.

| Controller | Gain crossover (rad/s) | Phase margin (degrees) | Gain margin | Phase crossover |
|---|---:|---:|---|---|
| P | 7.851522 | 101.055980 | Infinite | No finite crossing |
| PI | 8.746533 | 71.557865 | Infinite | No finite crossing |
| PID | 8.211821 | 90.111616 | Infinite | No finite crossing |

These infinite gain margins are properties of the chosen ideal, delay-free model, not assurances of hardware robustness. Limiting approach to −180° at infinite frequency is not a finite crossover. The independently calculated closed-loop characteristic roots of `D_G D_C+N_G N_C` all have negative real parts. Largest real parts are −12.5, −3.583308, and −5.325032 s⁻¹ for P, PI, PID. Full poles are exported without cancellation-based stability shortcuts.

Outputs: separate `bode_P.png`, `bode_PI.png`, `bode_PID.png`, `bode_comparison.png`, corresponding per-controller frequency CSVs, `stability_margins.csv`, `crossovers.csv`, `closed_loop_poles.csv`, `bode_settings.json`.

## Assumptions requiring confirmation before interpretation beyond illustration

Actual optical-to-electrical coupling, illumination spectrum and time variation, sensor transfer function and operating point, stage time constants and gain, actuator sign/range, saturation and windup, sensor noise and shot noise, motion, LED modulation/demodulation, ADC behavior, feedback delay and digital implementation must be established experimentally. The additive linear disturbance excludes nonlinear optical effects and multiplicative pulse attenuation. The periodic surrogate excludes physiological variability. No conclusions about preserved pulse morphology in a real compensation loop follow from the scalar level-control model. Controller tuning should be revisited only after identification, constraints and intended bandwidth are confirmed.

## Executed validation and preservation

The six new tests passed independently. The complete suite ran **149 tests in 125.913 s: 146 passed, one failed and two errored**, with no skips. All six new tests also passed in that complete run. Existing unsuccessful checks were:

- `test_alarm_budget.AlarmBudgetArtifactTests.test_inputs_outputs_and_preservation_verified`: `Historical artifact changed: docs/FULL_PIPELINE_GUIDE.md`.
- `test_local_30.Local30ArtifactTests.test_outputs_and_protected_files_are_unchanged`: `Protected file changed: docs/FULL_PIPELINE_GUIDE.md`.
- `test_target_hardening.SavedTargetTests.test_historical_outputs_preserved`: saved hash differs for `docs/MODEL_EVALUATION.md`.

Those documents were already modified when work began. A SHA-256 snapshot of **737 pre-existing files** across source, data, results, archives, tests, documentation, dashboard, requirements and README was compared after the complete suite: **zero changed or missing files**. The snapshot proves these mismatches were present before these additions. No historical manifests, tests or integrity expectations were adjusted to conceal them. The full test output is retained in `validation/full_suite.log`; successful standalone new-test output is in `validation/new_tests.log`.
