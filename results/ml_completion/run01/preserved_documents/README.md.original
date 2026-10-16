# Predictive Oxygen Controller

**Final study outcome:** on the 30 local recordings, the final alarm-budget policy detected 0/15 eligible events at 5/10/15 seconds and 0/14 at 30 seconds for all four learned models. The tested 1/5/10 false-starts-per-evaluable-hour budgets and 5% active-time cap are research assumptions, not clinically validated limits. Some held-out constraints were violated. No useful early warning, proven model superiority or clinical readiness was demonstrated; patient independence is unverified. Read the [final evaluation and historical reconciliation](docs/FINAL_REPORT.md) and [alarm-budget experiment](docs/ALARM_BUDGET_EXPERIMENT.md) before using earlier metrics or figures.

Start with **`python -m src.check_project --audit`** from this directory. It verifies installed dependencies and inspects the local source data without training or overwriting results.

Read [FULL_PIPELINE_GUIDE.md](docs/FULL_PIPELINE_GUIDE.md) for the complete data-flow diagram, file map, commands, model equations, target definitions and interpretation. [FILE_CLEANUP.csv](docs/FILE_CLEANUP.csv) explains every cleanup decision; historical source is recoverable from [archive/](archive/README.md).

This research pipeline compares a weighted ridge ELM, logistic regression, Gaussian naive Bayes and a shallow decision tree using causal PPG features and recording-wise validation. It emits scores/alerts and **does not actuate oxygen delivery**.

Two separate targets are evaluated:

- `future_low`: minimum valid observed SpO₂ in `(t,t+30s]` ≤94%, including ongoing lows.
- `new_onset`: the same complete valid future horizon, but current observed SpO₂ must be >94% and a qualifying >94 to ≤94 transition must occur. A single low reading qualifies; persistence and clinical validity are not assumed. Observed 91–92% episodes form a separately reported subset.

## Reproduce the earlier recall-oriented evaluation

```bash
python -m src.check_project --audit
python -m src.run_experiments --target future_low --output results/final/future_low
python -m src.run_experiments --target new_onset --output results/final/new_onset
python -m unittest discover -s tests -v
python -m src.check_project --links --verify-run results/final/new_onset
```

Full runs train fixed models, perform grouped validation, benchmark, generate reports and figures. They take minutes; there is no hyperparameter search. Native BLAS threads are limited to one inside fitting/benchmarking. Running targets sequentially avoids CPU competition during timing.

To regenerate reports and figures without repeating the evaluation:

```bash
python -m src.target_reporting --output results/final/new_onset
python -m src.figures --output results/final/new_onset
```

The figure command performs fixed diagnostic refits for block-permutation importance; it does not select features, tune models or change predictions. It verifies that diagnostic refits reproduce saved held-out scores.

Earlier recall-oriented artifacts are in [results/final/future_low/](results/final/future_low/) and [results/final/new_onset/](results/final/new_onset/). Each has `EVALUATION.md`, metrics, predictions, manifests, models and a `figures/INDEX.md` with PNG/PDF links and interpretations. Historical `results/validated/` and earlier `results/hardened/` evidence remain unchanged.

Large prepared datasets, raw pulse tables, model binaries and compressed validation tables are local generated artifacts, ignored by Git. On a fresh checkout, run the two full commands above to regenerate them. Historical-only integration tests skip when their optional local historical artifacts are absent; final-run integration tests execute once final figures exist.

## What to edit

- Targets, feature order and fixed parameters: [src/config.py](src/config.py).
- Ground truth: [src/labeling.py](src/labeling.py).
- Causal signal processing: [src/preprocessing.py](src/preprocessing.py).
- Pulse/rolling features: [src/features.py](src/features.py).
- ELM and baseline wrappers: [src/model.py](src/model.py).
- Evaluation and threshold policy: [src/evaluation.py](src/evaluation.py).
- Figure generation: [src/figures.py](src/figures.py).

Leave raw data, saved predictions/manifests, and archived historical source untouched. Source/config/data changes invalidate prepared-data caches; edited cached artifacts also force regeneration. Run tests after changing behavior.

## Inference example

After generating `new_onset` models:

```python
from src.controller import Controller
c = Controller.load('results/final/new_onset/models/Logistic.joblib')
result = c.predict({
    'time_s': 1.2, 'PPG_Amplitude': 0.2, 'Pulse_Interval_s': 0.8,
    'Pulse_Rate_BPM': 75.0, 'Amplitude_Slope': 0.0,
    'spo2': 98.0, 'spo2_time_s': 1.0,
})
print(result)
```

SpO₂ supplies eligibility, not a model input. Low/missing/stale current SpO₂ suppresses onset alerts while causal pulse history continues to update. `update_waveform()` provides the raw 125Hz stream interface. Only load locally trusted joblib artifacts.

## Interpretation

At the earlier recall-oriented thresholds, the models produce excessive false positives; high pooled future-low recall is partly driven by ongoing low saturation. Historical onset precision is around 3%, and event sensitivity under nearly continuous alerts does not prove useful early warning. ELM speed does not establish detection superiority. See [FINAL_REPORT.md](docs/FINAL_REPORT.md) for current measured results and limitations, and [HARDENING_SUMMARY.md](docs/HARDENING_SUMMARY.md) for the preceding target-definition comparison.

Only the 30 locally available BIDMC recordings are evaluated. Recording identity is not verified patient identity. CSV timing precision, missing numerics, unknown monitor delay, few onset-bearing recordings and unadjudicated one-reading excursions limit interpretation. Computational latency excludes physiological acquisition and is distinct from the prediction horizon.

Current figure hub: [PNG/PDF indices](results/validated/figures/INDEX.md). Final handoff: [FINAL_REPORT](docs/FINAL_REPORT.md).

The [reliability audit](docs/RELIABILITY_AUDIT.md) independently verifies event lead times and false alarms, adds fixed-window one-to-one alarm matching and compares simple baselines without retraining. Reproduce with `python -m src.reliability_evaluation --output results/reliability/reproduction` (choose a new destination).
