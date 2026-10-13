# Predictive Oxygen Controller

A reproducible research comparison of causal pulse features for predicting whether **minimum observed SpO₂ in (t, t+30 seconds] is ≤94%**. This software returns research risk scores and alerts; it does not control oxygen delivery.

The active pipeline repairs future waveform leakage, incomplete target horizons, recording selection and threshold validation in the original experiments. It compares a weighted ridge ELM, regularized logistic regression, Gaussian naive Bayes, and a shallow decision tree on identical held-out recording rows. Use the measured results in [MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md); historical CSVs at the root of `results/` are not the authoritative evaluation.

## Run from the repository root

The verified environment is Python 3.14.2; dependency versions are pinned in `requirements.txt`. To recreate it in your chosen virtual environment, run `python -m pip install -r requirements.txt`. Local data contain paired `data/bidmc_01_Signals.csv` / `bidmc_01_Numerics.csv` through record 30. No download is performed. Missing paired files cause an explicit error. Additional waveform files require corresponding numeric files.

```bash
# Validate raw data, regenerate causal prepared data, audit legacy labels
python -m src.dataset

# Run automated core and saved-artifact checks
python -m unittest discover -s tests -v

# Complete all fixed-model grouped validation/held-out experiments and benchmarks/reports
python -m src.run_experiments

# Benchmark persisted models without training
python -m src.benchmark_latency

# Regenerate plots/reports from saved predictions, without refitting
python -m src.reporting

# Optional raw-stream demonstration using the saved full-data reference model
python -m src.replay --record 12
```

The runner reuses prepared data only when raw and generation-code hashes match. `--rebuild` forces regeneration; `--output PATH` isolates a run. `--skip-benchmark` is for evaluation-only runs and makes no fresh latency claim. Runs replace only files in the selected authoritative output folder. There is no silent checkpoint resume or hyperparameter search. Native BLAS threads are limited to one for experiments/timing. Sensor acquisition and monitor latency are separate from computational timing.

## Causal and label contracts

Raw waveform filtering uses forward-only third-order 0.5–5Hz Butterworth SOS, zero initial filter state. A local maximum is confirmed by its following sample (8ms at 125Hz), with 0.03 amplitude and 0.30s refractory criteria based only on available data. Amplitude uses the preceding trough; intervals use preceding peaks; intervals above 2s are counted as rejected. Features are timestamped at confirmation, not at an earlier peak. No waveform-wide scale or retrospective peak optimization is used. Waveform CSV times are rounded after 100s: 125Hz row-clock reconstruction is allowed only when all timestamps agree within 0.005001s. This preserves every acquisition row and rejects incompatible timing. Numeric timestamps retain their original coordinate; record 23 starts at -1s and is not independently rebased.

Ten ordered inputs: PPG_Amplitude, Pulse_Interval_s, Pulse_Rate_BPM, Amplitude_Slope, Rate_Mean_5, Rate_Std_10, Rate_Change_5, Amplitude_Mean_5, Amplitude_Std_10, Interval_Std_10. Windows use pulses, not seconds. Means use available history; sample standard deviations are zero for a singleton; five-pulse differences are zero before six pulses exist. A bounded ten-pulse buffer is reset per recording. History includes pulses with unavailable targets: dropping target rows never erases causal feature history. NaN measurements remain missing, with medians learned only on training rows; sklearn's entirely empty training columns use documented zero imputation. Infinity, missing column keys, inconsistent interval/rate, invalid ranges and unordered timestamps raise errors. Missingness is exposed in controller metadata.

Labels use strictly future numeric samples ≤30 seconds ahead; require complete source coverage at 1Hz (maximum gap 1.01s), valid SpO₂ and no missing future observation. Incomplete tails and missing futures are unknown, never negative. Current saturation does not select primary rows. `spo2`, future minimum, time, recording ID and label are diagnostics/ground truth and never predictors. All thirty recordings participate. The broader target includes ongoing low saturation; a secondary current-SpO₂>94 subset and independently defined raw onset events assess impending detection. This distinction matters when interpreting pooled recall.

## Validation and models

The outer test is one entire recording. All other recordings generate three-fold GroupKFold OOF scores; the largest threshold achieving ≥92% timestamp recall is selected on those scores, then the model is refitted on outer-training data. Imputers and scalers are independently fitted in every inner/outer training set. The test recording does not select features, hyperparameters, weights or thresholds. ELM has 50 fixed random tanh units, ridge=1 and positive weight=5; output weights solve weighted ridge normal equations by Cholesky factorization. Logistic C=1 uses balanced weights; Gaussian NB has equal priors and variance smoothing 1e-6; tree depth=3, leaf minimum=50, balanced weights. All are fixed before evaluation, seed=42.

ELM scores are uncalibrated and need not lie in [0,1]. Classifier probabilities with changed class priors/weights are also not validated calibrated physiological risks. NaN marks undefined metrics; macro denominators are explicit. AP denotes average precision. Pooled AP across refits has score-comparability limitations. Alert episode rates and positive timestamp rates are separately defined in the evaluation report.

## Inference

```python
from src.controller import Controller

controller = Controller.load('results/validated/models/Logistic.joblib')
# Invoke for each NEW, strictly increasing pulse timestamp; reset controller for another record.
result = controller.predict({
    'time_s': 1.2, 'PPG_Amplitude': 0.2, 'Pulse_Interval_s': 0.8,
    'Pulse_Rate_BPM': 75.0, 'Amplitude_Slope': 0.0,
})
print(result)  # score, alert, validation-derived threshold, version, missingness
# Or call controller.update_waveform(time_s, ppg) on every 125Hz input.
# It returns None until a pulse is confirmed, then the same result contract.
```

The saved full-data models are reproducible demonstration refits, each with a full-data grouped OOF threshold; these models were not used to produce held-out metrics. Logistic is the prespecified transparent demonstration default, not a claim that outer tests independently validate model selection. Load only locally trusted joblib files. Saved metadata includes target, threshold, hyperparameters, seed, feature order, training IDs, data hash and version.

## Authoritative outputs

`results/validated/` contains dataset/raw quality and preprocessing audits; independent legacy `label_audit.csv`; causal dataset and full pulse history; per-record predictions and metrics; pooled metrics and `model_comparison.csv`; validation metrics and compressed OOF predictions; selected operating points in `threshold_analysis.csv` and complete tradeoffs in `threshold_analysis.csv.gz`; event matches/summary; impending timestamp metrics; feature/score/record diagnostics and plots; exploratory uncertainty estimates; measured `latency_benchmark.csv` and environment; source/output hashes, grouped splits, run manifest; and trusted model binaries plus readable JSON metadata. Large generated data/OOF tables and binaries are ignored by Git but remain local and reproducible. No source files or historical results were deleted or overwritten.

See [PROJECT_AUDIT.md](docs/PROJECT_AUDIT.md), [MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md), and [LATENCY_REPORT.md](docs/LATENCY_REPORT.md). Original replaced scripts are preserved in `archive/legacy_src/`; other legacy scripts remain in place because of their imports and existing user changes. Use `python -m src...` for the maintained entry points; old experiment commands retain historical behavior.

BIDMC source: [PhysioNet](https://physionet.org/content/bidmc/1.0.0/). Only 30 of the public 53 recordings are local. The available evidence is retrospective, with few prospective onsets, missing numeric measurements, chronic low saturation and unverified patient identities/sensor delay. Results support a demonstration and honest model comparison; they do not establish clinical benefit or safe oxygen actuation.
