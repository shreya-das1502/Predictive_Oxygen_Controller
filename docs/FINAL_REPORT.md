# Final implementation and verification report

The existing causal pipeline was hardened and evaluated with four fixed models; no broad search was performed. Historical targets/results remain separate. The current models produce excessive false alerts and do not establish useful clinical prediction or ELM superiority.

## Dataset reconciliation and targets

The retained legacy [dataset](../results/elm_dataset.csv) has 17,089 rows, 466 positives and 26 recordings. The current complete-window future-low dataset has 20,157 rows, 4,587 positives and 30 recordings. These are different populations and feature-extraction timelines, not a performance improvement attributable to relabeling alone. [Per-record count reconciliation](../results/final/dataset_reconciliation.csv) provides exact counts.

The archived legacy builder (`src/elm_experiment.py` in `archive/legacy_sources.zip`) filtered out current SpO₂ ≤94%, thereby removing recordings 04/08/09/22. It also used noncausal filtering/global prominence and inferred a 100 Hz clock from rounded timestamps. The current [preprocessing](../src/preprocessing.py) uses the verified 125 Hz sample clock and causal filtering; [features](../src/features.py) uses past/current pulses only. The saved independent legacy label audit found 15,706 fully observable agreeing labels, 1,096 incomplete horizons and 287 invalid horizons (1,383 unverifiable labels). Original results are retained rather than silently recomputed.

The current [label implementation](../src/labeling.py) requires complete valid numeric coverage over `(t,t+30s]` and maximum numeric gap 1.01 s. `future_low` is minimum observed SpO₂ ≤94%, including ongoing lows. `new_onset` additionally requires a fresh current reading >94% and an adjacent valid >94%→≤94% transition within that horizon. One low sample qualifies; no sustained-event or clinical episode definition is implied. This yields 16,050 rows, 501 positives, 26 recordings and six positive-containing recordings. Of the future-low rows, 4,107 are ineligible for onset: 4100 have current SpO₂ ≤94% and 7 lack current SpO₂. On fully valid currently normal windows the two labels agree. Pulse times changed, so the remaining legacy/current difference cannot be assigned a one-to-one row waterfall.

## Evaluation and measured results

Fresh evaluations rebuilt both datasets (`cache_reused=false`). Each outer test recording is excluded from fitting and three-fold grouped training-only OOF threshold selection. Median imputation/scaling are fitted inside each training fold. All models see identical eligible held-out rows within each target. The threshold policy remains the largest training-OOF threshold meeting 92% timestamp recall; it was not adjusted using these test results. Positive-containing and negative-only recordings, pooled metrics and defined-value macro metrics are reported separately.

Numbers below are pooled held-out timestamps. AP is average precision, not trapezoidal PR area. FP/FN count timestamps rather than distinct episodes.

### future_low

| Model | Recall | Precision | Specificity | F1 | AP | FP | FN | False episodes/hour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ELM | 0.9649 | 0.2390 | 0.0949 | 0.3831 | 0.2559 | 14093 | 161 | 140.69 |
| Logistic | 0.9320 | 0.2196 | 0.0243 | 0.3555 | 0.2403 | 15192 | 312 | 27.70 |
| GaussianNB | 0.9821 | 0.2257 | 0.0072 | 0.3670 | 0.1963 | 15458 | 82 | 13.99 |
| Tree | 0.8949 | 0.2146 | 0.0353 | 0.3462 | 0.1895 | 15021 | 482 | 28.52 |

### new_onset

| Model | Recall | Precision | Specificity | F1 | AP | FP | FN | False episodes/hour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ELM | 0.8902 | 0.0309 | 0.1010 | 0.0598 | 0.0240 | 13979 | 55 | 182.94 |
| Logistic | 0.8862 | 0.0282 | 0.0161 | 0.0547 | 0.0335 | 15299 | 57 | 21.60 |
| GaussianNB | 0.9002 | 0.0332 | 0.1563 | 0.0641 | 0.0576 | 13118 | 50 | 60.63 |
| Tree | 1.0000 | 0.0312 | 0.0000 | 0.0605 | 0.0396 | 15549 | 0 | 9.76 |

The historical approximately 96.5% recall/3% precision conflated populations: current future-low ELM precision is 23.90%, whereas onset precision is 3.09%. The [before/after metric table](../results/final/before_after_metrics.csv) verifies that historical hardened and fresh fixed-model metrics agree; this cleanup/visualization pass did not alter their predictions. Legacy v2 results are not a matched before/after comparison because eligibility, labeling validity and feature processing differ.

On onset, ELM alerts on 89.88% of eligible timestamps. Logistic alerts on 98.09%, GaussianNB on 84.54%, and the tree always alerts. GaussianNB has better pooled AP than ELM; no overall ELM superiority is supported. Low false-episode counts can arise from one very long continuous alert: interpret episodes/hour alongside alert fraction and specificity. Exposure covers eligible pulse segments, not all wall-clock recording duration. Record 26 accounts for 37 ELM false negatives, record 30 for 17, record 7 for one; record 12 has none. Detailed score/threshold timelines are plotted for 12/26/30.

All models detect 14/14 eligible observed onsets and 3/3 eligible 91–92% episodes; the severe-subset median lead is 9.784 s. An always-alert reference also detects all of them, and many observed transitions lack eligible prior opportunities. These small-denominator overlap-based event results do not demonstrate useful anticipation. The event tables document excluded events and opportunity rules.

## Actual latency

The fresh benchmarks were executed sequentially. The following pulse-to-alert computational measurements include rolling feature updates, the model path, eligibility and thresholding (starting from an already extracted pulse). Separate feature, transform/model, threshold, per-waveform-sample replay, pulse-producing update and warm model-loading measurements, raw timing samples, environment and repetitions are saved per target. P99 is reported only with at least 1,000 repetitions; 100 model-load repetitions support median/P95 only.

| Target | Model | Median ms | P95 ms | P99 ms |
| --- | --- | --- | --- | --- |
| future_low | ELM | 0.903 | 3.828 | 6.053 |
| future_low | Logistic | 0.612 | 1.027 | 1.141 |
| future_low | GaussianNB | 0.703 | 0.999 | 1.239 |
| future_low | Tree | 0.596 | 1.046 | 1.415 |
| new_onset | ELM | 0.489 | 0.743 | 0.904 |
| new_onset | Logistic | 0.616 | 0.864 | 1.157 |
| new_onset | GaussianNB | 0.707 | 0.810 | 1.236 |
| new_onset | Tree | 0.599 | 0.684 | 1.018 |

Timing varies on this shared host, including slower future-low ELM timings. These measurements exclude physiological sensor averaging, waiting for samples/pulses and the 30-second prediction horizon. They are not a real-time or hardware guarantee.

## Changes, cleanup and artifacts

- Targets/configuration/cache: [labeling](../src/labeling.py), [config](../src/config.py), [dataset](../src/dataset.py), [provenance](../src/provenance.py). Cache identities cover relevant preprocessing/features/labels/config/source fingerprints; artifact hashes detect edited prepared outputs.
- Evaluation/runtime: [evaluation](../src/evaluation.py), [runner](../src/run_experiments.py), [model](../src/model.py), [controller](../src/controller.py), [replay](../src/replay.py), [target reports](../src/target_reporting.py). Manifests record targets, features, seeds, parameters, fingerprints, thresholds and recording splits. Full-data demo models are separate from held-out predictions.
- Analysis/verification: [figures](../src/figures.py), [latency benchmark](../src/benchmark_latency.py), [quality gate](../src/check_project.py), and target/saved-result/figure tests. Raw latency distributions and source-bound numerical chart tables make figures auditable.
- Cleanup: 26 legacy source files and the prior README were archived byte-for-byte; 28 historical-only root outputs were archived. Two checkpoint CSVs were deleted only after matching retained counterparts exactly. Empty directories and disposable bytecode caches were removed. Raw data, historical validated/hardened evidence, legacy dataset and v2 results/predictions were retained. [Every file decision](FILE_CLEANUP.md), [dependency evidence](CLEANUP_DEPENDENCIES.json), [archive recovery](../archive/README.md).

[Requested figure hub](../results/validated/figures/INDEX.md) links 23 PNG/PDF figure pairs per target, without duplicating image data or overwriting historical predictions. Included are raw signals/SpO₂, actual PPG pulse peaks, interval/rate quality proxies, missingness/clock issues, class balance, feature distributions/correlation, exploratory held-out block-permutation importance, matched model metrics/PR/ROC/confusion, OOF thresholds, recording timelines/errors, events/lead times and raw timing distributions. PPG pulse peaks are not ECG R-peaks. Permutation analysis uses fixed refits whose scores must match saved predictions; it does not tune models.

[Future-low run](../results/final/future_low/) and [onset run](../results/final/new_onset/) contain `EVALUATION.md`, predictions, comparison/per-record tables, `run_manifest.json`, `dataset_cache.json`, `models/`, event diagnostics and latency samples. Large local prepared data/model files are Git-ignored and regenerate from the documented commands.

## Reproduction and verification

The complete pipeline diagram, module-by-module mapping and commands are in [FULL_PIPELINE_GUIDE](FULL_PIPELINE_GUIDE.md). Start with the environment/raw-data inspection:

```bash
python -m src.check_project --audit
python -m src.run_experiments --target future_low --output results/final/future_low
python -m src.run_experiments --target new_onset --output results/final/new_onset
python -m unittest discover -s tests -v
python -m src.check_project --links --verify-run results/final/future_low
python -m src.check_project --links --verify-run results/final/new_onset
python -m src.figures --output results/final/new_onset
```

For this pass, both fixed-model evaluations were actually executed with `--skip-benchmark`, then both latency commands, target-report commands and figure commands were executed separately. This avoided concurrent training during timing. Dataset CLI output was checked against the final onset dataset hash; an archived-source restore into a new temporary directory succeeded; the saved Logistic onset model was loaded in a separate replay process for record 12 (748 pulse outputs). No package installation, data download, broad search, clinical validation or hardware deployment was performed.

The full suite executed **76 tests in 23.165 seconds, all passed with zero skips**. Both saved-run gates passed, 141 local links resolved, all 85 protected historical files matched their hashes, and source/result archives restored or hash-verified successfully. The README inference example and separate saved-model replay executed successfully. Final test and consistency results are recorded in [quality_gate.json](../results/final/quality_gate.json). Remaining limitations include only 30 short recordings, few onset/severe events, rounded waveform timestamps, single-reading threshold events, no validated signal-quality gate, highly correlated pulse features, uncalibrated pooled scores across refits, and excessive alerts. Future research should prespecify an operational false-alarm objective and validate on additional recordings before considering deployment; this pass stops before that model-selection work.
