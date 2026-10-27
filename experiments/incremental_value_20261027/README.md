# Does PPG add information beyond SpO2 history?

This isolated experiment implements the user's refresh task in the actual `Predictive_Oxygen_Controller` checkout (called `Predictive_FiO2_ML` in the request). It does not rename the repository or expand its dataset. Historical source, tests, outputs and model artifacts are preserved.

Start with [AUDIT.md](AUDIT.md) and the frozen [config.json](config.json). The completed [REPORT.md](REPORT.md) records actual results after fitting, analysis and verification. No reliability, patient-independence or clinical-benefit claim follows from merely completing the code.

## Prespecified comparison

- Existing local30 recordings only; no new data or inferred patient identities.
- Primary label unchanged: minimum valid observed SpO2 ≤94% in `(t,t+30s]`, including ongoing lows.
- Three modalities: seven causal SpO2-history features, the existing PPG10 features, and their17-feature concatenation.
- Three fixed classifiers per modality: existing logistic regression, GaussianNB and weighted-ridge ELM. No search or outer-selected model winner. Modality comparisons are the feature-ablation experiment; context18 remains preserved but is not another search dimension here.
- Identical common timestamps, labels, records and outer folds for all nine learned conditions, two fixed references and five forecasters.
- Common cohort has20,150 rows: the original20,157 valid-label rows minus seven without a valid latest current SpO2. All4,587 positives remain. Report full-target and current-normal metrics separately; do not attribute persistence of already-low states to early warning.
- Leave-one-recording-out evaluation; three inner recording-grouped folds for learned-classifier threshold selection. Every imputer/scaler/model is fitted only on its permitted training rows.
- One declared research operating point: maximize training OOF one-to-one W30 detections subject to ≤5 off-event starts/evaluable hour and ≤5% activity. These are research assumptions, not clinical tolerances. Fixed persistence/trend alarms are not constrained or budget-selected.
- Original single-reading event definition retained. Sensitivity analysis scores the same frozen predictions/policies against ≥3 consecutive low readings spanning≥2 seconds; false-start burden is recomputed against that subset. It does not relabel/refit the primary models.
- Continuous forecasting predicts the same future minimum in percentage points: persistence, causal10-second trend, and fixed-alpha1 ridge using SpO2 history, PPG or both. Physical clipping is fixed before testing; raw predictions are saved.
- Whole-record paired bootstrap:2,000 draws, seed20261027; conditional on saved overlapping CV fits, not patient-level/refit uncertainty.

## Reproduce

From the repository root, use the prepared Python3.12 venv or install the pinned research requirements into an isolated environment:

```bash
cd /workspace/Predictive_Oxygen_Controller
source /workspace/.venvs/oxygen/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR=/workspace/.cache/oxygen-mpl
export XDG_CACHE_HOME=/workspace/.cache PYTHONDONTWRITEBYTECODE=1
python -m unittest experiments.incremental_value_20261027.test_study -v
python -m experiments.incremental_value_20261027.run --stage prepare
python -m experiments.incremental_value_20261027.run --stage fit
python -m experiments.incremental_value_20261027.run --stage analyse
python -m experiments.incremental_value_20261027.verify
```

`--stage all` runs preparation, fitting and analysis. The read-only verifier remains a separate step. Preparation and completed fits are reused only after hash checks; incomplete fold checkpoints fail closed for diagnosis. Frozen fitting code/config/raw-data changes are rejected on resume. To investigate a different scientific setting, make a separately named experiment directory and update its own module imports; do not overwrite this frozen study or reset any historical files.

Each scientific outer fit is saved with OOF scores, inner train/validation memberships, hashes, full candidate grid, cutoff and trusted local model object. `classification_fit_receipt.json` is written only after all270 learned outer folds are sealed. Regression uses90 outer fits with alpha fixed rather than tuned. Output models have experimental feature schemas; they are not replacements for the original dashboard's model files.

Analysis can be rerun deterministically from sealed predictions; it records its own source hash. No source changes after a fit may silently alter its scientific meaning. Reproduction in a fresh environment requires regenerating this new experiment's outputs, not relying on live processes or the presence of generated historical caches.

## Files and outputs

| Path | Contents |
|---|---|
| `study.py` | Causal numeric history, raw-data reconstruction, independent target masks, event audit and preserved matching/exposure wrappers |
| `run.py` | Frozen protocol, grouped fits/OOF thresholds, model checkpoints and continuous ridge forecasts |
| `analysis.py` | Held-out metrics, event sensitivity, sustained sensitivity, paired record bootstrap, calibration diagnostics and figures |
| `verify.py` | Separate numerical reconstruction of event starts/matching/exposure, labels/metrics, fold/cutoff and artifact checks |
| `test_study.py` | Twelve hand-checkable causal label/history/alarm and training-transform invariants |
| `audit/` | Before/after existing-suite logs, new tests, historical-file hashes and source index |
| `outputs/protocol.json` | Frozen config, all fitting-source/raw hashes, actual versions and candidate quantiles |
| `outputs/data_inventory.csv`, `raw_event_audit.csv` | Durations/clocks/missingness/pulse counts and all raw low episodes, including isolated/left-censored cases |
| `outputs/classification_predictions.csv.gz` | Every matched held-out score/label/cutoff/decision and explicit policy origin |
| `outputs/timestamp_summary.csv`, `timestamp_per_recording.csv` | Full-target and current-normal classification metrics kept separate |
| `outputs/event_summary.csv`, `event_per_recording.csv`, `event_matches.csv`, `event_audit_long.csv` | Original/sustained counts, exposure, alarm burden/activity, per-event detections/leads and quality context |
| `outputs/feature_modality_ablation.csv`, `paired_modality_differences.csv` | Matched information ablation and conditional paired event/burden differences |
| `outputs/calibration_bins.csv`, `probability_assessment.csv` | Diagnostic LR/GNB reliability/Brier; raw ELM/heuristic scores explicitly excluded from probability metrics |
| `outputs/forecast_predictions.csv.gz`, `forecast_summary.csv`, `forecast_per_recording.csv` | Future-min forecasts and errors in percentage points |
| `outputs/*bootstrap*`, `paired_forecast_differences.csv` | Shared recording multiplicities and conditional uncertainty; no row-independence assumption |
| `outputs/figures/` | Standalone PNG/PDF precision–recall, burden, lead, reliability, forecast-error and event-audit plots |
| `outputs/verification.json`, `independent_event_verification.json` | Actual post-run verification and independently reconstructed counts, not a clinical validation certificate |

## Interpretation safeguards

Latest SpO2 is now deliberately a predictor in modalities A/C and the references; future SpO2 remains solely the label. PPG-only model features remain PPG-derived, although the common comparison and event exposure use current SpO2 for eligibility. Offline complete future coverage is used to define observable outcomes/evaluable exposure, not as a proposed live future-aware gate.

Probability-shaped LR/GNB outputs are assessed, not assumed calibrated: balanced weights/equal priors alter interpretation. ELM scores and empirical ranks are not automatically probabilities. Missing metric denominators/leads remain undefined.

Signal-quality columns are descriptive proxies, not validated SQI or annotated beat success rates. Consecutive sample count and observed low span are reported separately; a one-reading episode has observed span0s but one numeric observation, not a proven zero-duration physiological event.

This cohort has already been examined repeatedly. A new fixed comparison on it is exploratory. No patient linkage, external validation, accepted clinical alarm tolerance, guaranteed minimum warning time or oxygen-control benefit is established.
