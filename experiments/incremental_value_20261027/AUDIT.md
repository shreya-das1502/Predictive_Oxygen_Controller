# Read-only audit before the incremental-information experiment

## Historical source and configuration

Historical commit: `ac5f0c02ba559381ff304ed6a852224bb0047dee`. `audit/historical_files_before.json` fingerprints all2,563 tracked files before new scientific outputs. Every new source/artifact is under this experiment; original source/results/models remain untouched.

| Stage | Maintained implementation inspected |
|---|---|
| Raw data | `src/dataset.py:load_csv/build_dataset` |
| Clock/filter/pulse detection | `src/preprocessing.py:canonical_waveform_times/WaveformState/PulseDetector/extract_pulses` |
| PPG10 | `src/features.py:FeatureState/add_features`; order fixed in `src/config.py` |
| PPG18 follow-up | `ml_research/core.py:ContextState/add_context`, preserved and not retuned here |
| Labels/events | `src/labeling.py:future_labels/onset_labels/desaturation_events` |
| Models/transforms | `src/model.py:ELM/FittedModel`; exact `src/config.py:MODEL_CONFIGS` |
| Validation | `src/evaluation.py:outer_splits/grouped_oof` |
| Historical cutoff | `src/evaluation.py:choose_threshold`,92% training timestamp recall |
| Final budget cutoff | `src/alarm_budget_experiment.py:topology/candidate_statistics/choose_candidate/select_partition` |
| Events/burden | `src/reliability_evaluation.py:raw_events/links/episodes/eligible_events/match_events/off_event_alarms` |
| Bootstrap | `src/local_30_evaluation.py:bootstrap_weights/bootstrap_metrics/interval` |
| Historical reporting | `results/final/<target>/run_manifest.json`, original `results/alarm_budget_experiment/protocol.json` and `run02/` |
| Follow-up lineage | `results/ml_completion/run01/protocol.json`, recovery/fit ledger and saved result tables; distinct targets/horizons/features from the present primary comparison |

The earlier audit in this session independently checked the saved historical metrics and original/follow-up aggregations. The present task additionally rebuilds the waveform-to-pulse clock and primary labels from raw inputs, checking them against the preserved historical held-out ELM reference. Full old model fitting reproduction remains limited by missing generated original prepared datasets, OOF tables and final demo binaries; those are not silently restored or overwritten.

## Data and target checks

- Thirty existing BIDMC Signals/Numerics pairs; no dataset expansion and no verified patient-ID linkage.
- Typical waveform60,001 rows over480s; documented125Hz sample clock checked against every rounded export time. Numerics481 rows at nominal1Hz. Per-record actual checks are in `outputs/data_inventory.csv`.
- Raw PPG missing values and numeric SpO2 missingness recorded per record; no automatic hard-event exclusions.
- Future-low target remains minimum valid observed SpO2≤94 in `(t,t+30s]`, complete valid horizons and maximum1.01s gaps; current instant excluded.
- Historical complete-target row/clock/label parity checked at20,157 rows and4,587 positives. Common comparison excludes seven unavailable current readings, retaining20,150 rows and all4,587 positives. All three modalities share this restriction, rather than giving the SpO2 baseline easier rows.
-22 raw low episodes;15 qualifying normal-to-low onsets, including one isolated94% reading. Four all-low recordings contribute no current-normal event exposure but remain in fitting and the reporting roster.
- Primary one-to-one actual-alarm-start matching and strict current-normal exposure are preserved. A W30 event window is not a guarantee of≥10s warning.
- Sensitivity event definition is frozen before fitting:≥3 consecutive low observations spanning≥2s. Raw events are not deleted; alternate eligibility, burden and counts are reported separately.

## Leakage assessment and deliberate differences

Original filtering is forward-only SOS recursion; peak availability uses next-sample confirmation time; PPG histories are causal. The new numeric histories use only observation times≤prediction time. Tests change/remove all later observations to verify invariance. Features are explicitly whitelisted and exclude labels, future minima, IDs and absolute times.

Each held-out record is excluded from every fit/transformation and threshold-selection array. Grouped inner OOF scores are used for thresholds. All selections are sealed before outcome analysis; parameters and one research budget are specified in config. No feature selection, learned probability calibration, horizon search or outer-winner selection occurs.

The new experiment deliberately adds current/past SpO2 as predictors in A/C; this is its research question, not a hidden change to historical PPG features. Conditioning the common dataset on current availability changes seven rows relative to the old PPG-only run, so new matched PPG scores should not be claimed as an exact reproduction of old fitted scores. Original files remain intact.

Complete future labels/exposure are retrospective observation criteria. They are not live predictor inputs, but they limit interpretation to evaluable periods and must not be turned into a live future-dependent quality gate. Patient overlap remains unverified. Repeated human inspection of this cohort is not removed by another round of grouped folds.

## Before-change existing tests

Executed `python -m unittest discover -s tests -v` in the pinned environment: **162 tests,13 errors,15 skips**; underlying command exit1. Log: `audit/existing_tests_before.log`.

Errors are missing historical prepared final datasets in preservation/quality gates, eight missing-demo-model subtest errors, a resulting demo UI missing control, and two absent generated OOF-table links. Skipped historical/hardening integrations require optional local artifacts. No original test/receipt is edited to make these pass. The new experiment must be evaluated on its own completed verified outputs, while retaining these original-suite limitations.

New initial tests:12/12 passed, covering future label boundaries/missingness, causal history and units/staleness, unchanged predictor schema, training-only transforms, fixed forecasts and one-to-one/nonclipped matching. Final rerun and preservation outcomes are recorded after execution.

## Completed verification

The same12 new tests passed after implementation. The existing suite remained162 tests,13 errors and15 skips; before/after logs preserve the actual failures. Independent reconstruction checked all22 new event conditions, classifier target alignment/metrics, training folds/cutoffs, forecast errors and recording intervals. All2,563 original tracked files remained byte-for-byte unchanged. A sealed-fit resume completed without refitting. See `outputs/verification.json` and `REPORT.md` for the completed result and scientific limitations.
