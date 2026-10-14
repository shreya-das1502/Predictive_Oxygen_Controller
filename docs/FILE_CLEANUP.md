# Cleanup decisions

Dependency evidence: [CLEANUP_DEPENDENCIES.json](CLEANUP_DEPENDENCIES.json). Exact archived bytes and recovery instructions: [archive README](../archive/README.md). No raw data were removed.

| File_or_Directory | Action | Reason | Dependents |
| --- | --- | --- | --- |
| src/advanced_experiment.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| src/check_dropped_samples.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| src/check_prediction_consistency.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| src/check_records.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| src/compare_models.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| src/conventional_experiment.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| src/diagnose_advanced_results.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| src/diagnose_record_26.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| src/diagnose_thresholds.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| src/elm_experiment.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| src/elm_threshold_check.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| src/elm_threshold_sweep.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| src/feature_audit.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| src/inspect_elm_dataset.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| src/inspect_elm_scores.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| src/inspect_events.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| src/label_audit.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| src/recall_first_elm.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| src/recording_aware_elm.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| src/recording_aware_elm_v2.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| src/reporting.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| src/verify_labels.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| archive/legacy_src/labeling.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| archive/legacy_src/benchmark_latency.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| archive/legacy_src/features.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| main.py | archived | Exact bytes retained in archive/legacy_sources.zip; superseded implementation or historical-only diagnostic | No active caller; archived scripts may import one another; restore as a unit |
| src/__init__.py | kept | Active authoritative pipeline module | Runner, inference, reporting or tests |
| src/benchmark_latency.py | kept | Active authoritative pipeline module | Runner, inference, reporting or tests |
| src/config.py | kept | Active authoritative pipeline module | Runner, inference, reporting or tests |
| src/controller.py | kept | Active authoritative pipeline module | Runner, inference, reporting or tests |
| src/dataset.py | kept | Active authoritative pipeline module | Runner, inference, reporting or tests |
| src/evaluation.py | kept | Active authoritative pipeline module | Runner, inference, reporting or tests |
| src/features.py | kept | Active authoritative pipeline module | Runner, inference, reporting or tests |
| src/labeling.py | kept | Active authoritative pipeline module | Runner, inference, reporting or tests |
| src/model.py | kept | Active authoritative pipeline module | Runner, inference, reporting or tests |
| src/preprocessing.py | kept | Active authoritative pipeline module | Runner, inference, reporting or tests |
| src/provenance.py | kept | Active authoritative pipeline module | Runner, inference, reporting or tests |
| src/replay.py | kept | Active authoritative pipeline module | Runner, inference, reporting or tests |
| src/run_experiments.py | kept | Active authoritative pipeline module | Runner, inference, reporting or tests |
| src/target_reporting.py | kept | Active authoritative pipeline module | Runner, inference, reporting or tests |
| data/ | kept | Raw PhysioNet inputs | Reproducibility, tests or reports |
| results/elm_dataset.csv | kept | Legacy label-audit input | Reproducibility, tests or reports |
| results/validated/ | kept | Historical comparison, saved-model and regression-test evidence | Reproducibility, tests or reports |
| results/hardened/ | kept | Previous target-specific experiments and preservation evidence | Reproducibility, tests or reports |
| docs/ | kept | Audit/evaluation/handoff evidence | Reproducibility, tests or reports |
| tests/ | kept | Core and regression validation | Reproducibility, tests or reports |
| requirements.txt | kept | Pinned runtime dependencies | Reproducibility, tests or reports |
| src/figures.py | kept | Authoritative figure generation or final quality gate | Runner, guide, and new tests |
| src/check_project.py | kept | Authoritative figure generation or final quality gate | Runner, guide, and new tests |
| README.md | consolidated | Replaces accumulated contradictory entry-point examples with one current quickstart; previous exact README archived | User handoff |
| notebooks/ | deleted | Empty, untracked directory; no files or references | None |
| results/PPG_Filtered_Comparison.png | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/recording_aware_elm_results.csv | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/advanced_dataset.csv | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/recall_first_elm_test_predictions.csv | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/recall_first_elm_search.csv | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/label_audit.csv | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/PPG_Peaks.png | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/recording_diagnostic_summary.csv | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/record_07_events.csv | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/recording_aware_elm_predictions.csv | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/logistic_metrics.csv | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/labeled_record_01.csv | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/filtered_ppg.csv | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/advanced_test_predictions.csv | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/recall_first_elm_test_metrics.csv | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/feature_audit.csv | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/model_comparison.csv | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/elm_score_diagnostics.csv | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/label_verification.csv | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/elm_metrics.csv | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/recording_diagnostic_features.csv | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/latency_benchmark.csv | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/elm_threshold_sweep.csv | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/SpO2.png | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/PPG.png | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/advanced_dataset_cleaned.csv | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/advanced_model_comparison.csv | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/features_record_01.csv | archived | Unique historical-only output retained byte-for-byte in archive/legacy_results.zip; no active consumer | Archived experiments/diagnostics only |
| results/recording_aware_elm_v2_results_checkpoint.csv | deleted | Byte-identical redundant checkpoint; exact content retained in results/recording_aware_elm_v2_results.csv | No active reader; archived script only writes checkpoint |
| results/recording_aware_elm_v2_predictions_checkpoint.csv | deleted | Byte-identical redundant checkpoint; exact content retained in results/recording_aware_elm_v2_predictions.csv | No active reader; archived script only writes checkpoint |
| results/recording_aware_elm_v2_results.csv | kept | Original v2 metrics/predictions supporting the historical 11.85% macro-recall report | Historical comparison and checkpoint content recovery |
| results/recording_aware_elm_v2_predictions.csv | kept | Original v2 metrics/predictions supporting the historical 11.85% macro-recall report | Historical comparison and checkpoint content recovery |
| src/tests/archive __pycache__/ | deleted | Disposable Python bytecode generated by local execution; no tracked bytecode; regenerated automatically | No source or saved model dependency |
