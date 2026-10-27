# Final scientific audit of the incremental PPG experiment

Audit directory created **27 October2026,21:50:12 Asia/Kolkata**. Source experiment: `experiments/incremental_value_20261027/`. All scientific reanalysis is exploratory on the same30 recordings. No original source, model, target, cutoff or result is overwritten.

Read [REPORT.md](REPORT.md) for results, [AUDIT_FINDINGS.md](AUDIT_FINDINGS.md) for correctness/scope findings and [reproduced_metrics.json](reproduced_metrics.json) for machine-readable reproduction status. The audit found no metric-changing evaluation defect. Eleven excluded sensitivity rows lacked policy metadata; the new failure table fills this reporting omission from the frozen prediction policy, without changing original files.

## Requested outputs

| File | Meaning |
|---|---|
| [event_failure_analysis.csv](event_failure_analysis.csv) | All22 raw episodes ×11 conditions ×2 definitions, including excluded events. Contains observed severity/duration, current/trend features, multiwindow PPG diagnostic trends, quality proxies, scores/cutoffs/crossings, starts, leads, exclusion/miss reasons and explicitly unknown physiological cause. |
| [paired_model_comparisons.csv](paired_model_comparisons.csv) | Paired event sensitivity, alarm burden/activity, common-event lead differences and forecast errors with conditional whole-record intervals. Negative forecast-error differences favor the first condition; positive lead differences mean longer warning on commonly detected events only. |
| [per_recording_metrics.csv](per_recording_metrics.csv) | Long-format classification/event and continuous forecast metrics. `Analysis` identifies the unit; blanks are not zeros. |
| [reproduced_metrics.json](reproduced_metrics.json) | Counts, raw/feature/fold checks, paired MAE findings, influence and limitations; undefined values are JSON null. |
| [reproduced_event_summary.csv](reproduced_event_summary.csv) | All22 event conditions, exact exposure, false starts, activity and pooled/per-record constraint checks. |
| [reproduced_event_matches.csv](reproduced_event_matches.csv) | Independently reconstructed actual-start-to-event assignments. |
| [reproduced_forecast_summary.csv](reproduced_forecast_summary.csv) | Same-timestamp future-minimum MAE/RMSE for five forecasters and both target populations. |
| [threshold_and_split_audit.csv](threshold_and_split_audit.csv) |270 outer fits: matching grouped splits/OOF rows, cutoff winner, independently counted selected training alarms, and training-only outer imputation/scaling statistics. |
| [uncertainty_intervals.csv](uncertainty_intervals.csv) | Conditional cluster-bootstrap intervals for counts-derived rates, detected-only lead medians and forecast errors. |
| [recording_paired_differences.csv](recording_paired_differences.csv) | Recording-level paired distributions; no-event sensitivities or absent common leads remain undefined. |
| [paired_common_event_leads.csv](paired_common_event_leads.csv) | Lead differences only on events detected by both compared conditions; avoids comparing different detected populations. |
| [forecast_influence.csv](forecast_influence.csv) | Signed contributions and leave-one-record contribution-removal effects; no model refitting. |
| [lead_coverage_diagnostic.csv](lead_coverage_diagnostic.csv) | Fraction of original eligible onsets with original matched warning≥L seconds; no cutoff changes/rematching and no zero-imputed missed leads. |
| [failure_reason_counts.csv](failure_reason_counts.csv) | Algorithmic reasons for eligible-event misses, distinguished from unknown physiological causes. |
| [failure_associations_descriptive.csv](failure_associations_descriptive.csv) | Small descriptive detected/missed feature medians; no causal interpretation or significance test. |
| [recording_inventory.csv](recording_inventory.csv), [raw_event_inventory.csv](raw_event_inventory.csv) | Exact same30 recording clocks/durations/missingness and reconstructed22 episodes/15 qualifying transitions. |
| [event_window_index.csv](event_window_index.csv) | Links to retrospective SpO2/raw-PPG/score panels for all15 qualifying onsets, including the ineligible22-second onset. |
| `figures/` | Standalone PNG/PDF audit figures and individual event panels. |
| `logs/`, `commands.json`, `tests/` | Actual commands/outcomes, unchanged historical failures, audit tests and initial corrected-fixture history. |
| `preservation_before.json`, `COMPLETION.json` | Fingerprints including all original tracked files and the entire previously untracked incremental experiment. |

## Reproduce without refitting

From the existing checkout with its pinned Python environment:

```bash
cd /workspace/Predictive_Oxygen_Controller
source /workspace/.venvs/oxygen/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR=/workspace/.cache/oxygen-mpl
export XDG_CACHE_HOME=/workspace/.cache PYTHONDONTWRITEBYTECODE=1
python -m experiments.final_audit_20261027T215012_IST.audit
python -m experiments.final_audit_20261027T215012_IST.plot_event_windows
python -m unittest experiments.final_audit_20261027T215012_IST.tests.test_audit -v
python -m unittest experiments.incremental_value_20261027.test_study -v
python -m unittest discover -s tests -v
```

`audit.py` independently reads masks/labels/numerics, reconstructs alarm starts/matching and all per-record metrics, verifies270 selected training policies and saved outer transform statistics, and reuses the original paired recording multiplicities after checking their seed. It writes only to this audit directory. It never calls the old experiment runner or verifier, since those may rewrite their output receipts.

Loading model objects is limited to the sealed trusted local artifacts generated by this repository; arbitrary downloaded joblib/pickle objects are not used. No new packages, network/data downloads, recording expansion, scientific model fits or threshold searches are needed.

The historical suite is expected to remain non-green without its optional original generated artifacts:162 tests,13 errors,15 skips. The new audit tests and the original incremental experiment's12 tests run separately. See the report for actual outcomes rather than assuming all commands pass.

## Scientific boundaries

- Validation is by recording, not verified patient. The cohort has been repeatedly examined.
- The primary target remains future measured SpO2 minimum≤94 within30s; this includes persistence of ongoing lows. Current-normal event metrics are separate.
- Original event matching/cutoffs are retained; the sustained-event truth is the source experiment's frozen sensitivity definition.
- Rates per **evaluable** current-normal hour use2.869176h. Counts divided by4h of source recordings are an additional descriptive normalization, not a measured whole-monitoring-time rate.
- Calibration plots/Brier are diagnostics of uncalibrated probability-shaped LR/GNB outputs. ELM/heuristic scores are not probabilities.
- Whole-record paired intervals do not refit overlapping CV models or establish independent confirmation; zero-detection bootstrap[0,0] is not certainty of population zero sensitivity.
- Per-event PPG statistics and retrospective plots are not validated SQI or proofs of why an event was missed. Physiology remains unknown where no reference supports it.
- Lead comparisons use common detected events. The minimum-lead curve is a descriptive filter on original matches, not a new optimized policy.
- No clinical reliability, benefit, journal readiness or oxygen-delivery control is established.
