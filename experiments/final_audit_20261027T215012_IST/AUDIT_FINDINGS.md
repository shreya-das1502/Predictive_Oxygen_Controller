# Findings, corrections and deliberately unchanged methods

Evidence: inspected source and sealed artifacts of `incremental_value_20261027`, raw local data, independently recomputed metrics and explicit tests. No new scientific models were fitted.

## Correctness findings

| ID | Finding | Classification and action |
|---|---|---|
| F01 |9/14,0/14,10/14 original LR counts and9/13,0/13,9/13 sustained counts reproduce exactly. All22 model/definition conditions and660 per-record event rows agree. | No count/matching defect found. Original files retained. |
| F02 |All270 saved classifier cutoffs are the declared feasible-grid winners. Their selected training event/start/activity/exposure counts independently reproduce from OOF scores/raw numerics. | No evaluation-label threshold selection found. Existing budget5/cap5% comparison is already supported; no new budget sweep justified. |
| F03 |Outer imputer medians/scaler mean/variance match permitted training data for every saved outer model; maximum checked scaler-mean error is0. All OOF row/label identities and inner/outer recording memberships agree. | Positive evidence against recording leakage/global outer normalization. Inner transform objects are not retained, so inner execution is supported by source/partition tests rather than independently refitted. Patient linkage remains unknown. |
| F04 |All20,150 future labels/minima, seven causal history features and8ms peak-confirmation timestamps agree with independent numeric masks/formulas. | No time-alignment or future-history-input defect found. PPG filtering/source prefix contracts remain causal. Complete future observability still defines a retrospective evaluation cohort. |
| F05 |All original forecast errors and paired sensitivity/MAE/RMSE intervals reproduce with identical sample IDs, target values and recording multiplicities. | The roughly0.006 MAE gain is numerically real as a descriptive point difference, not established practically meaningful/robust benefit. |
| F06 |11 excluded sensitivity rows in the old `event_audit_long.csv` have blank `Policy`: the isolated record07/event367 across11 conditions. | Verified reporting-metadata omission, with no effect on predictions/counts. New `event_failure_analysis.csv` fills the known frozen policy for every row, including excluded episodes. Old source/output untouched; regression test checks complete metadata. |

No metric-changing implementation defect or reason to alter the historical model fitting was established. This does not prove all possible leakage/bias absent: the audit cannot verify unique patient identity, replace missing inner objects, recover independent clinical ground truth or undo repeated human development on the cohort.

## Important interpretation limitations, not silently “fixed”

1. **Exposure:**4h of source duration is not4h of scored normal-state exposure. The established denominator is10,329.032s. A secondary source-duration normalization is labeled explicitly; it must not be marketed as a whole-monitoring-time false-alarm estimate.
2. **Future observability:**valid labels and exposure require future numeric coverage. This is legitimate retrospective outcome evaluability, but is not a future-aware live screening rule. Missing/censored time limits deployment extrapolation.
3. **Primary target includes ongoing lows:**high full-target AP can largely reflect current-state persistence. Current-normal metrics remain separate; the target is not silently replaced with onset-only training.
4. **Event validity:**one observed≤94 reading qualifies, without persistence/washout or clinical adjudication. The sustained sensitivity drops one eligible isolated event. No arterial oxygen reference or monitor-delay correction is available.
5. **Alarm constraints:**the original objective is pooled training feasibility, not guaranteed held-out or individual-record feasibility. Per-record violations are a robustness finding, not proof the selector was coded incorrectly.
6. **Lead estimates:**different models detect different events. Their individual median leads are descriptive, not a paired incremental lead estimate. The new common-event comparison gives LR median difference0s with wide uncertainty.
7. **Uncertainty:**record clusters are used, not overlapping windows. Conditional saved-CV bootstrap omits refit/model-selection uncertainty, assumes recordings are appropriate independent clusters and cannot establish patient independence. Zero-detection[0,0] empirical intervals are degenerate, not proof of exact population zero.
8. **Calibration:**LR/GNB probability-shaped scores can be diagnosed with Brier/reliability, but balanced class weights/equal priors and sparse independent events prevent a calibrated-risk claim. ELM/heuristic scores are excluded from probability calibration metrics.
9. **Quality and cause:**finite waveform samples are not proof of adequate PPG quality. Pulse count/interval CV/amplitude/trend statistics are unvalidated proxies. Threshold noncrossing is an algorithmic explanation; why the model lacked a warning is physiologically unknown.
10. **Adaptive reuse:**all results concern30 already examined recordings and a few eligible events. Paired intervals do not make another analysis independent confirmation. No multiplicity-corrected superiority claim is made.

## Tests and unavailable artifacts

The pre-existing suite again runs162 tests with13 errors and15 skips. Error identities match the preceding experiment's log: missing original prepared final datasets, eight missing-demo-model subtest errors, the resulting demo UI control error, and missing generated OOF links/quality gates. These are not new audit failures; no historical receipts or tests are relaxed.

The new audit has16 numerical/unit invariants plus4 saved-result/metadata integration checks. Its initial unit run had one **test-fixture error**: an exactly-at-onset alarm was tested with no prior prediction opportunity, so the correct result was excluded/undefined rather than eligible-but-missed. The fixture was corrected to include an earlier39s opportunity while keeping the alarm at40s; the exclusion boundary assertion was retained. The corrected16 units and final20-test suite pass. Both initial and corrected logs are preserved. This was not a discovered defect in historical event matching.

The source incremental experiment's12 tests also pass. No extra scientific refits or speculative features were introduced. The only direct reporting repair is isolated policy metadata; added paired lead, failure and influence outputs answer questions the earlier report did not quantify fully.
