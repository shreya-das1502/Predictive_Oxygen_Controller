# Results: does PPG add predictive value beyond SpO2 history?

## Main conclusion

**The new comparison completed. Recent SpO2 history provides much stronger prediction than the tested PPG-only features. A robust incremental benefit from adding PPG was not established.** This is a more informative result than another classifier-only comparison.

Logistic regression with SpO2 history detects9/14 eligible onsets at the declared learned alarm policy; combined features detect10/14. The paired conditional recording-bootstrap sensitivity difference is **+7.14 percentage points,95% interval −40.0 to+57.14 points**. Under the prespecified sustained-event definition, both detect9/13; the net gain disappears. GaussianNB and ELM detect fewer events after PPG is added. No preferred model or clinical policy is selected from these held-out outcomes.

This result does not prove PPG contains no physiological information. It limits the claim to these features, models, target and alarm policy on30 previously examined recordings.

## Methods and completed scope

All new code/data/results are under `experiments/incremental_value_20261027`. Historical source/results/models were not changed. The actual checkout is `Predictive_Oxygen_Controller`, corresponding to the project named `Predictive_FiO2_ML` in the request.

The protocol was frozen before fitting in `config.json` and `outputs/protocol.json`. It records raw/fitting-source hashes, exact installed research versions, model parameters, seed42, candidate quantiles, input sets, target, policy and bootstrap plan. No dataset expansion, hyperparameter search, held-out winner selection or fitted calibration was performed.

**Primary target preserved:** binary indicator that the minimum valid observed SpO2 in `(t,t+30s]` is≤94%, including ongoing lows. Future values appear only in outcomes, not predictors. Complete valid horizons and maximum1.01s numeric gaps are unchanged.

**Matched cohort:** raw preprocessing reproduced the historical20,157 eligible timestamps/labels and4,587 positives. Seven rows without valid latest current SpO2 are excluded from every condition, leaving20,150 rows,4,587 positives and30 recordings. Current-normal analyses use16,050 rows with501 positives. The new matched PPG fit is therefore not claimed to be an exact reproduction of the older fit that had seven additional rows; the older experiment remains intact.

**Input ablation:**

- A: seven causal SpO2-history features—current value,10/30s means,10/30s OLS slopes,30s sample SD and current-reading age.
- B: the existing PPG10 pulse/rolling features, unchanged.
- C: their17-feature concatenation. This is not the historical context18 feature set, which is preserved but not another search dimension in this minimal study.

The existing logistic regression, GaussianNB and weighted-ridge ELM implementations/settings are reused. Each condition has30 outer held-out recording fits, each with three inner recording-grouped OOF fits. All imputation/scaling/model fitting and threshold selection use training records only.270 outer classifier fits and90 outer ridge fits completed and were saved with provenance. Patient identity/linkage is unavailable, so no patient-level generalization is claimed.

**Alarm policy:** one scalar cutoff per modality/model/outer fold maximizes training OOF one-to-one W30 detections subject to≤5 off-event starts/evaluable hour and≤5% activity; deterministic existing ties retained. This single operating point is a research assumption, not a clinically validated tolerance. Training feasibility does not guarantee held-out or per-record adherence.

**References:** persistence and a causal10s linear-trend future-min forecast are thresholded at≤94 without fitting. They are not budget-selected; their burden must be compared explicitly rather than treated as constrained competitors.

**Event evaluation:** existing current-normal exposure and chronological earliest-unused actual starts in `[onset−30,onset)`; no shared credit or clipped restart.22 raw low episodes include15 qualifying onsets.14 are evaluable at W30. Alternate truth retains episodes with≥3 consecutive low readings spanning≥2s, giving14 raw qualifying/13 eligible sustained events. Primary labels/fits/cutoffs are unchanged for that sensitivity analysis; off-event burden is recomputed against the sustained subset.

**Forecasting:** continuous minimum SpO2 over the same future horizon was chosen because it exactly aligns with the primary label and missingness rules. Compare persistence, causal trend and fixed-alpha1 ridge for each input modality. All forecasts are clipped to[0,100] under the frozen plan, with raw outputs retained; errors are in saturation percentage points.

**Uncertainty:**2,000 shared whole-record bootstrap multiplicity draws, seed20261027. Paired comparisons retain all within-record contributions and do not treat pulse windows as independent. Intervals are conditional on saved overlapping CV fits, with no model refits or patient-independence guarantee. Multiple family comparisons remain exploratory, without a multiplicity-adjusted superiority claim.

## Primary event results

Exposure is10,329.032s=2.869176 evaluable hours in every condition. “Active” below is time weighted; it is not the fraction of positive pulse decisions.

| Modality | Model | Detected / eligible | Off-event starts/h | Active time | Median matched lead(s) |
|---|---|---:|---:|---:|---:|
| SpO2 history | Logistic |9/14|3.485|1.358%|14.960|
| PPG10 | Logistic |0/14|0|0%|Undefined|
| Combined | Logistic |10/14|4.879|0.954%|13.844|
| SpO2 history | GaussianNB |10/14|4.182|6.483%|13.268|
| PPG10 | GaussianNB |0/14|0|0%|Undefined|
| Combined | GaussianNB |6/14|3.485|5.012%|20.280|
| SpO2 history | ELM |6/14|4.182|0.330%|9.956|
| PPG10 | ELM |0/14|0|0%|Undefined|
| Combined | ELM |1/14|1.394|0.102%|23.600|
| SpO2 history | Persistence, fixed |0/14|0|0%|Undefined|
| SpO2 history | Trend10, fixed |7/14|40.430|5.714%|15.792|

GaussianNB with history violates the pooled active cap, and its combined version slightly exceeds it as well. The naive trend has high burden. PPG-only achieves silence under the learned constraints, not useful warning. A conditional lead based on one detected event is not a good lead-time guarantee.

Even the pooled-feasible logistic conditions are not individually robust: history has3 records exceeding the start budget and4 exceeding the activity cap; combined has3 start-budget and3 activity-cap violations. Record29 contributes only6.264s of evaluable current-normal exposure and is100% active in both LR conditions. Short, interrupted normal periods can be hidden by pooled exposure from long uneventful recordings. Do not interpret pooled≤5% as every patient's≤5%.

## Incremental value, uncertainty and event concentration

| Combined minus history | Original sensitivity difference | Conditional95% interval |
|---|---:|---:|
| Logistic |+0.0714|[−0.4000,+0.5714]|
| GaussianNB |−0.2857|[−0.7053,+0.2500]|
| ELM |−0.3571|[−0.5714,−0.2000]|

The intervals describe this recording-resampling analysis, not independent confirmatory model selection. For zero-detection conditions an empirical[0,0] interval reflects resampling all-zero observed successes; it does not prove population sensitivity exactly zero.

The logistic result is **not** simply the same nine events plus one. Adding PPG:

- Gains record07/event367 (the isolated94% observation), record12/event210 and record26/event144.
- Loses record30/events72 and98.
- Produces a net one-event gain with changed detections across several records. It is not consistent across recordings.

Removing the isolated event from the alternate truth cancels the aggregate net gain: history9/13, combined9/13; sensitivity difference0, interval[−0.5000,+0.4659]. Combined false starts/h rises to5.577 under sustained truth, versus3.485 for history, so it now exceeds the pooled research budget. The remaining event swaps are still present; do not claim the only changed prediction was the isolated event.

GaussianNB loses five record26 events and gains one record30 event when PPG is added. ELM loses five previously detected events and gains none. These are inconsistent modality effects, not evidence that an ELM architecture should be preferred.

See `outputs/paired_event_changes.csv`, `event_audit_long.csv`, `event_per_recording.csv` and `paired_modality_differences.csv` for identities, opportunity counts, leads and uncertainty.

## Timestamp discrimination and calibration

These metrics describe ranking/classification, not event alarms. Full-target positives include ongoing lows.

| Model | History full-target AP | PPG full-target AP | Combined full-target AP | History current-normal AP | PPG current-normal AP | Combined current-normal AP |
|---|---:|---:|---:|---:|---:|---:|
| Logistic |0.9869|0.2402|0.9886|0.4960|0.0305|0.5402|
| GaussianNB |0.9771|0.1962|0.9720|0.3279|0.0238|0.2707|
| ELM |0.9584|0.2560|0.9177|0.3616|0.0315|0.2277|

Current-normal prevalence is501/16050≈0.031215. PPG-only onset-opportunity ranking is near/below that prevalence reference, while history is substantially more informative. LR combined ranking improves descriptively over history, which is a signal worth investigating—not proof of no information whatsoever—but its operational event gain is uncertain and sensitive to event definition. Pooled AP combines scores from different fits; within-record tables are also saved.

LR's full-target Brier is0.03120 for history and0.02384 combined; current-normal0.03818 versus0.02885. These are descriptive diagnostics of **uncalibrated** probability-shaped outputs. Balanced LR class weights/GNB equal priors affect risk interpretation. Reliability bins/plots are saved; no fitted calibration or calibrated-risk claim is made. ELM and fixed heuristic scores are excluded from Brier/reliability probability assessment.

## Continuous future-minimum forecasts

All models use identical held-out rows and the same future-minimum target. Lower errors are better; units are SpO2 percentage points.

| Forecaster | Full-target MAE | Full-target RMSE | Current-normal MAE | Current-normal RMSE |
|---|---:|---:|---:|---:|
| Persistence |0.3163|0.6473|0.3314|0.6545|
| Causal Trend10 |0.7197|1.5189|0.7153|1.4592|
| Ridge, history |0.3369|0.4775|0.3441|0.4753|
| Ridge, PPG |2.3173|2.7299|1.9106|2.2658|
| Ridge, combined |0.3312|0.4743|0.3407|0.4746|

Combined minus history, full-target MAE:−0.00566,95% conditional interval[−0.02157,+0.00984]; RMSE:−0.00326,[−0.01187,+0.00713]. Current-normal MAE difference−0.00337,[−0.01905,+0.01387]. None establishes a robust PPG increment. Combined improves per-record MAE in16/30 records and worsens14/30; current-normal14/26 improve and12/26 worsen.

Persistence has the lowest MAE among these forecasters, while history/combined ridge reduce RMSE compared with persistence. These losses reward different behavior; there is no universal forecasting winner. PPG-only ridge is substantially worse than history. The naive extrapolation is worse than persistence, so an untrained linear trend is not automatically a strong warning model.

Forecasting is informative as a diagnostic: it quantifies how much of the predictable future minimum is accessible from present/past SpO2, and exposes PPG-only limitations. It is not evidence that low MAE/RMSE produces useful event alarms. Most timestamps are not new events, and forecasts must be assessed separately from warning utility.

## Signal/event quality versus model failure

`raw_event_audit.csv` contains record/onset, minimum saturation, consecutive low count, observed span and nominal support, pre-event numeric trajectory, eligible/common/current-normal opportunities, pulse count, interval CV and amplitude median. `event_audit_long.csv` adds each model's original/sustained eligibility, detection and policy result while retaining isolated and excluded episodes.

No validated SQI or annotated beat reference is available. Pulse counts/variability are quality proxies, not measured beat-detection success. No isolated event or difficult recording was dropped from the primary definition. Model failure cannot be attributed confidently to bad PPG, absent physiology, detector mistakes or conditional-independence violations merely because scores are poor. Those are hypotheses for a separate annotated signal/clinical audit.

The original event definition also allows short current-normal opportunities after low periods, with no healthy-baseline washout. This is preserved rather than silently corrected, and limits a clinical “new deterioration” interpretation.

## Explicit answers to the requested research questions

1. **Does history explain most apparent performance?** In this tested comparison, most predictive discrimination is available from current/past SpO2, especially for the future-low target that includes ongoing lows. This is an empirical input-ablation result, not a causal variance decomposition or proof about every possible PPG method.
2. **Does PPG alone provide useful information?** No useful operational warning was demonstrated by PPG10 with these fixed models/policy; all detect0/14. Do not generalize that to “PPG can never predict desaturation.”
3. **Does adding PPG help history?** LR has a small descriptive AP/Brier gain and net one-event primary gain, but wide paired uncertainty and no net sustained-event gain. GNB/ELM worsen. Robust incremental warning is not established.
4. **Is the gain consistent?** No. LR gains three and loses two event matches across records; its net gain depends on the isolated event. Forecast MAE gains are split16/30 versus14/30 records.
5. **Which model has the best trade-off?** LR remains a defensible transparent research reference. History/combined LR have favorable pooled trade-offs relative to the tested GNB/trend conditions, but per-record violations, uncertainty and cohort reuse prevent selecting a validated overall winner. ELM is not supported as preferred.
6. **Does the result survive alternate events?** The broad history-versus-PPG distinction remains. LR's net incremental event gain does not:9/13 versus9/13, with combined sustained-truth false-start burden above the research budget.
7. **Is continuous forecasting more informative?** It adds a useful quantitative diagnostic and loss trade-off. It does not establish a superior alarm task or clinical endpoint, and offers no robust PPG increment here.
8. **What is supported/unsupported/uncertain?** Supported: matched code execution, stronger history predictors and no robust PPG increment under the declared analysis. Unsupported: clinical reliability, guaranteed warning duration, ELM superiority, patient-independent generalization or oxygen control. Uncertain: other causal PPG representations, calibrated temporal models and whether observed LR ranking gains replicate independently.

## Verification, tests and limitations

The separate verifier reconstructs all22 event conditions from saved score/decision streams and raw numerics without using the production event matcher, checks common labels/current readings, all270 fold memberships/candidate winners/checkpoint hashes, forecasting errors and whole-record intervals. `outputs/verification.json` records its actual outcome. Numerical agreement is not clinical validation.

New tests:12/12 pass before/after. Existing suite before:162 tests,13 errors,15 skips (48.906s); after:162 tests,13 errors,15 skips (48.052s). Its missing generated historical datasets/models/OOF links remain unchanged; do not call the whole repository green. All2,563 historical tracked files are verified unchanged. No expensive full historical restoration or broader search was performed.

Other limitations: six onset-bearing recordings and14 primary eligible events; unknown patient linkage; previously examined cohort; single-reading reference/monitor delay; conditional CV bootstrap with overlapping training sets and no refitting; no independent test cohort; research-only alarm tolerances; future-dependent retrospective evaluability; no prospective workflow validation. Low pooled rates over2.869 evaluable hours do not establish long-term patient-level burden stability.

## Recommended next steps without dataset expansion

1. Present this matched information-ablation study as the refresh result, alongside the preserved original negative PPG experiment.
2. Review the15 current transitions and representative waveform excerpts with appropriate clinical/signal expertise. Annotate questionable events/peak errors without changing the saved primary analysis.
3. For a separately frozen exploratory extension, investigate one training-only calibration/threshold-transfer or reduced-feature hypothesis—not a classifier sweep. Evaluate minimum useful lead explicitly rather than using a30s matching window as a warning guarantee.
4. Keep current SpO2 history as a mandatory comparator and preserve per-record burden/activity reporting. Any new result on these same recordings remains exploratory; it cannot create independent confirmation.
5. Do not choose a deployment model, tune an oxygen actuator or claim patient benefit from these outcomes. A reproducible finding that PPG adds no established operational benefit is a valid, bounded research conclusion.

## Result artifacts

Machine-readable summary tables and all predictions are indexed in README. Figures are PNG/PDF under `outputs/figures`: full/current-normal PR curves, detection/burden/activity, matched lead distribution, reliability diagnostics, forecasting errors and event-level error heatmap. They derive from saved held-out outputs, not a hand-selected favorable run.
