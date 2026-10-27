# Final scientific audit and targeted-improvement decision

## Verdict

The reported results are numerically reproducible. **PPG did not show robust incremental warning value beyond SpO2 history in this experiment.** No metric-changing evaluation defect or evidence of held-out-record leakage was found. No scientific model was refitted, no cutoff was retuned and no historical file was overwritten.

One reporting omission was verified and repaired in the new audit outputs:11 excluded sensitivity rows lacked the known policy name. This does not change any scientific result. The useful improvements here are complete event-level diagnostics, paired common-event leads, explicit rate denominators and recording-influence analysis—not another classifier search.

## 1. Audit scope, data and causal validity

Source: `experiments/incremental_value_20261027/`, historical repository commit `ac5f0c02ba559381ff304ed6a852224bb0047dee`. All2,563 original tracked files and all1,784 files of the previously untracked incremental experiment were fingerprinted, totaling4,347 pre-existing files. The saved source experiment's1,783-file completion inventory and fitting input hashes were checked before analysis.

The same30 recordings are used: IDs01–30, typically480s each,125Hz waveform sampling and1Hz numerics, totaling14,400s=4 source hours. The rounded export clock agrees with the125Hz row clock. Missing numeric SpO2 observations are recorded in01(13),05(6),19(102),25(1); raw PPG contains no missing values. No recordings were downloaded, added or invented.

The primary target remains minimum valid observed SpO2≤94% in `(t,t+30s]`, including ongoing lows. The common20,150 timestamps retain4,587 positives; the seven additional historical target-valid rows without valid current SpO2 remain consistently excluded across modalities. Current-normal evaluation contains16,050 rows with501 positives. All future minima/labels, seven numeric history features and8ms pulse-confirmation timing were independently recomputed and agreed. Future saturation is an outcome, not a predictor.

Source inspection confirms forward-only filtering, past pulse histories, recording-separated fitting and training-only cutoffs. Every outer model's imputer medians/scaler mean/variance matched the permitted training matrix; maximum checked scaler-mean error was0. All270 model/modality/held-out-record OOF row/label alignments, grouped memberships, matching inner splits, saved feasible-grid winners and selected training-policy event/start/activity statistics agreed.

This scope does not independently refit every inner model or recompute every unselected grid candidate. Inner transform objects are not retained; their training-only behavior is supported by inspected source and partition/causal tests. Recording grouping is not verified patient grouping. No available identifiers establish unique patient independence, and repeated human examination of this cohort remains a limitation.

## 2. Reproduced event counts and alarm burden

Raw numerics reconstruct22 low episodes and15 qualifying normal-to-low onsets. The onset at record26/time22s is excluded from W30 sensitivity because a complete30s raw lead-in is unavailable. Thus14 original onsets are eligible. Requiring≥3 consecutive low readings spanning≥2s removes the isolated record07/time367s event, leaving13 eligible sustained onsets. Primary fits/cutoffs/labels are unchanged for this sensitivity analysis.

All22 condition/definition summaries and660 per-record event contributions independently reproduce, including actual new-start matching, no duplicate credit, excluded events, exposure, activity, false starts, leads and unmatched/redundant starts.

| Logistic input | Original detected | Sustained detected | Original off-event starts | Starts/evaluable hour | Original active fraction |
|---|---:|---:|---:|---:|---:|
| SpO2 history |9/14|9/13|10|3.4853|1.3578%|
| PPG10 |0/14|0/13|0|0|0%|
| Combined |10/14|9/13|14|4.8795|0.9543%|

**Denominators:**10,329.032s=2.869176 evaluable current-normal hours, not4 source hours. `Source_Duration_Normalized_Starts_Per_Hour` additionally reports10/4=2.5 for history LR and14/4=3.5 for combined LR. These lower values normalize only the starts observed in evaluable periods; they are not an estimate of false alarms over fully monitored4h. Per-record source duration, evaluable exposure and both normalizations are saved explicitly.

## 3. Were the alarm constraints met?

The source already supports a matched comparison at a predefined research budget:≤5 off-event starts/evaluable hour and≤5% activity, with cutoffs selected entirely in training. A new budget sweep was therefore unnecessary. These are research sensitivity assumptions, not clinical standards.

- Original history LR and combined LR meet both **pooled held-out** constraints.
- History LR exceeds the start budget on3 exposed records and the active cap on4; combined LR exceeds them on3 and3, respectively.
- Under sustained truth, combined LR has5.5765 starts/evaluable hour and fails the pooled start budget. History LR remains3.4853/h.
- History/combined ELM meet pooled constraints but have record-level violations and weaker event detection. Both GaussianNB modalities with saturation history exceed the pooled active cap (6.483% history,5.012% combined).
- Fixed Trend10 detects7/14 but produces40.43 starts/evaluable hour; it is not a budget-selected reference.
- Silent PPG conditions and persistence satisfy the numerical bounds but detect no events.

**No event-detecting condition meets both bounds on every exposed recording.** This is a robustness limitation, not a selector bug: the declared training objective is pooled feasibility, not guaranteed individual-record feasibility. Record29 has only6.264s of eligible current-normal exposure; both LR variants are100% active there. Pooled summaries can conceal short high-activity normal intervals.

## 4. Event-level failure investigation

`event_failure_analysis.csv` contains484 rows: all22 raw episodes,11 conditions and both event definitions. It retains detected, missed and excluded episodes. It includes onset/duration/severity, current SpO2 and10/30s trends at the last opportunity, PPG amplitude/interval/rate features and5/15/30s diagnostic trends, waveform/pulse quality proxies, scores/cutoffs/margins, positive windows, actual starts, matched leads, exclusion reasons, policy and a score trace.

For eligible original events:

| Logistic modality | Detected | Missed | Immediate algorithmic reason for each miss |
|---|---:|---:|---|
| History |9|5|Trained cutoff never crossed within eligible warning window|
| PPG |0|14|Trained cutoff never crossed within eligible warning window|
| Combined |10|4|Trained cutoff never crossed within eligible warning window|

The four combined misses are record26/events122 and241 and record30/events72 and98. Their observed low spans are9,21,5 and13s. They are all sustained under the frozen sensitivity definition; misses are not explained simply by isolated short readings. Maximum combined score-minus-cutoff margins are approximately−0.1593,−0.2322,−0.0238 and−0.4246. A negative margin identifies an unmet decision threshold, not its physiological cause.

Adding PPG gains matches at07:367,12:210 and26:144, but loses30:72 and30:98. The net gain is1; removing the isolated07 event cancels that net gain. The underlying swaps across records remain—this is not just the same nine detections plus one.

The combined missed/detected groups have similar median pre-event pulse amplitude proxies (about0.3060 versus0.3053 exported units). Interval CV and saturation trends vary; no small descriptive comparison establishes a causal quality or physiological explanation. All raw PPG pre-windows are finite where available, but finite signals can still contain motion, gain/contact effects or missed/extra peaks. No annotated beat reference or validated SQI is available, so “pulse detection success rate” and the physiological cause of each miss remain unknown rather than fabricated.

Fifteen retrospective event panels show raw PPG, observed SpO2 and held-out LR score margins, including the excluded22s onset. These facilitate inspection; they do not validate signal quality or provide arterial oxygen truth.

## 5. Incremental value and lead-time uncertainty

The paired combined-minus-history LR sensitivity difference is+0.07143,95% conditional whole-record percentile interval[−0.4000,+0.57143]. Sustained sensitivity difference is0,[−0.5000,+0.46591]. These intervals reproduce the original analysis and include no improvement. GaussianNB and ELM detect fewer events after PPG is added; no family-independent PPG gain is established.

Lead times are conditional on matched events. Original individual medians are14.960s(history) and13.844s(combined), but those involve different detected populations. On the **seven events both detect**, median paired lead difference is0s,95% conditional interval[−4.000,+10.064]s. These seven events occur in only four recording clusters. PPG-versus-history common-event leads are undefined because PPG detects none; they are not set to zero.

A diagnostic using original matches finds **both history and combined LR provide≥10s observed lead for6/14 eligible events**. At≥15s the counts are4/14 and5/14; at≥20s2/14 and3/14. This is a descriptive coverage curve with misses retained in the denominator, not a newly optimized policy or validated minimum-warning target.

The shared2,000 recording-multiplicity draws exactly reproduce seed20261027. Each recording's full event/exposure/forecast contributions are retained; overlapping pulse windows are not independently bootstrapped. Intervals are conditional on saved overlapping CV fits and omit refitting/adaptive-selection uncertainty. Unknown patient linkage and few event-bearing clusters limit power. No statistical-significance or independent-confirmation claim is made. Zero-detection empirical[0,0] bootstrap intervals do not prove population sensitivity exactly zero.

## 6. Continuous forecasting and recording influence

All five forecasters predict the same observed minimum over `(t,t+30s]`, with the same timestamps/targets and original fixed[0,100] clipping rule. No timing-delay correction was invented. Errors are saturation percentage points, pooled over pulse decisions; per-record errors are also saved.

| Forecaster | Full-target MAE | Full-target RMSE | Current-normal MAE | Current-normal RMSE |
|---|---:|---:|---:|---:|
| Persistence |0.316278|0.647308|0.331402|0.654484|
| Causal Trend10 |0.719746|1.518856|0.715293|1.459224|
| Ridge history |0.336895|0.477528|0.344064|0.475263|
| Ridge PPG |2.317296|2.729904|1.910584|2.265759|
| Ridge combined |0.331231|0.474268|0.340698|0.474601|

The full-target combined-minus-history MAE difference is **−0.005664**, not a mismatched-cohort artifact: both have20,150 identical timestamp/target pairs. Its95% conditional interval is[−0.021570,+0.009835]. RMSE difference−0.003260 also includes zero in its interval. No practical-importance threshold or clinical benefit is established.

Combined MAE improves16/30 records and worsens14/30. The median per-record difference is−0.003855. Removing any one record's saved contributions leaves full-target pooled differences between−0.008786 and−0.001895: **no single removal reverses that full-cohort sign**. Record08 has a strong favorable signed contribution; removing it shrinks the gain to−0.001895. Thus the tiny point gain is heterogeneous and influenced by a subset, but should not be falsely described as entirely caused by one record.

In current-normal rows,14/26 improve and12/26 worsen. Pooled difference is−0.003366. Removing record27 changes it to+0.000381, favoring history; its sign is more fragile in this clinically relevant subset. These are contribution-removal diagnostics without refitting, not new cross-validation trials.

Persistence has smaller MAE than combined ridge, while ridge reduces RMSE. Different losses emphasize different error behavior. Continuous accuracy neither proves event detection nor establishes useful minimum warning. Displayed SpO2 can lag physiology through transit/averaging/processing, and no independent reference establishes the size of that delay here.

## 7. Corrections and targeted-improvement decision

**Fixed in isolated audit outputs:**11 missing policy metadata entries; completed per-event failure diagnostics; common-event paired lead intervals; recording-level paired forecast/event differences; explicit source/evaluable denominators; leave-one-record influence and minimum-observed-lead coverage.

**Deliberately unchanged:**all historical sources/artifacts,30-record dataset, primary/sustained definitions, feature sets, model parameters, held-out folds, learned/fixed policies, cutoffs, forecast target and clipping. No new scientific models, features, SQI exclusion rules, budget search or calibration fit were run. The existing persistence/trend references were retained, including their poor results.

No verified metric defect justified refitting. The failure diagnostics do not supply validated quality labels or independent evidence sufficient to choose a new feature/regularization policy. Optimizing against these repeatedly inspected14 events would turn audit discoveries into further adaptive tuning.

**Single most useful next experiment within the same dataset:** a blinded reference-quality study of all15 qualifying observed onsets and prespecified matched non-event windows. Reviewers should be blinded to model scores; PPG beat annotators can additionally be blinded to event status/SpO2. Separately assess observed saturation episode validity/duration and peak/trough detection accuracy. Do not claim this establishes true arterial hypoxemia. If annotations reveal a systematic causal preprocessing/SQI defect, test one prespecified correction in a new grouped experiment with event coverage retained. If no such defect is found, keep the negative incremental-value conclusion rather than launching another classifier sweep.

## 8. Tests, reproducibility and preservation

- New audit:20/20 tests pass—16 numerical/unit invariants plus4 saved-result/metadata checks; no skips in that suite.
- Existing incremental experiment:12/12 targeted tests pass.
- Repository suite:162 tests,13 errors,15 skips,47.517s; its error identities match the previous missing-artifact failures. Missing original prepared datasets/demo models/OOF links are documented, not manufactured or fixed by altering receipts.
- The first audit-unit attempt had one fixture error about an at-onset alarm without any prior opportunity. The fixture was corrected while retaining the exact-onset exclusion assertion; initial/corrected logs are preserved. No historical matching code was changed.
- All4,347 pre-existing files, including the whole incremental experiment, are fingerprint-verified unchanged. New work stays in this timestamped audit directory. Scientific model refits:0.

The required machine-readable tables, source, tests, command records and standalone figures are indexed in README. `reproduced_metrics.json` is a numerical consistency record, not a certificate of clinical reliability or oxygen-delivery control readiness.
