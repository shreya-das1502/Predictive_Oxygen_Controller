# Reliability audit of saved predictive oxygen evaluations

This audit independently recomputed events and alarms from existing held-out timestamp predictions and raw numeric measurements. It did not refit models, change hyperparameters or thresholds, exclude difficult recordings, or overwrite existing results. The original reports remain historical evidence. The new analysis is in [results/reliability/final](../results/reliability/final/); its [manifest](../results/reliability/final/manifest.json) fingerprints inputs, code, outputs and the explicit analysis protocol.

## Findings and the 29.7-second claim

**The saved 29.704-second median is numerically reproducible as the earliest positive timestamp inside a clipped 30-second pre-onset window. It is not evidence that a new alarm first began 29.7 seconds before desaturation.** The archived `src/reporting.py` functions `observed_events` and `event_analysis` in [legacy_sources.zip](../archive/legacy_sources.zip) produce the original [MODEL_EVALUATION](MODEL_EVALUATION.md) event results. They require dense eligible pulse coverage of the whole 30-second window (first/last within 2 seconds, no pulse gap >2 seconds), then take `onset - min(positive timestamp in window)`. They do not constrain those event-analysis timestamps to current-normal saturation. Thus the historical “genuine pre-onset” event section is too strong: its severe-subset windows can include ongoing lows. The separate current-normal timestamp subsection does not imply the event subsection uses that restriction.

The original 22 observed low episodes produce 13 eligible events under the archived dense-coverage rule. All four models detect 13/13 with median 29.704 s; the three episodes containing an actual 91/92 reading have median 29.672 s. The independent row-by-row calculation reproduces all 88 original model/event rows with zero status, detection or lead-time discrepancies. See [original_lead_time_recomputation.csv](../results/reliability/final/original_lead_time_recomputation.csv), including the selected timestamp and actual containing alarm-run start.

For example, the original ELM alarm selected for record 12's onset at 210 s is the positive at **180.328 s** (29.672 s nominal lead); its continuous run began at **1.360 s**. For record 17's onset at 250 s, the selected positive is **220.312 s**, during the initial ongoing-low episode ending at 239 s; its run began at 205.248 s. For record 29's onset at 368 s, the selected positive at 338.656 s is also during an ongoing low, before recovery after 360 s. These are real timestamps, but their interpretation as useful new warnings is unsupported.

On the original **same 13 eligible events**, [baseline recomputation](../results/reliability/original_baseline_comparison.csv) gives always-alert 13/13 with the **same 29.704 s median**, never-alert 0/13, periodic 13/13 with median 26.952 s, and pulse-rate >100 bpm 12/13 with median 25.708 s among detections. All use the same original unrestricted pulse windows; these are diagnostic references for that historical metric, not meaningful new-warning claims. The executable [reference-check script](../results/reliability/original_reference_check.py) saves every baseline event match.

The newer [target_reporting.evaluate_events](../src/target_reporting.py#L25) instead requires a complete valid raw 30-second lead-in and at least one current-normal prediction opportunity. `future_low` explicitly filters current SpO₂ >94%; onset predictions already have that eligibility. This changes the denominator to 14 events, not a model improvement. All four models detect 14/14 by reusable positive timestamps; ELM onset median is 29.376 s, logistic 29.684 s. The severe-subset median is 9.784 s for all four. Those saved event rows also independently reproduce without discrepancies. The separately executed [production correction check](../results/reliability/correction_verification.json) confirms zero changed event rows in either real run after the boundary fix. Both calculations allow one long alarm, or even the same positive timestamp in overlapping windows, to credit multiple nearby events. [Shared positive timestamps](../results/reliability/shared_positive_timestamps.csv) enumerates 295 reusable positive timestamp entries across the two target/model analyses (for example, record 26 at 114.920 s can credit onsets at both 122 and 144 s); sharing is a documented historical metric property, not one-to-one event precision.

## Source trace and smallest correction

| Calculation | Source and boundary |
| --- | --- |
| Binary future-low target | [labeling.future_labels](../src/labeling.py#L6): strictly `(t,t+30]`, complete valid future data, numeric gaps ≤1.01 s, ≤94% positive. Ongoing lows remain included. |
| Raw observed event | [labeling.desaturation_events](../src/labeling.py#L27): first valid ≤94 reading after an immediately preceding valid >94 reading within 1.01 s; missing/invalid/gapped data split low episodes; initial lows are left censored. One low sample suffices. |
| Current-normal onset eligibility | [labeling.onset_labels](../src/labeling.py#L54): latest current observation age ≤1.01 s and >94%, plus full valid future target window. |
| Existing event eligibility and lead | [target_reporting.evaluate_events](../src/target_reporting.py#L25), especially lines 41–57: raw lead-in, current-normal subset, first positive within the window; duplicate event credit permitted. |
| Existing alarm run and exposure | [evaluation.alarm_statistics](../src/evaluation.py#L69): contiguous positive decisions, split by a negative decision or gap >2 s; false only when the entire run contains no positive 30-second target. Exposure sums all adjacent eligible pulse intervals ≤2 s. |
| Independent implementation | [reliability_evaluation](../src/reliability_evaluation.py): `raw_events`, `independent_historical`, `burden`, `eligible_events`, `episodes`, `match_events` and `off_event_alarms`. It does not call the production label/event/metric routines to recompute their outputs. |

**Verified defect:** the production raw-lead-in validator checked only gaps between retained numeric samples. A missing block at the left boundary could be accepted if all retained interior samples were contiguous. The one-line correction in `target_reporting.py:46` checks differences across `[window_start, retained timestamps, onset]`. A synthetic [regression test](../tests/test_reliability.py) demonstrates that the previous expression accepts an incomplete boundary window and the corrected expression excludes it. No saved report was regenerated or overwritten. The correction does not alter eligible events on these actual saved runs; the new strict validation and independent historical calculations agree here. Existing future-label boundary checks already included endpoints and were not changed.

## Independent timestamp and exposure verification

Full-target quantities below reproduce the saved comparisons exactly, without calling `alarm_statistics` or `metrics`. FP is the number of negative-label timestamps with a positive decision. Alert fraction is positive decisions divided by all eligible pulse decisions. Exposure is the sum of adjacent within-record time differences ≤2 s, with no duration added before the first or after the last decision. It is not the full 30×8 minutes of source recording time. A singleton contributes zero exposure; rates with zero exposure are undefined. The independent evaluator rejects duplicate/invalid pulse timestamps and binary-label violations.

| Target | Model | FP_Timestamps | False_Episodes_Target_Run | Exposure_s | Alert_Fraction | False_Episodes_Per_Hour |
| --- | --- | --- | --- | --- | --- | --- |
| future_low | ELM | 14093 | 513 | 13127.0320 | 0.9187 | 140.6868 |
| future_low | GaussianNB | 15458 | 51 | 13127.0320 | 0.9904 | 13.9864 |
| future_low | Logistic | 15192 | 101 | 13127.0320 | 0.9658 | 27.6986 |
| future_low | Tree | 15021 | 104 | 13127.0320 | 0.9489 | 28.5213 |
| new_onset | ELM | 13979 | 525 | 10331.0160 | 0.8988 | 182.9443 |
| new_onset | GaussianNB | 13118 | 174 | 10331.0160 | 0.8454 | 60.6330 |
| new_onset | Logistic | 15299 | 62 | 10331.0160 | 0.9809 | 21.6048 |
| new_onset | Tree | 15549 | 28 | 10331.0160 | 1.0000 | 9.7570 |

All model FP counts, alert fractions, false target-run counts and exposure totals matched saved values. This does not validate the clinical meaning of the inherited false-alarm definition: a long positive run with even one positive target is treated as a true run, regardless of its many false-positive timestamps. Always-alert can therefore have very few “false episodes.” Negative-only recordings, positive-containing recordings and all-positive/brief eligible segments remain included; no poor-outcome recordings were removed.

**A real denominator discrepancy:** onset ELM's inherited exposure is 10,331.016 s. Enforcing complete raw coverage and current-normal status throughout each inter-pulse edge gives **10,329.032 s**, a 1.984 s reduction. The record-7 edge from 366.488 to 368.472 s bridges the observed 94% reading at 367 s between eligible normal endpoints; the strict rule splits it. ELM's false target-run count becomes 526 rather than 525 and its corresponding rate 183.328 rather than 182.944/hour. The new event analysis uses the strict 10,329.032 s (2.869176 h) for every model and baseline in both target comparisons. This is an analysis denominator correction, not a model improvement. The affected edge is record 7, 366.488–368.472 s, spanning the 94% observation at 367 s. Both conventions remain reported in [timestamp_burden.csv](../results/reliability/final/timestamp_burden.csv).

Time-weighted alert fraction uses left-held decisions over those permitted edges. It differs from pulse-count alert fraction when pulse intervals vary; it is an explicit replay convention, not measured continuous monitor duty. Neither convention counts unobserved/invalid time. The original full future-low population remains 20,157 rows; the new event-focused current-normal population is 16,050 rows. Onset training and future-low training are kept separate; future-low event analysis is a clearly identified current-normal subset of its held-out predictions.

## Fixed pre-onset windows and matching protocol

The four windows are **[onset−W,onset)** for W=5,10,15,30 seconds: an alert at the left edge counts, one exactly at onset does not. These are nested look-back windows, not requirements for at least W seconds of warning. The prediction target horizon remains 30 seconds; no model is retrained for the smaller windows.

Eligibility is determined once from raw ground truth and the common eligible prediction timestamps, never from whether the model alerts: a qualifying crossing, complete valid raw coverage of that W-second lead-in including boundary gaps, and at least one eligible current-normal prediction opportunity. Gaps/invalid numerics break coverage. Saved eligible predictions also require their original complete 30-second future target. There are 15 eligible events at 5/10/15 s and 14 at 30 s; the early record-26 onset at 22 s has no complete 30 s raw lead-in. Initial/unknown-prior low episodes are excluded rather than relabeled negative. Identical windows, events, rows, exposure and rules apply to all four learned models and four prespecified simple baselines. Every event status/opportunity count, including exclusions, is saved.

Primary metric: **`episode_start_one_to_one`**. Build positive runs over the whole eligible recording stream, splitting at a negative prediction, >2 s gap or invalid/current-low inter-pulse edge. Do not reset an alarm at a window edge. In chronological event order, assign the earliest unused alarm start in that window. Each episode and event receives at most one match; later redundant alarm starts cannot receive another credit for the same event. An always-positive stream can still restart after a legitimate eligibility/signal gap; that is not learned prediction. This greedy earliest-event policy is explicit and deterministic, not an optimized maximum matching. Its trade-offs near overlapping events are part of the metric definition.

Secondary diagnostic: **`any_positive_timestamp_reusable`** permits positive timestamps and alarm episodes to credit multiple events, as the historical method did. Its higher sensitivity and boundary-clipped lead times must not be described as independent new warnings. It is labeled separately throughout.

For false alarms, **off-event alarm starts** have no observed qualifying onset in `(alarm_start,alarm_start+W]`. All observed qualifying onsets, including those excluded from sensitivity, are checked so a warning near an excluded real event is not falsely declared off-event. This count is divided by the full strict evaluable current-normal exposure, including negative-only recordings, rather than by positive event windows. It does not require a one-to-one credit: redundant starts near one real event are separately reflected by the larger unmatched-alarm count. `Unmatched_Alarms_Per_Hour_All_Events` includes both off-event and redundant/uncredited starts and is not synonymous with false alarms. These are threshold-crossing reference metrics, not clinically adjudicated nuisance alarms; sustained false-positive time still requires alert fractions to interpret.

Baselines: always alert; never alert; deterministic 1-second-on/9-second-off pulses with phase anchored at recording time zero; and PPG pulse rate >100 bpm. They have no fitted/test-selected parameters. None uses current/future SpO₂ as a predictor. The current-normal gate is common eligibility, not a predictive input. The periodic phase and rate cutoff were fixed before this comparison and were not varied to improve results.

### future_low model / baseline results

| Model | Window_s | Eligible_Events | Detected | Sensitivity | Off_Event_False_Alarms_Per_Hour | Lead_Median_s |
| --- | --- | --- | --- | --- | --- | --- |
| Always_Alert | 5 | 15 | 1 | 0.0667 | 14.9869 | 2.3680 |
| Always_Alert | 10 | 15 | 3 | 0.2000 | 14.2898 | 6.3680 |
| Always_Alert | 15 | 15 | 5 | 0.3333 | 13.5928 | 9.7840 |
| Always_Alert | 30 | 14 | 8 | 0.5714 | 12.1986 | 13.2680 |
| ELM | 5 | 15 | 4 | 0.2667 | 193.0868 | 2.6040 |
| ELM | 10 | 15 | 8 | 0.5333 | 190.6471 | 6.9080 |
| ELM | 15 | 15 | 9 | 0.6000 | 187.8588 | 11.5760 |
| ELM | 30 | 14 | 12 | 0.8571 | 182.2823 | 18.2000 |
| GaussianNB | 5 | 15 | 3 | 0.2000 | 26.4884 | 2.3680 |
| GaussianNB | 10 | 15 | 5 | 0.3333 | 25.0943 | 6.3680 |
| GaussianNB | 15 | 15 | 7 | 0.4667 | 23.0031 | 11.5760 |
| GaussianNB | 30 | 14 | 10 | 0.7143 | 20.9119 | 14.3800 |
| Logistic | 5 | 15 | 3 | 0.2000 | 41.4753 | 2.3680 |
| Logistic | 10 | 15 | 5 | 0.3333 | 40.4297 | 5.0880 |
| Logistic | 15 | 15 | 6 | 0.4000 | 39.0356 | 10.6800 |
| Logistic | 30 | 14 | 9 | 0.6429 | 37.6415 | 11.9280 |
| Never_Alert | 5 | 15 | 0 | 0.0000 | 0.0000 | undefined |
| Never_Alert | 10 | 15 | 0 | 0.0000 | 0.0000 | undefined |
| Never_Alert | 15 | 15 | 0 | 0.0000 | 0.0000 | undefined |
| Never_Alert | 30 | 14 | 0 | 0.0000 | 0.0000 | undefined |
| Periodic_10s_1s | 5 | 15 | 6 | 0.4000 | 356.8969 | 1.6680 |
| Periodic_10s_1s | 10 | 15 | 12 | 0.8000 | 354.8058 | 5.1120 |
| Periodic_10s_1s | 15 | 15 | 13 | 0.8667 | 352.7146 | 9.9600 |
| Periodic_10s_1s | 30 | 14 | 13 | 0.9286 | 348.5322 | 23.6000 |
| Pulse_Rate_GT100 | 5 | 15 | 7 | 0.4667 | 363.1705 | 1.8400 |
| Pulse_Rate_GT100 | 10 | 15 | 11 | 0.7333 | 358.6396 | 7.3040 |
| Pulse_Rate_GT100 | 15 | 15 | 12 | 0.8000 | 353.7602 | 10.9600 |
| Pulse_Rate_GT100 | 30 | 14 | 13 | 0.9286 | 346.4410 | 18.1280 |
| Tree | 5 | 15 | 1 | 0.0667 | 42.5209 | 2.3680 |
| Tree | 10 | 15 | 3 | 0.2000 | 41.8239 | 6.3680 |
| Tree | 15 | 15 | 5 | 0.3333 | 41.1268 | 9.7840 |
| Tree | 30 | 14 | 8 | 0.5714 | 39.7327 | 13.2680 |

### new_onset model / baseline results

| Model | Window_s | Eligible_Events | Detected | Sensitivity | Off_Event_False_Alarms_Per_Hour | Lead_Median_s |
| --- | --- | --- | --- | --- | --- | --- |
| Always_Alert | 5 | 15 | 1 | 0.0667 | 14.9869 | 2.3680 |
| Always_Alert | 10 | 15 | 3 | 0.2000 | 14.2898 | 6.3680 |
| Always_Alert | 15 | 15 | 5 | 0.3333 | 13.5928 | 9.7840 |
| Always_Alert | 30 | 14 | 8 | 0.5714 | 12.1986 | 13.2680 |
| ELM | 5 | 15 | 6 | 0.4000 | 197.6177 | 2.6760 |
| ELM | 10 | 15 | 10 | 0.6667 | 191.6927 | 7.5360 |
| ELM | 15 | 15 | 11 | 0.7333 | 189.2530 | 9.7840 |
| ELM | 30 | 14 | 13 | 0.9286 | 185.4191 | 19.5600 |
| GaussianNB | 5 | 15 | 1 | 0.0667 | 65.8726 | 2.3680 |
| GaussianNB | 10 | 15 | 3 | 0.2000 | 65.1755 | 6.3680 |
| GaussianNB | 15 | 15 | 6 | 0.4000 | 64.1299 | 10.6800 |
| GaussianNB | 30 | 14 | 9 | 0.6429 | 62.7358 | 12.8640 |
| Logistic | 5 | 15 | 3 | 0.2000 | 27.5340 | 2.3680 |
| Logistic | 10 | 15 | 5 | 0.3333 | 26.4884 | 5.0880 |
| Logistic | 15 | 15 | 6 | 0.4000 | 25.4428 | 10.6800 |
| Logistic | 30 | 14 | 9 | 0.6429 | 24.0487 | 13.2320 |
| Never_Alert | 5 | 15 | 0 | 0.0000 | 0.0000 | undefined |
| Never_Alert | 10 | 15 | 0 | 0.0000 | 0.0000 | undefined |
| Never_Alert | 15 | 15 | 0 | 0.0000 | 0.0000 | undefined |
| Never_Alert | 30 | 14 | 0 | 0.0000 | 0.0000 | undefined |
| Periodic_10s_1s | 5 | 15 | 6 | 0.4000 | 356.8969 | 1.6680 |
| Periodic_10s_1s | 10 | 15 | 12 | 0.8000 | 354.8058 | 5.1120 |
| Periodic_10s_1s | 15 | 15 | 13 | 0.8667 | 352.7146 | 9.9600 |
| Periodic_10s_1s | 30 | 14 | 13 | 0.9286 | 348.5322 | 23.6000 |
| Pulse_Rate_GT100 | 5 | 15 | 7 | 0.4667 | 363.1705 | 1.8400 |
| Pulse_Rate_GT100 | 10 | 15 | 11 | 0.7333 | 358.6396 | 7.3040 |
| Pulse_Rate_GT100 | 15 | 15 | 12 | 0.8000 | 353.7602 | 10.9600 |
| Pulse_Rate_GT100 | 30 | 14 | 13 | 0.9286 | 346.4410 | 18.1280 |
| Tree | 5 | 15 | 1 | 0.0667 | 14.9869 | 2.3680 |
| Tree | 10 | 15 | 3 | 0.2000 | 14.2898 | 6.3680 |
| Tree | 15 | 15 | 5 | 0.3333 | 13.5928 | 9.7840 |
| Tree | 30 | 14 | 8 | 0.5714 | 12.1986 | 13.2680 |

Lead distributions (minimum, Q25, median, Q75, maximum) and severe-subset results for all models/windows are in [window_summary.csv](../results/reliability/final/window_summary.csv); the individual matched timestamps, alarm IDs and leads are in [window_event_matches.csv](../results/reliability/final/window_event_matches.csv). Quantiles include detected events only; misses have undefined lead, not zero. Small denominators and conditioning on detection limit these summaries. An apparent longer median can arise by missing difficult late-detectable events.

At 30 s, the onset tree is identical to always-alert on these rows. The ELM's additional event detections coincide with many more alarm starts, rather than demonstrated improvement at matched false-alarm burden. No model superiority or clinically useful 29.7-second warning is established. The all-alert reusable-timestamp reference attains 100% sensitivity on all eligible events by construction.

## Causality, threshold and grouping audit

- [preprocessing.WaveformState](../src/preprocessing.py) uses causal `sosfilt`; the local PPG maximum is confirmed at the next sample and stamped with the actual availability time. `peak_time_s` precedes `time_s`; labels and matching use `time_s`. No retrospective `filtfilt` or global peak search appears in the active pipeline. Sensor averaging/transport delay and physiological delay remain unknown.
- [features.FeatureState](../src/features.py) appends only the current pulse and prior pulse history (at most ten); rolling means/stds and changes exclude future pulses. Histories reset between recordings. [dataset](../src/dataset.py) constructs full pulse history before eligibility exclusions. The feature list is the ten PPG amplitude/interval/rate/slope/history measurements. Neither current/future SpO₂ nor labels enter the fitted feature matrix; current SpO₂ supplies target eligibility only. Existing prefix and replay tests verify causal availability.
- [model.FittedModel.fit](../src/model.py) fits median imputation and scaling within each permitted training fold. [evaluation.grouped_oof](../src/evaluation.py#L37) receives only outer-training arrays; GroupKFold excludes each validation recording; the test recording is inaccessible. The largest threshold meeting the prespecified 92% training-OOF timestamp recall is selected using complete tie blocks. No threshold was changed for this audit.
- The independent [boundary-check script](../results/reliability/boundary_checks.py) verifies all recording split disjointness, each training sample appears exactly once in each permitted outer/model OOF set, no test-record OOF rows, no SpO₂ features, and every saved operating point equals the score of the required kth highest positive OOF sample. It also verifies held-out decisions are exactly `Score >= Threshold`. There are **120 future-low** and **104 onset** outer/model combinations, with **2,338,212** and **1,605,000** OOF rows respectively. [boundary_checks.json](../results/reliability/boundary_checks.json) records actual checks. Historical provenance cannot prove arbitrary unrecorded behavior, so code inspection and train-only transform regression tests supplement these saved-array checks.
- Group labels are BIDMC recording numbers parsed from filenames; they are not verified patient identifiers. The local data are paired Signals/Numerics CSVs without a patient linkage table. Thirty source recordings and 26 onset-eligible recording groups do **not** establish 30/26 unique patients. Results are recording-separated retrospective CV, **not verified patient-independent validation** and not an untouched external test set. Folds share much of their training data. No patient-level claim is warranted.

## Executed verification and preservation

The report is based on actual saved predictions and measured raw observations. Original event definitions were traced into the archived source; their values were independently recomputed rather than calling that implementation. The current production event/label routines were inspected before the single correction. New tests cover left/right event boundaries, incomplete lead-ins, missing/invalid readings, numeric/pulse gaps, hidden low intervals, shared alerts, repeated alarms per event, sustained alarms, exact leads, empty/singleton streams and zero exposure. Existing label-horizon, causal feature, fitted transform, threshold, model-loading and recording split tests remain part of the full suite.

**The complete suite passed: 91 tests, zero failures/errors/skips, in 11.330 s.** Independent historical/timestamp checks all have zero discrepancies. All 224 outer/model operating points and training-only row/split boundaries were independently checked. All **529** pre-existing final/validated/hardened artifacts and **60** raw source CSVs match their saved fingerprints; **183** local documentation links resolve. Final execution results and preservation hashes are recorded in [verification.json](../results/reliability/verification.json), with the complete [test log](../results/reliability/test_results.txt). [preservation.json](../results/reliability/preservation.json) fingerprints all pre-existing final/validated/hardened artifacts before this pass. The initial `audit_v1` folder contains only an incomplete first-attempt raw-event table after an empty-population handling error; it is not an evaluation result. `verified` contains the completed independent historical recomputation and its exact evaluator source snapshot. `final` adds the clarified off-event versus unmatched-alarm distinction. Earlier/new audit attempts remain separate, and none replaces original model results.

To reproduce in a new output directory (the evaluator refuses an existing destination):

```bash
python -m src.reliability_evaluation --output results/reliability/reproduction
PYTHONPATH=. python results/reliability/boundary_checks.py
python results/reliability/shared_alert_check.py
PYTHONPATH=. python results/reliability/original_reference_check.py
python -m unittest discover -s tests -v
python -m src.check_project --links
```

The auxiliary scripts regenerate only their own reliability diagnostics, never model artifacts. Input paths and source fingerprints are recorded. No model training, broad search, package install, dataset download, recording removal or hardware/clinical experiment was performed.

## Prioritized risks and smallest next experiment

1. **Resolve independent identity and event truth first.** Obtain a verified patient-to-recording map and reserve additional, untouched patients from a compatible data source. Specify the observed-onset threshold, required persistence/recovery/washout, artifact rejection, monitor timing and censoring rules before evaluation. A one-sample crossing, brief recovery/re-crossing and unknown physiological delay remain major uncertainties. This audit does not invent clinical annotations or merge events after inspecting model outcomes.
2. **Prespecify an operational alert objective on development data.** Select one primary window (for example 10 s), one-to-one alarm-start matching, a maximum false-alarm rate and alert-duty limit, event sensitivity and its minimum worthwhile improvement. If selecting an operating point, use only development-record/patient OOF predictions. Keep current models/features fixed initially. Do not choose a threshold or suppression policy from the outer results reported here.
3. **Smallest informative next experiment:** freeze the existing Logistic and ELM feature/model settings and one development-only operating point per model at the same prespecified alarm burden; compare against the same fixed always/never/periodic/rate references on an untouched cohort of verified independent patients. Run the exact causal replay and the fixed one-to-one event evaluator once. Report paired per-patient/event detections, misses, lead distributions, off-event starts/hour and time-weighted alert fraction, with patient-cluster uncertainty and explicit exposure/censoring. Set event accrual/sample size from a prespecified improvement margin and uncertainty requirement; the current six positive recording groups and three severe events cannot justify a precise clinical claim. If patient linkage or enough independent onsets is unavailable, the experiment cannot demonstrate patient-independent improvement—collect/verify those data before further model search.

A genuine improvement would mean more timely matched event detections **at the same allowed false-alarm burden and acceptable alert duty** on the independent cohort, not a higher clipped-window median or recall under near-continuous alerts. No such improvement is established by the current analysis.
