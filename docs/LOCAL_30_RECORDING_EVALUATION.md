# Evaluation restricted to the 30 local recordings

## Executive summary

This fixed-policy retrospective reanalysis uses **only the existing 30 local BIDMC recordings**. No data were downloaded or introduced. It preserves the saved leave-one-recording-out (LORO) predictions, models, thresholds and earlier artifacts. No model was refitted, no threshold/alarm-budget was selected, and no difficult recording was removed. New outputs are in [results/local_30_fixed_policy_v2](../results/local_30_fixed_policy_v2/), with input/output hashes, the full 30-record roster and reproducible recording-bootstrap weights.

At the audited 30-second pre-onset window, the onset-trained ELM detects **13/14** eligible onsets but produces **185.42 off-event alarm starts/hour** and is active during **89.09%** of evaluable time. Logistic detects **9/14**, with **24.05/hour** and **98.14%** active time. Always-alert detects **8/14** under new-alarm-start matching, with **12.20/hour** and **100%** active time; state/coverage gaps create its new starts. These are not matched-alarm-burden comparisons. Neither high sensitivity nor a long median detected-event lead establishes useful superiority. Only six recordings contain eligible onsets, and only three qualifying episodes contain observed 91/92 readings.

All 30 recordings are retained in reporting/resampling. Four entirely-low recordings (04/08/09/22) have no current-normal event-evaluation opportunities and receive zero event exposure and undefined event rates; their future-low timestamp results remain included. Source recording IDs are verified; **patient IDs are not**. LORO measures generalization to held-out recordings, not proven generalization to unseen patients. All results remain exploratory and do not establish clinical readiness.

## Methods and provenance

The required [reliability audit](RELIABILITY_AUDIT.md), [final report](FINAL_REPORT.md) and [pipeline guide](FULL_PIPELINE_GUIDE.md) were read before implementing this analysis. Their historical recommendations to collect additional data are superseded for this study: all present work and the recommended next experiment below use the same 30 local recordings.

The new [local evaluator](../src/local_30_evaluation.py) reuses the audited `current_normal`, `links`, `episodes`, `eligible_events`, `match_events`, `raw_events` and baseline functions in [reliability_evaluation.py](../src/reliability_evaluation.py). It verifies the previous audit's input/output hashes, regenerates all new-alarm event matches and asserts equality with the saved audited matches, including selected alarm IDs and actual timestamps. Existing boundary-gap fixes in [target_reporting.py](../src/target_reporting.py#L46) and strict inter-pulse coverage checks are preserved. The analysis runs no fitting or threshold selection; full-data demonstration models never supply test predictions.

### Targets, timing and alarm definitions

- **future_low**: minimum valid observed SpO₂ in `(t,t+30]` ≤94%, including ongoing low saturation; complete valid future coverage is required. Its full timestamp population has 20,157 rows across 30 recordings.
- **new_onset**: the same 30-second future contract, restricted to a fresh current observation >94%, with a qualifying adjacent valid >94→≤94 crossing. It has 16,050 rows across 26 nonempty eligible recording groups. Features are constructed on the full past pulse stream before target exclusions. Current/future SpO₂ and labels are excluded from the ten predictor features; current SpO₂ only gates eligibility. The two training populations remain separate.
- An observed low episode starts at the first valid ≤94 reading after a valid >94 reading no more than 1.01 s earlier. Missing/invalid data and gaps split episodes; initial/unknown-prior lows are left censored. One reading suffices; this is not clinically adjudicated persistent desaturation. Observed 91/92 readings form a descriptive subset, not exact-saturation prediction.
- Event windows are `[onset−W,onset)` for **W=5,10,15,30 s**. The left edge is included and an alert at onset is too late. These are nested look-back windows, **not minimum required warning times**; model targets remain 30 s. A complete valid raw W-second lead-in, qualifying crossing and at least one common current-normal prediction opportunity determine eligibility independently of model alerts. There are 15 eligible events at 5/10/15 s and 14 at 30 s. Record 26's onset at 22 s lacks a complete 30 s lead-in. This is a coverage exclusion, not removal of a poor result.
- Build positive-decision runs over each entire eligible recording, splitting on negative decisions, >2 s gaps or intervening invalid/current-low coverage. Do not restart at event-window boundaries. Process eligible events chronologically and assign the earliest unused alarm start in the window; each episode and event gets at most one credit. Nearby events cannot share one alarm credit. Gate re-entry may legitimately start a new run, including for always-alert; that behavior is not learned anticipation.
- **Off-event false alarm starts** have no observed qualifying onset in `(start,start+W]`, checking all observed qualifying onsets including sensitivity-excluded events. Repeated starts near one real event are redundant/unmatched, not off-event false alarms. Both off-event and unmatched counts are saved. The denominator is strict eligible current-normal exposure over all recordings, including negative-only ones, not just event windows: **10,329.032 s = 2.869176 h** for each target/model/baseline. Inter-pulse edges must be ≤2 s with valid complete raw coverage and no hidden current-low reading. No duration is invented outside observed endpoints. The record-7 edge 366.488–368.472 s is excluded because it crosses the observed 94% reading at 367 s.
- **Alarm-active fraction** is left-held positive-decision duration divided by permitted inter-pulse exposure. **Pulse alert fraction** is positive decisions divided by eligible pulse count. They differ with pulse rate. Zero-exposure rates are undefined, not zero; an isolated alarm may contribute a count without duration. Leads are `onset−actual matched alarm_start`, never a backdated PPG peak or clipped positive inside an older alarm. Lead distributions condition on detections; misses retain undefined leads. A longer detected-only median may reflect different/missed events rather than earlier detection of the same events.

Comparators are the fixed ELM, logistic, GaussianNB and tree, plus always-alert, never-alert, deterministic 1-second-on/9-second-off at time-zero phase, and PPG pulse rate >100 bpm. No baseline phase/cutoff was selected from these held-out results. All models have identical eligible rows/events within each target. For event analysis the future-low-trained models are explicitly evaluated on the current-normal subset; full future-low timestamp results remain separate. Baseline ranking metrics use binary baseline decisions, not inherited ELM scores.

### Outer isolation and alarm-burden selection audit

The saved [runner](../src/run_experiments.py) holds out a complete recording. [grouped_oof](../src/evaluation.py#L37) receives only outer-training rows and performs three grouped inner folds; [FittedModel.fit](../src/model.py) fits imputation, scaling and the fixed model on each permitted training fold. The largest training-OOF threshold meeting **92% timestamp recall** is selected, then the model/transforms are refitted on outer-training rows and evaluated on the outer record. Causal filtering, pulse confirmation and rolling features use no future samples beyond the actual reported availability time. All 30 source recordings are in the roster; future-low has 30 nonempty test folds, onset has 26 nonempty folds plus four explicitly empty target test records. No scores or event outcomes are fabricated for those empty records.

The new audit independently validates **120 future-low and 104 onset outer/model operating points**, using all **2,338,212** and **1,605,000** saved training-OOF rows. It checks exact sample/record/label provenance, outer/inner recording disjointness and coverage, once-only OOF row sets, the kth-positive-score cutoff equivalent to the recall rule, transferred held-out thresholds, and `Prediction == (Score >= Threshold)`. See [threshold_selection_audit.csv](../results/local_30_fixed_policy_v2/threshold_selection_audit.csv). Existing transform-fit and causal-prefix regression tests supplement these saved-array checks. Fold metadata alone cannot establish arbitrary unrecorded historical behavior.

**No development-only alarm-burden selection was performed in the existing or current reported analysis.** The threshold objective was timestamp recall, and the burden comparisons here are outcomes at those frozen operating points. It would be invalid to scan thresholds on outer predictions, pick a burden/sensitivity tradeoff, then report that same scan as independent performance. The saved training-OOF predictions could support selection of a prespecified budget within each outer-training set, but their burden at the selected threshold would be selection performance, not independent performance. Existing full-data demo thresholds also cannot be used to claim held-out performance.

For a future local budget experiment, retain outer LORO. Inside each outer-training set, choose a **prespecified** threshold policy using only grouped training-OOF scores/events/exposure; fit the model/transforms on that outer-training set and apply the resulting threshold once to the untouched outer record. If comparing/choosing budgets, postprocessing or models, add middle recording-grouped assessment folds: derive each threshold from inner OOF data of the middle-training records only, assess it on middle-held-out records, choose the policy entirely inside the outer-training set, then freeze/refit and evaluate the outer record. Do not call inner/middle outcomes used for selection independent performance. Reject unattainable budgets or follow an advance-specified fallback; never relax them using the outer result. No alarm-budget claims from such an unexecuted experiment are included here.

### Recording-level uncertainty

Use **2,000 unstratified whole-recording bootstrap draws**, seed **20261014**, from the full 30-record roster. Identical sampled multiplicities are used for every target, model, window and paired difference; the [weights](../results/local_30_fixed_policy_v2/bootstrap_recording_weights.csv.gz) are saved. Each recording copy contributes its complete counts, exposure and matched-event leads. Episodes are not rematched across concatenated recordings/copies. Negative-only and zero-opportunity records remain in sampling. Pooled sensitivities are sums of detected/eligible counts; pooled rates are sums of starts divided by sums of exposure; active fractions are duration ratios. Resampled detected leads are repeated according to record multiplicity and summarized by the usual median. Undefined zero-event/zero-detection replicates are excluded only from the relevant quantile calculation, with explicit valid-replicate counts.

Intervals are exploratory 2.5th–97.5th percentiles, **conditional on these saved CV fits**. They do not refit models, estimate all training/threshold-selection variation, make overlapping CV fits independent, or establish patient-level independence. Event imbalance (especially record 26) and only six positive records limit coverage. Paired intervals use common recording draws; they are not adjusted for multiple comparisons. Lead-median differences compare detected-only distributions, not paired lead gains on identical events. No winner, budget or policy is chosen from interval endpoints. AP/ROC from pooled scores combine different refits and remain descriptive; they are not bootstrapped as independent timestamps.

## Results: timestamp classification

These full-target metrics are distinct from event sensitivity and false alarm starts. FP/FN count timestamps, not distinct episodes.

| Target | Model | Samples | Recall | Precision | Specificity | PR_AUC_AP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| future_low | Always_Alert | 20157 | 1.0000 | 0.2276 | 0.0000 | 0.2276 | 15570 | 0 |
| future_low | ELM | 20157 | 0.9649 | 0.2390 | 0.0949 | 0.2559 | 14093 | 161 |
| future_low | Logistic | 20157 | 0.9320 | 0.2196 | 0.0243 | 0.2403 | 15192 | 312 |
| future_low | Never_Alert | 20157 | 0.0000 | undefined | 1.0000 | 0.2276 | 0 | 4587 |
| new_onset | Always_Alert | 16050 | 1.0000 | 0.0312 | 0.0000 | 0.0312 | 15549 | 0 |
| new_onset | ELM | 16050 | 0.8902 | 0.0309 | 0.1010 | 0.0240 | 13979 | 55 |
| new_onset | Logistic | 16050 | 0.8862 | 0.0282 | 0.0161 | 0.0335 | 15299 | 57 |
| new_onset | Never_Alert | 16050 | 0.0000 | undefined | 1.0000 | 0.0312 | 0 | 501 |

Additional models/baselines are in [timestamp_pooled.csv](../results/local_30_fixed_policy_v2/timestamp_pooled.csv). [timestamp_per_recording.csv](../results/local_30_fixed_policy_v2/timestamp_per_recording.csv) includes every record, full-target and current-normal populations, with undefined class metrics preserved. High pooled future-low recall includes ongoing lows and is not evidence of new-onset warning.

## Results: one-to-one event evaluation

All tables below use fixed saved thresholds. Off-event rates reference all qualifying onsets even for the severe subset. All 30 records contribute their available exposure. Lead distributions include detections only.

### future_low

| Model | Window_s | Eligible_Events | Detected | Missed_Events | Sensitivity | False_Alarm_Starts_Per_Hour | Alarm_Active_Fraction | Lead_Median_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Always_Alert | 5 | 15 | 1 | 14 | 0.0667 | 14.9869 | 1.0000 | 2.3680 |
| Always_Alert | 10 | 15 | 3 | 12 | 0.2000 | 14.2898 | 1.0000 | 6.3680 |
| Always_Alert | 15 | 15 | 5 | 10 | 0.3333 | 13.5928 | 1.0000 | 9.7840 |
| Always_Alert | 30 | 14 | 8 | 6 | 0.5714 | 12.1986 | 1.0000 | 13.2680 |
| ELM | 5 | 15 | 4 | 11 | 0.2667 | 193.0868 | 0.9179 | 2.6040 |
| ELM | 10 | 15 | 8 | 7 | 0.5333 | 190.6471 | 0.9179 | 6.9080 |
| ELM | 15 | 15 | 9 | 6 | 0.6000 | 187.8588 | 0.9179 | 11.5760 |
| ELM | 30 | 14 | 12 | 2 | 0.8571 | 182.2823 | 0.9179 | 18.2000 |
| GaussianNB | 5 | 15 | 3 | 12 | 0.2000 | 26.4884 | 0.9929 | 2.3680 |
| GaussianNB | 10 | 15 | 5 | 10 | 0.3333 | 25.0943 | 0.9929 | 6.3680 |
| GaussianNB | 15 | 15 | 7 | 8 | 0.4667 | 23.0031 | 0.9929 | 11.5760 |
| GaussianNB | 30 | 14 | 10 | 4 | 0.7143 | 20.9119 | 0.9929 | 14.3800 |
| Logistic | 5 | 15 | 3 | 12 | 0.2000 | 41.4753 | 0.9745 | 2.3680 |
| Logistic | 10 | 15 | 5 | 10 | 0.3333 | 40.4297 | 0.9745 | 5.0880 |
| Logistic | 15 | 15 | 6 | 9 | 0.4000 | 39.0356 | 0.9745 | 10.6800 |
| Logistic | 30 | 14 | 9 | 5 | 0.6429 | 37.6415 | 0.9745 | 11.9280 |
| Never_Alert | 5 | 15 | 0 | 15 | 0.0000 | 0.0000 | 0.0000 | undefined |
| Never_Alert | 10 | 15 | 0 | 15 | 0.0000 | 0.0000 | 0.0000 | undefined |
| Never_Alert | 15 | 15 | 0 | 15 | 0.0000 | 0.0000 | 0.0000 | undefined |
| Never_Alert | 30 | 14 | 0 | 14 | 0.0000 | 0.0000 | 0.0000 | undefined |
| Periodic_10s_1s | 5 | 15 | 6 | 9 | 0.4000 | 356.8969 | 0.1010 | 1.6680 |
| Periodic_10s_1s | 10 | 15 | 12 | 3 | 0.8000 | 354.8058 | 0.1010 | 5.1120 |
| Periodic_10s_1s | 15 | 15 | 13 | 2 | 0.8667 | 352.7146 | 0.1010 | 9.9600 |
| Periodic_10s_1s | 30 | 14 | 13 | 1 | 0.9286 | 348.5322 | 0.1010 | 23.6000 |
| Pulse_Rate_GT100 | 5 | 15 | 7 | 8 | 0.4667 | 363.1705 | 0.2602 | 1.8400 |
| Pulse_Rate_GT100 | 10 | 15 | 11 | 4 | 0.7333 | 358.6396 | 0.2602 | 7.3040 |
| Pulse_Rate_GT100 | 15 | 15 | 12 | 3 | 0.8000 | 353.7602 | 0.2602 | 10.9600 |
| Pulse_Rate_GT100 | 30 | 14 | 13 | 1 | 0.9286 | 346.4410 | 0.2602 | 18.1280 |
| Tree | 5 | 15 | 1 | 14 | 0.0667 | 42.5209 | 0.9546 | 2.3680 |
| Tree | 10 | 15 | 3 | 12 | 0.2000 | 41.8239 | 0.9546 | 6.3680 |
| Tree | 15 | 15 | 5 | 10 | 0.3333 | 41.1268 | 0.9546 | 9.7840 |
| Tree | 30 | 14 | 8 | 6 | 0.5714 | 39.7327 | 0.9546 | 13.2680 |

### new_onset

| Model | Window_s | Eligible_Events | Detected | Missed_Events | Sensitivity | False_Alarm_Starts_Per_Hour | Alarm_Active_Fraction | Lead_Median_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Always_Alert | 5 | 15 | 1 | 14 | 0.0667 | 14.9869 | 1.0000 | 2.3680 |
| Always_Alert | 10 | 15 | 3 | 12 | 0.2000 | 14.2898 | 1.0000 | 6.3680 |
| Always_Alert | 15 | 15 | 5 | 10 | 0.3333 | 13.5928 | 1.0000 | 9.7840 |
| Always_Alert | 30 | 14 | 8 | 6 | 0.5714 | 12.1986 | 1.0000 | 13.2680 |
| ELM | 5 | 15 | 6 | 9 | 0.4000 | 197.6177 | 0.8909 | 2.6760 |
| ELM | 10 | 15 | 10 | 5 | 0.6667 | 191.6927 | 0.8909 | 7.5360 |
| ELM | 15 | 15 | 11 | 4 | 0.7333 | 189.2530 | 0.8909 | 9.7840 |
| ELM | 30 | 14 | 13 | 1 | 0.9286 | 185.4191 | 0.8909 | 19.5600 |
| GaussianNB | 5 | 15 | 1 | 14 | 0.0667 | 65.8726 | 0.8504 | 2.3680 |
| GaussianNB | 10 | 15 | 3 | 12 | 0.2000 | 65.1755 | 0.8504 | 6.3680 |
| GaussianNB | 15 | 15 | 6 | 9 | 0.4000 | 64.1299 | 0.8504 | 10.6800 |
| GaussianNB | 30 | 14 | 9 | 5 | 0.6429 | 62.7358 | 0.8504 | 12.8640 |
| Logistic | 5 | 15 | 3 | 12 | 0.2000 | 27.5340 | 0.9814 | 2.3680 |
| Logistic | 10 | 15 | 5 | 10 | 0.3333 | 26.4884 | 0.9814 | 5.0880 |
| Logistic | 15 | 15 | 6 | 9 | 0.4000 | 25.4428 | 0.9814 | 10.6800 |
| Logistic | 30 | 14 | 9 | 5 | 0.6429 | 24.0487 | 0.9814 | 13.2320 |
| Never_Alert | 5 | 15 | 0 | 15 | 0.0000 | 0.0000 | 0.0000 | undefined |
| Never_Alert | 10 | 15 | 0 | 15 | 0.0000 | 0.0000 | 0.0000 | undefined |
| Never_Alert | 15 | 15 | 0 | 15 | 0.0000 | 0.0000 | 0.0000 | undefined |
| Never_Alert | 30 | 14 | 0 | 14 | 0.0000 | 0.0000 | 0.0000 | undefined |
| Periodic_10s_1s | 5 | 15 | 6 | 9 | 0.4000 | 356.8969 | 0.1010 | 1.6680 |
| Periodic_10s_1s | 10 | 15 | 12 | 3 | 0.8000 | 354.8058 | 0.1010 | 5.1120 |
| Periodic_10s_1s | 15 | 15 | 13 | 2 | 0.8667 | 352.7146 | 0.1010 | 9.9600 |
| Periodic_10s_1s | 30 | 14 | 13 | 1 | 0.9286 | 348.5322 | 0.1010 | 23.6000 |
| Pulse_Rate_GT100 | 5 | 15 | 7 | 8 | 0.4667 | 363.1705 | 0.2602 | 1.8400 |
| Pulse_Rate_GT100 | 10 | 15 | 11 | 4 | 0.7333 | 358.6396 | 0.2602 | 7.3040 |
| Pulse_Rate_GT100 | 15 | 15 | 12 | 3 | 0.8000 | 353.7602 | 0.2602 | 10.9600 |
| Pulse_Rate_GT100 | 30 | 14 | 13 | 1 | 0.9286 | 346.4410 | 0.2602 | 18.1280 |
| Tree | 5 | 15 | 1 | 14 | 0.0667 | 14.9869 | 1.0000 | 2.3680 |
| Tree | 10 | 15 | 3 | 12 | 0.2000 | 14.2898 | 1.0000 | 6.3680 |
| Tree | 15 | 15 | 5 | 10 | 0.3333 | 13.5928 | 1.0000 | 9.7840 |
| Tree | 30 | 14 | 8 | 6 | 0.5714 | 12.1986 | 1.0000 | 13.2680 |

ELM's higher 30-second onset event sensitivity coincides with much higher alarm-start burden, while logistic has higher active duty. The periodic/rate rules also detect 13/14 at 30 s but with very high start rates. The tree's onset decisions equal always-alert. Thus no useful ELM superiority follows from sensitivity alone. Historical reusable-positive matching and its roughly 29.7-second clipped lead are preserved in prior reports and are not relabeled as new alarm starts here.

### Recording-level results, onset target at 30 seconds

The following table retains the full roster. Detected/missed are counts; 0/0 means no eligible event, not zero sensitivity. Empty exposure gives undefined rates. Full recording-level results for every model/window and severe subset are in [event_per_recording.csv](../results/local_30_fixed_policy_v2/event_per_recording.csv).

| Record | Eligible | Exposure s | ELM detected/missed | ELM starts/h | ELM active | Logistic detected/missed | Logistic starts/h | Logistic active |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 404.8880 | 0/0 | 26.6740 | 0.9989 | 0/0 | 17.7827 | 1.0000 |
| 2 | 0 | 449.0400 | 0/0 | 424.9065 | 0.6921 | 0/0 | 48.1026 | 0.9898 |
| 3 | 0 | 448.3520 | 0/0 | 64.2352 | 0.9285 | 0/0 | 48.1764 | 0.9717 |
| 4 | 0 | 0.0000 | 0/0 | undefined | undefined | 0/0 | undefined | undefined |
| 5 | 0 | 413.3120 | 0/0 | 69.6810 | 0.9910 | 0/0 | 17.4203 | 1.0000 |
| 6 | 0 | 448.7200 | 0/0 | 16.0456 | 0.9984 | 0/0 | 16.0456 | 0.9935 |
| 7 | 1 | 446.4880 | 1/0 | 16.1259 | 0.9993 | 0/1 | 16.1259 | 1.0000 |
| 8 | 0 | 0.0000 | 0/0 | undefined | undefined | 0/0 | undefined | undefined |
| 9 | 0 | 0.0000 | 0/0 | undefined | undefined | 0/0 | undefined | undefined |
| 10 | 0 | 448.4640 | 0/0 | 385.3152 | 0.5558 | 0/0 | 8.0274 | 1.0000 |
| 11 | 0 | 449.3280 | 0/0 | 16.0239 | 0.9992 | 0/0 | 8.0120 | 1.0000 |
| 12 | 1 | 445.5680 | 0/1 | 517.0928 | 0.8359 | 0/1 | 16.1591 | 1.0000 |
| 13 | 0 | 448.6000 | 0/0 | 337.0486 | 0.3321 | 0/0 | 24.0749 | 0.9858 |
| 14 | 0 | 449.2880 | 0/0 | 472.7480 | 0.8194 | 0/0 | 8.0127 | 1.0000 |
| 15 | 0 | 448.9200 | 0/0 | 336.8083 | 0.9369 | 0/0 | 8.0192 | 1.0000 |
| 16 | 0 | 448.6960 | 0/0 | 32.0930 | 0.9968 | 0/0 | 8.0232 | 1.0000 |
| 17 | 2 | 30.7520 | 2/0 | 0.0000 | 1.0000 | 2/0 | 0.0000 | 1.0000 |
| 18 | 0 | 448.9440 | 0/0 | 352.8280 | 0.7784 | 0/0 | 8.0188 | 1.0000 |
| 19 | 0 | 190.9120 | 0/0 | 75.4274 | 1.0000 | 0/0 | 75.4274 | 1.0000 |
| 20 | 0 | 448.9360 | 0/0 | 104.2465 | 0.9824 | 0/0 | 8.0190 | 1.0000 |
| 21 | 0 | 448.9200 | 0/0 | 184.4427 | 0.9411 | 0/0 | 8.0192 | 1.0000 |
| 22 | 0 | 0.0000 | 0/0 | undefined | undefined | 0/0 | undefined | undefined |
| 23 | 0 | 447.8240 | 0/0 | 96.4665 | 0.9468 | 0/0 | 24.1166 | 0.9952 |
| 24 | 0 | 448.6400 | 0/0 | 8.0243 | 1.0000 | 0/0 | 8.0243 | 1.0000 |
| 25 | 0 | 448.7920 | 0/0 | 136.3661 | 0.9774 | 0/0 | 8.0215 | 1.0000 |
| 26 | 6 | 354.5360 | 6/0 | 406.1647 | 0.8165 | 3/3 | 40.6165 | 1.0000 |
| 27 | 0 | 448.7440 | 0/0 | 32.0896 | 0.9879 | 0/0 | 8.0224 | 1.0000 |
| 28 | 0 | 449.1840 | 0/0 | 8.0145 | 0.9989 | 0/0 | 8.0145 | 1.0000 |
| 29 | 1 | 6.2640 | 1/0 | 0.0000 | 1.0000 | 1/0 | 0.0000 | 1.0000 |
| 30 | 3 | 406.9200 | 3/0 | 318.4901 | 0.9282 | 3/0 | 185.7859 | 0.5977 |

The ELM's missed eligible 30-second onset is record 12 at 210 s: positives can be present throughout its lead-in without a **new** alarm start there. This differs from timestamp false negatives. Records 26 and 30 remain included with their multiple recurring events; no exclusion is based on poor recall or high alarm burden. [event_matches.csv](../results/local_30_fixed_policy_v2/event_matches.csv) records each onset, eligibility reason, alarm ID, matched timestamp and lead. [detected_event_leads.csv](../results/local_30_fixed_policy_v2/detected_event_leads.csv) and per-record quantiles preserve the complete lead distributions.

### Bootstrap intervals and severe subset

Onset-target 30-second exploratory 95% intervals:

| Model | Metric | Lower_95 | Upper_95 | Defined_Replicates | Replicates |
| --- | --- | --- | --- | --- | --- |
| Always_Alert | Sensitivity | 0.3333 | 0.8571 | 1996 | 2000 |
| Always_Alert | False_Alarm_Starts_Per_Hour | 9.3023 | 16.3576 | 2000 | 2000 |
| Always_Alert | Alarm_Active_Fraction | 1.0000 | 1.0000 | 2000 | 2000 |
| Always_Alert | Pulse_Alert_Fraction | 1.0000 | 1.0000 | 2000 | 2000 |
| Always_Alert | Lead_Median_s | 6.3680 | 17.6920 | 1979 | 2000 |
| ELM | Sensitivity | 0.6250 | 1.0000 | 1996 | 2000 |
| ELM | False_Alarm_Starts_Per_Hour | 116.7834 | 256.8479 | 2000 | 2000 |
| ELM | Alarm_Active_Fraction | 0.8209 | 0.9499 | 2000 | 2000 |
| ELM | Pulse_Alert_Fraction | 0.8357 | 0.9496 | 2000 | 2000 |
| ELM | Lead_Median_s | 9.7840 | 25.2920 | 1992 | 2000 |
| Logistic | Sensitivity | 0.3333 | 1.0000 | 1996 | 2000 |
| Logistic | False_Alarm_Starts_Per_Hour | 13.6037 | 39.4632 | 2000 | 2000 |
| Logistic | Alarm_Active_Fraction | 0.9473 | 0.9990 | 2000 | 2000 |
| Logistic | Pulse_Alert_Fraction | 0.9452 | 0.9992 | 2000 | 2000 |
| Logistic | Lead_Median_s | 6.3680 | 17.6920 | 1979 | 2000 |

Only 1,996/2,000 all-event sensitivity replicates contain eligible events. Lead-median replicates also require detections (1,992 for ELM, 1,979 for logistic). Dropping undefined values does not create additional event information. All windows/models/subsets and valid replicate counts are in [recording_bootstrap_intervals.csv](../results/local_30_fixed_policy_v2/recording_bootstrap_intervals.csv).

Paired ELM minus logistic differences at 30 seconds, onset target:

| Metric | Difference_A_Minus_B | Lower_95 | Upper_95 | Defined_Replicates |
| --- | --- | --- | --- | --- |
| Sensitivity | 0.2857 | 0.0000 | 0.5000 | 1996 |
| False_Alarm_Starts_Per_Hour | 161.3704 | 93.5928 | 229.5323 | 2000 |
| Alarm_Active_Fraction | -0.0905 | -0.1700 | -0.0221 | 2000 |
| Pulse_Alert_Fraction | -0.0821 | -0.1520 | -0.0211 | 2000 |
| Lead_Median_s | 6.3280 | 0.0000 | 15.3520 | 1979 |

The sensitivity difference interval touches zero, and ELM has many more off-event starts. Negative active-fraction differences mean lower active duty for ELM, not zero burden. These outcomes do not establish dominance at a common operational constraint. Further prespecified paired comparisons against simple references are saved in [paired_recording_bootstrap.csv](../results/local_30_fixed_policy_v2/paired_recording_bootstrap.csv).

For the three episodes containing observed 91/92 readings, onset-target results at 30 seconds are:

| Model | Eligible_Events | Detected | Missed_Events | Sensitivity | Lead_Min_s | Lead_Median_s | Lead_Max_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Always_Alert | 3 | 2 | 1 | 0.6667 | 6.3680 | 8.0760 | 9.7840 |
| ELM | 3 | 2 | 1 | 0.6667 | 6.3680 | 8.0760 | 9.7840 |
| GaussianNB | 3 | 3 | 0 | 1.0000 | 6.3680 | 9.7840 | 12.8640 |
| Logistic | 3 | 2 | 1 | 0.6667 | 6.3680 | 8.0760 | 9.7840 |
| Never_Alert | 3 | 0 | 3 | 0.0000 | undefined | undefined | undefined |
| Periodic_10s_1s | 3 | 2 | 1 | 0.6667 | 9.7840 | 19.7280 | 29.6720 |
| Pulse_Rate_GT100 | 3 | 3 | 0 | 1.0000 | 3.3440 | 7.3040 | 12.8640 |
| Tree | 3 | 2 | 1 | 0.6667 | 6.3680 | 8.0760 | 9.7840 |

Three events cannot support a stable severe-desaturation reliability claim. Assignment is performed once against all eligible onsets, then the severe subset is selected; alarm credits are not awarded again. The severe-subset alarm-rate columns use the all-onset reference, not an invented severe-specific false-alarm definition.

## Patient identifiers and study limitations

The local source inventory contains exactly 30 Signals/Numerics pairs, 60 CSVs. [source_inventory.csv](../results/local_30_fixed_policy_v2/source_inventory.csv) records every filename/header and checks for patient/subject/person ID fields. None is present. No verified patient-to-recording linkage table was found during repository inspection. Filename numbers are recording identifiers, not fabricated patient IDs. This establishes held-out-recording evaluation only; the same patient could potentially occur in different recordings. Recording bootstrap intervals do not fix that uncertainty.

Other unresolved limitations: all 30 recordings have already been inspected repeatedly; post-hoc evaluation-rule work cannot make them a fresh confirmatory test set. Few positive records, short exposure, one-reading events, brief recoveries, unknown monitor/physiological delay, no validated signal-quality gate, repeated correlated events and unstable score calibration limit the conclusions. Current-normal eligibility and complete future ground-truth availability restrict evaluable exposure and may not equal online operating time. Gaps create alarm restarts, including for trivial references. Bootstrap intervals condition on shared CV fits and do not capture the full learning procedure. No patient-level, clinical-readiness or exact-SpO₂ prediction claim is made.

## Recommended next experiment: local data only

Prespecify one operational alarm budget and active-duty cap, one primary window (for example the 10-second look-back window), one-to-one matching, an event-sensitivity objective and an infeasibility rule **before** rerunning anything. Keep the current ten features and ELM/logistic hyperparameters fixed. Use the nested recording-grouped design above on these same 30 recordings: threshold selection only on outer-training OOF data, optional policy choice only through middle assessment splits, and final reporting only from outer-held-out decisions. Compare every model/reference at the stated constraints and report infeasible folds rather than choosing looser constraints from the test results. Save all splits, train-only thresholds, burden estimates and outer results separately.

This is the smallest useful next **local** experiment; it has not been executed in this pass because no operational budget was prespecified and the current thresholds must remain frozen. It can test whether a fixed budgeted policy improves the local recording-level tradeoff. It cannot undo prior observation of the cohort or demonstrate unseen-patient/clinical generalization. No additional dataset is requested, downloaded or proposed for this study. If the few local events cannot distinguish policies, report that limitation rather than expanding claims or searching broadly.

## Changed files and executed verification

- [src/local_30_evaluation.py](../src/local_30_evaluation.py): new read-only-to-input evaluator, independent training-OOF provenance checks, complete recording contributions, one-to-one match verification and paired cluster bootstrap; exclusive new output directory.
- [tests/test_local_30.py](../tests/test_local_30.py): split/threshold leakage guards, held-label invariance, recording bootstrap/undefined denominators, event boundaries/shared credit, exposure gaps and saved-output consistency.
- This new report and the separate output directory. Existing core modules, models, thresholds, raw data and historical reports/results were not changed. The first attempt `local_30_fixed_policy_v1` stopped during a missing-value representation comparison before producing evaluation metrics; its two partial inventory/preservation files are retained and are not study results. The completed evaluation is **v2**.

**Executed validation:** 106 tests passed in 10.516 s, with zero skips, failures or errors. The local aggregation/provenance gate and both existing saved-run gates passed; all 662 protected files matched hashes, all 60 local source CSVs remained unchanged, and 209 local documentation links resolved. Actual complete test and quality results are in [verification.json](../results/local_30_fixed_policy_v2/verification.json) and [test_results.txt](../results/local_30_fixed_policy_v2/test_results.txt). The [manifest](../results/local_30_fixed_policy_v2/manifest.json) records immutable inputs, code, fixed policy, bootstrap seed/weights and output hashes. [preservation.json](../results/local_30_fixed_policy_v2/preservation.json) protects 662 pre-existing raw/result/archive/report/source files captured before this run. No package installation, external data access, model training, threshold change or model search was performed.

Executed commands (new outputs only):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.local_30_evaluation --output results/local_30_fixed_policy_v2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.local_30_evaluation --output results/local_30_fixed_policy_v2 --verify
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.check_project --links --verify-run results/final/future_low
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.check_project --links --verify-run results/final/new_onset
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.check_project --audit
git diff --check
```

To reproduce the analysis, use a **new** output path, e.g. `python -m src.local_30_evaluation --output results/local_30_fixed_policy_reproduction`. The evaluator refuses an existing directory. The verification command is read-only. Complete tests are run after output generation so the new artifact checks execute rather than skip. Bootstrap output is deterministic for the saved inputs, 2,000 draws and seed 20261014.
