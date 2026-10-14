# Prespecified alarm-budget experiment on the 30 local recordings

## Executive result

**None of the four learned models detected an eligible onset under any of the three prespecified research budgets.** This holds for both original training targets and every evaluated 5/10/15/30-second look-back window. There are 15 eligible events at 5/10/15 seconds and 14 at 30 seconds. All learned-model timestamp true positives are also zero in the common current-normal population. No useful early-warning advantage over never-alert was demonstrated by this particular policy.

Only the existing 30 local recordings were used. All models/features/labels/preprocessing and the existing outer recording-wise score fits remain unchanged; historical thresholds and artifacts were not overwritten. The experiment selects **new fold-specific thresholds only**, saved separately. It uses the saved training-only inner OOF scores and outer-held-out scores from the same nested fixed-model fits, rather than retraining identical models or scoring full-data demonstration models as tests.

A clinically justified primary alarm budget cannot be inferred from this small retrospective dataset or its prior results. Accordingly this is a **prespecified sensitivity analysis at 1, 5 and 10 off-event alarm starts per evaluable hour**, with a fixed **5% alarm-active-time cap**. All budgets are reported; no budget/model/policy was chosen from outer outcomes. This is an offline research result, not clinical-device validation, independent-patient validation or an automated oxygen-delivery controller.

## Phase A audit and frozen specification

Before implementation the [reliability audit](RELIABILITY_AUDIT.md), [local-30 evaluation](LOCAL_30_RECORDING_EVALUATION.md), [pipeline guide](FULL_PIPELINE_GUIDE.md), [final report](FINAL_REPORT.md), and active evaluation/threshold/event/replay implementations were reviewed. Repository status and saved schemas were inspected. Existing uncommitted local-30 files were retained. The brief [Phase A audit](../results/alarm_budget_experiment/AUDIT.md) documents the source trace and resolved ambiguities.

- [evaluation.outer_splits/grouped_oof](../src/evaluation.py) and [run_experiments](../src/run_experiments.py) provide leave-one-recording-out fits and three grouped inner folds. Imputation/scaling and each fixed model are fitted only on the relevant training partition. Saved OOF rows are identified by **both** `Outer_Test_Record` and training `Record`, as well as Model/sample_id. A full-data or differently indexed OOF array would not be a valid substitute.
- [reliability_evaluation](../src/reliability_evaluation.py) defines observed crossings, complete numeric lead-in coverage, fresh-current-normal opportunities, strict inter-pulse exposure, actual positive-run starts, chronological one-to-one event matching and off-event starts. [target_reporting.py:46](../src/target_reporting.py#L46) retains the earlier boundary-gap correction. No existing event/exposure definition was changed.
- [replay](../src/replay.py) loads a full-data demonstration model; those outputs are explicitly unsuitable as held-out evaluation and were not used. Existing causal feature availability and recording-reset behavior remain unchanged.
- The available score caches support this threshold-only nested experiment without a new model fit. New code verifies exact training row/label/record/time provenance, original grouped split isolation, original recall-threshold provenance, raw/dataset hashes and all score-generating source hashes before using the caches. New thresholds are selected only from the correct outer-training OOF arrays.

The [protocol.json](../results/alarm_budget_experiment/protocol.json) was written before running new outer evaluation and was not changed afterward. Its SHA-256 is bound to each run/selection receipt. Prior outer results had already been inspected in earlier work; freezing a new protocol does not make this cohort an untouched confirmatory test set.

### Rationale and policy, specified before new outer outcomes

The 1/5/10 per-hour values are research sensitivity points, roughly one false start per 60/12/6 evaluable minutes. They span stringent through less stringent burden well below the earlier 24–185/hour onset logistic/ELM results. They are not asserted to be safe or acceptable clinical rates. Local data contain no clinician utility function, adjudicated nuisance-alarm annotation or evidence supporting a validated primary budget. There is no hidden choice of whichever sensitivity point looks best.

A **5% active-duty cap** was fixed alongside each budget because nearly continuous alarms can have few starts and misleadingly low episode burden. This is an explicit policy-selection constraint, not a change to the event metric or an empirically selected clinical limit. Always-alert cannot meet it. There is no new smoothing, cooldown, persistence rule, aggregation, suppression policy or model/feature search.

For every outer-training partition/model:

1. Take only that partition's recording-grouped OOF scores. Form the existing current-normal eligible stream with full causal feature histories retained upstream. Candidates are the unique linear score quantiles at 0, .01, …, 1 plus .995/.9975/.999, and `nextafter(max_score,+inf)` as a no-training-alert candidate. Score ties remain complete. This finite, frozen threshold grid is not an exhaustive optimum over every possible threshold.
2. Explicitly evaluate each candidate's **30-second** training-OOF event detections, off-event starts/hour and active duration. Start counts are nonmonotonic in threshold, so no monotonic/binary-search assumption is used.
3. Among candidates satisfying BOTH the specified start budget and active cap, maximize one-to-one 30-second event detections. Resolve ties by fewer off-event starts, less active time, then the highest threshold. If training has no eligible event, the same tie rules apply without examining test outcomes. The no-training-alert candidate guarantees training feasibility when exposure is positive; it is not a guarantee that unseen scores stay below that threshold. Do not relax constraints after seeing held-out results.
4. Store the threshold, exact `float.hex()` representation, candidate table, selected counts, allowed recordings and original threshold provenance. Only after all **672** selections (224 outer/model combinations × 3 budgets) are complete does new outer evaluation read the held-out prediction values.
5. Apply each selected scalar threshold once to its corresponding original outer-held-out recording's score. No outer labels, events, alarm rates, bootstrap intervals or pooled curves select or modify a threshold/model/policy. Original full-data demonstration model thresholds remain untouched.

The fixed-model inner fits and outer fits are exactly those used for the saved nested evaluation. Reusing their authenticated scores is sufficient because only the decision threshold changes; fitting imputation/scaling or models again would not add a new validation layer. Training-OOF outcomes used to select a candidate are selection diagnostics, **not independent performance estimates**. Inner score scales may transfer poorly to an outer refit; actual outer constraint violations are reported below rather than corrected with test-informed thresholds.

## Evaluation definitions and scope

`future_low` and `new_onset` preserve their existing, distinct training populations. The first target includes ongoing lows; the second requires a current fresh >94% reading and a qualifying future crossing. Both retain the 30-second `(t,t+30]` complete-valid-future label. Features contain only causal PPG amplitude/interval/rate/slope/history measurements; SpO₂ and future outcomes are not predictor inputs.

For the alarm experiment, **both targets' fitted models are evaluated on the audited common current-normal population**, 16,050 timestamps/501 positive timestamps. Full future-low timestamp performance on ongoing-low states remains in the historical outputs and is not relabeled as onset detection. Current SpO₂ gates eligibility only. Four wholly-low recordings (04/08/09/22) have zero eligible alarm exposure; all 30 remain in per-record tables and bootstrap sampling. Future-low has 30 nonempty model test folds; onset has 26 nonempty test folds, with four empty-policy-state records explicitly represented rather than fabricated model outcomes.

Observed low episodes require one valid ≤94 reading following a valid >94 reading within 1.01 s. Missing/invalid measurements and gaps split episodes; initial/unknown-prior lows are left censored. This is monitor-threshold ground truth, not a clinically adjudicated sustained event. The severe-interest subset contains actual 91/92 observations and only three eligible events at 30 seconds.

The four evaluation windows are **[onset−W,onset)**, W=5/10/15/30 seconds. Left endpoints count; onset-time alerts are too late. These are nested look-back windows, not guarantees of at least W seconds' warning. Eligible onsets require valid complete raw W-second lead-in and at least one eligible current-normal prediction opportunity. Earlier record-26 onset at 22 s is eligible for shorter windows but lacks 30 s of raw history.

An alarm is a **new positive-run start**, split by negative decisions, recording boundaries, gaps >2 s or invalid/intervening-low exposure. It is never manufactured by clipping a continuously active alarm at a window boundary. Events are processed chronologically; the earliest unused alarm start in a window gets one credit. One alarm and one event receive at most one match. Detected-only lead is onset minus that actual start. Misses retain undefined lead; matched medians/IQRs are undefined for models with no matched event. Assignment is performed against all eligible events before extracting the severe subset.

A false/off-event start has no qualifying observed onset in `(start,start+W]`, checking also onsets excluded from sensitivity. Redundant starts near a real event are not off-event starts and are saved separately as unmatched counts. False starts/hour is the integer numerator divided by **10,329.032 s / 3600 = 2.869176 evaluable hours** per model/target. This includes all negative-only recordings' eligible exposure, not just positive windows. Missing/low gaps, including the record-7 1.984 s bridge across the observed 94% reading, contribute no exposure. Rates at zero exposure are undefined. Alarm-active fraction uses left-held decisions over the permitted edges; pulse alert fraction counts decisions and is a distinct quantity.

**Budgets are selected against the 30-second false-start definition.** Shorter-window false rates have different event references and need not meet that budget. Even at 30 seconds, an outer recording may violate a constraint selected in training. All such violations remain in the result. A common training constraint is not identical achieved test burden.

Always-alert and never-alert are fixed, unselected references repeated across budget panels. Always-alert restarts when eligibility/coverage resumes; under one-to-one matching this yields some new starts and detections despite no predictive discrimination. Its 100% active duty violates the cap, so it is **not** a fair common-burden comparator. Never-alert satisfies the constraints by detecting nothing; showing performance no better than it is not useful early warning.

## Actual outer results

All learned models missed **all 15 eligible events at 5/10/15 s and all 14 at 30 s**, at every budget and both training targets. Their correctly matched lead-time distributions are empty; median and IQR are undefined. The following table gives the full numerator and denominator at the 30-second evaluation window. All values are held-out outcomes, not the training candidate diagnostics.

### future_low-trained models

| Model | Budget | Detected | Missed_Events | Eligible_Events | Off_Event_Alarm_Starts | Exposure_s | False_Alarm_Starts_Per_Hour | Alarm_Active_Fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Always_Alert | 1.0000 | 8 | 6 | 14 | 35 | 10329.0320 | 12.1986 | 1.0000 |
| Always_Alert | 5.0000 | 8 | 6 | 14 | 35 | 10329.0320 | 12.1986 | 1.0000 |
| Always_Alert | 10.0000 | 8 | 6 | 14 | 35 | 10329.0320 | 12.1986 | 1.0000 |
| ELM | 1.0000 | 0 | 14 | 14 | 0 | 10329.0320 | 0.0000 | 0.0000 |
| ELM | 5.0000 | 0 | 14 | 14 | 0 | 10329.0320 | 0.0000 | 0.0000 |
| ELM | 10.0000 | 0 | 14 | 14 | 0 | 10329.0320 | 0.0000 | 0.0000 |
| GaussianNB | 1.0000 | 0 | 14 | 14 | 1 | 10329.0320 | 0.3485 | 0.0001 |
| GaussianNB | 5.0000 | 0 | 14 | 14 | 1 | 10329.0320 | 0.3485 | 0.0001 |
| GaussianNB | 10.0000 | 0 | 14 | 14 | 1 | 10329.0320 | 0.3485 | 0.0001 |
| Logistic | 1.0000 | 0 | 14 | 14 | 0 | 10329.0320 | 0.0000 | 0.0000 |
| Logistic | 5.0000 | 0 | 14 | 14 | 0 | 10329.0320 | 0.0000 | 0.0000 |
| Logistic | 10.0000 | 0 | 14 | 14 | 1 | 10329.0320 | 0.3485 | 0.0000 |
| Never_Alert | 1.0000 | 0 | 14 | 14 | 0 | 10329.0320 | 0.0000 | 0.0000 |
| Never_Alert | 5.0000 | 0 | 14 | 14 | 0 | 10329.0320 | 0.0000 | 0.0000 |
| Never_Alert | 10.0000 | 0 | 14 | 14 | 0 | 10329.0320 | 0.0000 | 0.0000 |
| Tree | 1.0000 | 0 | 14 | 14 | 0 | 10329.0320 | 0.0000 | 0.0000 |
| Tree | 5.0000 | 0 | 14 | 14 | 0 | 10329.0320 | 0.0000 | 0.0000 |
| Tree | 10.0000 | 0 | 14 | 14 | 0 | 10329.0320 | 0.0000 | 0.0000 |

### new_onset-trained models

| Model | Budget | Detected | Missed_Events | Eligible_Events | Off_Event_Alarm_Starts | Exposure_s | False_Alarm_Starts_Per_Hour | Alarm_Active_Fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Always_Alert | 1.0000 | 8 | 6 | 14 | 35 | 10329.0320 | 12.1986 | 1.0000 |
| Always_Alert | 5.0000 | 8 | 6 | 14 | 35 | 10329.0320 | 12.1986 | 1.0000 |
| Always_Alert | 10.0000 | 8 | 6 | 14 | 35 | 10329.0320 | 12.1986 | 1.0000 |
| ELM | 1.0000 | 0 | 14 | 14 | 0 | 10329.0320 | 0.0000 | 0.0000 |
| ELM | 5.0000 | 0 | 14 | 14 | 0 | 10329.0320 | 0.0000 | 0.0000 |
| ELM | 10.0000 | 0 | 14 | 14 | 0 | 10329.0320 | 0.0000 | 0.0000 |
| GaussianNB | 1.0000 | 0 | 14 | 14 | 0 | 10329.0320 | 0.0000 | 0.0000 |
| GaussianNB | 5.0000 | 0 | 14 | 14 | 1 | 10329.0320 | 0.3485 | 0.0001 |
| GaussianNB | 10.0000 | 0 | 14 | 14 | 3 | 10329.0320 | 1.0456 | 0.0002 |
| Logistic | 1.0000 | 0 | 14 | 14 | 0 | 10329.0320 | 0.0000 | 0.0000 |
| Logistic | 5.0000 | 0 | 14 | 14 | 4 | 10329.0320 | 1.3941 | 0.0013 |
| Logistic | 10.0000 | 0 | 14 | 14 | 14 | 10329.0320 | 4.8795 | 0.0035 |
| Never_Alert | 1.0000 | 0 | 14 | 14 | 0 | 10329.0320 | 0.0000 | 0.0000 |
| Never_Alert | 5.0000 | 0 | 14 | 14 | 0 | 10329.0320 | 0.0000 | 0.0000 |
| Never_Alert | 10.0000 | 0 | 14 | 14 | 0 | 10329.0320 | 0.0000 | 0.0000 |
| Tree | 1.0000 | 0 | 14 | 14 | 57 | 10329.0320 | 19.8663 | 0.0089 |
| Tree | 5.0000 | 0 | 14 | 14 | 57 | 10329.0320 | 19.8663 | 0.0089 |
| Tree | 10.0000 | 0 | 14 | 14 | 57 | 10329.0320 | 19.8663 | 0.0089 |

### Other windows and lead distributions

The fixed references below are identical across the two target panels and all budgets. Learned models are uniformly 0/15 at 5/10/15 s and 0/14 at 30 s; their rows for every window are preserved in the machine-readable summaries, rather than omitted as difficult results.

| Model | Window_s | Detected | Missed_Events | Eligible_Events | Off_Event_Alarm_Starts | False_Alarm_Starts_Per_Hour | Alarm_Active_Fraction | Lead_Q25_s | Lead_Median_s | Lead_Q75_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Always_Alert | 5 | 1 | 14 | 15 | 43 | 14.9869 | 1.0000 | 2.3680 | 2.3680 | 2.3680 |
| Always_Alert | 10 | 3 | 12 | 15 | 41 | 14.2898 | 1.0000 | 4.3680 | 6.3680 | 8.0760 |
| Always_Alert | 15 | 5 | 10 | 15 | 39 | 13.5928 | 1.0000 | 6.3680 | 9.7840 | 11.5760 |
| Always_Alert | 30 | 8 | 6 | 14 | 35 | 12.1986 | 1.0000 | 8.9300 | 13.2680 | 19.8240 |
| Never_Alert | 5 | 0 | 15 | 15 | 0 | 0.0000 | 0.0000 | undefined | undefined | undefined |
| Never_Alert | 10 | 0 | 15 | 15 | 0 | 0.0000 | 0.0000 | undefined | undefined | undefined |
| Never_Alert | 15 | 0 | 15 | 15 | 0 | 0.0000 | 0.0000 | undefined | undefined | undefined |
| Never_Alert | 30 | 0 | 14 | 14 | 0 | 0.0000 | 0.0000 | undefined | undefined | undefined |

See [summary.csv](../results/alarm_budget_experiment/run02/summary.csv) for all windows, models, budgets, targets and severe subsets, including median/Q25/Q75/min/max lead. [event_matches.csv](../results/alarm_budget_experiment/run02/event_matches.csv) lists all eligible detected/missed event identifiers (record number + numeric event index), exclusions, exact alarm timestamps and IDs. Those recording identifiers are not patient identifiers. [lead_times.csv](../results/alarm_budget_experiment/run02/lead_times.csv) contains only correctly matched leads. There are no learned-model true detections of the three eligible 91/92 episodes either. No figure is needed to clarify these all-zero learned sensitivities; tables retain exact denominators.

### Timestamp classification is secondary

These are the same current-normal held-out rows used by the policy evaluation, not the historical full future-low population. Precision is undefined when there are no alerts, and zero when alerts exist but none is correct. Recall is zero for all learned models; none has a true-positive timestamp here. Timestamp metrics do not count independent clinical episodes. They describe the full common current-normal population; repeating them on severe-subset summary rows does not turn them into severe-only timestamp metrics.

| Target | Model | Budget | Samples | TP | FP | FN | Timestamp_Precision | Timestamp_Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| future_low | Always_Alert | 1.0000 | 16050 | 501 | 15549 | 0 | 0.0312 | 1.0000 |
| future_low | Always_Alert | 5.0000 | 16050 | 501 | 15549 | 0 | 0.0312 | 1.0000 |
| future_low | Always_Alert | 10.0000 | 16050 | 501 | 15549 | 0 | 0.0312 | 1.0000 |
| future_low | ELM | 1.0000 | 16050 | 0 | 0 | 501 | undefined | 0.0000 |
| future_low | ELM | 5.0000 | 16050 | 0 | 0 | 501 | undefined | 0.0000 |
| future_low | ELM | 10.0000 | 16050 | 0 | 0 | 501 | undefined | 0.0000 |
| future_low | GaussianNB | 1.0000 | 16050 | 0 | 1 | 501 | 0.0000 | 0.0000 |
| future_low | GaussianNB | 5.0000 | 16050 | 0 | 1 | 501 | 0.0000 | 0.0000 |
| future_low | GaussianNB | 10.0000 | 16050 | 0 | 1 | 501 | 0.0000 | 0.0000 |
| future_low | Logistic | 1.0000 | 16050 | 0 | 0 | 501 | undefined | 0.0000 |
| future_low | Logistic | 5.0000 | 16050 | 0 | 0 | 501 | undefined | 0.0000 |
| future_low | Logistic | 10.0000 | 16050 | 0 | 1 | 501 | 0.0000 | 0.0000 |
| future_low | Never_Alert | 1.0000 | 16050 | 0 | 0 | 501 | undefined | 0.0000 |
| future_low | Never_Alert | 5.0000 | 16050 | 0 | 0 | 501 | undefined | 0.0000 |
| future_low | Never_Alert | 10.0000 | 16050 | 0 | 0 | 501 | undefined | 0.0000 |
| future_low | Tree | 1.0000 | 16050 | 0 | 0 | 501 | undefined | 0.0000 |
| future_low | Tree | 5.0000 | 16050 | 0 | 0 | 501 | undefined | 0.0000 |
| future_low | Tree | 10.0000 | 16050 | 0 | 0 | 501 | undefined | 0.0000 |
| new_onset | Always_Alert | 1.0000 | 16050 | 501 | 15549 | 0 | 0.0312 | 1.0000 |
| new_onset | Always_Alert | 5.0000 | 16050 | 501 | 15549 | 0 | 0.0312 | 1.0000 |
| new_onset | Always_Alert | 10.0000 | 16050 | 501 | 15549 | 0 | 0.0312 | 1.0000 |
| new_onset | ELM | 1.0000 | 16050 | 0 | 0 | 501 | undefined | 0.0000 |
| new_onset | ELM | 5.0000 | 16050 | 0 | 0 | 501 | undefined | 0.0000 |
| new_onset | ELM | 10.0000 | 16050 | 0 | 0 | 501 | undefined | 0.0000 |
| new_onset | GaussianNB | 1.0000 | 16050 | 0 | 0 | 501 | undefined | 0.0000 |
| new_onset | GaussianNB | 5.0000 | 16050 | 0 | 1 | 501 | 0.0000 | 0.0000 |
| new_onset | GaussianNB | 10.0000 | 16050 | 0 | 3 | 501 | 0.0000 | 0.0000 |
| new_onset | Logistic | 1.0000 | 16050 | 0 | 0 | 501 | undefined | 0.0000 |
| new_onset | Logistic | 5.0000 | 16050 | 0 | 23 | 501 | 0.0000 | 0.0000 |
| new_onset | Logistic | 10.0000 | 16050 | 0 | 52 | 501 | 0.0000 | 0.0000 |
| new_onset | Never_Alert | 1.0000 | 16050 | 0 | 0 | 501 | undefined | 0.0000 |
| new_onset | Never_Alert | 5.0000 | 16050 | 0 | 0 | 501 | undefined | 0.0000 |
| new_onset | Never_Alert | 10.0000 | 16050 | 0 | 0 | 501 | undefined | 0.0000 |
| new_onset | Tree | 1.0000 | 16050 | 0 | 173 | 501 | 0.0000 | 0.0000 |
| new_onset | Tree | 5.0000 | 16050 | 0 | 173 | 501 | 0.0000 | 0.0000 |
| new_onset | Tree | 10.0000 | 16050 | 0 | 173 | 501 | 0.0000 | 0.0000 |

### Training selection behavior and held-out constraint failures

The following counts explain abstention without reinterpreting training selection performance as independent evidence. `No_Training_Alerts` means the selected threshold generated no training alert; unseen outer scores can exceed that threshold. It does not forcibly disable the model on the test recording.

| Target | Model | Budget | Outer_Folds | No_Training_Alerts | Maximum_Train_Active |
| --- | --- | --- | --- | --- | --- |
| future_low | ELM | 1.0000 | 30 | 29 | 0.0001 |
| future_low | ELM | 5.0000 | 30 | 26 | 0.0012 |
| future_low | ELM | 10.0000 | 30 | 22 | 0.0029 |
| future_low | GaussianNB | 1.0000 | 30 | 30 | 0.0000 |
| future_low | GaussianNB | 5.0000 | 30 | 27 | 0.0012 |
| future_low | GaussianNB | 10.0000 | 30 | 27 | 0.0012 |
| future_low | Logistic | 1.0000 | 30 | 30 | 0.0000 |
| future_low | Logistic | 5.0000 | 30 | 30 | 0.0000 |
| future_low | Logistic | 10.0000 | 30 | 24 | 0.0068 |
| future_low | Tree | 1.0000 | 30 | 29 | 0.0000 |
| future_low | Tree | 5.0000 | 30 | 29 | 0.0000 |
| future_low | Tree | 10.0000 | 30 | 29 | 0.0000 |
| new_onset | ELM | 1.0000 | 26 | 26 | 0.0000 |
| new_onset | ELM | 5.0000 | 26 | 26 | 0.0000 |
| new_onset | ELM | 10.0000 | 26 | 25 | 0.0010 |
| new_onset | GaussianNB | 1.0000 | 26 | 26 | 0.0000 |
| new_onset | GaussianNB | 5.0000 | 26 | 14 | 0.0009 |
| new_onset | GaussianNB | 10.0000 | 26 | 9 | 0.0010 |
| new_onset | Logistic | 1.0000 | 26 | 26 | 0.0000 |
| new_onset | Logistic | 5.0000 | 26 | 16 | 0.0029 |
| new_onset | Logistic | 10.0000 | 26 | 10 | 0.0053 |
| new_onset | Tree | 1.0000 | 26 | 26 | 0.0000 |
| new_onset | Tree | 5.0000 | 26 | 26 | 0.0000 |
| new_onset | Tree | 10.0000 | 26 | 26 | 0.0000 |


Each constraint-violation count below is out of **26 records with positive eligible exposure**; the other four remain zero-exposure rows. Pooled compliance can hide a violation on an individual short recording.

| Target | Model | Budget | Budget_Violating_Records | Active_Cap_Violating_Records |
| --- | --- | --- | --- | --- |
| future_low | ELM | 1.0000 | 0 | 0 |
| future_low | ELM | 5.0000 | 0 | 0 |
| future_low | ELM | 10.0000 | 0 | 0 |
| future_low | GaussianNB | 1.0000 | 1 | 0 |
| future_low | GaussianNB | 5.0000 | 1 | 0 |
| future_low | GaussianNB | 10.0000 | 1 | 0 |
| future_low | Logistic | 1.0000 | 0 | 0 |
| future_low | Logistic | 5.0000 | 0 | 0 |
| future_low | Logistic | 10.0000 | 0 | 0 |
| future_low | Tree | 1.0000 | 0 | 0 |
| future_low | Tree | 5.0000 | 0 | 0 |
| future_low | Tree | 10.0000 | 0 | 0 |
| new_onset | ELM | 1.0000 | 0 | 0 |
| new_onset | ELM | 5.0000 | 0 | 0 |
| new_onset | ELM | 10.0000 | 0 | 0 |
| new_onset | GaussianNB | 1.0000 | 0 | 0 |
| new_onset | GaussianNB | 5.0000 | 1 | 0 |
| new_onset | GaussianNB | 10.0000 | 1 | 0 |
| new_onset | Logistic | 1.0000 | 0 | 0 |
| new_onset | Logistic | 5.0000 | 3 | 0 |
| new_onset | Logistic | 10.0000 | 1 | 0 |
| new_onset | Tree | 1.0000 | 4 | 1 |
| new_onset | Tree | 5.0000 | 4 | 1 |
| new_onset | Tree | 10.0000 | 2 | 1 |

The onset tree selected zero training alerts in every fold/budget but produced 57 off-event outer starts, **19.8663/hour** pooled—above every budget—with one recording violating the active cap. New outer refit scores crossed training maxima, demonstrating operating-point transfer failure. These results were not repaired by lowering/raising thresholds on the test data. Onset logistic at budgets 5/10 produced 4/14 false starts respectively and no true event; ELM stayed silent. The data do not establish that an arbitrary other model/policy could not work, only that this frozen finite-grid selection procedure with the current fixed models failed to provide useful warning.

## Per-recording results and uncertainty

[per_recording.csv](../results/alarm_budget_experiment/run02/per_recording.csv) retains all 30 recordings for each model/target/budget/window/severe subset, including eligible/detected/missed counts, false/unmatched starts, exposure, active seconds/fraction, precision/recall and lead quantiles. [outer_constraint_status.csv](../results/alarm_budget_experiment/run02/outer_constraint_status.csv) gives per-record violations without excluding them. In particular recordings 12/26/30 remain included and their eligible events are all missed by the learned models. Four no-opportunity records are explicitly represented with zero exposure and undefined rates rather than silently discarded.

Uncertainty uses **2,000 paired unstratified whole-recording bootstrap draws**, seed **20261014**, sampled from all 30 records. The identical multiplicity matrix applies to every model/budget/target/window. Complete record contributions (counts, exposure, active time, matched leads) are repeated; streams are never stitched across different recordings or copies for alarm matching. Negative-only and zero-exposure records stay in sampling. Pool rates as sums of numerators divided by sums of exposure; do not average per-record false rates. Undefined zero-event/zero-detection draws are counted explicitly rather than converted to zero.

The following are illustrative sensitivity-analysis-panel intervals at budget 5 and 30 seconds for onset-trained models; the other budgets were also computed and remain in the output. Showing this panel is not selection of a validated operating budget.

| Model | Metric | Lower_95 | Upper_95 | Defined_Replicates | Replicates |
| --- | --- | --- | --- | --- | --- |
| ELM | Sensitivity | 0.0000 | 0.0000 | 1996 | 2000 |
| ELM | False_Alarm_Starts_Per_Hour | 0.0000 | 0.0000 | 2000 | 2000 |
| ELM | Alarm_Active_Fraction | 0.0000 | 0.0000 | 2000 | 2000 |
| ELM | Pulse_Alert_Fraction | 0.0000 | 0.0000 | 2000 | 2000 |
| ELM | Lead_Median_s | undefined | undefined | 0 | 2000 |
| GaussianNB | Sensitivity | 0.0000 | 0.0000 | 1996 | 2000 |
| GaussianNB | False_Alarm_Starts_Per_Hour | 0.0000 | 1.1881 | 2000 | 2000 |
| GaussianNB | Alarm_Active_Fraction | 0.0000 | 0.0004 | 2000 | 2000 |
| GaussianNB | Pulse_Alert_Fraction | 0.0000 | 0.0002 | 2000 | 2000 |
| GaussianNB | Lead_Median_s | undefined | undefined | 0 | 2000 |
| Logistic | Sensitivity | 0.0000 | 0.0000 | 1996 | 2000 |
| Logistic | False_Alarm_Starts_Per_Hour | 0.0000 | 3.2626 | 2000 | 2000 |
| Logistic | Alarm_Active_Fraction | 0.0000 | 0.0031 | 2000 | 2000 |
| Logistic | Pulse_Alert_Fraction | 0.0000 | 0.0033 | 2000 | 2000 |
| Logistic | Lead_Median_s | undefined | undefined | 0 | 2000 |
| Tree | Sensitivity | 0.0000 | 0.0000 | 1996 | 2000 |
| Tree | False_Alarm_Starts_Per_Hour | 0.3526 | 55.2778 | 2000 | 2000 |
| Tree | Alarm_Active_Fraction | 0.0001 | 0.0260 | 2000 | 2000 |
| Tree | Pulse_Alert_Fraction | 0.0001 | 0.0317 | 2000 | 2000 |
| Tree | Lead_Median_s | undefined | undefined | 0 | 2000 |

All zero-sensitivity bootstrap intervals are [0,0] because every observed event was missed. **That degenerate empirical interval is not proof that population sensitivity is exactly zero**: resampling these records cannot generate an unobserved success. All learned lead-median intervals are undefined with zero valid replicates. There are 1,996/2,000 draws with an eligible 30-second event. The bootstrap conditions on saved CV fits/selected policies; it does not estimate the full variation from refitting/reselecting, remove shared-training dependence between folds, or establish patient-level independence. Only six event-containing recordings and 14 eligible onsets (three severe-interest episodes) limit inference. No multiplicity-adjusted superiority claim is made.

[bootstrap_intervals.csv](../results/alarm_budget_experiment/run02/bootstrap_intervals.csv) saves every model/budget/window/target interval and valid-replicate count. [paired_bootstrap.csv](../results/alarm_budget_experiment/run02/paired_bootstrap.csv) reports ELM-minus-logistic differences using the same recording weights. [bootstrap_weights.csv.gz](../results/alarm_budget_experiment/run02/bootstrap_weights.csv.gz) allows exact numerical reconstruction. Since neither model detects events, no useful sensitivity advantage follows from the paired calculation.

## Preservation, tests and reproducibility

The [new implementation](../src/alarm_budget_experiment.py) and [regression tests](../tests/test_alarm_budget.py) are additions only. Existing data/features/labels/preprocessing/model configurations/outer grouping/event matching/replay modules, saved models, historical thresholds and earlier result directories were not modified. [preservation.json](../results/alarm_budget_experiment/preservation.json) fingerprints all 682 pre-existing raw/result/archive/report/source artifacts captured before this experiment, including the previous uncommitted local-30 work.

The initial run completed and saved every training selection, then stopped before producing outer evaluation artifacts on a zero-event record because an empty table lacked matcher columns. The correction is confined to the **new experiment's empty-table schema handling**, with regression tests. It does not change any event/matching rule. `run01` retains its training-only artifacts. `run02` revalidates every saved fold's training provenance and each frozen candidate winner, copies the frozen selections byte-for-byte and completes evaluation; its selection checksum equals run01's. This avoids repeating threshold selection or adapting it to outer outcomes. Both attempts and the frozen protocol remain preserved. The final experiment results are **run02**.

Tests cover outer-test/OOF provenance isolation, inner recording separation, refusal of out-of-partition raw/events, deterministic train-only budget choices, complete score ties, nonmonotonic start counts, active-cap enforcement, zero-event schema handling, shared-alert/lead boundaries, missing/low numeric exposure, fixed bootstrap seeds, unchanged frozen selections, exact saved outer decisions and independent audited recomputation of all saved record-level alarm/exposure counts. Existing causal-feature, train-only transformation, label-window, model-loading and historical quality tests are rerun unchanged.

**Final executed validation: 126 tests passed in 62.226 seconds, zero skips/failures/errors.** The experiment gate and both historical saved-run project gates passed; all 682 protected artifacts match their fingerprints. All 60 raw CSVs remain unchanged. The completed selections and candidate tables match run01 byte-for-byte; 246 documentation links resolve. Actual final test outcomes, project quality gates and preservation verification are recorded in [verification.json](../results/alarm_budget_experiment/verification.json) and [test_results.txt](../results/alarm_budget_experiment/test_results.txt). The run [manifest](../results/alarm_budget_experiment/run02/manifest.json) fingerprints inputs, source and numerical outputs; [selection_complete.json](../results/alarm_budget_experiment/run02/selection_complete.json) binds protocol and thresholds before outer evaluation.

Executed commands:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.alarm_budget_experiment --output results/alarm_budget_experiment/run01
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.alarm_budget_experiment --output results/alarm_budget_experiment/run02 --selection-source results/alarm_budget_experiment/run01
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.alarm_budget_experiment --output results/alarm_budget_experiment/run02 --verify
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.check_project --links --verify-run results/final/future_low
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.check_project --links --verify-run results/final/new_onset
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.check_project --audit
git diff --check
```

The run01 command stopped as documented; run02 is the completed result. For clean reproduction with the corrected code and the frozen protocol, choose a **new** directory under the same root, e.g. `python -m src.alarm_budget_experiment --output results/alarm_budget_experiment/reproduction`. The CLI refuses existing destinations and outputs outside this experiment root. It reuses authenticated nested score caches and never downloads or fits new model configurations. `--verify` is read-only. The optional `--selection-source` must have the same frozen protocol and an intact selection receipt; its winners and training partitions are revalidated.

Machine-readable outputs include training candidate/selected metrics, exact threshold representations, recording splits, timestamp-level outer decisions, all-event detection/miss identifiers, alarm starts, per-record and aggregate summaries, actual constraint violations, leads and paired uncertainty. Baselines are explicitly marked unselected. New thresholds are fold-specific offline evaluation artifacts, not deployment thresholds for full-data demonstration models. No latency or clinical intervention experiment is claimed here.

## Limitations and conclusion

These 30 recordings have already been examined repeatedly; this is a prespecified final analysis on an existing research cohort, not an untouched confirmatory validation. Patient-to-recording linkage remains unverified; LORO and recording bootstrap measure held-out-recording variation, not proven unseen-patient generalization. Labels use single observed monitor threshold crossings rather than clinical adjudication, monitor delay is unknown, opportunities/exposure are restricted by complete ground truth, and sustained alerts/eligibility restarts can distort naive event sensitivity. Conditional bootstrap and the finite quantile grid cannot establish the best possible threshold or model. The budget/active cap is a research utility assumption, not a clinical standard.

**Conclusion:** the current four fixed models, paired with this frozen training-only alarm-budget selector, provide no demonstrated useful early warning on the available held-out recordings. Their useful-sensitivity/common-burden objective was not achieved; some outer policies also violated the selected burden constraints. Never-alert meets the limits but detects nothing; always-alert detects events while violating the active cap and is not evidence of model reliability. No further model/feature tuning, budget search, recording removal or dataset acquisition was performed. There is no clinical-readiness or automated oxygen-delivery claim.
