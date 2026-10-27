# Does adding PPG reduce desaturation warning delay?

Analysis date: 28 October 2026 (Asia/Calcutta). Frozen 30-recording retrospective comparison. No new predictions, fits, cutoffs, event annotations or signal annotations were created.

## Main finding

**The existing evidence does not establish that adding PPG produces earlier useful warnings.** Logistic regression has a one-event original-definition gain (9/14 → 10/14), but no gain in ≥10-second coverage (6/14 each), and a zero-second median paired lead improvement. Sustained-event detection is tied (9/13 each), while ≥10-second coverage decreases (6/13 → 5/13). GaussianNB and ELM lose substantial detection coverage. Individual recording alarm constraints are violated. No physical oxygen-delivery delay or clinical benefit was measured.

## Sources and reproducibility

Source experiment: `experiments/incremental_value_20261027/`. Exact outputs: `outputs/classification_predictions.csv.gz`, `matched_dataset.csv.gz`, `raw_event_audit.csv`, `event_matches.csv`, `event_per_recording.csv`, `event_summary.csv`, and `fold_ledger.json`. Inputs include all 30 existing numeric recordings. The preceding `experiments/final_audit_20261027T215012_IST/` report, reconstructed metrics, causal audit source, paired-lead table and lead-coverage table provide independent consistency checks.

`source_hashes.json` records exact source/checkpoint hashes. Historical fit/preparation receipts were checked; 180 relevant outer folds were inspected for held-out-record exclusion, grouped inner partitions, completed checkpoint hashes and unchanged training-only threshold winners. No model was loaded for refitting. All three previously specified classifier families are reported; logistic is discussed in detail as the established transparent reference, not selected as a new held-out winner. `analyze.py`, commands and software versions are indexed by `manifest.json`.

The preservation baseline includes 5,052 pre-existing files. The final manifest reports the actual preservation result. Historical data/predictions/experiments are read-only throughout this analysis.

## Definitions and timing

Original reference is an observed SpO₂ episode ≤94%, starting after a valid immediately preceding normal observation. There are 22 observed episodes: seven left-censored/unknown-prior episodes and 15 qualifying onsets. One qualifying onset at record26/time22 has insufficient 30-second raw lead-in. Thus 14 eligible original events span six recordings. The alternate sustained definition additionally requires ≥3 consecutive low readings spanning ≥2 seconds; it excludes the isolated record7/time367 reading, leaving 13 eligible events across five recordings. Original and sustained truth are scored separately against the SAME saved predictions/cutoffs. The event-level CSV retains all 22 episode identities for every family and definition, including excluded and not-in-sustained rows: 132 rows total. Excluded rows have missing detection/lead values and explicit reasons; they are not counted as missed eligible events.

The target is minimum observed SpO₂ ≤94% in `(t,t+30 s]`. Decisions occur at original pulse-confirmation timestamps, not a uniform 1-second grid. Waveform sampling is 125 Hz, and the pulse confirmation is 8 ms after its peak; SpO₂ sampling is approximately 1 Hz. Only current-normal opportunities with complete future coverage enter the preserved event evaluation. Warning threshold remains fold-specific, selected on training-record out-of-fold predictions to maximize one-to-one detections subject to off-event starts ≤5/evaluable hour and active fraction ≤5%. These are research constraints, not validated clinical limits.

Alarm runs retain actual starts, split only by the established connectivity rules (normal, valid ≤2-second pulse edges). Events are processed chronologically and matched to the earliest unused actual alarm start in `[onset−30,onset)`. An alarm cannot be reused for two events or clipped/restarted at a window boundary. Overlap ambiguity is therefore governed by an existing rule; no new matching rule or matching sensitivity analysis was introduced. An independent stream implementation reproduced all 12 family/modality/definition comparisons.

Lead = onset − actual warning start. Additional lead = combined lead − history lead, defined ONLY when both models detect the event. All qualifying matches must have lead >0 and ≤30 s. The code and tests verify arithmetic, boundaries and one-to-one assignment. Misses have blank lead/timestamp fields, never zero-second surrogate leads.

**At-onset/post-onset warning latency is unavailable under this pre-onset matching rule.** The table explicitly identifies detected warnings as before onset and marks misses as no qualifying pre-onset warning. It does not mislabel a missed pre-onset warning as proof that no later alarm occurred. Current-low rows are excluded from operational event matching, so a valid after-onset response-time analysis cannot be inferred from these matched outputs. A detected warning with <10 s lead is termed insufficient lead, not a measured post-onset alarm.

Causal feature verification: all saved PPG peak times precede confirmation by 0.008 s; SpO₂ history implementation restricts observations to at/before that confirmation. Prediction rows, feature rows and threshold comparisons were checked for identity; historical causal/partition tests were rerun. The earlier audit independently reconstructed causal history and pulse features and inspected training transforms; its artifacts are hash-sealed here. The retrospective requirement for complete FUTURE reference coverage is used to define evaluability/exposure, not as a deployed causal gate. It remains a limitation. Inner fitted transform objects were not retained historically; this run does not claim a new full refit audit.

## Detection and useful-warning coverage

| Model | Definition | Detected: history → combined | ≥10 s: history → combined | Missed: history → combined | Missed or <10 s: history → combined |
| --- | --- | --- | --- | --- | --- |
| Logistic | original | 9/14 → 10/14 | 6/14 → 6/14 | 5 → 4 | 8 → 8 |
| Logistic | sustained3 | 9/13 → 9/13 | 6/13 → 5/13 | 4 → 4 | 7 → 8 |
| GaussianNB | original | 10/14 → 6/14 | 6/14 → 4/14 | 4 → 8 | 8 → 10 |
| GaussianNB | sustained3 | 10/13 → 6/13 | 6/13 → 4/13 | 3 → 7 | 7 → 9 |
| ELM | original | 6/14 → 1/14 | 3/14 → 1/14 | 8 → 13 | 11 → 13 |
| ELM | sustained3 | 6/13 → 1/13 | 3/13 → 1/13 | 7 → 12 | 10 → 12 |


Rates use all eligible events in each definition. Counts at previously used 0, 5, 10, 15, 20 and 30-second lead requirements are in `timely_warning_summary.csv`; these are fixed descriptive operating requirements, not clinically validated response-time needs. The 0-second condition is detection under the strict pre-onset rule; no observed match is actually at onset. `timely_warning_comparison.png` plots these all-event denominators with recording-cluster intervals.

For logistic original truth, the one-event net gain combines gains at 7:367, 12:210 and 26:144 with losses at 30:72 and 30:98. It is not the same nine detections plus one. Gains at 12:210 and 26:144 have only 6.320 and 5.576 s lead. The isolated 7:367 gain has 16.728 s lead and is removed from sustained truth. Among shared detections, 26:310 improves from 2.368 to 24.312 s and crosses the 10-second requirement, but the two lost record30 events had 17.136 and 19.560 s leads. These changes explain why the original ≥10-second count is unchanged and the sustained count worsens.

## Conditional paired timing

| Family (both definitions) | Paired events / recordings | Earlier / equal / later | Median Δ lead, s (95% CI) | Mean Δ lead, s (95% CI) | Defined bootstrap draws |
| --- | --- | --- | --- | --- | --- |
| Logistic | 7 / 4 | 2 / 3 / 2 | 0.000 [-4.000, 10.064] | 3.8080 [-4.000, 10.219] | 1977/2000 |
| GaussianNB | 5 / 3 | 1 / 4 / 0 | 0.000 [0.000, 5.072] | 2.0288 [0.000, 5.072] | 1920/2000 |
| ELM | 1 / 1 | 1 / 0 / 0 | 10.336 [10.336, 10.336] | 10.3360 [10.336, 10.336] | 1260/2000 |


Both definitions have the same paired detections here. Logistic's seven paired events span only four recordings. Its positive mean is influenced by two large improvements and has an interval crossing zero; the median is zero. ELM's apparently positive 10.336 s is based on ONE event in ONE recording while total detections collapse from six to one. Its degenerate conditional bootstrap interval cannot support a population timing claim.

| Recording:event | History lead, s | Combined lead, s | Additional lead, s |
| --- | --- | --- | --- |
| 17:250 | 9.784 | 9.784 | +0.000 |
| 17:454 | 25.600 | 25.600 | +0.000 |
| 26:57 | 20.616 | 19.264 | -1.352 |
| 26:294 | 13.112 | 23.176 | +10.064 |
| 26:310 | 2.368 | 24.312 | +21.944 |
| 29:368 | 6.368 | 6.368 | +0.000 |
| 30:30 | 14.960 | 10.960 | -4.000 |


**The both-detected subset is conditional and may favor easier-to-detect events. It cannot alone establish overall superiority.** Misses and model disagreements remain in the coverage comparisons above. `warning_lead_time_comparison.png` displays every eligible event; missed warnings occupy a separately labelled band, not an artificial numerical lead.

## Recording-cluster uncertainty

The existing scheme is preserved: 2,000 resamples of 30 whole recording IDs with replacement, NumPy seed 20261027. The same bootstrap multiplicities apply to both modalities and all contrasts. Within-record events stay together. Mean/median paired lead is recomputed from repeated whole-record contributions; coverage differences retain all events, including misses. Draws with no eligible events or no paired detections are undefined and excluded from the corresponding percentile interval, with defined-draw counts reported. There are 1,995/2,000 defined original-coverage draws, 1,991 sustained-coverage draws, and 1,977 logistic paired-timing draws. The saved integer weight matrix makes resampling reproducible.

| Family | Definition | Coverage threshold | Combined−history rate | Recording-bootstrap 95% CI |
| --- | --- | --- | --- | --- |
| Logistic | original | Detection | +0.0714 | [-0.4000, +0.5714] |
| Logistic | original | ≥10 s | +0.0000 | [-0.4310, +0.3379] |
| Logistic | sustained3 | Detection | +0.0000 | [-0.5000, +0.4659] |
| Logistic | sustained3 | ≥10 s | -0.0769 | [-0.5333, +0.1429] |
| GaussianNB | original | Detection | -0.2857 | [-0.7053, +0.2500] |
| GaussianNB | original | ≥10 s | -0.1429 | [-0.4160, +0.2500] |
| GaussianNB | sustained3 | Detection | -0.3077 | [-0.7143, +0.2667] |
| GaussianNB | sustained3 | ≥10 s | -0.1538 | [-0.4286, +0.2667] |
| ELM | original | Detection | -0.3571 | [-0.5714, -0.2000] |
| ELM | original | ≥10 s | -0.1429 | [-0.2500, +0.0000] |
| ELM | sustained3 | Detection | -0.3846 | [-0.6667, -0.2500] |
| ELM | sustained3 | ≥10 s | -0.1538 | [-0.2682, +0.0000] |


Only six original event-bearing recordings, five sustained-event recordings and four logistic paired-timing recordings contribute outcome information; 30 recordings are NOT established as 30 independent patients. Intervals are conditional on saved overlapping cross-validation fits, with no retraining or uncertainty about model fitting. Recordings with zero eligible events remain in the bootstrap universe. This previously examined cohort and multiple model/lead comparisons make results exploratory. No multiplicity-adjusted confirmatory superiority or significance is claimed. Some exploratory intervals may exclude zero, especially for ELM coverage losses, but this cannot certify population superiority or clinical reliability.

## Alarm burden and feasibility

Total source duration is 14,400 s (4 hours), but evaluable current-normal exposure is only 10,329.032 s = 2.869176 hours, contributed by 26 recordings. The four zero-exposure recordings retain rows with undefined burden rates. Every rate below uses evaluable exposure, NOT 4 hours of source duration.

| Model | Definition | Modality | Off-event starts / evaluable hour | Alarm starts | Active time, s (%) | Record violations: starts / activity |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic | original | SpO2_history | 3.485 | 29 | 140.248 (1.358%) | 3 / 4 |
| Logistic | original | Combined | 4.879 | 33 | 98.568 (0.954%) | 3 / 3 |
| Logistic | sustained3 | SpO2_history | 3.485 | 29 | 140.248 (1.358%) | 3 / 4 |
| Logistic | sustained3 | Combined | 5.577 | 33 | 98.568 (0.954%) | 3 / 3 |
| GaussianNB | original | SpO2_history | 4.182 | 26 | 669.648 (6.483%) | 4 / 5 |
| GaussianNB | original | Combined | 3.485 | 26 | 517.720 (5.012%) | 4 / 4 |
| GaussianNB | sustained3 | SpO2_history | 4.182 | 26 | 669.648 (6.483%) | 4 / 5 |
| GaussianNB | sustained3 | Combined | 3.485 | 26 | 517.720 (5.012%) | 4 / 4 |
| ELM | original | SpO2_history | 4.182 | 26 | 34.040 (0.330%) | 5 / 2 |
| ELM | original | Combined | 1.394 | 9 | 10.504 (0.102%) | 2 / 1 |
| ELM | sustained3 | SpO2_history | 4.182 | 26 | 34.040 (0.330%) | 5 / 2 |
| ELM | sustained3 | Combined | 1.394 | 9 | 10.504 (0.102%) | 2 / 1 |


Off-event false alarms are actual alarm starts with no qualifying onset in `(start,start+30 s]`, using all observed qualifying onsets, including those excluded from sensitivity. They differ from unmatched starts: redundant starts near a true onset need not be false. Sustained truth recomputes off-event burden using that alternate event set. Active duration is time-weighted over valid edges, not the sum of timestamp counts or a device-on duration. An alarm's last positive timestamp is provided in `alarm_episodes.csv`; it is not substituted for the prespecified exposure-weighted active-duration calculation.

Logistic original: history has 10 off-event starts, 29 alarm starts, 140.248 active seconds; combined has 14 off-event starts, 33 starts, 98.568 active seconds. Both meet pooled original research constraints, but BOTH violate constraints on individual exposed recordings. History exceeds the start budget in records7,26,30 and active cap in17,26,29,30; combined exceeds starts in3,7,26 and activity in17,26,29. Record29 has only 6.264 s evaluable exposure and 100% activity in both. Combined's record7 off-event rate is 72.566/evaluable hour. These unstable short-exposure rates do not demonstrate long-term rates, but they preclude a claim that all recordings meet the stated constraints. Under sustained truth, combined's pooled off-event rate is 5.577/h, exceeding the 5/h budget.

`recording_level_summary.csv` contains all 360 recording/family/modality/definition rows, zero-event/missing-denominator cases, timely/missed counts, durations and separate budget flags. Research budget adherence must not be described as clinical safety.

## Prediction versus intervention

The available artifacts contain signal timestamps and retrospective predictions. No timestamped clinician alarm acknowledgement, oxygen-command, actuator, flow-delivery or patient-response logs are available for this matched experiment. Therefore:

1. Predictive lead time: calculable as above.
2. Alarm response time: unavailable.
3. Physical oxygen-delivery response time: unavailable.
4. Clinical outcome or benefit: unavailable.

A positive predictive lead indicates only potential time to initiate an intervention. It does not measure monitor reporting delay, clinician action, oxygen transport, physiological response or patient benefit. The reference is displayed SpO₂ rather than arterial oxygen ground truth. No oxygen settings or autonomous treatment are recommended.

## Explicit research conclusions and freeze decision

- Did combined detect more? Logistic: yes by one original event, but tied for sustained truth. GaussianNB: 10→6; ELM: 6→1, under both event definitions.
- Did combined warn earlier on shared events? Inconsistently: logistic two earlier, three equal, two later. Median additional lead is **0.000 s**, 95% conditional interval **[−4.000,+10.064] s**; mean +3.808 s, interval [−4.000,+10.219].
- ≥10-second warning counts? Logistic original **6/14 each**; sustained **6/13 history versus 5/13 combined**. All other families are reported above rather than selectively omitted.
- Did it reduce missed or insufficient warnings overall? Logistic original misses decrease 5→4, but missed-or-<10-second events remain 8/14 each. Sustained misses are tied 4/13, while missed-or-insufficient increases 7/13→8/13. No consistent useful-warning improvement is demonstrated.
- Acceptable individual alarm burden? **No.** Both logistic modalities violate per-recording constraints; combined sustained burden also fails the pooled start budget.
- Incremental PPG predictive value? **Not established by this evaluation.** The mixed logistic gain does not robustly improve useful warning timing, depends on event definition and is accompanied by event losses and burden violations. This does not prove PPG can never help under other methods or populations.
- Earlier oxygen intervention? Some observed warnings create a theoretical intervention opportunity. Neither an overall improvement in that opportunity nor reduced actual oxygen-delivery delay or patient benefit has been demonstrated.
- Freeze? **Freeze this bounded retrospective analysis and the existing models as research artifacts. Do not label the oxygen controller validated or deployment-ready.** Essential outstanding work includes independent human PPG reference review (the 60-window annotation workflow still awaits actual reviewers), clinical review of event validity, independent/patient-aware replication and prospective alarm/device timing validation before claims in those domains. None of these were fabricated or replaced by synthetic software tests. Do not retune these thresholds to manufacture a gain.

## Verification and deviations

All 12 comparisons were reconstructed from saved scores/decisions and raw numeric references, checked against historical per-recording counts and an independent stream matcher. All 72 lead-coverage cells match the previous audit. Output schema, 132 event rows, event counts, missing-value semantics, timestamp arithmetic, one-to-one matches, bootstrap grouping and plot/table consistency were checked.

Focused analysis tests: **6/6 pass**. Relevant existing tests: **67 run, 66 pass, one pre-existing artifact error**. The error is `tests.test_alarm_budget.AlarmBudgetArtifactTests.test_inputs_outputs_and_preservation_verified`, missing `results/final/future_low/causal_dataset.csv`. This belongs to an older alarm-budget receipt, not the frozen incremental predictions used here. It is not reconstructed. Full repository suite was not run and is not claimed to pass. Logs retain exact traces. During development an output round-trip test caught boolean-valued single-event timely counts in the new recording CSV; they were explicitly cast to integers, results regenerated, and final focused tests passed. No historical metrics were affected.

No scientific protocol changes: models, cohort, scores, decisions, thresholds, horizon, event rules, exposure and matching remain fixed. New work adds explicit paired timing intervals and useful-lead reporting under already used requirements. After-onset latency and oxygen-delivery outcomes remain unavailable because required timing evidence is absent. No matching sensitivity was needed because a documented convention already existed. See `limitations_and_claims.md`, `verification.json`, `manifest.json` and logs for the precise scope.
