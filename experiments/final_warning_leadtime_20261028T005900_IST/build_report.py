from pathlib import Path
import pandas as pd
r=Path(__file__).resolve().parent;s=pd.read_csv(r/'overall_summary.csv');p=pd.read_csv(r/'paired_lead_time_summary.csv');d=pd.read_csv(r/'paired_warning_rate_differences.csv');e=pd.read_csv(r/'event_level_lead_times.csv');per=pd.read_csv(r/'recording_level_summary.csv')
def md(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n'
main=md(['Model','Definition','Detected: history → combined','≥10 s: history → combined','Missed: history → combined','Missed or <10 s: history → combined'],[[m,de,f'{int(a.Detected)}/{int(a.Eligible_Events)} → {int(b.Detected)}/{int(b.Eligible_Events)}',f'{int(a.Timely10)}/{int(a.Eligible_Events)} → {int(b.Timely10)}/{int(b.Eligible_Events)}',f'{int(a.Missed_Events)} → {int(b.Missed_Events)}',f'{int(a.MissedOrInsufficient10)} → {int(b.MissedOrInsufficient10)}'] for m in ['Logistic','GaussianNB','ELM'] for de in ['original','sustained3'] for a in [s[(s.Model==m)&(s.Definition==de)&(s.Modality=='SpO2_history')].iloc[0]] for b in [s[(s.Model==m)&(s.Definition==de)&(s.Modality=='Combined')].iloc[0]]])
paired=md(['Family (both definitions)','Paired events / recordings','Earlier / equal / later','Median Δ lead, s (95% CI)','Mean Δ lead, s (95% CI)','Defined bootstrap draws'],[[q.Model,f'{q.PairedEvents} / {q.PairedRecordings}',f'{q.Earlier} / {q.Equal} / {q.Later}',f'{q.MedianAdditionalLead_s:.3f} [{q.MedianLower95_s:.3f}, {q.MedianUpper95_s:.3f}]',f'{q.MeanAdditionalLead_s:.4f} [{q.MeanLower95_s:.3f}, {q.MeanUpper95_s:.3f}]',f'{q.MedianDefinedBootstrapReplicates}/2000'] for q in p[p.Definition.eq('original')].itertuples()])
burden=md(['Model','Definition','Modality','Off-event starts / evaluable hour','Alarm starts','Active time, s (%)','Record violations: starts / activity'],[[q.Model,q.Definition,q.Modality,f'{q.FalseAlarmStartsPerEvaluableHour:.3f}',int(q.Alarm_Starts),f'{q.Alarm_Active_s:.3f} ({100*q.AlarmActiveFraction:.3f}%)',f'{q.RecordStartBudgetViolations} / {q.RecordActiveCapViolations}'] for q in s.itertuples()])
ci=md(['Family','Definition','Coverage threshold','Combined−history rate','Recording-bootstrap 95% CI'],[[q.Model,q.Definition,'Detection' if q.MinimumLead_s==0 else '≥10 s',f'{q.CombinedMinusHistoryRate:+.4f}',f'[{q.Lower95:+.4f}, {q.Upper95:+.4f}]'] for q in d[d.MinimumLead_s.isin([0,10])].itertuples()])
g=e[e.Model.eq('Logistic')&e.Definition.eq('original')&e.History_Detected.eq(True)&e.Combined_Detected.eq(True)];detail=md(['Recording:event','History lead, s','Combined lead, s','Additional lead, s'],[[f'{q.Record}:{q.Event_ID}',f'{q.History_Lead_s:.3f}',f'{q.Combined_Lead_s:.3f}',f'{q.Additional_Lead_s:+.3f}'] for q in g.itertuples()])
report=f'''# Does adding PPG reduce desaturation warning delay?

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

{main}

Rates use all eligible events in each definition. Counts at previously used 0, 5, 10, 15, 20 and 30-second lead requirements are in `timely_warning_summary.csv`; these are fixed descriptive operating requirements, not clinically validated response-time needs. The 0-second condition is detection under the strict pre-onset rule; no observed match is actually at onset. `timely_warning_comparison.png` plots these all-event denominators with recording-cluster intervals.

For logistic original truth, the one-event net gain combines gains at 7:367, 12:210 and 26:144 with losses at 30:72 and 30:98. It is not the same nine detections plus one. Gains at 12:210 and 26:144 have only 6.320 and 5.576 s lead. The isolated 7:367 gain has 16.728 s lead and is removed from sustained truth. Among shared detections, 26:310 improves from 2.368 to 24.312 s and crosses the 10-second requirement, but the two lost record30 events had 17.136 and 19.560 s leads. These changes explain why the original ≥10-second count is unchanged and the sustained count worsens.

## Conditional paired timing

{paired}

Both definitions have the same paired detections here. Logistic's seven paired events span only four recordings. Its positive mean is influenced by two large improvements and has an interval crossing zero; the median is zero. ELM's apparently positive 10.336 s is based on ONE event in ONE recording while total detections collapse from six to one. Its degenerate conditional bootstrap interval cannot support a population timing claim.

{detail}

**The both-detected subset is conditional and may favor easier-to-detect events. It cannot alone establish overall superiority.** Misses and model disagreements remain in the coverage comparisons above. `warning_lead_time_comparison.png` displays every eligible event; missed warnings occupy a separately labelled band, not an artificial numerical lead.

## Recording-cluster uncertainty

The existing scheme is preserved: 2,000 resamples of 30 whole recording IDs with replacement, NumPy seed 20261027. The same bootstrap multiplicities apply to both modalities and all contrasts. Within-record events stay together. Mean/median paired lead is recomputed from repeated whole-record contributions; coverage differences retain all events, including misses. Draws with no eligible events or no paired detections are undefined and excluded from the corresponding percentile interval, with defined-draw counts reported. There are 1,995/2,000 defined original-coverage draws, 1,991 sustained-coverage draws, and 1,977 logistic paired-timing draws. The saved integer weight matrix makes resampling reproducible.

{ci}

Only six original event-bearing recordings, five sustained-event recordings and four logistic paired-timing recordings contribute outcome information; 30 recordings are NOT established as 30 independent patients. Intervals are conditional on saved overlapping cross-validation fits, with no retraining or uncertainty about model fitting. Recordings with zero eligible events remain in the bootstrap universe. This previously examined cohort and multiple model/lead comparisons make results exploratory. No multiplicity-adjusted confirmatory superiority or significance is claimed. Some exploratory intervals may exclude zero, especially for ELM coverage losses, but this cannot certify population superiority or clinical reliability.

## Alarm burden and feasibility

Total source duration is 14,400 s (4 hours), but evaluable current-normal exposure is only 10,329.032 s = 2.869176 hours, contributed by 26 recordings. The four zero-exposure recordings retain rows with undefined burden rates. Every rate below uses evaluable exposure, NOT 4 hours of source duration.

{burden}

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
'''
(r/'REPORT.md').write_text(report)
(r/'limitations_and_claims.md').write_text('''# Supported claims and limits

Supported: a reproducible, frozen-model retrospective comparison on the same 30 recordings; logistic original detection 9/14 vs10/14; original ≥10-second coverage6/14 each; sustained detection9/13 each and timely6/13 vs5/13; logistic paired median0 s with interval−4 to+10.064 s; inconsistent gains/losses and individual recording burden violations. GaussianNB/ELM coverage worsens with PPG.

Unsupported: robust incremental PPG superiority, consistent ≥10-second advance, clinical safety, independent patient generalization, reduced clinician response time, reduced physical oxygen-delivery delay, clinical benefit, deployment readiness, or validated pulse-detection accuracy.

Limits: only14/13 eligible events in6/5 recordings; patient linkage unknown;7 paired logistic events in4 recordings; prior cohort examination; conditional bootstrap over overlapping saved fits, not model refits;3 families and multiple descriptive lead requirements; intermittent current-normal exposure and future-dependent retrospective evaluability; monitor SpO₂ timing/isolated low truth; no clinical adjudication or real human PPG annotations; no independent/prospective cohort; no alarm acknowledgement/actuator/flow logs. Uninterpretable or missing reference is not zero error. Post-onset warnings are not assessed by the existing pre-onset matching rule; missed means no qualifying pre-onset alarm, not evidence of absence of later activity.

Freeze the analysis/models as bounded research artifacts. Human pulse annotations and clinical reference review remain essential to signal-quality and event-validity claims. Independent patient-aware evaluation and prospective alarm/device measurements remain essential to generalization and delivery-delay claims. No refitting or threshold changes were made in this analysis.
''')
(r/'README.md').write_text('''# Frozen warning lead-time analysis

Read [REPORT.md](REPORT.md) for results and [limitations_and_claims.md](limitations_and_claims.md) for claim boundaries. Input and output hashes, parameters, commands, verification status and preservation results are in manifest.json. No human annotations or intervention records were invented.

Primary tables: event_level_lead_times.csv (all22 episodes ×3 families ×2 definitions;132 rows), recording_level_summary.csv (30 recordings ×3 families ×2 modalities ×2 definitions;360 rows), paired_lead_time_summary.csv (6 contrasts), overall_summary.csv, timely_warning_summary.csv, paired_warning_rate_differences.csv. CSV blank means undefined/not applicable, never a missed-event zero lead. Event timestamps are seconds from recording start. Model identifies the classifier family; each event row contains BOTH modalities.

Alarm details: alarm_episodes.csv, plus exposure-weighted duration and burden in recording_level_summary.csv. Figures are warning_lead_time_comparison.png and timely_warning_comparison.png. Bootstrap weights are recording_bootstrap_weights.npy (2000×30, columns record1..30). source_hashes.json and preservation_before.json retain provenance.

Reproduce WITHOUT overwriting this directory, from repository root:

```bash
MPLCONFIGDIR=/tmp/ppg-leadtime-mpl XDG_CACHE_HOME=/tmp/ppg-leadtime-cache /workspace/.venvs/oxygen/bin/python experiments/final_warning_leadtime_20261028T005900_IST/analyze.py --out experiments/final_warning_leadtime_reproduction_NEW
```

The output directory must not already exist. Reproduction recomputes tables/figures/verification from frozen sources and does not fit models. This command does not regenerate narrative reports or the final archival manifest; those are preserved with the original run.

Focused tests:

```bash
MPLCONFIGDIR=/tmp/ppg-leadtime-mpl XDG_CACHE_HOME=/tmp/ppg-leadtime-cache /workspace/.venvs/oxygen/bin/python -m unittest discover -s experiments/final_warning_leadtime_20261028T005900_IST/tests -v
```

Existing focused suite: experiments.incremental_value_20261027.test_study, experiments.final_audit_20261027T215012_IST.tests.test_audit, tests.test_alarm_budget, tests.test_reliability. One older artifact test fails because results/final/future_low/causal_dataset.csv is absent; do not claim the repository suite is green.
''')
