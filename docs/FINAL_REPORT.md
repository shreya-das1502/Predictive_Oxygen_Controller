# Final implementation and verification report

**Completed follow-up (16 October 2026):** the original policy results below are preserved. The separately frozen ML study has now been finalized in [ML_IMPROVEMENT_REPORT.md](ML_IMPROVEMENT_REPORT.md). Three GaussianNB conditions detected the same single nonsevere event while meeting pooled burden assumptions; all violated the start budget on at least one recording. This does not establish robust early warning or a winning model.

## Final submission summary

The final planned study used only the **30 local recordings**. Under the frozen alarm-budget sensitivity analysis, **none of the four learned models detected an eligible event**: 0/15 at 5, 10 and 15 seconds and 0/14 at 30 seconds, for both original training targets and all tested budgets. The research assumptions were **1, 5 and 10 false alarm starts per evaluable hour with a 5% alarm-active-time cap**; they are not clinically validated limits. Some held-out recordings violated constraints met during training-only threshold selection. The selected policy did not demonstrate useful early warning or proven model superiority.

This report preserves the earlier implementation, numerical results and validation record below, explicitly labeled historical. The **Final evaluation: alarm-budget experiment** section reconciles the earlier 14/14 result with the final 0/14 result and provides the final discussion/conclusion. Full methods, per-recording/event outputs and uncertainty are in [ALARM_BUDGET_EXPERIMENT.md](ALARM_BUDGET_EXPERIMENT.md). No clinical validation, patient-independent validation or automated oxygen-delivery capability is established.

## Dataset reconciliation and targets

The retained legacy [dataset](../results/elm_dataset.csv) has 17,089 rows, 466 positives and 26 recordings. The current complete-window future-low dataset has 20,157 rows, 4,587 positives and 30 recordings. These are different populations and feature-extraction timelines, not a performance improvement attributable to relabeling alone. [Per-record count reconciliation](../results/final/dataset_reconciliation.csv) provides exact counts.

The archived legacy builder (`src/elm_experiment.py` in `archive/legacy_sources.zip`) filtered out current SpO₂ ≤94%, thereby removing recordings 04/08/09/22. It also used noncausal filtering/global prominence and inferred a 100 Hz clock from rounded timestamps. The current [preprocessing](../src/preprocessing.py) uses the verified 125 Hz sample clock and causal filtering; [features](../src/features.py) uses past/current pulses only. The saved independent legacy label audit found 15,706 fully observable agreeing labels, 1,096 incomplete horizons and 287 invalid horizons (1,383 unverifiable labels). Original results are retained rather than silently recomputed.

The current [label implementation](../src/labeling.py) requires complete valid numeric coverage over `(t,t+30s]` and maximum numeric gap 1.01 s. `future_low` is minimum observed SpO₂ ≤94%, including ongoing lows. `new_onset` additionally requires a fresh current reading >94% and an adjacent valid >94%→≤94% transition within that horizon. One low sample qualifies; no sustained-event or clinical episode definition is implied. This yields 16,050 rows, 501 positives, 26 recordings and six positive-containing recordings. Of the future-low rows, 4,107 are ineligible for onset: 4100 have current SpO₂ ≤94% and 7 lack current SpO₂. On fully valid currently normal windows the two labels agree. Pulse times changed, so the remaining legacy/current difference cannot be assigned a one-to-one row waterfall.

## Historical recall-oriented evaluation and measured results

Fresh evaluations rebuilt both datasets (`cache_reused=false`). Each outer test recording is excluded from fitting and three-fold grouped training-only OOF threshold selection. Median imputation/scaling are fitted inside each training fold. All models see identical eligible held-out rows within each target. For these historical results, the threshold policy was the largest training-OOF threshold meeting 92% timestamp recall; it was not adjusted using these test results. Positive-containing and negative-only recordings, pooled metrics and defined-value macro metrics are reported separately.

Numbers below are pooled held-out timestamps. AP is average precision, not trapezoidal PR area. FP/FN count timestamps rather than distinct episodes.

### future_low

| Model | Recall | Precision | Specificity | F1 | AP | FP | FN | False episodes/hour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ELM | 0.9649 | 0.2390 | 0.0949 | 0.3831 | 0.2559 | 14093 | 161 | 140.69 |
| Logistic | 0.9320 | 0.2196 | 0.0243 | 0.3555 | 0.2403 | 15192 | 312 | 27.70 |
| GaussianNB | 0.9821 | 0.2257 | 0.0072 | 0.3670 | 0.1963 | 15458 | 82 | 13.99 |
| Tree | 0.8949 | 0.2146 | 0.0353 | 0.3462 | 0.1895 | 15021 | 482 | 28.52 |

### new_onset

| Model | Recall | Precision | Specificity | F1 | AP | FP | FN | False episodes/hour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ELM | 0.8902 | 0.0309 | 0.1010 | 0.0598 | 0.0240 | 13979 | 55 | 182.94 |
| Logistic | 0.8862 | 0.0282 | 0.0161 | 0.0547 | 0.0335 | 15299 | 57 | 21.60 |
| GaussianNB | 0.9002 | 0.0332 | 0.1563 | 0.0641 | 0.0576 | 13118 | 50 | 60.63 |
| Tree | 1.0000 | 0.0312 | 0.0000 | 0.0605 | 0.0396 | 15549 | 0 | 9.76 |

The historical approximately 96.5% recall/3% precision conflated populations: current future-low ELM precision is 23.90%, whereas onset precision is 3.09%. The [before/after metric table](../results/final/before_after_metrics.csv) verifies that historical hardened and fresh fixed-model metrics agree; this cleanup/visualization pass did not alter their predictions. Legacy v2 results are not a matched before/after comparison because eligibility, labeling validity and feature processing differ.

On onset, ELM alerts on 89.88% of eligible timestamps. Logistic alerts on 98.09%, GaussianNB on 84.54%, and the tree always alerts. GaussianNB has better pooled AP than ELM; no overall ELM superiority is supported. Low false-episode counts can arise from one very long continuous alert: interpret episodes/hour alongside alert fraction and specificity. Exposure covers eligible pulse segments, not all wall-clock recording duration. Record 26 accounts for 37 ELM false negatives, record 30 for 17, record 7 for one; record 12 has none. Detailed score/threshold timelines are plotted for 12/26/30.

### Historical reusable-positive event metric

All models detect 14/14 eligible observed onsets and 3/3 eligible 91–92% episodes; the severe-subset median lead is 9.784 s. An always-alert reference also detects all of them, and many observed transitions lack eligible prior opportunities. These small-denominator overlap-based event results do not demonstrate useful anticipation. The event tables document excluded events and opportunity rules.

## Historical computational latency (unchanged)

The fresh benchmarks were executed sequentially. The following pulse-to-alert computational measurements include rolling feature updates, the model path, eligibility and thresholding (starting from an already extracted pulse). Separate feature, transform/model, threshold, per-waveform-sample replay, pulse-producing update and warm model-loading measurements, raw timing samples, environment and repetitions are saved per target. P99 is reported only with at least 1,000 repetitions; 100 model-load repetitions support median/P95 only.

| Target | Model | Median ms | P95 ms | P99 ms |
| --- | --- | --- | --- | --- |
| future_low | ELM | 0.903 | 3.828 | 6.053 |
| future_low | Logistic | 0.612 | 1.027 | 1.141 |
| future_low | GaussianNB | 0.703 | 0.999 | 1.239 |
| future_low | Tree | 0.596 | 1.046 | 1.415 |
| new_onset | ELM | 0.489 | 0.743 | 0.904 |
| new_onset | Logistic | 0.616 | 0.864 | 1.157 |
| new_onset | GaussianNB | 0.707 | 0.810 | 1.236 |
| new_onset | Tree | 0.599 | 0.684 | 1.018 |

Timing varies on this shared host, including slower future-low ELM timings. These measurements exclude physiological sensor averaging, waiting for samples/pulses and the 30-second prediction horizon. They are not a real-time or hardware guarantee.

## Historical implementation, cleanup and artifacts

- Targets/configuration/cache: [labeling](../src/labeling.py), [config](../src/config.py), [dataset](../src/dataset.py), [provenance](../src/provenance.py). Cache identities cover relevant preprocessing/features/labels/config/source fingerprints; artifact hashes detect edited prepared outputs.
- Evaluation/runtime: [evaluation](../src/evaluation.py), [runner](../src/run_experiments.py), [model](../src/model.py), [controller](../src/controller.py), [replay](../src/replay.py), [target reports](../src/target_reporting.py). Manifests record targets, features, seeds, parameters, fingerprints, thresholds and recording splits. Full-data demo models are separate from held-out predictions.
- Analysis/verification: [figures](../src/figures.py), [latency benchmark](../src/benchmark_latency.py), [quality gate](../src/check_project.py), and target/saved-result/figure tests. Raw latency distributions and source-bound numerical chart tables make figures auditable.
- Cleanup: 26 legacy source files and the prior README were archived byte-for-byte; 28 historical-only root outputs were archived. Two checkpoint CSVs were deleted only after matching retained counterparts exactly. Empty directories and disposable bytecode caches were removed. Raw data, historical validated/hardened evidence, legacy dataset and v2 results/predictions were retained. [Every file decision](FILE_CLEANUP.md), [dependency evidence](CLEANUP_DEPENDENCIES.json), [archive recovery](../archive/README.md).

[Requested figure hub](../results/validated/figures/INDEX.md) links 23 PNG/PDF figure pairs per target, without duplicating image data or overwriting historical predictions. Included are raw signals/SpO₂, actual PPG pulse peaks, interval/rate quality proxies, missingness/clock issues, class balance, feature distributions/correlation, exploratory held-out block-permutation importance, matched model metrics/PR/ROC/confusion, OOF thresholds, recording timelines/errors, events/lead times and raw timing distributions. PPG pulse peaks are not ECG R-peaks. Permutation analysis uses fixed refits whose scores must match saved predictions; it does not tune models.

[Future-low run](../results/final/future_low/) and [onset run](../results/final/new_onset/) contain `EVALUATION.md`, predictions, comparison/per-record tables, `run_manifest.json`, `dataset_cache.json`, `models/`, event diagnostics and latency samples. Large local prepared data/model files are Git-ignored and regenerate from the documented commands.

## Historical reproduction and verification

The commands and test counts in this section record the earlier implementation pass, not this documentation review or the final operating policy. Do not rerun generation commands against preserved result directories; use a new directory and the final experiment report for final-policy reproduction. The complete pipeline diagram, module-by-module mapping and commands are in [FULL_PIPELINE_GUIDE](FULL_PIPELINE_GUIDE.md). Start with the environment/raw-data inspection:

```bash
python -m src.check_project --audit
python -m src.run_experiments --target future_low --output results/final/future_low
python -m src.run_experiments --target new_onset --output results/final/new_onset
python -m unittest discover -s tests -v
python -m src.check_project --links --verify-run results/final/future_low
python -m src.check_project --links --verify-run results/final/new_onset
python -m src.figures --output results/final/new_onset
```

For this pass, both fixed-model evaluations were actually executed with `--skip-benchmark`, then both latency commands, target-report commands and figure commands were executed separately. This avoided concurrent training during timing. Dataset CLI output was checked against the final onset dataset hash; an archived-source restore into a new temporary directory succeeded; the saved Logistic onset model was loaded in a separate replay process for record 12 (748 pulse outputs). No package installation, data download, broad search, clinical validation or hardware deployment was performed.

The full suite executed **76 tests in 23.165 seconds, all passed with zero skips**. Both saved-run gates passed, 141 local links resolved, all 85 protected historical files matched their hashes, and source/result archives restored or hash-verified successfully. The README inference example and separate saved-model replay executed successfully. Final test and consistency results are recorded in [quality_gate.json](../results/final/quality_gate.json). Remaining limitations include only 30 short recordings, few onset/severe events, rounded waveform timestamps, single-reading threshold events, no validated signal-quality gate, highly correlated pulse features, uncalibrated pooled scores across refits, and excessive alerts. Future research should prespecify an operational false-alarm objective and validate on additional recordings before considering deployment; this pass stops before that model-selection work.

## Final evaluation: alarm-budget experiment

### Methods and reconciliation of 14/14 with 0/14

The final experiment retained the same fixed ELM, logistic regression, GaussianNB and shallow-tree configurations, features, target labels and outer leave-one-recording-out score fits. Preprocessing, imputation, scaling and model fitting had occurred within the relevant training folds. New fold-specific thresholds were selected solely from each outer-training partition's grouped inner OOF scores. No held-out outcomes selected thresholds, models or budgets. All selections were frozen before new outer evaluation; historical model files and thresholds were retained separately.

There was no clinically defensible primary budget available from these recordings. The prespecified **1, 5 and 10 false alarm starts per evaluable hour** and **5% active-time cap** are research sensitivity assumptions, not clinical tolerances. Within each outer-training partition, the fixed candidate grid maximized 30-second one-to-one event detections subject to both constraints; ties favored fewer off-event starts, less active time and a higher threshold. All budgets were reported without choosing one from outer performance. Training feasibility is not a guarantee of constraint attainment on a held-out recording.

The apparent change from 14/14 to 0/14 involves **both a matching-rule distinction and a different operating policy**. The earlier 14/14 counted any positive timestamp inside a pre-onset window, even if the alarm was already active; nearby events could share alerts. It did not mean 14 newly starting alarms or useful early warning. The intermediate audit applied one-to-one alarm-start matching while keeping the original recall-selected thresholds. The final experiment retained that audited matching rule and changed only threshold selection to the frozen alarm-budget policy. For the onset-trained models at the same 14 eligible 30-second events:

| Stage | Training-only threshold objective | Event matching | ELM | Logistic | GaussianNB | Tree |
| --- | --- | --- | --- | --- | --- | --- |
| Earlier historical result | At least 92% OOF timestamp recall | Any positive in window; reuse permitted | 14/14 | 14/14 | 14/14 | 14/14 |
| Audited fixed-policy result | Same recall-selected thresholds | One-to-one actual alarm starts | 13/14 | 9/14 | 9/14 | 8/14 |
| Final alarm-budget experiment | Event detections subject to research budget and active cap | Same audited one-to-one actual alarm starts | 0/14 | 0/14 | 0/14 | 0/14 |

The final row holds at all three budgets and for both target training populations; the intermediate row explicitly refers to onset-trained models. The original `future_low` target still includes ongoing lows in training; the final alarm policy evaluates both fitted target models on the same current-normal population. This is not a silent label change or a claim that historical numerical results were wrong. See the [reliability audit](RELIABILITY_AUDIT.md), [local-30 fixed-policy evaluation](LOCAL_30_RECORDING_EVALUATION.md) and [final experiment report](ALARM_BUDGET_EXPERIMENT.md) for the preserved stages and their denominators.

### Metric terminology and denominators

| Quantity | Definition and interpretation |
| --- | --- |
| Timestamp-level precision/recall | TP/(TP+FP) and TP/(TP+FN) over eligible pulse decisions. Overlapping positive timestamps are not distinct events. Historical classification tables above retain their original target populations and thresholds. |
| Event sensitivity | Detected eligible onsets / eligible onsets under the stated window and matching rule. Final matching assigns each actual alarm start and event at most once. |
| Historical false episodes/hour | Positive decision runs containing no positive future-target timestamp, divided by historical evaluable exposure. A long run with one positive target is not a false episode under that definition, despite many false-positive timestamps. |
| Final false alarm starts/hour | New alarm starts with no qualifying observed onset in `(start,start+W]`, divided by strict current-normal evaluable hours. This is not the historical false-episode metric; redundant/unmatched starts are also recorded separately. |
| Alarm-active fraction | Left-held positive-decision duration divided by permitted inter-pulse exposure. It is time-weighted, not the fraction of pulses that alert. |
| Pulse alert fraction | Positive decisions / eligible pulse decisions. The earlier 89.88% ELM and 98.09% logistic values refer to pulse counts at recall-selected thresholds, not active duration or final-budget thresholds. |
| Lead time | Onset time minus an actual matched alarm-start time. Median/IQR condition on correctly matched events; final learned-model leads are undefined because none was detected. |

Final exposure is **10,329.032 seconds (2.869176 evaluable hours)** per model/target, including negative-only recordings' eligible time. Four entirely-low recordings have no current-normal opportunities; they remain zero-exposure rows in the full 30-record roster, not performance-based exclusions. Invalid/missing/current-low gaps are not counted as eligible exposure. Historical false-episode rates use their original exposure conventions and must not be substituted for final false-start rates.

An alarm already active before `[onset−W,onset)` is not restarted at its left boundary. An alarm exactly at onset is too late. The 5/10/15/30-second windows are look-back intervals, not guarantees of at least that much warning. Budgets were selected against the 30-second false-start definition; shorter-window rates and held-out rates need not satisfy it.

### Final results

All four learned models missed every eligible event: **15/15 missed at each of 5, 10 and 15 seconds; 14/14 missed at 30 seconds**, across both target populations and all tested budgets. None detected any of the **three severe-interest events** at 30 seconds either. These events contain observed 91/92 readings; they are not clinically adjudicated severe events. The onset-trained model outcomes at 30 seconds are reproduced below without changing the saved results:

| Research budget (starts/h) | Model | Detected/eligible | Off-event starts | Exposure (s) | False starts/h | Active fraction |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | ELM | 0/14 | 0 | 10329.032 | 0.0000 | 0.000000 |
| 1 | GaussianNB | 0/14 | 0 | 10329.032 | 0.0000 | 0.000000 |
| 1 | Logistic | 0/14 | 0 | 10329.032 | 0.0000 | 0.000000 |
| 1 | Tree | 0/14 | 57 | 10329.032 | 19.8663 | 0.008912 |
| 5 | ELM | 0/14 | 0 | 10329.032 | 0.0000 | 0.000000 |
| 5 | GaussianNB | 0/14 | 1 | 10329.032 | 0.3485 | 0.000116 |
| 5 | Logistic | 0/14 | 4 | 10329.032 | 1.3941 | 0.001342 |
| 5 | Tree | 0/14 | 57 | 10329.032 | 19.8663 | 0.008912 |
| 10 | ELM | 0/14 | 0 | 10329.032 | 0.0000 | 0.000000 |
| 10 | GaussianNB | 0/14 | 3 | 10329.032 | 1.0456 | 0.000200 |
| 10 | Logistic | 0/14 | 14 | 10329.032 | 4.8795 | 0.003478 |
| 10 | Tree | 0/14 | 57 | 10329.032 | 19.8663 | 0.008912 |

The onset tree produced 57 off-event starts, about 19.87/hour, despite zero alerts under its training-selected thresholds. Some individual held-out recordings violated the training-selected start budget and one tree test recording violated the active cap. Thus common training constraints did not produce identical achieved outer-test burdens. There was no post-test threshold correction. Per-record violations, both target results and all windows remain in the [final experiment outputs](../results/alarm_budget_experiment/run02/).

Always-alert detected 8/14 under the final matching rule because state/coverage gaps can restart its alarms; its active fraction is 100%, so it violates the cap and is not a fair common-burden comparator. Never-alert meets the constraints by detecting nothing. Neither reference supports an early-warning superiority claim.

### Final discussion, limitations and conclusion

The selected policy did **not demonstrate useful early warning**. Mostly silent or false-only outer predictions, and failures to transfer some training constraints, are the actual findings. This conclusion is limited to the current fixed models and finite-grid threshold procedure; it does not establish that every possible detector or threshold would fail. No additional model, feature or budget tuning followed the result.

Inference is limited by **30 recordings**, only **14 eligible events at 30 seconds** from six event-containing recordings, **three severe-interest events**, retrospectively and repeatedly examined data, uncertain monitor delay, single-reading threshold events and no clinical validation. Patient-to-recording linkage is unverified: held-out-recording evaluation is not proven unseen-patient evaluation. Paired recording-bootstrap intervals condition on the saved fits/policies and do not establish patient independence or full training-procedure uncertainty. A degenerate [0,0] empirical sensitivity interval after zero observed detections is not proof that population sensitivity is exactly zero.

**Final academic and presentation conclusion:** this is an offline, reproducible comparison with a negative operating-policy result. High historical timestamp recall, reusable-positive 14/14 sensitivity, clipped-window lead times and sub-millisecond computational timing do not demonstrate clinically reliable anticipation, oxygen-delivery capability or proven model superiority. Every presentation of the earlier metrics must identify its target, threshold objective, matching rule and exposure denominator. Earlier figures describe their saved recall-policy runs and should not be presented as alarm-budget results. The full [ALARM_BUDGET_EXPERIMENT.md](ALARM_BUDGET_EXPERIMENT.md) is the definitive final experiment account; earlier reports, numerical tables and result artifacts remain preserved. The final planned experiment used no additional data and establishes no clinical readiness.

### Documentation consistency review

This documentation-only review checked 259 local links, verified the three-stage reconciliation and 12 displayed final-budget rows against the saved summaries, and checked all 96 learned-model target/budget/window combinations. All pre-existing numerical table rows were retained; the other 722 files in the pre-edit snapshot, including data, source, tests and experiment results, remained unchanged. The alarm-budget, reliability-audit and local-30 reports were retained byte-for-byte. Earlier experiment test counts above are historical execution records, not new model-test results from this review.

Historical preservation receipts include pre-review hashes of this report, the guide and the older model report. Those receipts were deliberately retained rather than rewritten; strict document-byte preservation checks will flag the authorized documentation revisions. The experimental artifacts and numerical results were not regenerated.

## Completed bounded ML follow-up — separate from the original final policy

All1120 completed model/fold groups,6720 training-selected cutoffs and144 saved evaluation conditions were reused; no scientific CV/OOF fits or new policies were selected. The original30s/base10/raw event-policy results reproduced all192 learned summary conditions. The new [complete report](ML_IMPROVEMENT_REPORT.md) and [machine-readable comparisons](../results/ml_completion/run01/before_after_comparison.csv) retain both targets, all fixed families, actual trainedH5/10/15/30, the context18-atH30 ablation, raw/training-negative-rank scores, all budgets and all matchingW. The original shorterW tables above still describe H30-trained models.

| New-onset GaussianNB/base10/negative-rank, configured budget10 | Detected / eligible | False starts/evaluable h | Alarm active% | Median lead s | Recording-bootstrap sensitivity95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| H=W10 | 1/15 | 1.394129 | 0.0377 | 9.824 | [0,0.428571] |
| H=W15 | 1/15 | 0.697064 | 0.0245 | 9.824 | [0,0.428571] |
| H=W30 | 1/14 | 1.394129 | 0.0505 | 13.440 | [0,0.428571] |

Source: [event_summary.csv](../results/ml_completion/run01/event_summary.csv), [recording intervals](../results/ml_completion/run01/recording_bootstrap_intervals.csv) and [per-record results](../results/ml_completion/run01/event_per_recording.csv). The same record07 onset367s supplies all three detections; it is a single94% reading, not one of the91/92-interest events. Every detecting condition violates budget10 on at least one recording; none is a demonstrated robust policy. All three severe-interest events remain missed in primary learned comparisons. ELM/logistic/tree, future-low and context18 show no primary event gain. No outer-result-selected winner is recommended.

The follow-up saves2000 shared recording-bootstrap draws(seed20261015), complete before/after tables, five traceable figures and actual raw latency samples. Warm transform/model/normalization/threshold median0.200–0.550ms, P950.244–1.342ms across measured conditions; feature medians0.081–0.159ms. These are distinct timers, exclude acquisition/monitor delay and do not show clinical benefit or guaranteed warning. Forty outer-training-only timing refits reproduced saved last-record scores exactly; scientific predictions/thresholds were not replaced.

A saved-array verification passed7820712 predictions,3456 event conditions and6720 cutoffs before authorized submission-document updates. Historical preservation assertions remain unchanged and their document exceptions are disclosed separately. Missing execution-time OOF/reference hashes for1074 recovered groups remain a provenance limitation, not reconstructed evidence. See [completion tracker](ML_COMPLETION_TRACKER.md) for final commands, checks, preservation accounting and remaining browser/patient limitations.
