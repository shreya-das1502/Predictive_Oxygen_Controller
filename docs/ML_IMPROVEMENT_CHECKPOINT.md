# ML experiment progress checkpoint — paused by user

No experiment is running. The evaluation command was stopped safely with Ctrl-C, exit **130**, during recording-bootstrap summary construction. No new experiment was launched after the pause request. Essential subsequent work was limited to read-only checks of saved tables/hashes and writing this checkpoint.

## Inspected and changed files

The baseline audit inspected the original `src/config.py`, `src/preprocessing.py`, `src/features.py`, `src/labeling.py`, `src/dataset.py`, `src/model.py`, `src/evaluation.py`, `src/run_experiments.py`, `src/reliability_evaluation.py`, `src/alarm_budget_experiment.py`, `src/local_30_evaluation.py`, `src/check_project.py` and `src/benchmark_latency.py`; the existing final/reliability/local-30/alarm-budget reports; saved target datasets, pulse tables, held-out predictions, training-OOF predictions, policies, manifests and split receipts. All 30 local numeric recordings supply ground truth; existing waveform/source fingerprints remain verified. This pass does not claim a new waveform replay or browser test.

New ML-task files (production source files were not changed):

- `ml_research/__init__.py`, `baseline.py`, `core.py`, `experiment.py`, `reporting.py`, `summarize.py`, `verify.py`: isolated original audit/experiment workflow, frozen before fitting.
- `ml_research/recover.py`, `checkpoint_evaluation.py`, `complete_evaluation.py`: interruption recovery and checkpointed evaluation; scientific model settings and event formulas unchanged.
- `tests/test_ml_research.py`, `tests/test_ml_recovery.py`, `tests/test_ml_checkpoint.py`: causal/split/selection and recovery regressions, including interruption and corruption tests.
- `docs/ML_IMPROVEMENT_BASELINE.md`, `docs/ML_IMPROVEMENT_VERIFICATION.md`, this checkpoint report.
- Separate outputs under `results/ml_improvement/{run01,recovery01,recovery02,evaluation01,verification01,checkpoint01}/`. These include new derived labels, predictions, selected policies, test logs and audit receipts. Historical directories are untouched.

The existing working-tree changes in README, FINAL_REPORT, FULL_PIPELINE_GUIDE, MODEL_EVALUATION, dashboard and signal-studies files predate this ML continuation and were preserved. `docs/ML_IMPROVEMENT_REPORT.md` has **not** been generated.

## Completed, stopped and pending work

**Completed:** verified baseline; all **1,120 model/fold groups** across both targets, four fixed families and five feature/horizon settings; two declared score spaces; **6,720 training-only cutoff selections**; all **144 evaluation conditions**, each integrity-checkpointed. Saved outputs include **103,680 event-per-recording rows**, event matches, full timestamp summaries/per-record metrics and the combined prediction table. Seven original frozen source hashes and **765 protected files** still match. Every complete condition checkpoint hash was verified. The checkpointed evaluator reproduces all **620 surviving timestamp-summary rows** from the interrupted evaluator within explicit numerical tolerance.

**Stopped:** recording-bootstrap aggregate/interval construction. No hang was established; it lacked incremental summary progress and was stopped at the user's request. The per-condition work remains saved.

**Pending:** bootstrap and paired-comparison summaries, latency measurements, figures, original event-policy parity assertion at the end of aggregation, final run manifest/read-only verifier, final link checks and `ML_IMPROVEMENT_REPORT.md`. The full experiment is therefore **not finalized or fully verified**.

## Actual baseline and provisional measurements

Verified original budget policy: every learned family, both original targets and budgets 1/5/10 detected **0/14** at W=30, **0/15** at W=5/10/15, and **0/3** severe-interest events. These earlier windows were matching windows for 30-second-trained models.

Historical recall-oriented ELM timestamp metrics, which are a different operating policy/population:

| Target | Recall | Precision | AP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| future_low | 96.49% | 23.90% | 0.255929 | 14,093 | 161 |
| new_onset | 89.02% | 3.09% | 0.023976 | 13,979 | 55 |

Read-only provisional aggregation of the saved new tables found only three detecting primary learned conditions, all **new-onset GaussianNB/base10/training-negative-rank at research budget 10**:

| Trained H / matching W | Detected | Off-event starts/evaluable hour | Active time | Lead |
| --- | ---: | ---: | ---: | ---: |
| 10 s | 1/15 | 1.394129 | 0.0377% | 9.824 s |
| 15 s | 1/15 | 0.697064 | 0.0245% | 9.824 s |
| 30 s | 1/14 | 1.394129 | 0.0505% | 13.440 s |

All refer to the **same record-07 onset at 367 s**, not three independent events; it is not a 91/92 event. Saved alert timestamps 357.176 s (H10/H15) and 353.560 s (H30) were checked against actual saved scores, thresholds and positive decisions. Although these pooled rates meet their research assumptions, **each condition violates the start budget on at least one individual recording**: record 26 at H10/H15; records 7 and 26 at H30. The result is not a robust detector or a selected winning model. All three severe-interest events remain missed. No ELM, logistic or future-low improvement in primary matched-horizon event detections was measured. These are provisional point estimates without the unfinished intervals/final verification; they must not be presented as a completed validated improvement.

Machine-readable provisional tables and the pause receipt are in [checkpoint01](../results/ml_improvement/checkpoint01/). No historical result was overwritten.

## Exact stopped command and tests

```bash
MPLCONFIGDIR=/tmp/oxygen-ml-mpl OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -u -m ml_research.complete_evaluation --source results/ml_improvement/recovery02 --output results/ml_improvement/evaluation01
```

Unified execution session **5504**, stopped with Ctrl-C; exit **130**. No experiment remains running.

Latest complete test command:

```bash
MPLCONFIGDIR=/tmp/oxygen-ml-tests-mpl OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

**159 tests in 220.373 s: 156 passed, one failure, two errors, no skips.** Unsuccessful tests are the same original preservation checks: `test_target_hardening.SavedTargetTests.test_historical_outputs_preserved`, `test_local_30.Local30ArtifactTests.test_outputs_and_protected_files_are_unchanged`, and `test_alarm_budget.AlarmBudgetArtifactTests.test_inputs_outputs_and_preservation_verified`. They concern prior authorized document changes; old expected hashes match Git HEAD. No receipt/assertion was changed. Both original saved-run quality gates passed. See [verification record](ML_IMPROVEMENT_VERIFICATION.md) and its durable logs.

## Errors and repeated work

Two environment interruptions lost in-memory evaluation summaries. The original fitting interruption left 1,074 groups; those were reused and **only 46 missing scientific groups** were fit. A first recovery attempt failed on duplicated metadata keyword columns before new fitting; it was repaired with a regression test, and its files/log were retained. Forty outer-training-only refits rebuilt timing objects during recovery02; they did not replace scientific predictions and were subsequently lost before any timing was measured.

The noncheckpointed evaluation was partially repeated after the second interruption because its per-record summaries existed only in memory. This rebuilt saved outcomes, not scientific model fits or selected thresholds. The replacement checkpoints now preserve each completed condition. Full tests were repeated when recovery code/tests changed; logs retain each count/result. The earlier 5–10 minute estimate was too optimistic and was corrected after the interruption was discovered.

The work is paused. No completed experiment should be restarted. Any authorized continuation should reuse all 144 verified condition checkpoints and frozen selections, then finish only the pending reporting/verification stages.
