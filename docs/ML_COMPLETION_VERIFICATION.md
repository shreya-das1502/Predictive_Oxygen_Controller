# ML completion verification — 2026-10-16

New outputs: `results/ml_completion/run01/`. Scientific inputs remain under `results/ml_improvement/`; no additional data were acquired. All 1,120 model/fold groups and 6,720 selected thresholds were reused. No new scientific CV experiment or model search ran. Forty fixed-parameter training-only fits restored timing objects; all agree exactly with saved scores on excluded record30.

## Executed commands and outcomes

Commands below were executed from the repository root. Logs are preserved in `results/ml_completion/run01/verification/`. Environment prefixes are intentional; no packages were installed.

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -u -m ml_research.finalize_saved --output results/ml_completion/run01 --stage aggregate
MPLCONFIGDIR=/tmp/ml-completion-mpl OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -u -m ml_research.finalize_saved --output results/ml_completion/run01 --stage timing
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -u -m ml_research.finalize_saved --output results/ml_completion/run01 --stage figures
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python results/ml_completion/run01/write_completion_report.py --output results/ml_completion/run01
```

Aggregation passed: 3,456 event conditions, 2,000 shared recording-bootstrap draws, 192 original-policy parity rows; 1,920 before/after comparisons and 5,760 paired comparisons retained. Timing passed: 82 components, 1,000 measured repetitions after 50 warmups per component; median/P95/P99 retained with raw samples. Figure generation initially failed on an all-NaN lead-time categorical axis. The minimal numeric-axis correction passed a regression test and generated the missing panel without overwriting the four completed images. The first report entrypoint failed to import `ml_research`; the corrected separate generator succeeded, retaining the failed original script. These were execution/rendering defects, not ML/label/threshold defects.

```bash
MPLCONFIGDIR=/tmp/ml-completion-test-mpl OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_ml_finalization.py -v
MPLCONFIGDIR=/tmp/ml-completion-test-mpl OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

Final targeted run: **3/3 passed**, 1.025s. Final complete suite: **162 run in 95.384s; 159 passed, one failure, two errors; no skips**. Dashboard tests are included. The earlier 161-test run also had the same three historical failures. Exact unresolved assertions:

- `test_target_hardening.SavedTargetTests.test_historical_outputs_preserved`, `tests/test_target_hardening.py:193`: expected historical fingerprint for `docs/MODEL_EVALUATION.md` differs from the committed current report.
- `test_local_30.Local30ArtifactTests.test_outputs_and_protected_files_are_unchanged`, `tests/test_local_30.py:99`, `src/local_30_evaluation.py:126`: `docs/FULL_PIPELINE_GUIDE.md` differs from its historical preservation receipt.
- `test_alarm_budget.AlarmBudgetArtifactTests.test_inputs_outputs_and_preservation_verified`, `tests/test_alarm_budget.py:103`, `src/alarm_budget_experiment.py:303`: same guide mismatch.

The previous readiness audit traces those expected document versions to ef927e7 and the current committed documentation to 49c11a0. Neither affected document was edited in this completion. No receipts, expected hashes or existing tests were changed. These remain red; they are not called passing tests.

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m src.check_project --audit --links --verify-run results/final/new_onset
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m src.check_project --verify-run results/final/future_low
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m ml_research.verify --output results/ml_completion/run01
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python results/ml_improvement/verification01/post_checks.py --output results/ml_completion/run01 --reference results/ml_improvement/recovery02 --receipt results/ml_completion/run01/independent_checks.json
```

Both original target quality gates passed (four models and 23 PNGs each). The completion verifier passed **before** authorized document/caption updates: 765 protected files, 7,820,712 predictions, 3,456 event conditions, 6,720 thresholds. Independent checks passed: all 620 surviving interrupted timestamp summaries, 144 checkpoint hashes and concatenated predictions, 2,505 window-specific matched alerts and their score/cutoff/lead boundaries, and all 82 raw latency distributions including P99. Those matches are repeated windows/conditions, not 2,505 independent clinical events.

After approved documentation changes, the frozen whole-tree preservation verifier is rerun separately and rejected exactly `docs/DASHBOARD_GUIDE.md`, `docs/FINAL_REPORT.md`, `docs/PROJECT_REVIEW_EVIDENCE.md` and `dashboard/app.py`. Its receipt is retained unchanged; the pre-documentation numerical pass is not presented as a final-tree pass. `verify_completion.py` and `completion_receipt.json` separately authenticate the exact authorized changes against byte-preserved originals and check every other baseline file unchanged. This supplemental receipt does not replace or relax the three historical test contracts.

## Startup and visual status

The sandbox startup attempt failed with `PermissionError` on socket creation. A bounded localhost health harness then ran with approved escalation:

```bash
python /tmp/ml_completion_health.py
# launches:
python -m streamlit run dashboard/app.py --global.developmentMode=false --server.headless=true --server.address=127.0.0.1 --server.port=8524 --browser.gatherUsageStats=false
```

**HTTP200, body `ok`; server terminated in finally.** The harness is copied into the verification directory. This checks server startup, not browser rendering; no browser automation was available. Browser visual verification remains **NOT VERIFIED**. The event-count/undefined-lead figure was separately viewed as an image; this is not dashboard visual verification.

## Changed-file inventory and interpretation

- New `ml_research/finalize_saved.py`: guarded completion of saved aggregation, exact cluster-bootstrap optimization, timing reconstruction and NaN-safe event panel. Original frozen study modules untouched.
- New `tests/test_ml_finalization.py`: exact bootstrap equivalence/edge cases, clone protection and undefined-lead plotting tests.
- Updated `README.md`: completed-study scope and safe new reproduction destinations.
- Updated `dashboard/app.py`: one scientific-scope caption only; original saved policy and playback behavior retained.
- Updated `docs/FINAL_REPORT.md`, `docs/PROJECT_REVIEW_EVIDENCE.md`, `docs/DASHBOARD_GUIDE.md`: append verified follow-up evidence and preserve original results.
- Updated current readiness audit/checklist and new completion tracker: resolve completed work while retaining historical findings and exceptions.
- New `docs/ML_IMPROVEMENT_REPORT.md`, `docs/DATA_PROVENANCE.md`, this verification report: completed methods/results, citation boundaries and executed checks.
- New `results/ml_completion/run01/`: cloned immutable inputs, complete tables/uncertainty/influence, timing-only artifacts, five figures, report/generators, original-document backups, numerical/protection receipts and durable logs. Full inventory is hashed in the completion receipt. No historical result was overwritten.

The original final policy remains 0/14. Three GaussianNB conditions detect the same nonsevere event, but none satisfies the chosen budget in every exposed recording. The single-event sensitivity interval is broad and no robust ELM/logistic/model superiority is established. Only 30 local recordings, 14 eligible 30s events and three severe-interest events; patient independence, clinical reliability and control/PID efficacy remain unsupported. The project supports an honest retrospective research demonstration after manual presentation-machine checks. Academic submission must disclose the negative result, retrospective development and unresolved integrity contracts; it is not an all-checks-green or clinically validated system.

## Final read-only protection and documentation checks

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m src.check_project --links
git diff --check
PYTHONDONTWRITEBYTECODE=1 python results/ml_completion/run01/verify_completion.py --seal
PYTHONDONTWRITEBYTECODE=1 python results/ml_completion/run01/verify_completion.py
```

Local links: **495 resolved**, no broken links. Whitespace check passed. AST parsing passed for the finalizer, its tests, dashboard app and completion verifier. Baseline check verified **1,775 files: 1,767 unchanged and exactly eight authorized changes**, each backed by a byte-identical original; all ten frozen ML source files and 929 initially sealed outputs are unchanged. The tracker is one of those eight because its initial current-task version was included in the snapshot; it was not a pre-existing historical artifact. The initial Git-status capture also occurred after creating current-task files and is not presented as a pristine pre-task snapshot. Completion sealing authenticates supplemental outputs/current documents separately from unchanged old receipts; the initial seal output is retained in `completion_verification.json` beside the receipt. The initial receipt and verifier version are retained; `final_completion_receipt.json` additionally seals this final log clarification and supplemental files.
