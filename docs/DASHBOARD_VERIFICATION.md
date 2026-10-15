# Final dashboard verification — 2026-10-15

## Outcome and scope

Dashboard verification passed the focused checks and both saved-run quality gates. The complete suite remains **not fully passing** because historical receipts intentionally preserve older report bytes. No historical receipt or assertion was changed, and no experiment, training, relabeling or policy search was run. Browser-rendered visual verification remains incomplete.

This pass changes only `tests/test_dashboard.py`, `tests/test_dashboard_ui.py`, this new report and an appended link in `docs/DASHBOARD_GUIDE.md`. Other dashboard implementation files and the earlier README/report changes remain as found.

## Exact failures and authorized history

The expected document hashes in all three receipts equal the blobs at Git `HEAD` **ef927e75e2c961fb4545e2b5b7e6893de41935f4**. The documentation review remains an uncommitted working-tree change; it is not a separate Git commit. Its diffs were compared with the preceding user authorization to add a final alarm-budget section, reconcile 14/14 versus 0/14, distinguish metrics and preserve limitations. The diffs contain those additions and historical labeling. All original Markdown table lines remain: 22 in FINAL_REPORT, 33 in FULL_PIPELINE_GUIDE, and 44 in MODEL_EVALUATION. README adds the same negative-outcome notice and qualifies earlier metrics. No experimental numerical output was changed.

| Exact test name | Assertion and receipt | Observed failure |
| --- | --- | --- |
| `test_target_hardening.SavedTargetTests.test_historical_outputs_preserved` | [test_target_hardening.py:193](../tests/test_target_hardening.py#L193): `assertEqual(fingerprint(path), digest, path)` for every entry in [historical_preservation.json](../results/hardened/historical_preservation.json) | Assertion failure for `docs/MODEL_EVALUATION.md` |
| `test_local_30.Local30ArtifactTests.test_outputs_and_protected_files_are_unchanged` | [test_local_30.py:99](../tests/test_local_30.py#L99): `assertEqual(verify_output(DEFAULT_OUTPUT)['status'], 'passed')`; [local_30_evaluation.py:126](../src/local_30_evaluation.py#L126) checks every [preservation.json](../results/local_30_fixed_policy_v2/preservation.json) entry | `ValueError: Protected file changed: docs/FULL_PIPELINE_GUIDE.md` before the outer assertion |
| `test_alarm_budget.AlarmBudgetArtifactTests.test_inputs_outputs_and_preservation_verified` | [test_alarm_budget.py:103](../tests/test_alarm_budget.py#L103): `assertEqual(verify(ROOT/'run02')['status'], 'passed')`; [alarm_budget_experiment.py:303](../src/alarm_budget_experiment.py#L303) checks every [preservation.json](../results/alarm_budget_experiment/preservation.json) entry | `ValueError: Historical artifact changed: docs/FULL_PIPELINE_GUIDE.md` before the outer assertion |

Independent enumeration, rather than stopping at each verifier's first exception, found:

- Hardened preservation: **85 entries**, only MODEL_EVALUATION differs.
- Local-30 preservation: **662 entries**, only FULL_PIPELINE_GUIDE, FINAL_REPORT and MODEL_EVALUATION differ.
- Alarm-budget preservation: **682 entries**, only those same three documents differ.

| Document | Historical expected SHA-256 (also Git HEAD) | Current authorized-review SHA-256 |
| --- | --- | --- |
| `docs/MODEL_EVALUATION.md` | `ecb841126b8afe0d4f9c687c78ea87ee5d5765e7db92fdb9f1f30c12df71f2a4` | `c3a84b03fa47c3282128a5a2a14bb278cd56df82172d68a59e4a626033800ffd` |
| `docs/FULL_PIPELINE_GUIDE.md` | `02838962cb11fd2c422390b4c1999e8a55976c5c714f2c0b3e31a79c041c8485` | `c9285e989c2ad636fcf637193cd0a9625bc95192acd71e884bb877ccd7af65a4` |
| `docs/FINAL_REPORT.md` | `2f2d91429a73d93559944f0fe452445316007d1ad44c18613cb62dd51b9871e3` | `a17ddeab1181fe57e3b8dfbbc739388a79d35124d08ad8ee7f8a1f6aa84f4a21` |

**Baseline decision:** these are valid historical baselines, incompatible with today's authorized report revisions; they are not corrupted model/data receipts. They remain unchanged. Local-30 protects the hardened receipt, its own manifest hashes its preservation receipt, and alarm-budget protects both earlier receipts/manifests. Rebaselining would therefore propagate into historical experiment manifests, beyond a documentation-only correction. The smaller preservation-safe action is to record the exact authorized differences here and retain the strict failures. No file-type exemption, broad skip, catch-and-ignore, receipt deletion or altered assertion was introduced. A green immutable-history gate is not claimed.

Independently verified all **84 local-30 inputs and 12 outputs**, and all **11 final alarm-budget inputs and 15 outputs**, against their original manifest hashes. These include the experiment's predictions and selected policies. A before/after snapshot of all 703 existing files under `src`, `data`, `results` and `archive` checks that this verification pass changes none of them.

## Playback, outcomes and safety

[backend.py](../dashboard/backend.py) advances waveform indices only through `searchsorted(cursor, side='right')`, passes only the latest numeric observation at or before each sample to the existing controller, and resets causal state on backward navigation and discontinuities. Full local arrays are loaded for replay, but future values do not enter controller calls. Prefix mutation tests exercise the real eight saved model/target combinations: replacing all later PPG values with 999 and later SpO₂ with 40 leaves all earlier pulse and model outputs identical. These are isolated in-memory test perturbations; no data file is written.

Streamlit tests verify distinct replay engines, waveform states and pulse lists for all three cases, then distinct controller instances, feature histories, waveform histories, models and output lists in demo mode. Advancing Case 2 leaves Cases 1 and 3 at their own cursors. Existing tests cover backward replay, missing data, expiration, resets and delayed verification.

The default mode presents saved held-out final-policy decisions with their original fold-specific thresholds. The causal demo loads a separately saved full-data model and its own threshold; it explicitly states **“DEMONSTRATION ONLY”** and **“Not held-out performance or the final budget policy.”** Demo event credit and lead metrics are not computed. Audited event verification appears only in saved-policy mode after onset +30 seconds; future event minima/end times are not exposed. These distinctions are asserted in UI tests; no claim is made that wording eliminates every possible user misunderstanding.

The final-result panel precedes case controls and prominently states **0/14 events at 30 s and 0/15 at 5/10/15 s** for all learned models. Tests also inspect the rendered Streamlit warning that outputs are **not treatment recommendations**, clinician review remains essential, and **no oxygen delivery is controlled**. Source inspection found no treatment actuation path or guaranteed-treatment countdown. Quality text explicitly says there is no validated SQI. Historical results remain in their separately labeled expander. The saved summary denominators, missed counts and undefined lead times are checked against the actual artifact for every target/budget/window.

## Commands and results

Commands executed from the repository root:

```bash
PYTHONPATH=/tmp/oxygen-dashboard-deps OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_dashboard*.py' -v
PYTHONPATH=/tmp/oxygen-dashboard-deps OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m src.check_project --verify-run results/final/new_onset
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m src.check_project --verify-run results/final/future_low
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m src.check_project --links
git diff --check
PYTHONPATH=/tmp/oxygen-dashboard-deps OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python /tmp/oxygen_dashboard_health.py
```

- Focused dashboard tests: **10/10 passed in 7.054 seconds**, including actual Streamlit AppTest interactions.
- Complete suite: **136 tests in 67.730 seconds; 133 passed, one failure and two errors; no skips**. The only unsuccessful checks are the exact three historical-document preservation tests listed above. All 10 dashboard tests executed successfully in this run as well.
- Saved-run gates: **both passed**, verifying 16,050 onset rows and 20,157 future-low rows, four models and 23 PNG figures per target, metrics, cache and figure source hashes. Research package versions still exactly match requirements.
- Link/whitespace checks: **287 local links verified**; `git diff --check` passed. Changed Python files also passed AST parsing and a trailing-whitespace check. The final before/after SHA-256 snapshot check found **zero changes among all 703 protected files**.
- Streamlit health: **200 `ok`**. The sandbox initially blocked socket creation; the permitted retry succeeded. The temporary harness starts `python -m streamlit run dashboard/app.py --global.developmentMode=false --server.headless=true --server.address=127.0.0.1 --server.port=8517 --browser.gatherUsageStats=false`, polls `http://127.0.0.1:8517/_stcore/health`, and terminates the process. No server is intentionally left running.

History verification used `git status --short`, `git log -6 --oneline`, `git show -s --format='%H%n%s' HEAD`, `git diff -- README.md docs/FINAL_REPORT.md docs/FULL_PIPELINE_GUIDE.md docs/MODEL_EVALUATION.md`, `git show HEAD:<document>` and Python `hashlib.sha256`/JSON enumeration of each original receipt and manifest. Temporary test logs are `/tmp/dashboard-final-focused.log` and `/tmp/dashboard-final-suite.log`; these are not historical experiment outputs.

## Remaining issues

The three strict historical-document failures remain; this is a known preservation-contract discrepancy, not a wholly green suite. No browser tool is exposed, `playwright` and `selenium` are absent, and neither `chromium` nor `google-chrome` is on PATH. Consequently **visual appearance, mobile responsiveness, clipping and browser auto-refresh have not been browser-tested**. AppTest and HTTP health checks are not substitutes for that visual smoke test. Optional dashboard dependencies currently reside in `/tmp`; use the durable setup in [DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md) after an environment reset. None of these checks establishes clinical readiness, patient-independent generalization or useful early warning.
