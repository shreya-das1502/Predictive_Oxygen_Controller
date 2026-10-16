# Oxygen Research Studio dashboard

**Completion-stage scope:** the dashboard still plays the preserved original run02 final policy (0/14), with its own separate full-data demo. [ML_IMPROVEMENT_REPORT.md](ML_IMPROVEMENT_REPORT.md) now documents the bounded follow-up. One record07 GaussianNB event appears in three conditions, all with individual-record budget violations; no policy was selected or substituted into playback. New timing-only model objects are not deployment/demo artifacts.

**Current interface: one selected recording at a time.** Use **Select Patient Recording** near the top of the page. Historical three-case implementation and verification notes below are labeled as earlier stages.

This is a local, offline research demonstration, not a validated medical device or treatment recommendation. It never controls oxygen. Clinician assessment and prescribed protocols remain essential. The visual design uses lavender, white rounded panels and violet charts, following the written HealthPulse aesthetic; no reference image was available for pixel-level comparison.

## Setup and launch

From the repository root, use the existing compatible research environment. Optional UI dependencies are isolated from [research requirements](../requirements.txt) in [requirements-dashboard.txt](../requirements-dashboard.txt):

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dashboard.txt
python -m streamlit run dashboard/app.py --server.address 127.0.0.1 --global.developmentMode=false --browser.gatherUsageStats=false
```

Open the local URL printed by Streamlit, normally `http://127.0.0.1:8501`. In Codespaces, use its port-forwarding preview. Do not expose identifiable data publicly. Installation needs network access; running the dashboard uses existing local data only. No datasets are downloaded. Streamlit 1.66.0 and Plotly 6.9.0 were tested. Missing optional packages produce an installation message; missing recording/model/result files produce case-specific errors rather than triggering training. Existing models must be trusted local artifacts; never substitute an untrusted joblib file.

For the temporary environment used during implementation:

```bash
PYTHONPATH=/tmp/oxygen-dashboard-deps python -m streamlit run dashboard/app.py --server.address 127.0.0.1 --global.developmentMode=false --browser.gatherUsageStats=false
```

The temporary dependencies may disappear when the environment is rebuilt. Use the normal setup above for a durable installation. Research package versions were not changed.

## Single-recording workflow (current)

The dashboard opens with one recording selected. **Select Patient Recording** lists all recordings currently available through the existing backend, using `Recording 01`, `Recording 02`, etc. The underlying numeric recording IDs are retained; the list is not limited to three cases. Missing local recording pairs are omitted. If none are available, the dashboard explains which local files to restore and never fabricates or downloads data. The requested patient-recording label does not establish verified patient identities.

Use **Previous Recording** and **Next Recording** to step through available recordings; buttons are disabled at the ends. Selecting a recording updates the observation cards, pulse waveform, scores, saved cutoff, alerts and delayed retrospective outcomes together. A switch creates a fresh replay/controller/history, resets the timeline to zero and pauses playback. Returning to a previous recording starts fresh rather than restoring its old pulse or alert state. Model, target and source choices are retained when possible; changing those settings also resets playback.

The sidebar offers **Recording playback**, **Research results**, and **About and limitations**. Recording playback contains one full-width observation/status panel, one set of controls and a large waveform with prediction details beside it. There is no three-case grid, case inspector or comparison page. The research page is explicitly **study-wide pooled evidence**, not metrics for the selected recording; recording-specific outputs/outcomes appear on playback. Switching recordings from another page also resets the session before returning to playback.

1. Choose a local recording from **Select Patient Recording**.
2. Choose the existing model. Expand **Prediction question and playback source** if needed; retain the distinction between saved held-out final-policy playback and full-data demonstration inference.
3. Press **Play**, or move **Playback time (seconds)**. Pause and Reset operate only on the selected recording. Seeking backward rebuilds its causal prefix. Recorded time advances about one second per refresh, not at guaranteed real-time speed.
4. Read observed SpO₂ (%) and pulse (BPM), experimental status and last valid decision time. The score/cutoff are algorithmic values, not clinical probabilities. No alert is not a safety conclusion.
5. Inspect the PPG and prediction/event timeline. Saved retrospective event verification remains delayed until onset +30 seconds. Demo inference receives no held-out event credit or lead-time metrics.

The PPG panel shows only the last 12 seconds available at the cursor, with maxima marked only after the existing causal detector confirms them. Numeric SpO₂ is the latest observed value only when valid and no older than 1.01 seconds. Pulse rate is PPG-derived and shown only when a confirmed pulse is recent. Signal quality is a missingness/continuity proxy, **not validated SQI**. There are no fabricated patient identities, manual vital-entry predictors or placeholder physiological data.

Scores are uncalibrated experimental outputs. The last-pulse timestamp, actual saved threshold, eligibility and 30-second horizon are explicit. A saved decision expires from the status card after two seconds without a new decision. Missing predictions are never converted into negative predictions. Future-low includes ongoing lows; new-onset requires current SpO₂ >94% and valid freshness. A single adjacent >94→≤94 reading qualifies as an observed onset, not a clinically sustained event.

## Two deliberately separate modes

**Saved held-out final policy** is the default. It reads [outer predictions](../results/alarm_budget_experiment/run02/outer_predictions.csv.gz) from the completed final experiment. It displays the selected recording's actual LORO scores, fold-specific threshold and saved decision at the original availability timestamp. Selecting a research budget loads its existing policy, without selecting or tuning a new threshold. This is playback of prior evaluation, not live inference from a full-data binary. Evaluation rows include retrospective complete-future-window eligibility; their absence near invalid readings/tails is not an online normality decision.

**Causal full-data demo model** loads `results/final/<target>/models/<model>.joblib` and calls the existing [Controller](../src/controller.py), [WaveformState](../src/preprocessing.py) and [FeatureState](../src/features.py). Every waveform sample is processed in order, including history before the chart window. Current SpO₂ is passed only for eligibility; no current/future SpO₂ enters predictor features. The saved model's original threshold is used. These all-data fits are demonstrations, **not held-out performance or the final alarm-budget policy**. No final fold threshold is paired with a full-data fit. No refitting occurs. Demo event credit/lead-time estimates are deliberately not invented.

## Prediction versus outcome verification

Orange lines mark qualifying onsets only after their numeric timestamps have been reached. [Labeling](../src/labeling.py) receives only the observed numeric prefix for these markers. Scores and thresholds occupy a separate timeline from the retrospective outcome table.

In saved-policy playback, actual starts come from [alarm_starts.csv](../results/alarm_budget_experiment/run02/alarm_starts.csv); missed-event markers and leads come from [event_matches.csv](../results/alarm_budget_experiment/run02/event_matches.csv). Verification is conservatively delayed until **onset +30 seconds**, so full horizons of all pre-onset opportunities can be inspected. No future event end, minimum or severity is disclosed. The saved audited rule assigns the earliest unused alarm start in `[onset−30,onset)` chronologically, one credit per event and alarm. Lead is onset minus actual start, not a clipped alarm timestamp or backdated waveform peak. Missed/unevaluable events have undefined leads. A demo controller's alerts are not substituted into these held-out metrics.

## Research results and denominators

The prominent results panel reads [summary.csv](../results/alarm_budget_experiment/run02/summary.csv), separately selecting training target, budget and 5/10/15/30-second look-back window. All four learned models detected **0/14 eligible events at 30 seconds and 0/15 at 5/10/15 seconds**, for both training targets and all tested budgets. All three severe-interest events were missed. Lead times are undefined. The 1/5/10 false-starts/hour budgets and 5% active-time cap are research assumptions, not clinical limits; some held-out recordings violated constraints met in training. No useful early warning or proven superiority was demonstrated.

The table includes always-alert and never-alert; always-alert fails the active-time cap. Sensitivity counts eligible events; false starts/hour divides off-event starts by eligible inter-pulse exposure (10,329.032 seconds pooled), not full wall-clock duration. Active fraction is positive left-held duration/exposure. Timestamp precision and recall count prediction rows; undefined values remain NaN. See [alarm-budget methods](ALARM_BUDGET_EXPERIMENT.md), [reliability audit](RELIABILITY_AUDIT.md) and [local evaluation](LOCAL_30_RECORDING_EVALUATION.md).

The expandable historical table is explicitly earlier onset timestamp evaluation. Its false episodes/hour and pulse alert fraction are **different metrics** from audited false starts/hour and duration-based active fraction. The old 14/14 reusable-positive result is preserved, with always-alert equivalence and the later one-to-one correction explained. It is never combined with final-policy results.

## Architecture and preservation

- [app.py](../dashboard/app.py): Streamlit navigation, plain-language case summaries, Plotly charts and recording-specific session state. Cached input tables are read-only; settings persist across pages.
- [presentation.py](../dashboard/presentation.py): read-only status explanations, unit/empty-value formatting and the color palette; it never scores or fits a model.
- [styles.css](../dashboard/styles.css) and [theme configuration](../.streamlit/config.toml): explicit light surfaces, readable foregrounds, widget states and stacking rules. This theme file changes only UI appearance, not research configuration.
- [backend.py](../dashboard/backend.py): local catalog, record validation, chronological replay, artifact filtering and prefix-only event visibility. This adapter imports existing inference/label interfaces and contains no training call.
- [backend tests](../tests/test_dashboard.py) and [UI tests](../tests/test_dashboard_ui.py): selection, independent state, backwards resets, missing readings, prefix-only controller inputs, delayed verification, real denominators, case navigation and controller-mode interaction.

No research source, raw data, models, thresholds, datasets or result files are changed. Existing documentation edits from the preceding task are retained. In-memory Streamlit caches assume artifacts remain fixed during a session; restart the app after replacing any inputs. Raw file loading can inspect acquisition-clock validity across a recording, but no future waveform values enter controller inference or plots. Invalid acquisition clocks are rejected. Nonfinite waveform samples and detected discontinuities reset causal state without inventing samples.

## Earlier implementation verification record (historical)

Executed commands and results are recorded below. Optional UI checks were run with dependencies in `/tmp`, without changing pinned research packages.

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_dashboard.py -v
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_pipeline.py -v
PYTHONPATH=/tmp/oxygen-dashboard-deps OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_dashboard*.py' -v
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m src.check_project --links
git diff --check
```

Verification outcomes:

- Backend: 8/8 passed. Existing pipeline: 35/35 passed. Optional focused dashboard run: 9/9 passed before interruption; the final complete run executed those nine again successfully after the final adapter changes.
- Final complete suite, with optional UI dependencies: **135 tests in 130.040 s; 132 passed, 1 failure and 2 errors; no skips**. Exact command for this run:

  ```bash
  PYTHONPATH=/tmp/oxygen-dashboard-deps OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
  ```

  The three unresolved checks are `test_target_hardening.SavedTargetTests.test_historical_outputs_preserved` (old `docs/MODEL_EVALUATION.md` hash), `test_alarm_budget.AlarmBudgetArtifactTests.test_inputs_outputs_and_preservation_verified` and `test_local_30.Local30ArtifactTests.test_outputs_and_protected_files_are_unchanged` (old `docs/FULL_PIPELINE_GUIDE.md` hash). These documents were already edited by the preceding authorized documentation task. Neither the tests nor historical receipts were changed to hide this discrepancy. The suite is **not wholly passing**.
- `python -m src.check_project --audit` passed with 30 local recording pairs and the exact pinned research package versions unchanged. Both `python -m src.check_project --verify-run results/final/new_onset` and `python -m src.check_project --verify-run results/final/future_low` passed, verifying 4 models/23 PNG figures per target, saved metrics, datasets, cache and figure sources. These commands used `OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1`.
- Independent SHA-256 checks (`python /tmp/oxygen_dashboard_preservation.py`) verified all **11 inputs and 15 outputs** in the final alarm-budget manifest. Its 682-entry historical preservation list differs only for the three previously edited reports: `docs/FULL_PIPELINE_GUIDE.md`, `docs/FINAL_REPORT.md`, `docs/MODEL_EVALUATION.md`. No research artifact mismatch was found. The temporary script reads the existing manifests, hashes every listed file and asserts that any historical differences are limited to those existing documentation changes.
- Streamlit started on localhost, and `GET /_stcore/health` returned **200, `ok`**. The server was stopped after the check. Executed health harness: `PYTHONPATH=/tmp/oxygen-dashboard-deps OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python /tmp/oxygen_dashboard_health.py`. The harness launched `python -m streamlit run dashboard/app.py --global.developmentMode=false --server.headless=true --server.address=127.0.0.1 --server.port=8517 --browser.gatherUsageStats=false`, polled localhost, and terminated the process. Socket binding required sandbox permission. An earlier attempt without `--global.developmentMode=false` hit the temporary install's development-mode/port conflict; the explicit flag resolved it.
- Local Markdown links and whitespace checks passed; final link count is reported by the command above. Browser screenshots/responsiveness were not visually inspected: no browser automation tool was available. Streamlit AppTest exercised controls and script output, rather than pixel rendering.

Added files are `dashboard/__init__.py`, `dashboard/backend.py`, `dashboard/app.py`, `requirements-dashboard.txt`, `tests/test_dashboard.py`, `tests/test_dashboard_ui.py` and this guide. Existing README and three report edits were preserved, not extended by the dashboard task.

UI automation tests server-side Streamlit interaction, not rendered browser appearance or clinical performance. No training run, new experiment or policy search is part of this dashboard work. Only 30 recordings are available; patient identity/independence is unverified, data were examined retrospectively, and no clinical reliability is established.

## Subsequent final verification

The [final dashboard verification](DASHBOARD_VERIFICATION.md) traces the three preservation failures to the authorized documentation diff and original hashes, records fresh tests and quality gates, and adds checks for three independent demo states and future-suffix invariance across all eight model/target combinations. It retains the immutable historical baselines and explicitly identifies incomplete browser visual verification.


## Earlier three-case usability redesign (historical)

The product header says **RECORDED DATA · OFFLINE DEMO**, followed by “Offline predictive oxygen research demonstration” and “Research only. Does not recommend or control oxygen delivery.” A compact explanation describes recorded observations, experimental outputs and later outcome verification. The final **0/14 at 30 s; 0/15 at 5/10/15 s** finding remains visible above every page, rather than buried in a table.

1. On **Overview**, read the three card headers and select BIDMC recordings. Each card puts observed oxygen saturation (%) and PPG pulse rate (BPM) before advanced settings, and labels missing observations as Unavailable with a reason.
2. Press **Play** on a case, or move its **Playback time (seconds)** slider. Play, Pause and Reset affect only that case. A recording/model/target/source/budget change resets only that case.
3. Read **Prediction status**, its explanation and the last valid model decision timestamp. A below-cutoff output is not a safety conclusion. “Prediction horizon: 30 seconds” is not a countdown to treatment.
4. On Overview, choose **Inspect a recording** to view its larger trailing PPG waveform and score/event timeline. **Recording playback** offers an individual case view with the same settings and state. PPG is explicitly identified as a light-based pulse signal, not an ECG.
5. **Three-case comparison** shows each source/target/status and saved alarm starts in the available prefix. Event counts appear only after delayed verification; “none verified yet” is Unavailable, not an invented zero-event conclusion. Demo alert starts are separate and receive no held-out matched/missed counts or lead metrics.
6. **Research results** separates the final alarm-budget evaluation from the earlier recall-oriented policy. Vertical model cards replace a very wide technical table; technical timestamp details are expandable. **About and limitations** explains the local 30-record scope, single-reading event definition, patient uncertainty and lack of clinical validation.

Plain-language target choices preserve the original definitions: **New oxygen drop to 94% or lower** maps to `new_onset`; **Any oxygen reading at 94% or lower (including ongoing lows)** maps to `future_low`. The question tooltip explains current-reading requirements and the 30-second horizon. Playback-source descriptions identify previously evaluated recordings versus full-data demonstrations; selecting either never trains a model. Research budget controls remain disabled in demo mode. Model score/cutoff tooltips explain that neither is a calibrated clinical probability. A known saved cutoff can be shown before the first decision; an unavailable score is never shown as zero.

### Status definitions

| Visible status | Meaning |
| --- | --- |
| Experimental alert triggered | The last valid pulse score reached its saved cutoff; it is not an oxygen-treatment instruction. |
| No experimental alert at this time | The last valid pulse score was below its saved cutoff; it does not establish safety. |
| Prediction unavailable | No valid current decision is available, or the selected new-drop question does not allow a currently low/missing reading. |
| Current reading is stale | The most recent oxygen reading is more than 1.01 seconds old, preventing a current new-drop decision. |
| Insufficient valid signal | No confirmed PPG pulse is recent enough to support a current pulse-based decision. |

Every status has an icon, text and explanation, not just a color. The last valid decision time remains separately labeled even when no current decision exists. Current observation validity does not silently remove the existing future-low model's ability to predict ongoing lows.

### Contrast checks and remaining visual work

The repository's UI-only theme defaults to light mode. Lavender outer surfaces, white panels and purple accents retain the visual identity. Explicit styles cover sidebar text, metric cards, alerts, dropdowns/options, button hover/focus, disabled controls, expanders, tooltips and custom statuses. Chart backgrounds/fonts/axes/legends/hover labels use an explicit Plotly white theme, without Streamlit recoloring the palette. Chart markers use circle/triangle/diamond/cross symbols plus labels. Responsive CSS wraps columns below 850 pixels; no claim is made that this has been visually verified.

Solid-color WCAG calculations use `dashboard.presentation.contrast` (sRGB relative luminance). Tests require at least 4.5:1 for normal text and 3:1 for meaningful chart elements:

| Pair | Contrast |
| --- | --- |
| Navy text / white | 14.32:1 |
| Secondary text / lavender page | 6.11:1 |
| Secondary text / sidebar lavender | 5.67:1 |
| Purple text / white or sidebar | 8.98:1 / 7.47:1 |
| Warning text / pale warning surface | 8.34:1 |
| Experimental-alert text / pale pink | 7.73:1 |
| Information text / pale blue | 9.29:1 |
| Disabled text / disabled surface | 6.13:1 |
| White text / dark purple or navy | 8.98:1 / 14.32:1 |
| Chart pulse / onset / cutoff against white | 7.31:1 / 6.61:1 / 6.27:1 |
| Meaningful border / white | 4.61:1 |

These calculations are **not an audit of actual rendered component styles**. AppTest exercises the Streamlit element tree and controls, not CSS layout, hover pixels, browser theme overrides or canvas rendering. Browser automation/tools, Playwright/Selenium and Chromium/Chrome/Firefox were unavailable. The reported low-contrast defaults were addressed with explicit theme/style pairs, but **browser visual and rendered-contrast verification remains incomplete**. This is not a claim of WCAG conformance.

Manual visual acceptance steps (launch from the repository root using the commands above):

1. At **1440 × 1000** browser pixels and 100% zoom, open Overview. Verify three recording IDs, labeled %/BPM readings, explicit statuses and Play controls are readable. Inspect header/sidebar, the research result and safety note. No text should be yellow-on-yellow, dark-on-dark or white-on-pale.
2. Start Case 1, leave Case 2 paused and change Case 3's recording/model. Confirm independent clocks and reset behavior. Inspect the large PPG chart, tick labels, legend, markers and hover tooltip; score and cutoff must remain legible and distinguished by solid/dashed lines.
3. Open every selector and advanced source/target section. Check selected and hover options, keyboard Tab focus, question-mark help tooltips, expander text, and the **disabled** demo budget control. Inspect computed foreground/background colors in browser developer tools and measure actual contrast, including nested widget text and tooltip surfaces.
4. Switch to demo mode and confirm its full-data/non-held-out caption is visible. Inspect Research results: verify the final 0/14 and 0/15 result, severe-event misses, exposure, active-time percentages, undefined leads and the separately labeled historical policy.
5. Exercise empty states at time zero and currently-low new-drop records (for example record 04); inspect the explanatory status, not a healthy-looking zero default. On a disposable copy lacking a model/result file, verify the case-specific remedy and continued access to other views. Do not remove protected files from the working project.
6. Repeat at **768 × 1024** and **390 × 844**, plus 200% browser zoom. Confirm cards, controls and vertical result cards stack; no clipped status text, required horizontal scrolling, overlapping chart legend or unreachable navigation. Check warning/error/info text, buttons, selected options, chart hover and keyboard focus at each size. Check OS/browser dark preferences and confirm the deliberate light component pairs remain readable.
7. Play, pause, reset and seek backward; wait past an onset +30 s to inspect retrospective verification. Confirm past predictions never reveal a later event early, and “no alert” is never interpreted as “oxygen not required.” Record browser/version, screenshots, computed contrast pairs and any failures before declaring visual acceptance.

### Redesign verification record

Commands and final outcomes are recorded after execution below. The three historical-document preservation failures are intentionally retained; see [DASHBOARD_VERIFICATION.md](DASHBOARD_VERIFICATION.md) for their exact assertions and original hashes. No research source, model, label, feature, threshold, dataset, result or receipt is changed by the redesign.

Executed from the repository root for this redesign:

```bash
PYTHONPATH=/tmp/oxygen-dashboard-deps OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_dashboard*.py' -v
PYTHONPATH=/tmp/oxygen-dashboard-deps OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m src.check_project --verify-run results/final/new_onset
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m src.check_project --verify-run results/final/future_low
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m src.check_project --links
git diff --check
PYTHONPATH=/tmp/oxygen-dashboard-deps OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python /tmp/oxygen_dashboard_health.py
```

- Dashboard: **15/15 passed in 9.753 s**, including three-case independence, model changes, all five navigation views, missing-model errors/remedies, explicit empty states, statuses, color-pair calculations, unchanged causal prefix outputs and separation of demo/held-out evidence.
- Complete suite: **141 tests in 75.455 s; 138 passed, one failure and two errors; no skips**. The unsuccessful checks are still `test_target_hardening.SavedTargetTests.test_historical_outputs_preserved`, `test_local_30.Local30ArtifactTests.test_outputs_and_protected_files_are_unchanged` and `test_alarm_budget.AlarmBudgetArtifactTests.test_inputs_outputs_and_preservation_verified`. They flag the prior authorized MODEL_EVALUATION/FULL_PIPELINE_GUIDE edits; the receipt scan also identifies FINAL_REPORT, as detailed in the retained verification report. Their assertions/receipts were not changed. This is not a wholly green suite.
- Saved-run gates: **both passed**, covering 16,050 onset rows/20,157 future-low rows, four models and 23 PNG figures per target, saved metrics, cache and figure hashes. No regeneration was needed.
- **293 local links verified**; tracked-diff whitespace and changed-code AST/trailing-whitespace checks passed.
- Streamlit health: **200 `ok`**, using the existing temporary localhost harness. The initial sandbox socket restriction required a permitted retry; the harness stopped the server afterward. This is not a browser visual check.
- Protected-file SHA-256 snapshot: **703 files unchanged** under `src`, `data`, `results` and `archive`. The playback backend, controller, preprocessing, features, labels, models, saved thresholds and all historical artifacts/receipts remain untouched. No model fit, threshold tuning or new experiment was performed.
- Browser visual inspection: **not performed**; actual rendered contrast and narrow-layout acceptance remain incomplete. Use the precise manual checklist above before claiming accessibility/visual acceptance.

Exact files changed for this redesign: `dashboard/app.py`, new `dashboard/presentation.py`, new `dashboard/styles.css`, new `.streamlit/config.toml`, `tests/test_dashboard_ui.py`, new `tests/test_dashboard_presentation.py`, and `docs/DASHBOARD_GUIDE.md`. Earlier README/report and dashboard-verification edits were preserved. The new [presentation tests](../tests/test_dashboard_presentation.py) check text/status definitions, empty-value semantics, the light theme and solid-color contrast; [UI tests](../tests/test_dashboard_ui.py) exercise real Streamlit interactions, not CSS pixel output.


## Single-recording change verification

This update changes only `dashboard/app.py`, `tests/test_dashboard_ui.py` and this guide. The backend's `recordings()` supplies supported availability; the UI does not hardcode a three-option list. Existing replay, controller, feature, target, threshold and event-verification interfaces remain unchanged. On a failed selection, stale prior-recording engine/config state is cleared and a recording-specific remedy appears. Idle slots from an older three-case hot-reload session are removed from UI session state only; no disk artifacts are removed.

Manual visual smoke test when a browser is available:

1. Launch from the repository root using the setup command above. At 1440×1000, confirm exactly one `Select Patient Recording` dropdown, one observation/status panel, one timeline control and one large PPG waveform; no Case 1/2/3 grid or comparison navigation should remain.
2. Select Recording 17, choose the source/target intentionally, start playback, then choose Recording 12. Confirm all titles/readings/plots/outcomes switch to 12, its time is zero, playback is paused and no prior alert remains. Return to 17 and confirm a fresh start.
3. Exercise Previous/Next, including disabled boundary buttons. Verify missing recording pairs are omitted without changing the numeric IDs of other entries.
4. Switch records while on Research results, then return to playback. Verify the chosen record remains selected and state is reset. Check that pooled research results remain labeled study-wide, while saved and demo provenance are distinct.
5. Repeat at 390×844 and 200% zoom, checking stacked controls, readable lavender/white/purple styling, chart legends and safety text. Inspect actual rendered contrast; automated color-pair tests do not replace visual checking.

Exact verification commands and outcomes follow after execution. The known three historical-document integrity failures remain protected and must not be represented as a green complete suite.

Commands executed from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_dashboard*.py' -v
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m src.check_project --verify-run results/final/new_onset
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m src.check_project --verify-run results/final/future_low
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m src.check_project --links
git diff --check
python /tmp/single_recording_health.py
```

The temporary health harness starts `python -m streamlit run dashboard/app.py --global.developmentMode=false --server.headless=true --server.address=127.0.0.1 --server.port=8517 --browser.gatherUsageStats=false`, checks `http://127.0.0.1:8517/_stcore/health`, then terminates the server. The sandbox initially blocked its listening socket; the permitted retry returned **200 `ok`**. This does not verify browser rendering. No browser tool, Playwright/Selenium package or Chromium/Chrome/Firefox executable was available, so **single-recording visual verification remains incomplete**.

Both saved-run gates passed (16,050 onset rows; 20,157 future-low rows; four models and 23 PNG figures per target, with metrics/cache/figure hashes verified). All **293 local links** and whitespace checks passed. The protected-file snapshot contains **703 unchanged source/data/result/archive files**. No model, dataset, evaluation result, threshold, label, feature or historical receipt was modified. The causal backend tests remain intact; only obsolete simultaneous-three-case UI assertions were replaced by stronger selected-recording reset and isolation assertions. Final test counts are recorded below.

Final results for the single-recording update:

- **17/17 dashboard tests passed in 10.779 s**, including all available options, selected-record readings/titles/state, Previous/Next boundaries with missing recordings, demo/controller isolation, pause/reset/model changes, failed-selection cleanup, page navigation, old-session migration, causal-prefix invariance and prediction/outcome separation.
- **Complete suite: 143 tests in 102.993 s; 140 passed, one failure and two errors; no skips.** The only unsuccessful checks are the preserved `test_target_hardening.SavedTargetTests.test_historical_outputs_preserved`, `test_local_30.Local30ArtifactTests.test_outputs_and_protected_files_are_unchanged` and `test_alarm_budget.AlarmBudgetArtifactTests.test_inputs_outputs_and_preservation_verified`. They still flag the pre-existing authorized historical-document edits; no protection was weakened.
- Both saved-run quality gates, 293 links, whitespace checks, changed-Python syntax checks and localhost health passed. All 703 protected files remained unchanged.
- Browser visual verification was **not performed**. The manual single-recording checklist above remains outstanding.
