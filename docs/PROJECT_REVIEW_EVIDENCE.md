# Project review evidence: Predictive Feedforward-PID Oxygen Control

**Completion-stage notice:** original-policy conclusions and historical inventories below are retained. The completed bounded follow-up is documented in [ML_IMPROVEMENT_REPORT.md](ML_IMPROVEMENT_REPORT.md): one nonsevere event in three GaussianNB conditions, pooled-compliant but per-record violations; no robust winner. Its actual short-H fits/context ablation supersede those specific absences in §10.3. All other unsupported clinical/control/patient claims remain unsupported.

Repository audit date: 15 October 2026 (Asia/Kolkata). This is a separate review document; it does not replace historical reports. **The strongest final finding is negative: at each research alarm budget, both training-target families detect 0/14 eligible events at 30 s and 0/15 at each 5/10/15 s window; all three eligible 91/92-interest events are missed.** In plain terms, the learned policies detect none. High timestamp recall under earlier recall-driven thresholds is a different experiment and does not establish useful early warning.

Scope: machine learning, causal signal processing, retrospective evaluation, existing illustrative simulations, and a proposed predictive feedforward-PID architecture. No PCB/circuit implementation, fabrication or embedded deployment is included or proposed as a current task. No patient actuation is recommended.

Evidence labels: **Observed** means inspected saved repository evidence; **Derived** means arithmetic/grouping of those saved values, without fitting or new simulation; **Theory** means conditional mathematical reasoning; **Proposed** means work not demonstrated by the repository. Counts are exact; displayed floating-point values use up to 12 significant digits from CSV/JSON, not graph estimates. Undefined means the denominator/crossover/matched sample does not exist; it is not zero. Source tables retain full stored precision.

Reading order for a short review: the verdict in section 9, the five display priorities in section 5, and final-policy results in section 3.5. The other sections supply the detailed defense and source trail.

## 1. Project objective and actual implementation

**Research motivation / interpretation:** recognizing a future observed desaturation before onset could provide time for assessment or a future simulated anticipatory compensation strategy. A classifier finding a future-low label does not identify its cause, predict an oxygen dose, or establish that an intervention would help. This repository measures an offline prediction problem, not clinical benefit.

**Observed implementation:** raw recorded PPG → causal filtering/pulse confirmation → ten causal pulse/history features → fixed binary model score → training-derived cutoff → research alert. Current SpO₂ is used for target construction and, for onset inference, eligibility; it is not one of the ten predictors. `src.controller.Controller` is this inference interface, not a physical feedback controller. Separate `signal_studies` now contains an explicitly illustrative conditioning-amplifier feedback model. Neither branch implements an oxygen-delivery plant or evaluated feedforward-PID integration.

### 1.1 Exact targets and horizons

| Item | Implemented definition | Evidence |
|---|---|---|
| `future_low` | Minimum valid observed SpO₂ in `(t,t+30 s]` is ≤94%; ongoing low saturation is included; complete valid future coverage required | `src/labeling.py:future_labels`, `src/config.py:target_definition` |
| `new_onset` | Current latest valid SpO₂ must be >94% and no older than 1.01 s; a qualifying normal-to-low crossing occurs in `(t,t+30 s]`; complete valid future required | `src/labeling.py:onset_labels`, `desaturation_events` |
| Qualifying onset | Valid ≤94% sample immediately after a valid >94% sample, gap ≤1.01 s; one low sample suffices; no persistence/washout requirement | Same functions; `src/reliability_evaluation.py:raw_events` |
| Unknown targets | Incomplete future, missing/invalid SpO₂ or incompatible gaps are unknown, not negative; low/unknown current readings are onset-ineligible | `future_labels`, `onset_labels` |
| Trained label horizon | 30 s for both targets | Both `results/final/<target>/run_manifest.json` |
| Event windows evaluated | 5, 10, 15, 30 s, matching an actual new alarm start in `[onset−W,onset)` | `results/alarm_budget_experiment/protocol.json`; local-30 and final summaries |
| Meaning of W | Look-back interval, not at least W seconds of warning, and not independently trained 5/10/15 s classifiers | Same fixed 30 s score fits are reused |
| Severe-interest | Low episode includes an actual 91% or 92% observation; not every value ≤92%, and not an adjudicated clinical severity category | `SEVERE_SPO2_VALUES`, raw event `Has_91_92` |

The local files contain 30 paired eight-minute BIDMC recordings, IDs 01–30: 60,001 waveform rows per recording at verified 125 Hz and 481 numeric rows nominally at 1 Hz. Total source duration is 14,400 s (4 h), not 4 h of continuously evaluable normal-state exposure. The number of independently established subjects is **Not available in the current saved results**. IDs identify recordings, not verified unique patients. `source_inventory.csv` contains no patient-ID columns. Waveform channels are `RESP`, `PLETH`, `V`, `AVR`, `II`; numerics include `HR`, `PULSE`, `RESP`, `SpO2`. The maintained predictors use PLETH only; other channels' availability is not evidence of their use.

### 1.2 Feature definitions and causal timing

| Predictor | Actual computation and unit |
|---|---|
| `PPG_Amplitude` | Filtered peak minus preceding observed trough; arbitrary PPG units |
| `Pulse_Interval_s` | Time between accepted consecutive peaks; seconds |
| `Pulse_Rate_BPM` | 60 / interval; beats/min; mathematically dependent on interval |
| `Amplitude_Slope` | Change in successive peak amplitudes divided by interval; amplitude units/s |
| `Rate_Mean_5`, `Amplitude_Mean_5` | Means of up to five current/past pulses |
| `Rate_Std_10`, `Amplitude_Std_10`, `Interval_Std_10` | Sample standard deviations over up to ten current/past pulses; singleton initialized to zero |
| `Rate_Change_5` | Current rate minus rate five pulses earlier; zero until six pulses exist |

`src/features.py:FeatureState.update` retains a bounded ten-pulse history and resets between recordings. These windows count pulses, not seconds. No SpO₂, future minimum, recording ID, absolute time, ECG timing or PTT enters model features. `src/preprocessing.py:WaveformState` uses the existing third-order 0.5–5 Hz forward-only Butterworth SOS filter, zero initial state. Peak confirmation requires the next sample (8 ms at 125 Hz), past-trough amplitude ≥0.03 and a ≥0.30 s refractory interval; >2 s intervals are rejected. Predictor timestamp is confirmation time. This causal contract permits streaming inference but does not validate peak accuracy against annotated beats.

### 1.3 Proposal discrepancies and implementation status

| Claim/concept | Repository finding | Review-safe statement |
|---|---|---|
| LightGBM / XGBoost evaluated | No corresponding implementation, dependency, fitted metadata or valid comparison outputs found in active code or source archive | Not evaluated here; do not name as project results |
| PTT used | No proximal/distal arterial timing estimator, ECG R-peak-to-PPG pairing or PTT feature exists | PTT is absent; ECG-to-PPG delay would generally be pulse arrival time, also containing pre-ejection delay, not automatically true PTT |
| Proposed feature/horizon agreement | No separate original proposal/presentation with a verifiable feature list was located | Cannot independently establish every proposal mismatch; actual implementation is the ten features above and a 30 s trained label |
| Physical PID called `Controller` | `src/controller.py` scores/alerts only | ML inference interface, not oxygen feedback |
| P/PI/PID comparison | Now exists in `signal_studies/core.py`, `run.py` and saved outputs | Completed illustrative conditioning-system simulation only |
| Predictive feedforward-PID oxygen control | No risk-to-drive map, oxygen plant, constraints or combined experiment | Proposed architecture, not a completed/validated oxygen controller |
| Real-time/clinical validation | Local offline replay and timing, retrospective recording-wise scores | Causal software demonstration; clinical benefit/generalization unvalidated |

The archived random forest is an additional historical model, discussed below; the maintained four-model comparison should not be described as the only algorithm ever tried.

## 2. Why these models, and what their evidence supports

**Observed source:** `src/model.py:FittedModel` fits a training-only median imputer (`keep_empty_features=True`) and StandardScaler before each estimator. Each inner and outer training partition gets its own transforms. `src/config.py:MODEL_CONFIGS` fixes model settings and seed 42 before this comparison. The rationale below is a technical interpretation of the implementations, not a claim that a documented prospective model-selection trial established these as optimal.

| Model | Implemented settings | Why reasonable / computational implications | Interpretability and main limits |
|---|---|---|---|
| Extreme Learning Machine (ELM) | 50 fixed random tanh units; normal weights, uniform biases; weighted ridge output solved by Cholesky; ridge=1, positive weight=5, seed=42 | Compact nonlinear baseline without iterative hidden-layer optimization; ten-to-50 feature transform and linear output; very low measured model-only time | Random basis/output weights are not clinical feature explanations; needs scaling; few events and random basis can generalize poorly; score is unbounded and uncalibrated |
| Logistic Regression | sklearn `LogisticRegression`, C=1, balanced weights, max_iter=2000, random_state=42 | Regularized linear decision boundary; useful transparent reference with few predictors and limited data | Coefficients explain scaled associations, not causation; correlated interval/rate/history obscure individual attribution; balanced training shifts probability interpretation; nonlinear relations may be missed |
| Gaussian Naive Bayes | sklearn `GaussianNB`, equal priors [0.5,0.5], var_smoothing=1e-6 | Low-parameter statistical baseline; fast fitting and class-conditional Gaussian score evaluation | Conditional independence is explicitly implausible for rate=60/interval and overlapping rolling features; equal priors differ from observed prevalence; Gaussian assumptions and calibration are unvalidated |
| Shallow Decision Tree | sklearn `DecisionTreeClassifier`, depth≤3, min leaf=50, balanced weights, random_state=42 | At most seven split nodes/eight leaves; inexpensive interpretable nonlinear rule baseline | Can inspect paths, but few coarse scores create threshold ties; can collapse to always-alert; unstable partitioning with sparse positive recordings; a readable rule is not a validated physiological mechanism |
| Historical Random Forest | Archived `src/advanced_experiment.py`: 200 trees, depth≤8, min leaf=5, balanced_subsample, seed, n_jobs=-1 | Bagged nonlinear interaction model; heavier historical comparator with measured batch timing | Individual tree ensemble is less transparent; preserved results used a different selected/legacy pipeline, not the final matched causal evaluation; cannot be ranked against current models as a fair test |

All four maintained estimators support causal inference because features use only available history and the fitted transform/model needs no future input. This is conditional on obeying the streaming timestamp/reset/eligibility contract. Simpler models limit fitting cost and make a small-data study easier to audit; measured results do not prove that simplicity beats all more complex models. No LightGBM/XGBoost result is available.

Class imbalance is 4,587/20,157 for future-low but only 501/16,050 for onset. Positive timestamps cluster within a small number of recordings/events; they are not independent examples. Never-alert onset accuracy would be 15,549/16,050, despite detecting nothing. Hence accuracy alone rewards silence.

| Metric | Meaning and denominator here | Misinterpretation to avoid |
|---|---|---|
| Precision | TP/(TP+FP): correct positive-labeled pulse timestamps among alerts | Not event positive predictive value; undefined if no alerts |
| Recall | TP/(TP+FN): positive pulse timestamps covered | Can approach one by alerting almost continuously |
| F1 | 2TP/(2TP+FP+FN) | Threshold/population-specific; not a clinical utility metric |
| PR-AUC_AP | sklearn average precision over score ranking, not trapezoidal PR integration | Compare to prevalence; pooled scores from different refits have comparability limits |
| ROC-AUC | Positive-vs-negative score ordering | Large negative denominator can obscure a poor precision/false-alert tradeoff |
| Event sensitivity | Matched eligible onset count / eligible onset count | Requires explicit onset, coverage and one-to-one start matching |
| False-start burden | Off-event alarm starts / strict evaluable hours | Not FP timestamps/hour; near-event redundant alarms are separately unmatched |
| Active fraction | Left-held positive duration / permitted evaluable edge duration | Not pulse alert fraction; few starts can coexist with nearly continuous activity |
| Warning lead | Onset minus matched actual pre-onset start, detected events only | Misses have undefined lead; a 30 s window is not a 30 s guaranteed warning |

Low latency is useful only if the alert has predictive value. A rapid false alert or rapid silence is not a useful feedforward signal.

## 3. Numerical evidence, separated by experiment

### 3.1 Data accounting and splits

Observed counts from target-specific `dataset_audit.csv`, `dataset_summary.json`, raw CSVs and manifests. No new labels/models were generated for this document.

| Target | Raw_recordings | Nonempty_target_recordings | Pulse_rows_before_target | Evaluable_rows | Positive | Negative | Prevalence | Positive_recordings | All_positive_recordings | Unavailable_targets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| future_low | 30 | 30 | 21948 | 20157 | 4587 | 15570 | 0.22756362554 | 10 | 6 | 1791 |
| new_onset | 30 | 26 | 21948 | 16050 | 501 | 15549 | 0.031214953271 | 6 | 2 | 5898 |

For future-low, unknown targets comprise 413 invalid-future-SpO₂ and 1,378 incomplete horizons. Onset exclusions comprise 4,357 current-low, 155 invalid-current, 265 invalid-future and 1,121 incomplete horizons; these categories reflect the implementation's precedence and must not be combined with the first target's categories as independent cases. All 21,948 pulse rows remain available upstream; unavailable targets do not erase feature history. Four wholly-low recordings 04/08/09/22 have zero onset-target rows and no eligible final alarm exposure. There are 20 negative-only recordings in each nonempty target population.

`raw_data_quality.csv` reports missing numeric SpO₂ at record 01:13, 05:6, 19:102, 25:1 (122 total readings), with no missing raw PLETH. Waveform export has 9,500 duplicate displayed timestamps per record from rounding; the validated 125 Hz row clock retains every row, rather than deduplicating. Record 23 numerics span −1 to 479 s and are not independently rebased. Observed agreement with the reconstruction does not prove monitor physiological delay is known.

**Splitting:** leave one whole nonempty target recording out (30 future-low folds, 26 onset folds). Outer training uses 29 or 25 recordings; inner GroupKFold uses three disjoint recording partitions. There is no single fixed train/validation/test percentage. All rows of the outer-training set receive an OOF score; cutoff selection sees no outer test score/label. The table gives exact pulse-row counts, with min–max inner counts across that outer fold's three splits. Four empty onset records are represented in downstream 30-record policy tables, not fitted as fictitious onset folds.

**future_low split counts (derived from saved recording memberships and dataset audit):**

| Test_record | Test_rows | Test_positives | Outer_train_rows | Inner_train_rows_min | Inner_train_rows_max | Inner_validation_min | Inner_validation_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 619 | 0 | 19538 | 12802 | 13256 | 6282 | 6736 |
| 2 | 700 | 0 | 19457 | 12752 | 13201 | 6256 | 6705 |
| 3 | 580 | 0 | 19577 | 12837 | 13263 | 6314 | 6740 |
| 4 | 690 | 690 | 19467 | 12760 | 13207 | 6260 | 6707 |
| 5 | 675 | 0 | 19482 | 12770 | 13216 | 6266 | 6712 |
| 6 | 624 | 0 | 19533 | 12797 | 13253 | 6280 | 6736 |
| 7 | 682 | 49 | 19475 | 12765 | 13213 | 6262 | 6710 |
| 8 | 748 | 748 | 19409 | 12715 | 13168 | 6241 | 6694 |
| 9 | 577 | 577 | 19580 | 12840 | 13266 | 6314 | 6740 |
| 10 | 621 | 0 | 19536 | 12800 | 13256 | 6280 | 6736 |
| 11 | 751 | 0 | 19406 | 12712 | 13168 | 6238 | 6694 |
| 12 | 702 | 50 | 19455 | 12750 | 13200 | 6255 | 6705 |
| 13 | 509 | 0 | 19648 | 12848 | 13334 | 6314 | 6800 |
| 14 | 709 | 0 | 19448 | 12747 | 13194 | 6254 | 6701 |
| 15 | 804 | 0 | 19353 | 12681 | 13129 | 6224 | 6672 |
| 16 | 822 | 0 | 19335 | 12663 | 13111 | 6224 | 6672 |
| 17 | 626 | 626 | 19531 | 12795 | 13251 | 6280 | 6736 |
| 18 | 686 | 0 | 19471 | 12762 | 13210 | 6261 | 6709 |
| 19 | 241 | 0 | 19916 | 13114 | 13602 | 6314 | 6802 |
| 20 | 667 | 0 | 19490 | 12774 | 13222 | 6268 | 6716 |
| 21 | 921 | 0 | 19236 | 12575 | 13069 | 6167 | 6661 |
| 22 | 615 | 615 | 19542 | 12802 | 13260 | 6282 | 6740 |
| 23 | 515 | 0 | 19642 | 12848 | 13328 | 6314 | 6794 |
| 24 | 718 | 0 | 19439 | 12741 | 13189 | 6250 | 6698 |
| 25 | 927 | 0 | 19230 | 12575 | 13063 | 6167 | 6655 |
| 26 | 660 | 344 | 19497 | 12779 | 13228 | 6269 | 6718 |
| 27 | 660 | 0 | 19497 | 12779 | 13228 | 6269 | 6718 |
| 28 | 636 | 0 | 19521 | 12785 | 13248 | 6273 | 6736 |
| 29 | 685 | 685 | 19472 | 12763 | 13211 | 6261 | 6709 |
| 30 | 787 | 203 | 19370 | 12695 | 13142 | 6228 | 6675 |

**new_onset split counts (derived from saved recording memberships and dataset audit):**

| Test_record | Test_rows | Test_positives | Outer_train_rows | Inner_train_rows_min | Inner_train_rows_max | Inner_validation_min | Inner_validation_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 618 | 0 | 15432 | 10066 | 10467 | 4965 | 5366 |
| 2 | 700 | 0 | 15350 | 10015 | 10413 | 4937 | 5335 |
| 3 | 580 | 0 | 15470 | 10104 | 10467 | 5003 | 5366 |
| 5 | 674 | 0 | 15376 | 10032 | 10430 | 4946 | 5344 |
| 6 | 624 | 0 | 15426 | 10060 | 10464 | 4962 | 5366 |
| 7 | 680 | 49 | 15370 | 10026 | 10430 | 4940 | 5344 |
| 10 | 621 | 0 | 15429 | 10063 | 10465 | 4964 | 5366 |
| 11 | 751 | 0 | 15299 | 9978 | 10384 | 4915 | 5321 |
| 12 | 697 | 47 | 15353 | 10018 | 10413 | 4940 | 5335 |
| 13 | 509 | 0 | 15541 | 10153 | 10538 | 5003 | 5388 |
| 14 | 709 | 0 | 15341 | 10010 | 10404 | 4937 | 5331 |
| 15 | 804 | 0 | 15246 | 9934 | 10353 | 4893 | 5312 |
| 16 | 822 | 0 | 15228 | 9916 | 10353 | 4875 | 5312 |
| 17 | 43 | 43 | 16007 | 10585 | 10810 | 5197 | 5422 |
| 18 | 686 | 0 | 15364 | 10022 | 10424 | 4940 | 5342 |
| 19 | 237 | 0 | 15813 | 10391 | 10810 | 5003 | 5422 |
| 20 | 667 | 0 | 15383 | 10034 | 10437 | 4946 | 5349 |
| 21 | 921 | 0 | 15129 | 9866 | 10276 | 4853 | 5263 |
| 23 | 515 | 0 | 15535 | 10153 | 10532 | 5003 | 5382 |
| 24 | 718 | 0 | 15332 | 10005 | 10397 | 4935 | 5327 |
| 25 | 926 | 0 | 15124 | 9861 | 10271 | 4853 | 5263 |
| 26 | 531 | 221 | 15519 | 10153 | 10516 | 5003 | 5366 |
| 27 | 660 | 0 | 15390 | 10039 | 10444 | 4946 | 5351 |
| 28 | 636 | 0 | 15414 | 10060 | 10452 | 4962 | 5354 |
| 29 | 11 | 11 | 16039 | 10617 | 10810 | 5229 | 5422 |
| 30 | 710 | 130 | 15340 | 10010 | 10403 | 4937 | 5330 |

Source: [results/final/dataset_reconciliation.csv](../results/final/dataset_reconciliation.csv) — record-by-record legacy/future-low/onset count reconciliation.

### 3.2 Fixed-model recall-threshold timestamp results

These are the preserved **92% training-OOF recall policy**, not final alarm-budget thresholds. Counts below pool each held-out pulse exactly once. Each model evaluates the same target-specific rows. Confusion-matrix convention is actual-negative row `[TN,FP]`, actual-positive row `[FN,TP]`; all four counts are included. Metrics use fractions, not percent.

Source: [results/final/future_low/model_comparison.csv](../results/final/future_low/model_comparison.csv).

| Model | TP | FP | TN | FN | Recall | Precision | Specificity | F1 | PR_AUC_AP | ROC_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ELM | 4426 | 14093 | 1477 | 161 | 0.964900806627 | 0.238997786058 | 0.0948619139371 | 0.383103955683 | 0.255929312519 | 0.540807193096 |
| Logistic | 4275 | 15192 | 378 | 312 | 0.931981687377 | 0.219602404068 | 0.0242774566474 | 0.355450236967 | 0.240296002759 | 0.536869701996 |
| GaussianNB | 4505 | 15458 | 112 | 82 | 0.982123392195 | 0.225667484847 | 0.00719332048812 | 0.36700610998 | 0.196266895915 | 0.44632758323 |
| Tree | 4105 | 15021 | 549 | 482 | 0.894920427295 | 0.214629300429 | 0.0352601156069 | 0.346223590436 | 0.189493284679 | 0.336992280409 |

| Model | Balanced_Accuracy | MCC | Alert_Fraction | Macro_Recall | Macro_Precision | Macro_F1 | Macro_PR_AUC_AP | Macro_Recall_Defined_Recordings | Macro_Precision_Defined_Recordings | Training_And_Inner_Validation_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ELM | 0.529881360282 | 0.0917008415661 | 0.918737907427 | 0.937053988782 | 0.2287562551 | 0.239606427521 | 0.238516094587 | 10 | 30 | 10.07290743 |
| Logistic | 0.478129572012 | -0.100860680535 | 0.965768715583 | 0.926580592259 | 0.229172464725 | 0.234408635534 | 0.288179712442 | 10 | 30 | 9.941956449 |
| GaussianNB | 0.494658356342 | -0.0458775076679 | 0.990375551917 | 0.958578919616 | 0.228985103216 | 0.241905980311 | 0.207491870945 | 10 | 30 | 6.600787649 |
| Tree | 0.465090271451 | -0.132875140117 | 0.948851515603 | 0.916464471404 | 0.230758562716 | 0.221535433959 | 0.217648606133 | 10 | 30 | 13.197701339 |

Training cost is total measured outer fitting plus inner validation time in seconds for this saved run, not per inference or an externally comparable benchmark. Macro metrics average only defined per-record values; pooled metrics aggregate counts/scores first. Macro AP includes only records with both classes (four future-low, six onset); it can disagree substantially with pooled AP.

**Constant references for this full target population:**

Source: [results/final/future_low/constant_references.csv](../results/final/future_low/constant_references.csv).

| Model | TP | FP | TN | FN | Recall | Precision | F1 | PR_AUC_AP | ROC_AUC | Alert_Fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Always_Alert | 4587 | 15570 | 0 | 0 | 1 | 0.22756362554 | 0.370756547042 | 0.22756362554 | 0.5 | 1 |
| Never_Alert | 0 | 0 | 15570 | 4587 | 0 | Undefined | 0 | 0.22756362554 | 0.5 | 0 |

Source: [results/final/new_onset/model_comparison.csv](../results/final/new_onset/model_comparison.csv).

| Model | TP | FP | TN | FN | Recall | Precision | Specificity | F1 | PR_AUC_AP | ROC_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ELM | 446 | 13979 | 1570 | 55 | 0.890219560878 | 0.0309185441941 | 0.100971123545 | 0.0597614900174 | 0.0239758420019 | 0.411536564147 |
| Logistic | 444 | 15299 | 250 | 57 | 0.88622754491 | 0.028203010862 | 0.0160782043861 | 0.0546663383403 | 0.0335001273296 | 0.52218888482 |
| GaussianNB | 451 | 13118 | 2431 | 50 | 0.900199600798 | 0.0332375267153 | 0.156344459451 | 0.0641080312722 | 0.0575949774109 | 0.654903775316 |
| Tree | 501 | 15549 | 0 | 0 | 1 | 0.031214953271 | 0 | 0.0605401486315 | 0.0395818484365 | 0.529165991125 |

| Model | Balanced_Accuracy | MCC | Alert_Fraction | Macro_Recall | Macro_Precision | Macro_F1 | Macro_PR_AUC_AP | Macro_Recall_Defined_Recordings | Macro_Precision_Defined_Recordings | Training_And_Inner_Validation_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ELM | 0.495595342212 | -0.00507840893496 | 0.898753894081 | 0.946900298581 | 0.106054736637 | 0.120795692635 | 0.25717522579 | 6 | 26 | 7.843502566 |
| Logistic | 0.451152874648 | -0.124029834883 | 0.980872274143 | 0.926923076923 | 0.104676754696 | 0.11941081236 | 0.329108487952 | 6 | 26 | 7.475883567 |
| GaussianNB | 0.528272030125 | 0.0272000549825 | 0.845420560748 | 0.87703218767 | 0.109355602897 | 0.12597139626 | 0.276267512434 | 6 | 26 | 4.840515604 |
| Tree | 0.5 | Undefined | 1 | 1 | 0.105337889336 | 0.121464022308 | 0.223725967844 | 6 | 26 | 9.718242558 |

Training cost is total measured outer fitting plus inner validation time in seconds for this saved run, not per inference or an externally comparable benchmark. Macro metrics average only defined per-record values; pooled metrics aggregate counts/scores first. Macro AP includes only records with both classes (four future-low, six onset); it can disagree substantially with pooled AP.

**Constant references for this full target population:**

Source: [results/final/new_onset/constant_references.csv](../results/final/new_onset/constant_references.csv).

| Model | TP | FP | TN | FN | Recall | Precision | F1 | PR_AUC_AP | ROC_AUC | Alert_Fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Always_Alert | 501 | 15549 | 0 | 0 | 1 | 0.031214953271 | 0.0605401486315 | 0.031214953271 | 0.5 | 1 |
| Never_Alert | 0 | 0 | 15549 | 501 | 0 | Undefined | 0 | 0.031214953271 | 0.5 | 0 |

A no-information constant score has AP equal to target prevalence and ROC-AUC 0.5; the two constant alert policies have the same ranking metrics but different decisions. Onset tree alerts at every timestamp under this cutoff, even though its unthresholded scores retain a nonconstant ranking. ELM future-low AP exceeds the constant AP slightly, but recall-policy false positives dominate. Onset GaussianNB has the best pooled AP/ROC-AUC/F1 among these learned models; onset Logistic has the highest macro AP. These are distinct comparison statements.

### 3.3 Earlier event counting versus corrected alarm-start counting

The saved target reports count any positive timestamp in an event's look-back window and may credit an ongoing alarm repeatedly. Their detected-only lead can be a timestamp clipped to the window rather than the alarm's actual start. They are preserved historical diagnostics, not the final detection definition.

Source: [results/final/future_low/event_summary.csv](../results/final/future_low/event_summary.csv) — historical any-positive matching, 30 s.

| Model | Subset | Detected | Eligible_Events | Sensitivity | Median_Lead_Time_s | Excluded_Events |
| --- | --- | --- | --- | --- | --- | --- |
| ELM | all_qualifying_onsets | 14 | 14 | 1 | 29.684 | 8 |
| ELM | 91_92_episode_subset | 3 | 3 | 1 | 9.784 | 4 |
| Logistic | all_qualifying_onsets | 14 | 14 | 1 | 29.684 | 8 |
| Logistic | 91_92_episode_subset | 3 | 3 | 1 | 9.784 | 4 |
| GaussianNB | all_qualifying_onsets | 14 | 14 | 1 | 29.684 | 8 |
| GaussianNB | 91_92_episode_subset | 3 | 3 | 1 | 9.784 | 4 |
| Tree | all_qualifying_onsets | 14 | 14 | 1 | 29.684 | 8 |
| Tree | 91_92_episode_subset | 3 | 3 | 1 | 9.784 | 4 |

Source: [results/final/new_onset/event_summary.csv](../results/final/new_onset/event_summary.csv) — historical any-positive matching, 30 s.

| Model | Subset | Detected | Eligible_Events | Sensitivity | Median_Lead_Time_s | Excluded_Events |
| --- | --- | --- | --- | --- | --- | --- |
| ELM | all_qualifying_onsets | 14 | 14 | 1 | 29.376 | 8 |
| ELM | 91_92_episode_subset | 3 | 3 | 1 | 9.784 | 4 |
| Logistic | all_qualifying_onsets | 14 | 14 | 1 | 29.684 | 8 |
| Logistic | 91_92_episode_subset | 3 | 3 | 1 | 9.784 | 4 |
| GaussianNB | all_qualifying_onsets | 14 | 14 | 1 | 29.388 | 8 |
| GaussianNB | 91_92_episode_subset | 3 | 3 | 1 | 9.784 | 4 |
| Tree | all_qualifying_onsets | 14 | 14 | 1 | 29.684 | 8 |
| Tree | 91_92_episode_subset | 3 | 3 | 1 | 9.784 | 4 |

All 14/14 and 3/3 claims above require this caveat. Raw event extraction instead finds 22 low episodes, 15 qualifying onsets, seven episodes containing a 91/92 observation, and three qualifying onsets in that interest subset. Initial/unknown-prior low episodes are left censored. At 30 s one qualifying onset (record 26 at 22 s) lacks sufficient raw lead-in, leaving 14 eligible; shorter windows admit 15. The repeated-positive rule is not proof that an actual new warning starts before all events.

The independent reliability pass retains both methods in `results/reliability/final/window_summary.csv` (`Method`), with 41 independent checks and zero saved discrepancies in `independent_checks.csv`. The corrected local-30 pass reuses authenticated fits/cutoffs and performs chronological earliest-unused **actual positive-run start** matching. Episodes split at negative decisions, record boundaries, >2 s gaps or intervening low/invalid coverage; no start is invented at a look-back boundary. Off-event means no qualifying onset in `(start,start+W]`, including sensitivity-excluded onsets. Extra unmatched starts near real events are not automatically off-event starts.

### 3.4 Corrected local-30 fixed recall-policy results and nonlearned baselines

Source `results/local_30_fixed_policy_v2/event_summary.csv`, `Subset=all`, joined only to the matching `91_92` row for its detected count. All rates use the strict common current-normal exposure **10,329.032 s / 3600 = 2.86917555556 h**, 16,050 timestamps, six recordings containing eligible events, 26 with nonzero exposure; all 30 recordings remain in the roster. W is seconds. `Duty_pct` is 100×time-weighted active fraction. Lead entries are detected-only seconds. Off-event rates for severe subsets retain all-event burden; there is no invented severe-only exposure denominator.

Nonlearned controls: `Always_Alert`, `Never_Alert`, `Periodic_10s_1s` (time modulo 10 <1 s) and `Pulse_Rate_GT100` (pulse-derived rate >100 BPM), implemented in `src/reliability_evaluation.py:baseline_predictions`. They establish whether sensitivity can arise from broad/repetitive alerts without learned anticipation.

**future_low trained scores with their original recall cutoffs, corrected event accounting:**

| Model | Window_s | Detected | Eligible_Events | Severe_detected_of_3 | False_Alarm_Starts_Per_Hour | Duty_pct | Lead_Min_s | Lead_Q25_s | Lead_Median_s | Lead_Q75_s | Lead_Max_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Always_Alert | 5 | 1 | 15 | 0 | 14.9868835724 | 100 | 2.368 | 2.368 | 2.368 | 2.368 | 2.368 |
| Always_Alert | 10 | 3 | 15 | 2 | 14.2898192202 | 100 | 2.368 | 4.368 | 6.368 | 8.076 | 9.784 |
| Always_Alert | 15 | 5 | 15 | 2 | 13.592754868 | 100 | 2.368 | 6.368 | 9.784 | 11.576 | 14.96 |
| Always_Alert | 30 | 8 | 14 | 2 | 12.1986261636 | 100 | 2.368 | 8.93 | 13.268 | 19.824 | 25.6 |
| ELM | 5 | 4 | 15 | 1 | 193.086825561 | 91.7884270278 | 1.816 | 2.23 | 2.604 | 3.056 | 3.704 |
| ELM | 10 | 8 | 15 | 2 | 190.647100328 | 91.7884270278 | 2.368 | 5.486 | 6.908 | 8.05 | 9.784 |
| ELM | 15 | 9 | 15 | 2 | 187.85884292 | 91.7884270278 | 2.368 | 7.68 | 11.576 | 13.232 | 14.296 |
| ELM | 30 | 12 | 14 | 2 | 182.282328102 | 91.7884270278 | 6.368 | 12.224 | 18.2 | 25.942 | 29.704 |
| GaussianNB | 5 | 3 | 15 | 0 | 26.4884453838 | 99.2862061033 | 2.16 | 2.264 | 2.368 | 3.036 | 3.704 |
| GaussianNB | 10 | 5 | 15 | 2 | 25.0943166794 | 99.2862061033 | 2.16 | 2.368 | 6.368 | 7.184 | 9.784 |
| GaussianNB | 15 | 7 | 15 | 2 | 23.0031236228 | 99.2862061033 | 2.368 | 8.076 | 11.576 | 13.436 | 14.464 |
| GaussianNB | 30 | 10 | 14 | 2 | 20.9119305662 | 99.2862061033 | 2.368 | 10.232 | 14.38 | 24.354 | 29.704 |
| Logistic | 5 | 3 | 15 | 0 | 41.4753289563 | 97.4526945023 | 2.16 | 2.264 | 2.368 | 3.036 | 3.704 |
| Logistic | 10 | 5 | 15 | 2 | 40.429732428 | 97.4526945023 | 2.16 | 2.368 | 5.088 | 6.368 | 9.784 |
| Logistic | 15 | 6 | 15 | 2 | 39.0356037236 | 97.4526945023 | 2.368 | 7.222 | 10.68 | 11.84 | 14.96 |
| Logistic | 30 | 9 | 14 | 2 | 37.6414750191 | 97.4526945023 | 2.368 | 9.784 | 11.928 | 20.616 | 29.704 |
| Never_Alert | 5 | 0 | 15 | 0 | 0 | 0 | Undefined | Undefined | Undefined | Undefined | Undefined |
| Never_Alert | 10 | 0 | 15 | 0 | 0 | 0 | Undefined | Undefined | Undefined | Undefined | Undefined |
| Never_Alert | 15 | 0 | 15 | 0 | 0 | 0 | Undefined | Undefined | Undefined | Undefined | Undefined |
| Never_Alert | 30 | 0 | 14 | 0 | 0 | 0 | Undefined | Undefined | Undefined | Undefined | Undefined |
| Periodic_10s_1s | 5 | 6 | 15 | 0 | 356.89694833 | 10.1002301087 | 0.552 | 1.646 | 1.668 | 3.13 | 3.728 |
| Periodic_10s_1s | 10 | 12 | 15 | 2 | 354.805755273 | 10.1002301087 | 0.552 | 1.67 | 5.112 | 7.972 | 9.96 |
| Periodic_10s_1s | 15 | 13 | 15 | 2 | 352.714562216 | 10.1002301087 | 3.728 | 7.384 | 9.96 | 11.728 | 13.784 |
| Periodic_10s_1s | 30 | 13 | 14 | 2 | 348.532176103 | 10.1002301087 | 9.784 | 20.992 | 23.6 | 26.44 | 29.784 |
| Pulse_Rate_GT100 | 5 | 7 | 15 | 2 | 363.1705275 | 26.019863236 | 0.00800000000001 | 1.312 | 1.84 | 3.536 | 4.768 |
| Pulse_Rate_GT100 | 10 | 11 | 15 | 3 | 358.63960921 | 26.019863236 | 3.344 | 6.252 | 7.304 | 8.944 | 9.96 |
| Pulse_Rate_GT100 | 15 | 12 | 15 | 3 | 353.760158745 | 26.019863236 | 3.344 | 8.792 | 10.96 | 13.008 | 14.232 |
| Pulse_Rate_GT100 | 30 | 13 | 14 | 3 | 346.440983047 | 26.019863236 | 3.344 | 12.632 | 18.128 | 25.6 | 28.912 |
| Tree | 5 | 1 | 15 | 0 | 42.5209254846 | 95.4632728411 | 2.368 | 2.368 | 2.368 | 2.368 | 2.368 |
| Tree | 10 | 3 | 15 | 2 | 41.8238611324 | 95.4632728411 | 2.368 | 4.368 | 6.368 | 8.076 | 9.784 |
| Tree | 15 | 5 | 15 | 2 | 41.1267967802 | 95.4632728411 | 2.368 | 6.368 | 9.784 | 11.576 | 14.96 |
| Tree | 30 | 8 | 14 | 2 | 39.7326680758 | 95.4632728411 | 2.368 | 8.93 | 13.268 | 19.824 | 25.6 |

**new_onset trained scores with their original recall cutoffs, corrected event accounting:**

| Model | Window_s | Detected | Eligible_Events | Severe_detected_of_3 | False_Alarm_Starts_Per_Hour | Duty_pct | Lead_Min_s | Lead_Q25_s | Lead_Median_s | Lead_Q75_s | Lead_Max_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Always_Alert | 5 | 1 | 15 | 0 | 14.9868835724 | 100 | 2.368 | 2.368 | 2.368 | 2.368 | 2.368 |
| Always_Alert | 10 | 3 | 15 | 2 | 14.2898192202 | 100 | 2.368 | 4.368 | 6.368 | 8.076 | 9.784 |
| Always_Alert | 15 | 5 | 15 | 2 | 13.592754868 | 100 | 2.368 | 6.368 | 9.784 | 11.576 | 14.96 |
| Always_Alert | 30 | 8 | 14 | 2 | 12.1986261636 | 100 | 2.368 | 8.93 | 13.268 | 19.824 | 25.6 |
| ELM | 5 | 6 | 15 | 0 | 197.617743851 | 89.0917948555 | 1.232 | 1.576 | 2.676 | 3.14 | 4.864 |
| ELM | 10 | 10 | 15 | 2 | 191.692696857 | 89.0917948555 | 1.232 | 3.368 | 7.536 | 9.516 | 9.936 |
| ELM | 15 | 11 | 15 | 2 | 189.252971624 | 89.0917948555 | 1.312 | 7.172 | 9.784 | 13.204 | 14.96 |
| ELM | 30 | 13 | 14 | 2 | 185.419117687 | 89.0917948555 | 6.368 | 14.96 | 19.56 | 25.6 | 28.44 |
| GaussianNB | 5 | 1 | 15 | 0 | 65.8725812835 | 85.0368166155 | 2.368 | 2.368 | 2.368 | 2.368 | 2.368 |
| GaussianNB | 10 | 3 | 15 | 2 | 65.1755169313 | 85.0368166155 | 2.368 | 4.368 | 6.368 | 8.076 | 9.784 |
| GaussianNB | 15 | 6 | 15 | 3 | 64.129920403 | 85.0368166155 | 2.368 | 7.222 | 10.68 | 12.542 | 14.96 |
| GaussianNB | 30 | 9 | 14 | 3 | 62.7357916986 | 85.0368166155 | 2.368 | 9.784 | 12.864 | 19.56 | 25.6 |
| Logistic | 5 | 3 | 15 | 0 | 27.5340419122 | 98.1371342445 | 0.808 | 1.588 | 2.368 | 3.38 | 4.392 |
| Logistic | 10 | 5 | 15 | 2 | 26.4884453838 | 98.1371342445 | 0.808 | 2.368 | 5.088 | 6.368 | 9.784 |
| Logistic | 15 | 6 | 15 | 2 | 25.4428488555 | 98.1371342445 | 2.368 | 7.222 | 10.68 | 12.818 | 14.96 |
| Logistic | 30 | 9 | 14 | 2 | 24.0487201511 | 98.1371342445 | 2.368 | 9.784 | 13.232 | 19.56 | 25.6 |
| Never_Alert | 5 | 0 | 15 | 0 | 0 | 0 | Undefined | Undefined | Undefined | Undefined | Undefined |
| Never_Alert | 10 | 0 | 15 | 0 | 0 | 0 | Undefined | Undefined | Undefined | Undefined | Undefined |
| Never_Alert | 15 | 0 | 15 | 0 | 0 | 0 | Undefined | Undefined | Undefined | Undefined | Undefined |
| Never_Alert | 30 | 0 | 14 | 0 | 0 | 0 | Undefined | Undefined | Undefined | Undefined | Undefined |
| Periodic_10s_1s | 5 | 6 | 15 | 0 | 356.89694833 | 10.1002301087 | 0.552 | 1.646 | 1.668 | 3.13 | 3.728 |
| Periodic_10s_1s | 10 | 12 | 15 | 2 | 354.805755273 | 10.1002301087 | 0.552 | 1.67 | 5.112 | 7.972 | 9.96 |
| Periodic_10s_1s | 15 | 13 | 15 | 2 | 352.714562216 | 10.1002301087 | 3.728 | 7.384 | 9.96 | 11.728 | 13.784 |
| Periodic_10s_1s | 30 | 13 | 14 | 2 | 348.532176103 | 10.1002301087 | 9.784 | 20.992 | 23.6 | 26.44 | 29.784 |
| Pulse_Rate_GT100 | 5 | 7 | 15 | 2 | 363.1705275 | 26.019863236 | 0.00800000000001 | 1.312 | 1.84 | 3.536 | 4.768 |
| Pulse_Rate_GT100 | 10 | 11 | 15 | 3 | 358.63960921 | 26.019863236 | 3.344 | 6.252 | 7.304 | 8.944 | 9.96 |
| Pulse_Rate_GT100 | 15 | 12 | 15 | 3 | 353.760158745 | 26.019863236 | 3.344 | 8.792 | 10.96 | 13.008 | 14.232 |
| Pulse_Rate_GT100 | 30 | 13 | 14 | 3 | 346.440983047 | 26.019863236 | 3.344 | 12.632 | 18.128 | 25.6 | 28.912 |
| Tree | 5 | 1 | 15 | 0 | 14.9868835724 | 100 | 2.368 | 2.368 | 2.368 | 2.368 | 2.368 |
| Tree | 10 | 3 | 15 | 2 | 14.2898192202 | 100 | 2.368 | 4.368 | 6.368 | 8.076 | 9.784 |
| Tree | 15 | 5 | 15 | 2 | 13.592754868 | 100 | 2.368 | 6.368 | 9.784 | 11.576 | 14.96 |
| Tree | 30 | 8 | 14 | 2 | 12.1986261636 | 100 | 2.368 | 8.93 | 13.268 | 19.824 | 25.6 |

The fixed-policy sample baselines below use each target's **full target** population, not the common normal-only event population. Their AP uses the binary baseline decision as score; source `timestamp_pooled.csv`, `Population=full_target`. Do not combine the future-low full-target denominator with normal-only event rates.

| Target | Model | Samples | TP | FP | TN | FN | Recall | Precision | F1 | PR_AUC_AP | ROC_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| future_low | Always_Alert | 20157 | 4587 | 15570 | 0 | 0 | 1 | 0.22756362554 | 0.370756547042 | 0.22756362554 | 0.5 |
| future_low | Never_Alert | 20157 | 0 | 0 | 15570 | 4587 | 0 | Undefined | 0 | 0.22756362554 | 0.5 |
| future_low | Periodic_10s_1s | 20157 | 449 | 1571 | 13999 | 4138 | 0.0978853281012 | 0.222277227723 | 0.135916452248 | 0.227046164755 | 0.49849308152 |
| future_low | Pulse_Rate_GT100 | 20157 | 922 | 4746 | 10824 | 3665 | 0.201002834096 | 0.162667607622 | 0.179814724525 | 0.214519342016 | 0.448092939206 |
| new_onset | Always_Alert | 16050 | 501 | 15549 | 0 | 0 | 1 | 0.031214953271 | 0.0605401486315 | 0.031214953271 | 0.5 |
| new_onset | Never_Alert | 16050 | 0 | 0 | 15549 | 501 | 0 | Undefined | 0 | 0.031214953271 | 0.5 |
| new_onset | Periodic_10s_1s | 16050 | 47 | 1570 | 13979 | 454 | 0.0938123752495 | 0.0290661719233 | 0.0443814919736 | 0.0310133709889 | 0.496420625852 |
| new_onset | Pulse_Rate_GT100 | 16050 | 136 | 4742 | 10807 | 365 | 0.271457085828 | 0.0278802788028 | 0.0505670198922 | 0.0303097322577 | 0.483242852516 |

The complete per-record outcomes are in `event_per_recording.csv`, `timestamp_per_recording.csv`, `event_matches.csv` and detected leads in `detected_event_leads.csv`. `results/reliability/final/timestamp_burden.csv` also distinguishes historical full-target edges from strict normal-state edges. Historical `model_comparison.csv:False_Alerts_Per_Hour` counts runs with no positive target anywhere in the run; it is **not** the corrected off-event-start metric. In particular, old onset exposure 2.86972666667 h includes a short bridge excluded by strict event evaluation; do not merge its rate denominator with 2.86917555556 h.

### 3.5 Final alarm-budget experiment — decisive operating-policy result

Observed source: `results/alarm_budget_experiment/protocol.json`, `run02/manifest.json`, `selected_thresholds.csv`, `summary.csv`. This changes thresholds only, not features, labels, score fits or the original saved cutoffs. There are **672** training-selected policies = (30+26) nonempty target folds × four models × three budgets. Training candidates are unique linear score quantiles 0,.01,…,1 plus .995/.9975/.999, plus a threshold just above the training maximum. Candidates satisfy BOTH 1/5/10 off-event starts per hour and ≤5% active duration. Among feasible candidates maximize training-only 30 s detections; ties prefer fewer off-event starts, less active time, then higher threshold. Start counts need not be monotone in threshold. This finite grid is not a proof of the best achievable operating policy.

Every selection uses only the correct outer-training grouped OOF score rows, identified by `Outer_Test_Record` and `Record`. Selection completes before held-out score values are read. No new fit is necessary because only the cutoff changes. Training outcomes are selection diagnostics, not independent validation estimates. Previously inspecting this cohort prevents calling the frozen protocol an untouched confirmatory trial.

The four learned models, **both training targets and every budget**, detect **0/15 at 5, 10 and 15 s and 0/14 at 30 s**. Their severe-interest detections are **0/3** in each window. No learned matched leads exist; min, quartiles, median and max are all undefined. All 501 positive timestamps are missed under the final policies. This policy did not demonstrate useful early warning; the all-zero detection is not hidden behind the earlier high-recall cutoffs.

**future_low trained models, final policy at 30 s; exact counts and achieved burden:**

| Model | Budget | Detected | Eligible_Events | Off_Event_Alarm_Starts | False_Alarm_Starts_Per_Hour | Duty_pct | Pulse_alert_pct | TP | FP | TN | FN | Timestamp_Precision | Timestamp_Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Always_Alert | 1 | 8 | 14 | 35 | 12.1986261636 | 100 | 100 | 501 | 15549 | 0 | 0 | 0.031214953271 | 1 |
| Always_Alert | 5 | 8 | 14 | 35 | 12.1986261636 | 100 | 100 | 501 | 15549 | 0 | 0 | 0.031214953271 | 1 |
| Always_Alert | 10 | 8 | 14 | 35 | 12.1986261636 | 100 | 100 | 501 | 15549 | 0 | 0 | 0.031214953271 | 1 |
| ELM | 1 | 0 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 15549 | 501 | Undefined | 0 |
| ELM | 5 | 0 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 15549 | 501 | Undefined | 0 |
| ELM | 10 | 0 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 15549 | 501 | Undefined | 0 |
| GaussianNB | 1 | 0 | 14 | 1 | 0.348532176103 | 0.0116177392034 | 0.00623052959502 | 0 | 1 | 15548 | 501 | 0 | 0 |
| GaussianNB | 5 | 0 | 14 | 1 | 0.348532176103 | 0.0116177392034 | 0.00623052959502 | 0 | 1 | 15548 | 501 | 0 | 0 |
| GaussianNB | 10 | 0 | 14 | 1 | 0.348532176103 | 0.0116177392034 | 0.00623052959502 | 0 | 1 | 15548 | 501 | 0 | 0 |
| Logistic | 1 | 0 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 15549 | 501 | Undefined | 0 |
| Logistic | 5 | 0 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 15549 | 501 | Undefined | 0 |
| Logistic | 10 | 0 | 14 | 1 | 0.348532176103 | 0.00472454727607 | 0.00623052959502 | 0 | 1 | 15548 | 501 | 0 | 0 |
| Never_Alert | 1 | 0 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 15549 | 501 | Undefined | 0 |
| Never_Alert | 5 | 0 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 15549 | 501 | Undefined | 0 |
| Never_Alert | 10 | 0 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 15549 | 501 | Undefined | 0 |
| Tree | 1 | 0 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 15549 | 501 | Undefined | 0 |
| Tree | 5 | 0 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 15549 | 501 | Undefined | 0 |
| Tree | 10 | 0 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 15549 | 501 | Undefined | 0 |

**new_onset trained models, final policy at 30 s; exact counts and achieved burden:**

| Model | Budget | Detected | Eligible_Events | Off_Event_Alarm_Starts | False_Alarm_Starts_Per_Hour | Duty_pct | Pulse_alert_pct | TP | FP | TN | FN | Timestamp_Precision | Timestamp_Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Always_Alert | 1 | 8 | 14 | 35 | 12.1986261636 | 100 | 100 | 501 | 15549 | 0 | 0 | 0.031214953271 | 1 |
| Always_Alert | 5 | 8 | 14 | 35 | 12.1986261636 | 100 | 100 | 501 | 15549 | 0 | 0 | 0.031214953271 | 1 |
| Always_Alert | 10 | 8 | 14 | 35 | 12.1986261636 | 100 | 100 | 501 | 15549 | 0 | 0 | 0.031214953271 | 1 |
| ELM | 1 | 0 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 15549 | 501 | Undefined | 0 |
| ELM | 5 | 0 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 15549 | 501 | Undefined | 0 |
| ELM | 10 | 0 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 15549 | 501 | Undefined | 0 |
| GaussianNB | 1 | 0 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 15549 | 501 | Undefined | 0 |
| GaussianNB | 5 | 0 | 14 | 1 | 0.348532176103 | 0.0116177392034 | 0.00623052959502 | 0 | 1 | 15548 | 501 | 0 | 0 |
| GaussianNB | 10 | 0 | 14 | 3 | 1.04559652831 | 0.0199825114299 | 0.018691588785 | 0 | 3 | 15546 | 501 | 0 | 0 |
| Logistic | 1 | 0 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 15549 | 501 | Undefined | 0 |
| Logistic | 5 | 0 | 14 | 4 | 1.39412870441 | 0.134223613597 | 0.143302180685 | 0 | 23 | 15526 | 501 | 0 | 0 |
| Logistic | 10 | 0 | 14 | 14 | 4.87945046545 | 0.347835111751 | 0.323987538941 | 0 | 52 | 15497 | 501 | 0 | 0 |
| Never_Alert | 1 | 0 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 15549 | 501 | Undefined | 0 |
| Never_Alert | 5 | 0 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 15549 | 501 | Undefined | 0 |
| Never_Alert | 10 | 0 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 15549 | 501 | Undefined | 0 |
| Tree | 1 | 0 | 14 | 57 | 19.8663340379 | 0.891158048499 | 1.07788161994 | 0 | 173 | 15376 | 501 | 0 | 0 |
| Tree | 5 | 0 | 14 | 57 | 19.8663340379 | 0.891158048499 | 1.07788161994 | 0 | 173 | 15376 | 501 | 0 | 0 |
| Tree | 10 | 0 | 14 | 57 | 19.8663340379 | 0.891158048499 | 1.07788161994 | 0 | 173 | 15376 | 501 | 0 | 0 |

In those tables, samples=16,050, positives=501, negatives=15,549 for every row. Never-alert is budget-compliant silence. Always-alert is an unselected reference, not a burden-matched competitor: 100% duty violates the cap. Its corrected detections and leads across windows are:

| Window_s | Detected | Eligible_Events | False_Alarm_Starts_Per_Hour | Lead_Min_s | Lead_Q25_s | Lead_Median_s | Lead_Q75_s | Lead_Max_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 1 | 15 | 14.9868835724 | 2.368 | 2.368 | 2.368 | 2.368 | 2.368 |
| 10 | 3 | 15 | 14.2898192202 | 2.368 | 4.368 | 6.368 | 8.076 | 9.784 |
| 15 | 5 | 15 | 13.592754868 | 2.368 | 6.368 | 9.784 | 11.576 | 14.96 |
| 30 | 8 | 14 | 12.1986261636 | 2.368 | 8.93 | 13.268 | 19.824 | 25.6 |

Timestamp precision is undefined for a silent policy and zero for an alerting learned policy with no true positives. Timestamp F1 for these learned final policies is derived as zero; threshold-independent AP/ROC can be computed on the same normal-state score population, but **are not stored in `run02/summary.csv`**. The full-target historical future-low AP must not be substituted. Independently trained 5/10/15 s classifiers and their AP/ROC are **Not available in the current saved results**.

**Budget transfer:** all selected training candidates satisfy both constraints; this does not guarantee each outer record will. The following is a grouping of `outer_constraint_status.csv`, counting True flags across all 30 records. Undefined flags at zero exposure are not counted as False successes. The pooled new-onset tree rate approximately 19.8663/hour exceeds every nominal budget even though its pooled active fraction is <5%. All three new-onset tree budget panels have one per-record active-cap violation; pooled compliance cannot conceal it.

| Target | Model | Budget | Outer_start_budget_violations | Outer_active_cap_violations |
| --- | --- | --- | --- | --- |
| future_low | ELM | 1 | 0 | 0 |
| future_low | ELM | 5 | 0 | 0 |
| future_low | ELM | 10 | 0 | 0 |
| future_low | GaussianNB | 1 | 1 | 0 |
| future_low | GaussianNB | 5 | 1 | 0 |
| future_low | GaussianNB | 10 | 1 | 0 |
| future_low | Logistic | 1 | 0 | 0 |
| future_low | Logistic | 5 | 0 | 0 |
| future_low | Logistic | 10 | 0 | 0 |
| future_low | Tree | 1 | 0 | 0 |
| future_low | Tree | 5 | 0 | 0 |
| future_low | Tree | 10 | 0 | 0 |
| new_onset | ELM | 1 | 0 | 0 |
| new_onset | ELM | 5 | 0 | 0 |
| new_onset | ELM | 10 | 0 | 0 |
| new_onset | GaussianNB | 1 | 0 | 0 |
| new_onset | GaussianNB | 5 | 1 | 0 |
| new_onset | GaussianNB | 10 | 1 | 0 |
| new_onset | Logistic | 1 | 0 | 0 |
| new_onset | Logistic | 5 | 3 | 0 |
| new_onset | Logistic | 10 | 1 | 0 |
| new_onset | Tree | 1 | 4 | 1 |
| new_onset | Tree | 5 | 4 | 1 |
| new_onset | Tree | 10 | 2 | 1 |

The budget is selected using the **30 s** off-event definition. Shorter-window false-start rates can differ and are not promised to satisfy it. References are repeated across budgets without being selected. No cooldown/smoothing/persistence rule was added to repair outer outcomes.

### 3.6 Uncertainty, thresholds, calibration and severe events

Whole-record paired bootstrap uses 2,000 draws of the 30-record roster, seed 20261014, conditional on saved CV fits. It captures recording resampling variation, not independent-patient generalization or refitting/selection uncertainty. Undefined replicates (no eligible event) remain undefined. A zero-to-zero bootstrap interval for an all-zero detector is a consequence of resampling zero successes, not a precise population guarantee.

Selected 30 s, all-event confidence intervals below come directly from the corresponding saved interval CSVs. The full 640-row local and 1,440-row final interval tables retain all models/windows/subsets; paired comparisons are stored separately. These are exploratory uncertainty summaries, not multiple-comparison-corrected superiority tests.

Source: [results/local_30_fixed_policy_v2/recording_bootstrap_intervals.csv](../results/local_30_fixed_policy_v2/recording_bootstrap_intervals.csv).

| Target | Model | Metric | Lower_95 | Upper_95 | Defined_Replicates | Replicates |
| --- | --- | --- | --- | --- | --- | --- |
| future_low | Always_Alert | Sensitivity | 0.333333333333 | 0.857142857143 | 1996 | 2000 |
| future_low | Always_Alert | False_Alarm_Starts_Per_Hour | 9.30227453838 | 16.3575844468 | 2000 | 2000 |
| future_low | ELM | Sensitivity | 0.6 | 1 | 1996 | 2000 |
| future_low | ELM | False_Alarm_Starts_Per_Hour | 94.1901542822 | 312.278507814 | 2000 | 2000 |
| future_low | GaussianNB | Sensitivity | 0.5 | 1 | 1996 | 2000 |
| future_low | GaussianNB | False_Alarm_Starts_Per_Hour | 9.86538936664 | 41.0335302271 | 2000 | 2000 |
| future_low | Logistic | Sensitivity | 0.333333333333 | 1 | 1996 | 2000 |
| future_low | Logistic | False_Alarm_Starts_Per_Hour | 18.0289342632 | 64.5431308587 | 2000 | 2000 |
| future_low | Never_Alert | Sensitivity | 0 | 0 | 1996 | 2000 |
| future_low | Never_Alert | False_Alarm_Starts_Per_Hour | 0 | 0 | 2000 | 2000 |
| future_low | Periodic_10s_1s | Sensitivity | 0.6 | 1 | 1996 | 2000 |
| future_low | Periodic_10s_1s | False_Alarm_Starts_Per_Hour | 335.928684324 | 356.858185016 | 2000 | 2000 |
| future_low | Pulse_Rate_GT100 | Sensitivity | 0.857142857143 | 1 | 1996 | 2000 |
| future_low | Pulse_Rate_GT100 | False_Alarm_Starts_Per_Hour | 233.478703298 | 470.502665023 | 2000 | 2000 |
| future_low | Tree | Sensitivity | 0.333333333333 | 0.857142857143 | 1996 | 2000 |
| future_low | Tree | False_Alarm_Starts_Per_Hour | 13.2945695756 | 74.4101739559 | 2000 | 2000 |
| new_onset | Always_Alert | Sensitivity | 0.333333333333 | 0.857142857143 | 1996 | 2000 |
| new_onset | Always_Alert | False_Alarm_Starts_Per_Hour | 9.30227453838 | 16.3575844468 | 2000 | 2000 |
| new_onset | ELM | Sensitivity | 0.625 | 1 | 1996 | 2000 |
| new_onset | ELM | False_Alarm_Starts_Per_Hour | 116.783405063 | 256.847914694 | 2000 | 2000 |
| new_onset | GaussianNB | Sensitivity | 0.454545454545 | 1 | 1996 | 2000 |
| new_onset | GaussianNB | False_Alarm_Starts_Per_Hour | 31.3345381537 | 99.5861956051 | 2000 | 2000 |
| new_onset | Logistic | Sensitivity | 0.333333333333 | 1 | 1996 | 2000 |
| new_onset | Logistic | False_Alarm_Starts_Per_Hour | 13.6036970482 | 39.4631873001 | 2000 | 2000 |
| new_onset | Never_Alert | Sensitivity | 0 | 0 | 1996 | 2000 |
| new_onset | Never_Alert | False_Alarm_Starts_Per_Hour | 0 | 0 | 2000 | 2000 |
| new_onset | Periodic_10s_1s | Sensitivity | 0.6 | 1 | 1996 | 2000 |
| new_onset | Periodic_10s_1s | False_Alarm_Starts_Per_Hour | 335.928684324 | 356.858185016 | 2000 | 2000 |
| new_onset | Pulse_Rate_GT100 | Sensitivity | 0.857142857143 | 1 | 1996 | 2000 |
| new_onset | Pulse_Rate_GT100 | False_Alarm_Starts_Per_Hour | 233.478703298 | 470.502665023 | 2000 | 2000 |
| new_onset | Tree | Sensitivity | 0.333333333333 | 0.857142857143 | 1996 | 2000 |
| new_onset | Tree | False_Alarm_Starts_Per_Hour | 9.30227453838 | 16.3575844468 | 2000 | 2000 |

Source: [results/alarm_budget_experiment/run02/bootstrap_intervals.csv](../results/alarm_budget_experiment/run02/bootstrap_intervals.csv).

| Target | Model | Budget | Metric | Lower_95 | Upper_95 | Defined_Replicates | Replicates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| future_low | Always_Alert | 5 | Sensitivity | 0.333333333333 | 0.857142857143 | 1996 | 2000 |
| future_low | Always_Alert | 5 | False_Alarm_Starts_Per_Hour | 9.30227453838 | 16.3575844468 | 2000 | 2000 |
| future_low | ELM | 5 | Sensitivity | 0 | 0 | 1996 | 2000 |
| future_low | ELM | 5 | False_Alarm_Starts_Per_Hour | 0 | 0 | 2000 | 2000 |
| future_low | GaussianNB | 5 | Sensitivity | 0 | 0 | 1996 | 2000 |
| future_low | GaussianNB | 5 | False_Alarm_Starts_Per_Hour | 0 | 1.18805721783 | 2000 | 2000 |
| future_low | Logistic | 5 | Sensitivity | 0 | 0 | 1996 | 2000 |
| future_low | Logistic | 5 | False_Alarm_Starts_Per_Hour | 0 | 0 | 2000 | 2000 |
| future_low | Never_Alert | 5 | Sensitivity | 0 | 0 | 1996 | 2000 |
| future_low | Never_Alert | 5 | False_Alarm_Starts_Per_Hour | 0 | 0 | 2000 | 2000 |
| future_low | Tree | 5 | Sensitivity | 0 | 0 | 1996 | 2000 |
| future_low | Tree | 5 | False_Alarm_Starts_Per_Hour | 0 | 0 | 2000 | 2000 |
| new_onset | Always_Alert | 5 | Sensitivity | 0.333333333333 | 0.857142857143 | 1996 | 2000 |
| new_onset | Always_Alert | 5 | False_Alarm_Starts_Per_Hour | 9.30227453838 | 16.3575844468 | 2000 | 2000 |
| new_onset | ELM | 5 | Sensitivity | 0 | 0 | 1996 | 2000 |
| new_onset | ELM | 5 | False_Alarm_Starts_Per_Hour | 0 | 0 | 2000 | 2000 |
| new_onset | GaussianNB | 5 | Sensitivity | 0 | 0 | 1996 | 2000 |
| new_onset | GaussianNB | 5 | False_Alarm_Starts_Per_Hour | 0 | 1.18805721783 | 2000 | 2000 |
| new_onset | Logistic | 5 | Sensitivity | 0 | 0 | 1996 | 2000 |
| new_onset | Logistic | 5 | False_Alarm_Starts_Per_Hour | 0 | 3.26262500218 | 2000 | 2000 |
| new_onset | Never_Alert | 5 | Sensitivity | 0 | 0 | 1996 | 2000 |
| new_onset | Never_Alert | 5 | False_Alarm_Starts_Per_Hour | 0 | 0 | 2000 | 2000 |
| new_onset | Tree | 5 | Sensitivity | 0 | 0 | 1996 | 2000 |
| new_onset | Tree | 5 | False_Alarm_Starts_Per_Hour | 0.352638828876 | 55.277796695 | 2000 | 2000 |

Thresholds differ by outer fold and score scale; no universal physiological risk cutoff is validated. Historical selected cutoffs are in `results/final/<target>/threshold_analysis.csv` (one selected row per model/test fold); all unique-score alternatives are in `.csv.gz`, and original OOF outcomes in `validation_metrics.csv`. Final selected cutoffs and exact hexadecimal floats are in `run02/selected_thresholds.csv`. The table below groups those saved selections; training detections are **not** held-out detections.

| Policy | Target | Model | Budget | Folds | Threshold_min | Threshold_max | Silent_training_selections |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Recall OOF | future_low | ELM | n/a | 30 | -0.089767717104 | 0.237319683951 | n/a |
| Recall OOF | future_low | GaussianNB | n/a | 30 | 0.000111121371537 | 0.108900398803 | n/a |
| Recall OOF | future_low | Logistic | n/a | 30 | 0.0875398495755 | 0.268766442632 | n/a |
| Recall OOF | future_low | Tree | n/a | 30 | 0.00123081836188 | 0.139296618337 | n/a |
| Recall OOF | new_onset | ELM | n/a | 26 | -0.143865778984 | 0.0164070888474 | n/a |
| Recall OOF | new_onset | GaussianNB | n/a | 26 | 5.12758272628e-27 | 0.0278018619541 | n/a |
| Recall OOF | new_onset | Logistic | n/a | 26 | 0.00169608906867 | 0.205370077219 | n/a |
| Recall OOF | new_onset | Tree | n/a | 26 | 0 | 0.0855693119545 | n/a |
| Alarm budget | future_low | ELM | 1 | 30 | 1.47327935569 | 1.70658940655 | 29 |
| Alarm budget | future_low | ELM | 5 | 30 | 1.33367357036 | 1.70658940655 | 26 |
| Alarm budget | future_low | ELM | 10 | 30 | 1.28403271771 | 1.70658940655 | 22 |
| Alarm budget | future_low | GaussianNB | 1 | 30 | 0.999931986923 | 1 | 30 |
| Alarm budget | future_low | GaussianNB | 5 | 30 | 0.999925552253 | 1 | 27 |
| Alarm budget | future_low | GaussianNB | 10 | 30 | 0.999925552253 | 1 | 27 |
| Alarm budget | future_low | Logistic | 1 | 30 | 0.998337303213 | 0.999999939506 | 30 |
| Alarm budget | future_low | Logistic | 5 | 30 | 0.998337303213 | 0.999999939506 | 30 |
| Alarm budget | future_low | Logistic | 10 | 30 | 0.982359550637 | 0.999999939506 | 24 |
| Alarm budget | future_low | Tree | 1 | 30 | 0.801142305069 | 0.995912561071 | 29 |
| Alarm budget | future_low | Tree | 5 | 30 | 0.801142305069 | 0.995912561071 | 29 |
| Alarm budget | future_low | Tree | 10 | 30 | 0.801142305069 | 0.995912561071 | 29 |
| Alarm budget | new_onset | ELM | 1 | 26 | 0.662349319524 | 1.0854342433 | 26 |
| Alarm budget | new_onset | ELM | 5 | 26 | 0.662349319524 | 1.0854342433 | 26 |
| Alarm budget | new_onset | ELM | 10 | 26 | 0.60584498809 | 1.0854342433 | 25 |
| Alarm budget | new_onset | GaussianNB | 1 | 26 | 1 | 1 | 26 |
| Alarm budget | new_onset | GaussianNB | 5 | 26 | 0.999999999993 | 1 | 14 |
| Alarm budget | new_onset | GaussianNB | 10 | 26 | 0.999999999993 | 1 | 9 |
| Alarm budget | new_onset | Logistic | 1 | 26 | 0.99742722407 | 0.999998500389 | 26 |
| Alarm budget | new_onset | Logistic | 5 | 26 | 0.939087723151 | 0.999996813369 | 16 |
| Alarm budget | new_onset | Logistic | 10 | 26 | 0.939087723151 | 0.999983660448 | 10 |
| Alarm budget | new_onset | Tree | 1 | 26 | 0.876731143058 | 0.974744155478 | 26 |
| Alarm budget | new_onset | Tree | 5 | 26 | 0.876731143058 | 0.974744155478 | 26 |
| Alarm budget | new_onset | Tree | 10 | 26 | 0.876731143058 | 0.974744155478 | 26 |

Calibration curves, Brier scores, expected calibration error, independently calibrated probabilities and validated risk-to-dose calibration are **Not available in the current saved results**. ELM scores may lie outside [0,1]; weighted logistic/tree and equal-prior NB outputs are not established calibrated probabilities. Do not map these raw scores to oxygen quantities. The severe-interest subset has only three eligible onsets from three recordings; all are missed in the final policy, versus historical 3/3 under repeated-positive matching. This is too small for a reliable severity-specific generalization claim.

### 3.7 Latency: measurements, not physiological warning time

Sources: target-specific `latency_benchmark.csv`, `latency_samples.csv.gz`, `benchmark_environment.json`; `src/benchmark_latency.py:timed`, `summarize`, `benchmark_all`. Values below are milliseconds. Saved environment: AMD EPYC 7763 64-Core Processor, Python 3.14.2, numerical dependencies pinned in `requirements.txt`; native numerical threads limited to one. Standard components use 100 warmups and 2,000 repetitions. Loading uses 100 repetitions (five warmups), and waveform timing processes the configured 20,001 samples. P99 is deliberately undefined for <1,000 samples; it must not be reconstructed as if reliable.

Model-only times use already transformed features and exclude preprocessing/scaling. Pulse-to-alert includes feature calculation, transforms, model, eligibility/threshold path. Raw-sample timing mostly measures samples that do not produce a pulse; the separate pulse-producing row is necessary. Independent timers do not add exactly, and differing warm state/scheduling mean component comparisons are conditional measurements. Saved full-data demo models are used for software benchmarking, not outer test performance.

Source: [results/final/future_low/latency_benchmark.csv](../results/final/future_low/latency_benchmark.csv).

| Model | Component | Median_ms | P95_ms | P99_ms | P99_Reliable | Repetitions |
| --- | --- | --- | --- | --- | --- | --- |
| ELM | model_loading | 0.850118 | 3.20399435 | Undefined | False | 100 |
| ELM | feature_calculation | 0.23043 | 0.305329 | 2.31279482 | True | 2000 |
| ELM | model_only | 0.009408 | 0.011161 | 0.02805863 | True | 2000 |
| ELM | transforms_and_model | 0.546095 | 3.55518675 | 5.47211816 | True | 2000 |
| ELM | thresholding | 0.000321 | 0.000381 | 0.000441 | True | 2000 |
| ELM | pulse_to_alert | 0.9031665 | 3.82777865 | 6.0530557 | True | 2000 |
| ELM | waveform_sample_to_alert | 0.133359 | 0.210754 | 2.228823 | True | 19901 |
| ELM | waveform_pulse_producing_sample | 1.286618 | 5.05344575 | Undefined | False | 236 |
| GaussianNB | model_loading | 0.501126 | 0.56944875 | Undefined | False | 100 |
| GaussianNB | feature_calculation | 0.112911 | 0.1581145 | 0.19271982 | True | 2000 |
| GaussianNB | model_only | 0.209306 | 0.383798 | 0.45172153 | True | 2000 |
| GaussianNB | transforms_and_model | 0.5313725 | 1.00061655 | 1.45309287 | True | 2000 |
| GaussianNB | thresholding | 0.00017 | 0.000181 | 0.000201 | True | 2000 |
| GaussianNB | pulse_to_alert | 0.702512 | 0.99914485 | 1.23911213 | True | 2000 |
| GaussianNB | waveform_sample_to_alert | 0.06963 | 0.147776 | 0.973579 | True | 19901 |
| GaussianNB | waveform_pulse_producing_sample | 1.0263365 | 1.93426825 | Undefined | False | 236 |
| Logistic | model_loading | 0.499909 | 0.60054385 | Undefined | False | 100 |
| Logistic | feature_calculation | 0.1133665 | 0.251482 | 0.32018681 | True | 2000 |
| Logistic | model_only | 0.131931 | 0.26413155 | 0.44241846 | True | 2000 |
| Logistic | transforms_and_model | 0.4383385 | 0.77677105 | 0.96361565 | True | 2000 |
| Logistic | thresholding | 0.00017 | 0.000181 | 0.000201 | True | 2000 |
| Logistic | pulse_to_alert | 0.611908 | 1.0271536 | 1.14126568 | True | 2000 |
| Logistic | waveform_sample_to_alert | 0.06975 | 0.129843 | 0.809312 | True | 19901 |
| Logistic | waveform_pulse_producing_sample | 0.857176 | 1.17347925 | Undefined | False | 236 |
| Tree | model_loading | 0.6159305 | 0.70091845 | Undefined | False | 100 |
| Tree | feature_calculation | 0.1121445 | 0.202958 | 0.26224985 | True | 2000 |
| Tree | model_only | 0.1086525 | 0.19537105 | 0.23347963 | True | 2000 |
| Tree | transforms_and_model | 0.4176555 | 0.7411405 | 0.85070906 | True | 2000 |
| Tree | thresholding | 0.00017 | 0.000181 | 0.0002 | True | 2000 |
| Tree | pulse_to_alert | 0.5959335 | 1.04608725 | 1.41543318 | True | 2000 |
| Tree | waveform_sample_to_alert | 0.06953 | 0.146153 | 0.768696 | True | 19901 |
| Tree | waveform_pulse_producing_sample | 0.8044175 | 1.53651775 | Undefined | False | 236 |

Source: [results/final/new_onset/latency_benchmark.csv](../results/final/new_onset/latency_benchmark.csv).

| Model | Component | Median_ms | P95_ms | P99_ms | P99_Reliable | Repetitions |
| --- | --- | --- | --- | --- | --- | --- |
| ELM | model_loading | 0.812623 | 4.879151 | Undefined | False | 100 |
| ELM | feature_calculation | 0.226067 | 0.28877595 | 0.982612 | True | 2000 |
| ELM | model_only | 0.0106245 | 0.011802 | 0.03650541 | True | 2000 |
| ELM | transforms_and_model | 0.426782 | 0.5764095 | 1.00152378 | True | 2000 |
| ELM | thresholding | 0.00017 | 0.000191 | 0.00034 | True | 2000 |
| ELM | pulse_to_alert | 0.4886175 | 0.74296365 | 0.90381946 | True | 2000 |
| ELM | waveform_sample_to_alert | 0.070632 | 0.136916 | 0.677765 | True | 19901 |
| ELM | waveform_pulse_producing_sample | 0.729743 | 1.1582205 | Undefined | False | 236 |
| GaussianNB | model_loading | 0.4977545 | 0.5605842 | Undefined | False | 100 |
| GaussianNB | feature_calculation | 0.111478 | 0.198567 | 0.25126061 | True | 2000 |
| GaussianNB | model_only | 0.2097265 | 0.39279405 | 0.42870861 | True | 2000 |
| GaussianNB | transforms_and_model | 0.532189 | 0.81520955 | 1.01011016 | True | 2000 |
| GaussianNB | thresholding | 0.00017 | 0.000181 | 0.0002 | True | 2000 |
| GaussianNB | pulse_to_alert | 0.7068655 | 0.8097176 | 1.23601215 | True | 2000 |
| GaussianNB | waveform_sample_to_alert | 0.071493 | 0.166451 | 0.937611 | True | 19901 |
| GaussianNB | waveform_pulse_producing_sample | 1.029979 | 2.0019315 | Undefined | False | 236 |
| Logistic | model_loading | 0.507949 | 0.59285835 | Undefined | False | 100 |
| Logistic | feature_calculation | 0.1120345 | 0.1505195 | 0.18269346 | True | 2000 |
| Logistic | model_only | 0.127388 | 0.164912 | 0.1875439 | True | 2000 |
| Logistic | transforms_and_model | 0.435673 | 0.5171075 | 0.7442234 | True | 2000 |
| Logistic | thresholding | 0.00017 | 0.000181 | 0.0002 | True | 2000 |
| Logistic | pulse_to_alert | 0.61588 | 0.8637048 | 1.15714437 | True | 2000 |
| Logistic | waveform_sample_to_alert | 0.070832 | 0.1445 | 0.812107 | True | 19901 |
| Logistic | waveform_pulse_producing_sample | 0.8646795 | 1.632809 | Undefined | False | 236 |
| Tree | model_loading | 1.0588775 | 1.17991495 | Undefined | False | 100 |
| Tree | feature_calculation | 0.111388 | 0.1526396 | 0.18212251 | True | 2000 |
| Tree | model_only | 0.108282 | 0.1398845 | 0.16445551 | True | 2000 |
| Tree | transforms_and_model | 0.4181005 | 0.7031735 | 0.80759091 | True | 2000 |
| Tree | thresholding | 0.00017 | 0.000171 | 0.000201 | True | 2000 |
| Tree | pulse_to_alert | 0.598583 | 0.6837226 | 1.01815063 | True | 2000 |
| Tree | waveform_sample_to_alert | 0.070662 | 0.156302 | 0.773796 | True | 19901 |
| Tree | waveform_pulse_producing_sample | 0.839042 | 1.6090825 | Undefined | False | 236 |

ELM is fastest for **model-only** in both runs (medians 0.009408 and 0.0106245 ms). End-to-end pulse-to-alert median is fastest for Tree in the future-low run (0.5959335 ms) and ELM in the onset run (0.4886175 ms). Future-low Logistic has a smaller P99 than Tree despite its slightly larger median. Thus there is no universally fastest model across paths/percentiles/runs. The computation excludes sensor/monitor latency, filter response, waiting for the next pulse, and the 8 ms local-maximum confirmation. Sub-millisecond computation does not establish a clinically sufficient warning lead or a real-time deployment deadline.

### 3.8 Feature diagnostics, signal quality and historical experiments

`src/figures.py:FigureRun.permutation` refits the fixed models excluding the inspected record and checks reproduction of its saved scores before perturbation. It permutes each feature independently in ten-pulse blocks, seeds 42/43/44, three repeats, on mixed-class records only. The following values are **derived means of saved AP_Drop**; they are the two largest mean drops per model, not a new feature-selection experiment.

| Target | Model | Feature | Mean_AP_drop | Mixed_records | Repeats |
| --- | --- | --- | --- | --- | --- |
| future_low | ELM | Interval_Std_10 | 0.0247284622538 | 4 | 3 |
| future_low | ELM | Pulse_Interval_s | 0.0105740308276 | 4 | 3 |
| future_low | GaussianNB | Interval_Std_10 | 0.0120335357672 | 4 | 3 |
| future_low | GaussianNB | Rate_Change_5 | 0.00588538291523 | 4 | 3 |
| future_low | Logistic | Interval_Std_10 | 0.0811433376259 | 4 | 3 |
| future_low | Logistic | Amplitude_Slope | 0.000162386067418 | 4 | 3 |
| future_low | Tree | Interval_Std_10 | 0.0110010249905 | 4 | 3 |
| future_low | Tree | Pulse_Interval_s | 9.9896448785e-05 | 4 | 3 |
| new_onset | ELM | Interval_Std_10 | 0.0545234363388 | 4 | 3 |
| new_onset | ELM | Rate_Std_10 | 0.0522403336524 | 4 | 3 |
| new_onset | GaussianNB | Amplitude_Mean_5 | 0.004613767935 | 4 | 3 |
| new_onset | GaussianNB | PPG_Amplitude | 0.00214095582815 | 4 | 3 |
| new_onset | Logistic | Interval_Std_10 | 0.136356131293 | 4 | 3 |
| new_onset | Logistic | Pulse_Rate_BPM | 0.0355412453176 | 4 | 3 |
| new_onset | Tree | Amplitude_Mean_5 | 0.0139348378087 | 4 | 3 |
| new_onset | Tree | Rate_Mean_5 | 0.0085116617421 | 4 | 3 |

Correlated features, rate/interval dependence, overlapping history and implausible independently permuted combinations prevent a causal physiological interpretation. Negative drops in the full CSV also exist and should not be suppressed. These diagnostics did not select features or prove a feature ablation improves final early warning. Retrained feature-removal ablations with independently selected policies are **Not available in the current saved results**.

The detector has 21,980 candidates, 30 initial unpaired candidates and two rejected >2 s intervals, leaving 21,948 pulses. All saved target feature entries are finite; `dataset_summary.json` shows zero unexpected amplitude/interval/rate ranges, duplicates or missing features in these prepared rows. These are algorithmic checks, not independent beat annotation, SNR validation or artifact rejection. Pulse rate is derived, not an independent reference. True peak sensitivity/precision, motion labels, sensor delay validation and clinically adjudicated signal-quality performance are **Not available in the current saved results**.

**Historical result hierarchy:** `results/validated/` is the earlier repaired causal future-low run; `results/hardened/` adds target-specific hardening; `results/final/` is the fresh preserved comparison with full figures; `results/reliability/final/` corrects event/exposure interpretation; `results/local_30_fixed_policy_v2/` audits the fixed recall cutoffs; `results/alarm_budget_experiment/run02/` is the latest changed-policy result. They are not independent cohorts. `results/final/before_after_metrics.csv` contains 72 matched rows with zero `Difference` between hardening and fresh final metrics; this is reproducibility, not improvement. `run01` stores completed frozen threshold selections reused by run02, not a second external evaluation.

The older `results/elm_dataset.csv` contains 17,089 rows, 466 positives in 26 selected records. Its independent label audit verifies 15,706 complete valid labels with zero disagreements but flags 1,096 incomplete horizons and 287 invalid horizons (1,383 unverifiable targets). Noncausal `sosfiltfilt`, full-waveform prominence, selected current-SpO₂ rows and nonrepresentative splits invalidate interpreting the original experiment as the current causal task. The preserved recording-aware-v2 ELM confusion totals are:

| Historical_folds | Samples | Positives | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- |
| 26 | 17089 | 466 | 182 | 7359 | 9264 | 284 |

Source: [results/recording_aware_elm_v2_results.csv](../results/recording_aware_elm_v2_results.csv) — 26 historical outer records; per-record AP/thresholds are not pooled AP.

Additional preserved numeric comparisons below are read directly from `archive/legacy_results.zip` at the stated internal paths. These tables explain past claims; **none is a valid substitute for the matched current evaluation**. Different splits, targets, selected populations and incomplete/noncausal preprocessing prevent a common winner ranking. Old per-sample/batch latency excludes the current full streaming path. No new archive scripts were executed.

**Archive member `results/elm_metrics.csv`:**

| Training_and_test_inference_time_s | Accuracy | Precision | Recall | Average_Precision | TP | TN | FP | FN | Test_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.156666212 | 0.972498537156 | 0 | 0 | 0.0911738629202 | 0 | 4986 | 1 | 140 | 5127 |

**Archive member `results/logistic_metrics.csv`:**

| Model | Training_Time_ms | Inference_Total_ms | Inference_Median_us_per_sample | Chosen_Threshold | Accuracy | Precision | Recall | Average_Precision | TP | TN | FP | FN | Test_Samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 138.010866 | 0.275624999858 | Undefined | 0.532643257139 | 0.730836746635 | 0.046103183315 | 0.451612903226 | 0.0390820056965 | 42 | 2456 | 869 | 51 | 3418 |

**Archive member `results/model_comparison.csv`:**

| Model | Train_Time_s | Inference_ms_per_sample | Accuracy | Precision | Recall | PR_AUC_AP | TP | TN | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.111785926 | 0.000319127198934 | 0.808525033829 | 0.047138047138 | 1 | 0.0581859433483 | 14 | 1181 | 283 | 0 |
| ELM | 0.0144104380001 | 0.00110137347795 | 0.833558863329 | 0.0538461538462 | 1 | 0.0673553277854 | 14 | 1218 | 246 | 0 |

**Archive member `results/advanced_model_comparison.csv`:**

| Model | Training_Time_ms | Threshold | Accuracy | Precision | Recall | F1 | Average_Precision | TP | FP | TN | FN | Inference_Median_ms_per_sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Weighted ELM | 49.8441750001 | 0.488583093237 | 0.886666666667 | 0 | 0 | 0 | 0.141753189894 | 0 | 16 | 1197 | 137 | 0.0021321303703 |
| Logistic Regression | 40.5746320002 | 0.460500513168 | 0.881481481481 | 0 | 0 | 0 | 0.113938453614 | 0 | 23 | 1190 | 137 | 0.000117471111109 |
| Random Forest | 2176.40028 | 0.108887425832 | 0.89037037037 | 0 | 0 | 0 | 0.0982326739949 | 0 | 11 | 1202 | 137 | 0.0402761874075 |

**Archive member `results/recall_first_elm_test_metrics.csv`:**

| Model | Target_Recall | Hidden_Nodes | Ridge | Positive_Weight_Multiplier | Validation_Threshold | Threshold | Accuracy | Precision | Recall | F1 | Average_Precision | TP | TN | FP | FN | Positive_Samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Recall-first ELM | 0.92 | 50 | 1 | 2 | 0.409857558318 | 0.409857558318 | 0.86962962963 | 0.102040816327 | 0.036496350365 | 0.0537634408602 | 0.113317299306 | 5 | 1169 | 44 | 132 | 137 |

**Archive member `results/latency_benchmark.csv`:**

| Model | Training_Time_ms | Median_Inference_ms | P95_Inference_ms | Mean_Inference_ms | Throughput_samples_per_s | Repetitions |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 11.5582579999 | 0.166766 | 0.232807 | 0.167484805 | 5970.69089342 | 1000 |
| ELM | 33.9129189999 | 0.007914 | 0.008095 | 0.008049449 | 124232.105825 | 1000 |

The archived advanced model study excludes recordings 17/29, validates on 12 and tests on 07/30; all three models miss all 137 positives in that saved test. Its random forest is historical evidence of an attempted model, not evidence that a forest was part of the final four-model comparison. The original comparison/ELM/conventional scripts use random row partitions in parts of this archive; correlated rows within a recording can enter both sides. The v2 ELM chooses a small positive/negative recording validation subset rather than the maintained three-fold grouping. Numeric archives retain these limitations rather than being silently corrected.

## 4. Which model performs better?

**Interpretation of saved results; no overall winner is forced.**

| Setting / criterion | Best supported statement | Why that is not a general victory |
|---|---|---|
| Future-low, pooled AP/F1/precision/ROC | ELM: AP approximately 0.255929, F1 approximately 0.383104 (see exact extracted table above) | Weak ranking over prevalence, 14,093 FP and about 92% pulse alerts; includes chronic lows |
| Future-low, pooled recall | GaussianNB has highest recall among learned models | Near-continuous alerts and very low specificity; recall alone is not useful anticipation |
| Onset, pooled AP/ROC/F1 | GaussianNB: AP about 0.057595, ROC about 0.654904, F1 about 0.064108 | Precision remains about 3.32%; final budget policy detects zero events |
| Macro AP on both targets | Logistic has highest defined-record macro AP | Macro averaging emphasizes different records than pooled ranking; few mixed-class records |
| Original recall policy, corrected event sensitivity | ELM has the most learned-model detections in each 5/10/15/30 s window for both targets | At 30 s this is 12/14 future-low or 13/14 onset, with approximately 182/185 off-event starts/hour and approximately 92%/89% duty |
| Final budget-constrained early warning | No learned model demonstrates useful sensitivity at any tested budget/window | All have zero detections; silent ELM/other rows tie never-alert on detection while some models add false alarms |
| Model-only latency | ELM fastest in both saved target benchmarks | Excludes transforms/features; full-path winner depends on target and percentile |
| Interpretability | Logistic coefficients and shallow tree rules are easiest to inspect | No saved coefficient/path explanation proves physiological causality or decision reliability |
| Overall research outcome | An auditable causal offline comparison with a negative final policy result | Not validated feedforward oxygen control or demonstrated clinical early warning |

The exact ELM values should be read from section 3.2 rather than mixing older reports. At recall-driven cutoffs, ELM's relative event advantage is inseparable from very high burden; it is not a matched-burden superiority test. Periodic/always-alert controls also detect events with no learned discrimination. The budget experiment shows that a model useful at a permissive threshold need not transfer to a useful constrained policy. It does not prove all future models or all possible threshold grids must fail; it establishes failure of these fixed models/features/policy on this cohort.

## 5. Exact graphs and result tables to display

There are **no final alarm-budget PNG/PDF figures** in `results/alarm_budget_experiment/run02/`. Its exact tables should lead the presentation. Existing `results/final/<target>/figures/` charts belong to the earlier recall policy despite the directory name “final.” `src/figures.py` reads that run's saved predictions; it does not display the new budget cutoffs. Their provenance is auditable, but their operating policy is historical. Do not relabel their 14/14 bars, leads or false-alert counts as run02 evidence.

### Top five display items if time is limited

1. `results/alarm_budget_experiment/run02/summary.csv`: filter to learned models, `Subset=all`, W=30; show both targets/all budgets, and the 0/15 result at shorter W. State 0/3 severe detections and undefined leads.
2. `results/local_30_fixed_policy_v2/event_summary.csv`: show 30 s corrected detections, off-event starts/hour and active duty with fixed baselines. Explain why higher sensitivity at extreme burden is not useful control input.
3. `results/final/new_onset/figures/ranking_curves.png`: historical onset ranking/precision limits, with prevalence and model differences; no budget-policy claims.
4. `results/final/new_onset/figures/ppg_pulse_detection.png`: actual recorded PPG, causal filter, peaks, interval/rate; distinguish preprocessing from clinical peak validation.
5. `signal_studies/outputs/control_comparison.png`: explicitly illustrative P/PI/PID conditioning model; show integral DC rejection and modest PID transient differences, then state this is not oxygen or ML feedforward integration.

### Prioritized existing evidence and missing figures

All existing links below resolve from this document. “Current” means current for the stated experiment, not universally authoritative for every policy.

| Priority | Exact existing path | What it shows / supported claim | Presentation points | Trust/currentness | Action | Caveat |
|---|---|---|---|---|---|---|

| Essential | [results/alarm_budget_experiment/run02/summary.csv](../results/alarm_budget_experiment/run02/summary.csv) | Final sensitivity, counts, burden, duty, leads by target/budget/W; Final learned policy detects none | Name 14/15 denominators; severe misses; silent is not successful warning | Current final policy; table, not a graph | Use existing table; optional new plot listed below | No matched learned leads; do not plot them as zero |
| Essential | [results/alarm_budget_experiment/run02/outer_constraint_status.csv](../results/alarm_budget_experiment/run02/outer_constraint_status.csv) | Per-record held-out budget/cap flags; Training feasibility does not guarantee test feasibility | Show onset tree; separate pooled vs per-record; retain zero-exposure unknowns | Current final policy | Use existing table | Constraints were research choices, not clinical limits |
| Essential | [results/local_30_fixed_policy_v2/event_summary.csv](../results/local_30_fixed_policy_v2/event_summary.csv) | Corrected fixed-policy sensitivity and burden across all windows; High recall can come with extreme alarm activity | Actual starts; include simple controls; distinguish W from lead | Current corrected accounting of old cutoff policy | Use existing table | Not alarm-budget outcomes |
| Essential | [results/final/new_onset/figures/ranking_curves.png](../results/final/new_onset/figures/ranking_curves.png) | PR and ROC ranking of held-out 30 s onset scores; Small AP and disagreement across models | Compare AP to prevalence; NB pooled ranking; pooled-refit caveat | Trustworthy for saved recall-score experiment; hashes checked | No regeneration needed for that experiment | Not independent 5/10/15 s training or a budget operating point |
| Essential | [results/final/new_onset/figures/ppg_pulse_detection.png](../results/final/new_onset/figures/ppg_pulse_detection.png) | Recorded record-12 raw/causal filtered PPG, accepted peaks, intervals/rates; Implemented causal signal processing | Confirmation is one sample later; rate=60/interval; no ECG R peaks | Trustworthy recorded-data diagnostic | No regeneration needed | No annotated beat or ambient-light ground truth |
| Useful | [results/final/future_low/figures/ranking_curves.png](../results/final/future_low/figures/ranking_curves.png) | Future-low PR/ROC counterpart; Broad target differs from incident onset | Includes ongoing lows; compare prevalence; ELM modest ranking | Historical cutoff experiment, current saved figure receipt | No regeneration needed | Cannot infer impending-event detection from broad positive labels |
| Useful | [results/final/new_onset/figures/confusion_matrices.png](../results/final/new_onset/figures/confusion_matrices.png) | Recall-policy held-out timestamp matrices; Many FP and near-always alerts | Pulse denominator; tree always-alert; compare with final table | Valid for old policy only | Final-policy confusion plot would be a new figure | Not independent event counts |
| Useful | [results/final/future_low/figures/model_metrics.png](../results/final/future_low/figures/model_metrics.png) | Pooled/macro model metrics for broad target; ELM metric-specific relative result | Name aggregation; broad target; weak absolute discrimination | Valid saved comparison, not budget policy | No regeneration for historical use | Do not call best clinical predictor |
| Useful | [results/final/new_onset/figures/model_metrics.png](../results/final/new_onset/figures/model_metrics.png) | Onset pooled/macro comparison; NB vs Logistic metric disagreement | Pooled AP vs macro AP; six positive records; precision low | Valid historical comparison | No regeneration for historical use | 100% tree recall is not success |
| Useful | [results/final/new_onset/figures/latency_distributions.png](../results/final/new_onset/figures/latency_distributions.png) | Warm timing distribution and components; Causal local software is computationally lightweight | Model-only vs total; P95 tails; exclude physiological delay | Current saved benchmark for demo artifacts | No new benchmarking needed for review | Cannot turn computation time into achieved warning lead |
| Optional | [results/final/new_onset/figures/block_permutation_importance.png](../results/final/new_onset/figures/block_permutation_importance.png) | Mean block-permutation AP drops; Which fitted features influence held-out ranking | Ten-pulse blocks; three repeats; correlations limit attribution | Valid exploratory diagnostic, not feature selection | No regeneration or refit in this task | Not causal feature importance or an ablation result |
| Optional | [results/final/new_onset/figures/missingness_and_clock.png](../results/final/new_onset/figures/missingness_and_clock.png) | Missing numerics and exported waveform clock; Why targets can be unknown and row clock is reconstructed | Missing is not negative; retain waveform rows; monitor delay unknown | Valid dataset diagnostic | No regeneration needed | Clock consistency is not sensor timing validation |
| Optional | [results/final/new_onset/figures/events_and_lead_time.png](../results/final/new_onset/figures/events_and_lead_time.png) | Historical any-positive detection and leads plus timing; Explain the old result if challenged | Reusable positives; clipped-window leads; immediately contrast corrected table | Trustworthy historical rendering, NOT final event evidence | Replace with proposed corrected/final plot for headline use | Do not show 14/14 without matching-rule warning |
| Optional | [results/final/new_onset/figures/false_alarm_burden.png](../results/final/new_onset/figures/false_alarm_burden.png) | Historical target-run FP episodes/timestamps; Explain why old burden differs | Episode definition; exposure edges; active time needed | Valid for old definitions, not run02 | Use corrected tables instead for main burden claim | Not strict off-event-start burden |
| Useful | [signal_studies/outputs/control_comparison.png](../signal_studies/outputs/control_comparison.png) | Simulated P/PI/PID response and isolated input-offset effect; Comparison exists for a defined illustrative conditioning plant | Same inputs; integral removes DC error; chosen gains not optimized | Current illustrative simulation; source receipts checked | No new simulation needed | No oxygen plant, constraints or noise |
| Optional | [signal_studies/outputs/bode_comparison.png](../signal_studies/outputs/bode_comparison.png) | Open-loop conditioning-system magnitude/phase; Defined model margins and pole stability | Radians/s; finite gain crossover; infinity is model-specific | Current illustrative simulation | No regeneration needed | No finite −180° crossing is not hardware robustness |
| Optional | [signal_studies/outputs/waveform_step.png](../signal_studies/outputs/waveform_step.png) | Synthetic clean/disturbed/filtered PPG and matched residual; Transient response to additive offset step | Step at 20 s; residual vs filter distortion; finite recovery | Current synthetic evidence | No regeneration needed | Simplified additive optical coupling; not sensor validation |


The rows above form the prioritized figure/table catalogue. Existing `precision_recall.png` files are redundant with the richer `figures/ranking_curves.png`. ROC is useful as a secondary panel, not the lead claim with 3.12% onset prevalence. Do not spend review time showing every temporal/example chart; if challenged, the stored example selections identify exact records/timestamps in `results/final/new_onset/figures/example_selection.csv`.

**Proposed missing plots — not generated in this task:**

| Proposed filename (does not currently exist) | Exact saved input | Purpose / required caution |
|---|---|---|
| `results/alarm_budget_experiment/run02/figures/budget_sensitivity_and_burden.png` | `summary.csv`, `bootstrap_intervals.csv`, `outer_constraint_status.csv` in run02 | Separate panels for event sensitivity, false starts/hour, duty and violations by budget; show 0/14, not empty success bars |
| `results/alarm_budget_experiment/run02/figures/missed_onsets_by_window.png` | run02 `event_matches.csv`, `raw_events.csv` | All learned misses, coverage exclusions and three interest events; actual event identities |
| `results/local_30_fixed_policy_v2/figures/fixed_policy_event_burden.png` | local-30 `event_summary.csv`, `recording_bootstrap_intervals.csv` | Corrected sensitivity vs burden with nonlearned controls; do not join unrelated policies as an optimized ROC curve |
| `results/local_30_fixed_policy_v2/figures/matched_start_leads.png` | local-30 `detected_event_leads.csv`, `event_matches.csv` | Actual-start detected-only lead distribution; show number of misses separately; final learned-model panel must say undefined, not zeros |

## 6. PID versus P versus PI: what is demonstrated and what remains theoretical

### 6.1 Audit of the comparison now present

The previous signal studies are **existing evidence**, inspected here without rerunning them. This supersedes an earlier audit finding that controller comparisons were absent. However, it does not supersede the absence of a validated oxygen-delivery plant or a fair oxygen-controller comparison. The distinction matters when answering “why PID?”.

`signal_studies/core.py` implements separate `P`, `PI`, `PID` components and `plant`, `loop`, `closed_loop`, `control_run`, `margins`. The explicitly illustrative normalized conditioning amplifier is

`0.2 x1_dot + x1 = u+d`, `0.05 y_dot + y = x1`, hence `G(s)=1/[(0.2s+1)(0.05s+1)]`.

Here u is an abstract signed conditioning compensation drive and d an assumed additive illumination-related input offset. It regulates a scalar level; it does not model oxygen delivery, physiological oxygenation, or a pulse-preserving cancellation system. It is separate from the recorded-data bandpass and from ML alerts. All states start at zero; r=1 at t=0 and d=0.3 at t=10 s; duration 20 s; continuous-time propagation on a 0.002 s output grid, no digital delay or saturation.

`C_P=2`; `C_PI=2+8/s`; `C_PID=2+8/s+0.08s/(1+0.01s)`. Shared Kp and Ki where applicable make inputs/gains explicit, but this is not equal-bandwidth/equal-control-effort optimized tuning. Derivative acts on error, including the reference step, with a 0.01 s filter. The loop maps reference through `GC/(1+GC)` and input offset through `G/(1+GC)`.

The following tables are the saved numerical evidence, not new simulations. Reference settling means last exit from ±2% of the requested reference followed by at least one second in band before the 10 s disturbance. Disturbance recovery is the isolated offset-induced output entering ±0.006 and staying through the remainder of the record with at least one second remaining. Missing settling is not an invented infinity. Overshoot is relative to the requested unit reference unless marked equilibrium-relative.

Source: [signal_studies/outputs/control_metrics.csv](../signal_studies/outputs/control_metrics.csv).

| controller | kp | ki | kd | tau_d | settling_2pct_s | overshoot_pct | tail_error | settling_to_equilibrium_2pct_s | overshoot_equilibrium_pct | disturbance_settling_2pct_s | disturbance_peak | disturbance_tail | disturbance_iae | disturbance_tail_rejection_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P | 2 | Undefined | Undefined | Undefined | Undefined | 0 | 0.333333333333 | 0.342 | 3.78041321672 | Undefined | 0.103780413217 | 0.1 | 0.99166666665 | 66.6666666667 |
| PI | 2 | 8 | Undefined | Undefined | 0.572 | 1.55431223448e-13 | 2.99760216649e-15 | 0.572 | 1.55431223448e-13 | 0.964 | 0.0897014537902 | 8.57287308172e-16 | 0.0374999999833 | 100 |
| PID | 2 | 8 | 0.08 | 0.01 | 0.502 | 0.00441309880763 | -7.32747196253e-15 | 0.502 | 0.00441309880763 | 0.93 | 0.0798924094217 | -1.77298113551e-15 | 0.0374999999833 | 100 |

Source: [signal_studies/outputs/stability_margins.csv](../signal_studies/outputs/stability_margins.csv).

| controller | gain_margin | gain_margin_db | phase_margin_deg | gain_crossover_rad_s | phase_crossover_rad_s | closed_loop_stable | largest_pole_real_part |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P | Infinite | Infinite | 101.055979939 | 7.85152230419 | Undefined | True | -12.5 |
| PI | Infinite | Infinite | 71.5578649338 | 8.74653288868 | Undefined | True | -3.58330791249 |
| PID | Infinite | Infinite | 90.1116163196 | 8.21182137674 | Undefined | True | -5.32503156615 |

P remains below the reference with 1/3 DC error; its own equilibrium is 2/3. PI and PID reject constant input offset under this model. PID has a slightly shorter settling/recovery time and smaller disturbance peak at these chosen gains, but PI has smaller reference overshoot. All gain margins are mathematically infinite because no finite negative-real-axis crossover exists; undefined phase-crossover frequencies are not zero. Closed-loop characteristic roots are all in the left half plane, independently of the margin display. Infinite margins do not establish delay/noise/nonlinear robustness.

`signal_studies/REPORT.md`, `outputs/closed_loop_poles.csv`, `outputs/crossovers.csv` and settings receipts explain definitions and coefficients. The report includes six passing new tests and a prior full-suite result of 149 tests with one failure/two errors due to pre-existing documentation hashes. This task did not run those simulation-generating tests.

For the synthetic signal branch, `synthetic_ppg` is a deterministic 1.2 Hz three-harmonic AC surrogate at 125 Hz, 40 s, with offset 0.4 and an additional +0.6 at 20 s. Existing causal Butterworth coefficients are reused. `waveform_metrics.csv` separates reconstruction RMSE against clean input (which includes filter phase/amplitude distortion) from disturbance-only residual against a separately filtered clean input. It is more than a constant-DC test, but only a simplified additive illumination model, not a full optical sensor simulation. The key stored measurements are:

Source: [signal_studies/outputs/waveform_metrics.csv](../signal_studies/outputs/waveform_metrics.csv).

| scenario | window | start_s | end_s_exclusive | raw_rmse | reconstruction_rmse_unaligned | disturbance_only_rmse | residual_baseline_mean | residual_peak_abs | recovery_2pct_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constant | steady_pre | 10 | 20 | 0.4 | 0.0959139774747 | 6.8032247967e-08 | 1.00899698494e-08 | 3.82839713431e-07 | 2.816 |
| constant | early_post | 20 | 22 | 0.4 | 0.0893072436019 | 1.10706360304e-13 | 2.87136980859e-14 | 2.97678548478e-13 | 2.816 |
| constant | steady_post | 30 | 40 | 0.4 | 0.0959139788164 | 6.77955030442e-15 | -1.86918258649e-15 | 1.58206781009e-14 | 2.816 |
| step | steady_pre | 10 | 20 | 0.4 | 0.0959139774747 | 6.8032247967e-08 | 1.00899698494e-08 | 3.82839713431e-07 | 2.816 |
| step | early_post | 20 | 22 | 1 | 0.163537257891 | 0.130842354559 | 0.00330278137672 | 0.41674959676 | 2.816 |
| step | steady_post | 30 | 40 | 1 | 0.0959139749438 | 1.0204837125e-07 | 1.51349513978e-08 | 5.74259570743e-07 | 2.816 |

Thus offset transients recover in 2.816 s under the stated 2% criterion, but unaligned waveform RMSE remains approximately 0.095914 even after rejection. No “perfect clean pulse recovery” or real sensor claim follows.

### 6.2 Defensible theoretical justification

**Theory:** P is simple and responds directly to error. For this type-zero plant it retains a constant reference/disturbance error; integral action is needed for zero DC error. Increasing Kp can reduce error but may reduce robustness or increase oscillations for other dynamics/delays. It is not universally true that every P-controlled plant has steady error.

PI adds an integrator and can remove constant error when the loop remains stable and the actuation can supply the required drive. Excessive integral action can increase overshoot, settling, and windup under limits. It has no derivative term; many applications nevertheless need no derivative to meet their performance objectives.

PID's filtered derivative can improve damping/transients for appropriate plant/gains. It does not automatically outperform PI. Unfiltered differentiation amplifies high-frequency noise; filtered derivative limits but does not eliminate that problem. PPG pulsatility, peak errors, quantization and SpO₂ monitor filtering/delay can make derivative action misleading. A future simulation should compare derivative-on-measurement (which avoids reference-step derivative kick) against the current error-derivative implementation, filtering choices, actuator saturation, rate limits and integral anti-windup. These are mathematical/controller-model tasks, not current hardware tasks.

**Review answer:** “I use PID as a proposed architecture to study proportional correction, zero-offset integral action, and potential derivative damping. The current illustrative conditioning simulation shows metric-specific differences at fixed gains, but it does not prove PID is the best oxygen controller or better than well-tuned PI.”

### 6.3 Proposed minimum comparison for the intended control concept

The following is **future simulation work, not completed evidence**:

1. Define an explicitly scoped, defensible oxygen-response plant and uncertainty/operating conditions before selecting gains; identify what remains illustrative.
2. Use identical references, perturbations, initial conditions, measurement delay/noise, sampling, drive/rate bounds and saturation for P, PI, PID-only and combined feedforward variants.
3. Prespecify a tuning method and fair objective (for example matched tracking/bandwidth/effort constraints); do not claim fairness merely from equal gains or tune on the final comparison scenarios.
4. Report settling, target/equilibrium overshoot, steady error, disturbance peak/recovery/integral error **and control effort** (e.g. peak and integral squared drive). Control effort, actuator-constrained outcomes and noise sensitivity are **Not available in the current saved results**.
5. Compare derivative filters/on-measurement variants and anti-windup with noise/delay/parameter sweeps. Use stability margins where a linear model is appropriate and inspect poles; assess nonlinear constrained responses separately.
6. Repeat across appropriate operating conditions and uncertainty, with independent evaluation scenarios. Only then justify PID over PI if the relevant metrics and robustness support it.

## 7. Proposed feedforward-PID integration

| Integration requirement | Current finding | Status |
|---|---|---|
| Trained causal ML score / eligibility-aware alert | `src/model.py`, `src/controller.py`, saved demonstrations and held-out scores | Implemented, retrospective evaluation available |
| ML score driving a feedforward control input | No such call path to a physical/physiological plant | Proposed |
| Defined probability/risk-to-action mapping | Scores not calibrated; no control gain/map/dose law | Proposed; not available in saved results |
| Oxygen-delivery / oxygenation response plant | No such model in ML or signal studies | Proposed |
| Actuator saturation / rate constraints | Absent from illustrative study | Proposed |
| PID feedback loop | Exists only for the abstract conditioning amplifier | Implemented illustration, not oxygen feedback |
| Combined feedforward+feedback evaluation | No joint score-to-drive-to-output experiment | Proposed |
| P/PI/PID-only vs feedforward-PID baseline | Only standalone illustrative conditioning controllers compared | Proposed for intended system |

**Theory:** a future simulation could express `u_raw = u_ff(score, context) + u_fb(r−y)` followed by explicit constraints, where feedforward estimates a disturbance-related compensation and feedback corrects observed error. The feedforward term requires a validated causal mapping and a model of how actuation changes the output; it is not the same as a thresholded research alert. PID can correct residual error but cannot make an unreliable feedforward signal informative. Saturation/anti-windup require coordination with both contributions.

`future_low=1` means a low observation occurs somewhere in a future interval and may already be ongoing. It gives neither exact onset, causal mechanism, required compensation, treatment response nor safe drive magnitude. Final policies do not even establish useful early-warning detections at the tested burdens. Any future simulated integration needs appropriately defined outputs, constraints, delayed/noisy measurement models, calibration/utility assumptions and validation against independent scenarios. No claim about opening an oxygen valve or changing a patient treatment follows from this repository.

## 8. Limitations, risks and likely review questions

### Material limitations

- **Population:** 30 recordings, unverified unique patient count, only six recordings with onset positives and three with eligible severe-interest events. Recording separation does not prove patient separation; correlated recordings/pulses can reduce effective sample size.
- **Ground truth:** monitor observations at nominal 1 Hz; one-reading ≤94% crossing; no sustained duration, clinical adjudication or validated latency. 91/92-interest is a narrow exact-value definition. Missing/low/unknown-prior states constrain eligibility.
- **Imbalance and dependence:** 501 onset-positive pulse rows are not 501 independent episodes. Macro and pooled results answer different questions; constant/periodic baselines expose sensitivity inflation.
- **Policy dependence:** original high recall, corrected alarm starts, and final budget thresholds are distinct. Infinite/missing lead entries cannot be filled with zeros. An event window does not guarantee useful warning duration.
- **Burden:** recall policies spend much of the available time active. Few starts alone are misleading; final thresholds suppress true detections as well as false alerts and can still violate outer constraints.
- **Signal quality:** no annotated peak validation, motion assessment, patient/device calibration or real illumination experiments; synthetic offset rejection is not sensor validation.
- **Validation/generalization:** cohort already inspected; fixed hyperparameters reduce search but do not make reuse confirmatory. Inner-to-outer score distributions can shift. No external cohort, independently established patient split or calibrated clinical utility.
- **Timing:** offline causal replay and host CPU timings are software measurements, not monitor delay, acquisition timing or achieved intervention lead.
- **Control:** illustrative conditioning dynamics/gains do not validate oxygen dynamics; effort, saturation, noise/delay robustness and feedforward coupling remain absent. No clinically validated control comparison.
- **Integrity:** existing historical documentation hash mismatches prevent a blanket “all quality gates pass” claim. They remain reported and unmodified.

### Short, honest review answers

| Question | Answer grounded in this repository |
|---|---|
| Why these ML models? | Four fixed lightweight baselines span random nonlinear features, a linear classifier, Gaussian class models and simple nonlinear rules; model choices are auditable and fit the limited feature/data scale, not proven optimal. |
| Which is best? | ELM leads selected broad-target pooled metrics; NB leads onset pooled ranking/F1; Logistic leads macro AP; ELM leads recall-policy event counts at extreme burden. None succeeds under final alarm budgets. |
| Why not accuracy? | A silent predictor correctly labels most onset timestamps yet detects no events; prevalence and event burden matter more than raw correctness count. |
| What do false positives mean? | Positive pulse decisions with no future positive label; they differ from actual off-event alarm starts, which count new runs and use a strict exposure denominator. |
| Did it warn early enough? | Final policies produce no matched learned event warnings; their lead distributions are undefined. Older near-window-start positives are not independent starts. |
| How is leakage addressed? | Forward-only filtering, next-sample confirmation timestamps, past-only rolling features, complete targets, recording-level outer/inner splits, training-only transforms/cutoffs, saved OOF partition identity and authenticated score reuse. Patient independence and retrospective cohort reuse remain limitations. |
| Why PID rather than PI? | It is a proposed way to investigate damping plus integral correction, not an established necessity. The existing simulation has metric-specific PID/PI tradeoffs and no intended oxygen plant. |
| Is PID experimentally superior? | No. Current evidence is a chosen-gain conditioning simulation; no validated physiological plant or constrained/noisy matched oxygen-control comparison exists. |
| How can ML safely interact with PID? | Only as a future explicitly modeled/limited feedforward signal after validating predictive utility and a risk-to-drive law; the present output is a research score/alert with no control mapping. |
| Does a 30 s horizon mean 30 s warning? | No. It means a future interval or look-back eligibility window. Actual detected-only lead must be measured from a real prior alarm start. |
| What about LightGBM/PTT? | Neither is implemented/evaluated in the maintained pipeline. ECG exists but no PTT estimator or feature exists; its mere presence is insufficient. |
| What is novel/useful if final detection is zero? | The useful deliverable is a reproducible causal/recording-separated evaluation, explicit exposure/matching audit and candid negative policy result, plus bounded illustrative signal/control studies; not a proven new clinical control method. |
| Next research steps? | Improve and independently validate prediction at a prespecified burden; verify quality/labels/subject dependence; compare the proposed controller concept only in a documented constrained/uncertain simulation. |

## 9. Final project outcome

**Implemented:** causal PPG filtering/detection and history features; four fixed-model training/inference wrappers; grouped nested thresholding and held-out predictions; event/exposure audits and controls; alarm-budget threshold-only evaluation; offline dashboard/replay; host latency measurements; separate synthetic PPG and illustrative P/PI/PID response/Bode studies.

**Demonstrated:** reproduction/accounting of saved retrospective scores and outcomes; large false-positive/activity burden under recall-first cutoffs; zero learned-model detections at every tested final alarm budget/window; metric-specific simulated controller tradeoffs for the stated abstract plant. None demonstrates a useful final early-warning policy, a validated oxygen-delivery plant, ML-to-drive integration, clinical benefit or PID superiority for the proposed oxygen architecture.

**Best-supported model statement:** there is no defensible overall winner. ELM has a modest relative broad-target ranking/F1 advantage and very fast model-only computation; NB ranks onset timestamps better on pooled AP/ROC, while Logistic has stronger macro AP. Those rankings do not survive as useful event warnings under the final policy. Never-alert remains as effective at detection as every final learned policy, with no nuisance alerts; that is a negative prediction finding, not a recommended clinical alarm strategy.

**Three highest-priority next research tasks (software/research only):**

1. Establish predictive utility under a prespecified burden and independent evaluation plan: analyze the severe scarcity/record dependence, score transfer and threshold selection without reusing the same outer cohort as a fresh confirmatory claim.
2. Strengthen signal/label validity and uncertainty: annotated or otherwise justified pulse-quality checks, timing/unknown-target analysis, patient identity where legitimately available, and calibration only with adequate independent data. Evaluate any additional feature/model as a new controlled study, not a retrospective winner search.
3. Develop the proposed feedforward/feedback concept as a clearly scoped simulation with a defensible response model, constraints, delay/noise and fair P/PI/PID/combined baselines; quantify control effort and robustness before arguing for PID.

Legitimate review claim: “This project implements and audits a causal lightweight retrospective prediction pipeline and separate illustrative signal/control studies. The final constrained alert policy did not achieve useful early-warning detection, and the oxygen feedforward-PID integration remains proposed.”

## 10. Reproducibility, source index and checks

### 10.1 Commands that actually exist

Run from the repository root in the pinned `requirements.txt` environment. The following are **reference reproduction commands, not executed to create new experiments during this document task**. Use new destinations; existing research outputs/hashes should remain untouched.

```bash
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/oxygen-review-matplotlib

# New isolated copies of the two existing fixed-model experiments:
python -m src.run_experiments --target future_low --output /tmp/oxygen-review-reproduction/future_low
python -m src.run_experiments --target new_onset --output /tmp/oxygen-review-reproduction/new_onset

# Reports/figures for those copied runs (permutation generation refits diagnostics):
python -m src.target_reporting --output /tmp/oxygen-review-reproduction/new_onset
python -m src.figures --output /tmp/oxygen-review-reproduction/new_onset
python -m src.figures --output /tmp/oxygen-review-reproduction/future_low
# Fresh timing only on a copied run; does not prove timings will match exactly:
python -m src.benchmark_latency --output /tmp/oxygen-review-reproduction/new_onset

# A new fixed-policy audit from the existing authenticated saved fits:
# Choose a destination that does not exist; no model refit or cutoff change.
python -m src.local_30_evaluation --output /tmp/oxygen-review-local30-copy
# Independent event audit, new output directory:
python -m src.reliability_evaluation --output /tmp/oxygen-review-reliability-copy

# Alarm-budget runner requires a NEW directory under this experiment root.
# It authenticates and reuses the existing results/final score caches.
python -m src.alarm_budget_experiment --output results/alarm_budget_experiment/review_reproduction
# Optional exact frozen selection reuse, instead of selecting again:
python -m src.alarm_budget_experiment --selection-source results/alarm_budget_experiment/run01 --output results/alarm_budget_experiment/review_frozen_reproduction

# Existing simulation reproduction, separate from ML; not run for this review:
python -m signal_studies.run --study waveform --output /tmp/oxygen-review-signal-copy
python -m signal_studies.run --study control --output /tmp/oxygen-review-signal-copy
python -m signal_studies.run --study bode --output /tmp/oxygen-review-signal-copy
```

The ML runner defaults to `results/hardened/<target>` and may update files there, so explicit new output paths are essential. Alarm-budget/local-30 runners reject existing output directories. Even when run to a fresh destination, whole-project preservation verification may report the existing documentation mismatches; that is not permission to rewrite historical hashes. Timing results vary with scheduling/environment and are not expected to reproduce bit for bit.

Read-only verification commands and selected artifact tests:

```bash
python -m src.check_project --links --verify-run results/final/future_low
python -m src.check_project --verify-run results/final/new_onset
python -m src.local_30_evaluation --verify --output results/local_30_fixed_policy_v2
python -m src.alarm_budget_experiment --verify --output results/alarm_budget_experiment/run02

# Tests import test_pipeline as a top-level module; expose tests/ explicitly:
PYTHONPATH=tests python -m unittest test_saved_results test_alarm_budget.AlarmBudgetArtifactTests test_local_30.Local30ArtifactTests test_target_hardening.SavedTargetTests -v
# Relevant unit checks; these are tests, not new held-out experiments:
PYTHONPATH=tests python -m unittest test_pipeline.LabelTests test_pipeline.FeatureTests test_reliability.ReliabilityTests test_alarm_budget.AlarmBudgetTests -v
# Complete suite for future development; includes temporary generated-artifact tests:
python -m unittest discover -s tests -v
# Simulation tests also regenerate temporary simulations; NOT executed in this task:
python -m unittest tests.test_signal_studies -v
```

### 10.2 Exact source index

- [src/config.py](../src/config.py): feature order, models, seed, 30 s horizon, severe-interest values, timing config.

- [src/preprocessing.py](../src/preprocessing.py): WaveformState / PulseDetector / extract_pulses / acquisition clock.

- [src/features.py](../src/features.py): FeatureState.update / add_features, causal bounded histories.

- [src/labeling.py](../src/labeling.py): future_labels / onset_labels / desaturation_events.

- [src/model.py](../src/model.py): ELM and FittedModel, training-only transforms and persistence.

- [src/evaluation.py](../src/evaluation.py): outer_splits / grouped_oof / choose_threshold / metrics / historical alarm_statistics.

- [src/run_experiments.py](../src/run_experiments.py): nested fixed-model experiment and independent demo refits.

- [src/controller.py](../src/controller.py): predict / update_waveform, scores and alerts only.

- [src/reliability_evaluation.py](../src/reliability_evaluation.py): current_normal / links / episodes / eligible_events / match_events / off_event_alarms.

- [src/local_30_evaluation.py](../src/local_30_evaluation.py): audit_selection / verify_threshold_fold / bootstrap_metrics / fixed-policy summaries.

- [src/alarm_budget_experiment.py](../src/alarm_budget_experiment.py): select_partition / candidate_statistics / choose_candidate / verify / changed-policy experiment.

- [src/figures.py](../src/figures.py): FigureRun, plots from saved evidence and permutation diagnostics.

- [src/benchmark_latency.py](../src/benchmark_latency.py): independent component timing and tail reliability rule.

- [results/final/future_low/run_manifest.json](../results/final/future_low/run_manifest.json): broad-target fits/splits/parameters/source hashes.

- [results/final/new_onset/run_manifest.json](../results/final/new_onset/run_manifest.json): onset fits/splits/parameters/source hashes.

- [results/final/future_low/per_record_predictions.csv](../results/final/future_low/per_record_predictions.csv): held-out broad-target pulse scores, thresholds and decisions.

- [results/final/new_onset/per_record_predictions.csv](../results/final/new_onset/per_record_predictions.csv): held-out onset pulse scores, thresholds and decisions.

- [results/final/future_low/validation_oof_predictions.csv.gz](../results/final/future_low/validation_oof_predictions.csv.gz): inner-training OOF rows indexed by outer test record.

- [results/final/new_onset/validation_oof_predictions.csv.gz](../results/final/new_onset/validation_oof_predictions.csv.gz): onset inner-training OOF provenance.

- [results/final/future_low/figures/figure_manifest.json](../results/final/future_low/figures/figure_manifest.json): source/image/table hashes for historical broad-target figure set.

- [results/final/new_onset/figures/figure_manifest.json](../results/final/new_onset/figures/figure_manifest.json): source/image/table hashes for historical onset figure set.

- [results/local_30_fixed_policy_v2/manifest.json](../results/local_30_fixed_policy_v2/manifest.json): fixed-policy source and output hashes; bootstrap assumptions.

- [results/local_30_fixed_policy_v2/paired_recording_bootstrap.csv](../results/local_30_fixed_policy_v2/paired_recording_bootstrap.csv): paired whole-record differences, not patient-level significance.

- [results/alarm_budget_experiment/protocol.json](../results/alarm_budget_experiment/protocol.json): frozen policy constraints, candidate grid, matching and bootstrap.

- [results/alarm_budget_experiment/run02/manifest.json](../results/alarm_budget_experiment/run02/manifest.json): threshold-only provenance, 672 selections, inputs/output hashes.

- [results/alarm_budget_experiment/run02/selected_thresholds.csv](../results/alarm_budget_experiment/run02/selected_thresholds.csv): exact fold/budget cutoff, hex representation and selection counts.

- [results/alarm_budget_experiment/run02/per_recording.csv](../results/alarm_budget_experiment/run02/per_recording.csv): all 30 per-record contributions at every policy/window/subset.

- [results/alarm_budget_experiment/run02/event_matches.csv](../results/alarm_budget_experiment/run02/event_matches.csv): eligible/missed/excluded onsets and actual alarm identities.

- [results/alarm_budget_experiment/run02/alarm_starts.csv](../results/alarm_budget_experiment/run02/alarm_starts.csv): actual starts and off-event classification.

- [results/alarm_budget_experiment/run02/lead_times.csv](../results/alarm_budget_experiment/run02/lead_times.csv): matched leads only; learned-event leads absent.

- [results/alarm_budget_experiment/run02/paired_bootstrap.csv](../results/alarm_budget_experiment/run02/paired_bootstrap.csv): conditional paired uncertainty across final policies.

- [results/reliability/final/window_summary.csv](../results/reliability/final/window_summary.csv): side-by-side original/revised event-matching interpretations.

- [archive/legacy_sources.zip](../archive/legacy_sources.zip): historical source, including random forest experiment.

- [archive/legacy_results.zip](../archive/legacy_results.zip): preserved original metric tables and figures.

- [signal_studies/core.py](../signal_studies/core.py): illustrative conditioning plant/controllers, metrics and margins.

- [signal_studies/REPORT.md](../signal_studies/REPORT.md): synthetic/illustrative assumptions and precise metric definitions.

- [signal_studies/outputs/control_settings.json](../signal_studies/outputs/control_settings.json): simulation parameters, environment and source receipts.

- [signal_studies/outputs/closed_loop_poles.csv](../signal_studies/outputs/closed_loop_poles.csv): all illustrative closed-loop poles.

- [signal_studies/outputs/crossovers.csv](../signal_studies/outputs/crossovers.csv): finite gain/phase crossovers only.

- [signal_studies/validation/full_suite.log](../signal_studies/validation/full_suite.log): prior full-suite evidence, not a newly rerun full suite for this document.

### 10.3 Missing evidence register

The following are **Not available in the current saved results**: independently established subject count/mapping; external/patient-independent test performance; separately trained 5/10/15 s models; clinical event adjudication; validated signal-quality/beat ground truth; monitor physiological delay; calibrated risk probabilities or calibration error; retrained predictive-feature ablations; LightGBM/XGBoost results; PTT features; control effort, actuator-constrained/noisy/uncertain oxygen-plant comparisons; trained-risk-to-feedforward mapping; joint feedforward-PID outcomes; clinical utility/safety/efficacy. These gaps are not silently filled by the synthetic conditioning tests.

### 10.4 Validation performed for this new document

- **Only this document was created.** A SHA-256 snapshot of all 769 pre-existing non-Git/non-bytecode files was taken before writing; the final comparison found no changed or missing existing file. No new ML fits, controller simulations, plots, result tables or test artifacts were saved in the repository.
- **Numerical cross-checks passed:** target-specific confusion counts recomputed from saved held-out decisions; 20,157/4,587 and 16,050/501 sample/positive counts confirmed; all 288 final summary rows agree with grouped per-record counts, exposure and active duration; all learned detection/TP counts are zero with the stated all-event and severe denominators. The independent legacy-label audit has 15,706 verified labels, zero disagreement among those verified labels, plus the stated 1,383 unavailable labels.
- **Saved evidence receipts passed:** all 11 input and 15 output hashes in the final alarm-budget manifest; all 84 input and 12 output hashes in the local-30 manifest; all three existing simulation source receipts. These verify the inspected evidence without changing historical preservation expectations.
- **Existing environment/target quality checks passed:** all seven pinned dependencies; `verify_run` for future-low (20,157 rows) and onset (16,050 rows), four models and 23 PNG figures per target, matching source/table/image receipts and prepared caches. The repository-wide local documentation-link check passes, including this document; all 72 explicit relative links in this new document resolve. Two review-priority ML figures were visually inspected in addition to the saved source/manifest checks; illustrative plots were inspected during their prior creation.
- **Relevant artifact tests:** the corrected command in section 10.1 ran 23 tests in 77.194 s: **20 passed, one failed, two errored**. The nine saved-result tests pass, including causal feature/label correspondence, all-model persisted replay parity, score/count agreement and split provenance. Alarm-budget/local-30 outcome, selection, threshold and exposure tests pass except the preservation checks below.
- **Existing unsuccessful checks remain:** `AlarmBudgetArtifactTests.test_inputs_outputs_and_preservation_verified` and `Local30ArtifactTests.test_outputs_and_protected_files_are_unchanged` stop at the existing `docs/FULL_PIPELINE_GUIDE.md` hash mismatch; `SavedTargetTests.test_historical_outputs_preserved` fails at `docs/MODEL_EVALUATION.md`. Independently comparing complete preservation maps also identifies `docs/FINAL_REPORT.md`; the alarm-budget and local-30 maps each have exactly those three documentation mismatches. All were present in the before snapshot. No model/data/result mismatch was found in the inspected manifests, and no hashes/tests were altered.
- **Invocation correction recorded:** an initial direct `tests.test_target_hardening` invocation lacked `tests/` on the import path and failed to import its top-level `test_pipeline` dependency; exposing `PYTHONPATH=tests` resolves that import. This invocation error is separate from the three remaining historical integrity failures.
- **Scope of tests:** the complete 149-test suite was not rerun for this documentation-only task because it includes temporary simulation generation. Its prior recorded result (146 pass, one failure, two errors) is described in `signal_studies/REPORT.md` and its existing validation log. No claim is made that the current whole-project historical integrity gates all pass.

## 11. ML completion evidence update

The §10.3 absence register was accurate for its earlier review stage. Actual trainedH5/10/15/30 and one causal-context ablation are now finalized retrospectively, with saved event/timestamp tables, bootstrap uncertainty, original-policy parity, figures and new timing in [results/ml_completion/run01](../results/ml_completion/run01/). They are not additional independent patients or a new unexamined test cohort. The original final policy remains0/14; the bounded follow-up yields the same one record07 event in three GaussianNB conditions, all with individual-record budget violations. Do not substitute it for dashboard saved-policy playback or select a policy from outer outcomes. No clinical/PID/feedforward or patient-independent evidence was created. [ML_COMPLETION_TRACKER.md](ML_COMPLETION_TRACKER.md) records verification and remaining limitations; previous findings are preserved rather than silently erased.
