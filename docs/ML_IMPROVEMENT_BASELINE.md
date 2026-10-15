# Verified ML improvement baseline

The existing 30 local recordings are the entire cohort. No raw dataset, trained model, saved output, historical report or integrity receipt was changed. This audit was completed before fitting experimental models.

## Verified design

`src/config.py:MODEL_CONFIGS` defines ELM (50 tanh nodes, ridge 1, positive weight 5, seed 42), balanced logistic regression (C=1), GaussianNB (equal priors, smoothing 1e-6) and a balanced depth-3 tree (leaf minimum 50). `src/run_experiments.py`, `src/evaluation.py:grouped_oof` and `src/model.py:FittedModel.fit` implement outer leave-one-recording-out and inner three-fold recording-grouped OOF predictions; imputation/scaling fit inside each training fold. Thresholds receive only outer-training OOF predictions. Verified recording separation does not establish patient independence: source headers and inventories provide no verified patient identifiers.

`src/labeling.py` defines future-low as minimum SpO2 <=94 over (t,t+30], complete valid coverage, maximum gap 1.01 s; ongoing lows count. New-onset additionally requires fresh current SpO2 >94 and an adjacent valid >94 to <=94 crossing. One reading suffices; no clinical persistence requirement. Current SpO2 gates eligibility and supplies labels; it is not a predictor. `src/preprocessing.py` and `src/features.py` provide forward-only filtering, next-sample pulse confirmation, and current/past pulse features. No PTT is implemented.

**Horizon correction:** all existing models were trained for 30 seconds. The reported 5/10/15/30 seconds in event tables are look-back matching windows, not four trained forecast horizons. Strict current-normal event exposure is 10,329.032 s (2.869176 h); 15 events qualify at 5/10/15 s, 14 at 30 s, and three are the observed-91/92 subset. One-to-one chronological earliest unused alarm starts in [onset-W,onset) receive credit. Off-event starts have no qualifying onset in (start,start+W]; active fraction uses left-held duration, not alerted pulse fraction.

## Reproduced data and timestamp results

    Target  Rows  Positives  Recordings  Positive_Recordings
future_low 20157       4587          30                   10
 new_onset 16050        501          26                    6

```text
    Target      Model   Recall  Precision  Specificity  PR_AUC_AP   TP    FP  FN   TN
future_low        ELM 0.964901   0.238998     0.094862   0.255929 4426 14093 161 1477
future_low GaussianNB 0.982123   0.225667     0.007193   0.196267 4505 15458  82  112
future_low   Logistic 0.931982   0.219602     0.024277   0.240296 4275 15192 312  378
future_low       Tree 0.894920   0.214629     0.035260   0.189493 4105 15021 482  549
 new_onset        ELM 0.890220   0.030919     0.100971   0.023976  446 13979  55 1570
 new_onset GaussianNB 0.900200   0.033238     0.156344   0.057595  451 13118  50 2431
 new_onset   Logistic 0.886228   0.028203     0.016078   0.033500  444 15299  57  250
 new_onset       Tree 1.000000   0.031215     0.000000   0.039582  501 15549   0    0
```

The full future-low population has 20,157 rows/4,587 positives/30 recordings; onset has 16,050/501/26 nonempty groups, with six positive-containing recordings. Four all-low recordings have zero new-onset opportunities, not missing negative rows. Overlapping future windows create many correlated positive pulse labels per event.

## Reproduced final operating-policy result

All four learned models have **0/14** eligible 30 s detections, **0/15** at 5/10/15 s and **0/3** severe-interest detections, for both training-target families and budgets 1/5/10 starts/evaluable hour with 5% active cap. All matched learned leads are undefined, not zero. These budgets are research assumptions, not clinically validated limits. Event rows/counts and timestamp counts were recomputed from original saved arrays, and original manifest input/output SHA-256 values and both saved-run quality gates passed. New machine-readable copies/recomputations are in the isolated output directory.

## Failure analysis: evidence versus hypotheses

**Verified:** earlier onset ELM AP is 0.023976 against 501/16050=0.031215 prevalence; logistic AP 0.033500, GaussianNB 0.057595 and tree 0.039582. Poor discrimination precedes threshold choice; high future-low recall partly measures ongoing lows. The scalar-cutoff policy imposes both start and duration constraints, so frequent or sustained alerts can be infeasible. Training-selected policies are often silent; the table below counts them. Nonzero training detections did occur, so zero outer detection is not explained by universally zero training sensitivity. Some outer-record constraints fail despite training feasibility. Scores from inner refits and outer refits can have different distributions. No verified evaluation defect was found in this audit.

**Hypotheses to test, not established causes:** short/low-dimensional pulse features may not capture pre-onset physiology; causal longer context may help, but waveform/peak quality is not clinically validated. Inner-to-outer score-scale transfer may impair raw cutoffs; an empirical training-negative score rank may improve scale comparability, but is not probability calibration. Shorter truly trained horizons may change discrimination; existing shorter look-back counts do not answer that question. Few independent event-bearing recordings, correlated samples, unknown monitor delay and missing readings fundamentally limit inference. A high-capacity family or broad search is not justified for this pass.

```text
    Target      Model  Selections  Training_Detections  Silent_Selections
future_low        ELM          90                   14                 77
future_low GaussianNB          90                    6                 84
future_low   Logistic          90                    6                 84
future_low       Tree          90                    3                 87
 new_onset        ELM          78                    1                 77
 new_onset GaussianNB          78                   29                 49
 new_onset   Logistic          78                   38                 52
 new_onset       Tree          78                    0                 78
```

## Commands and preservation

`OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m ml_research.baseline --output results/ml_improvement/run01` verifies the manifests, both original target runs, event aggregation and timestamp results and creates only new files. The output path must not already exist. Explicit saved-run checks: `python -m src.check_project --verify-run results/final/new_onset` and the corresponding `future_low` command.

Three pre-existing full-suite failures concern authorized historical document changes: `test_target_hardening.SavedTargetTests.test_historical_outputs_preserved`, `test_local_30.Local30ArtifactTests.test_outputs_and_protected_files_are_unchanged`, and `test_alarm_budget.AlarmBudgetArtifactTests.test_inputs_outputs_and_preservation_verified`. Their receipts/assertions remain protected. Their old hashes are accurate historical baselines; this task will not rewrite them to obtain a green suite.

The cohort has already been examined repeatedly. Any ensuing controlled local experiment is exploratory held-out-recording evaluation, not a fresh confirmatory trial, unseen-patient validation or evidence for oxygen actuation. No improvement is promised.
