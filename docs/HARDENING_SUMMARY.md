# Targeted hardening change summary

The existing future-low target and all historical outputs in results/validated are preserved. No scripts were deleted/archived, raw data changed, packages installed, data downloaded, hyperparameters searched, or outer-test outcomes used to adjust thresholds.

## Definitions and implementation

A: `future_low` remains minimum valid observed SpO2 in (t,t+30s] <=94, with complete valid future coverage. B: `new_onset` additionally requires latest current observed SpO2 >94 within 1.01s, and a qualifying observed crossing in the same future horizon. A crossing is a valid >94 reading immediately followed within 1.01s by a valid <=94 reading. One low sample suffices; there is no persistence or normal-state washout requirement. Missing/invalid readings and gaps prevent qualifying transitions. Initial low episodes are left censored. This is a defensible observed-monitor definition, not a clinical event annotation. Neither target predicts exact SpO2.

On complete valid current-normal windows these binary labels agree; new-onset training nevertheless excludes ongoing-low and unknown-current states. Healthy periods can be brief between recurring episodes. SpO2 is used only for eligibility/ground truth, never as a model input. Features retain the complete past pulse stream before eligible rows are selected.

## Files changed and why

- src/config.py: named target definitions, separate output roots, shared horizon/threshold/gap, severe-subset values, benchmark settings, threshold-strategy description.
- src/labeling.py: reusable observed episodes and explicit onset eligibility/status; future-low numerical labels preserved.
- src/dataset.py: both named labels, selected target, full-history features before exclusions, event/audit artifacts, new default output; historical legacy dataset remains untouched.
- src/provenance.py (new): content-verified versioned receipts including raw/legacy source, generation-module, target and feature-order fingerprints; generated artifacts verified on every cache hit; atomic receipt replacement.
- src/run_experiments.py: target selection, protected historical output path, cache reuse, richer manifest and model target metadata; existing grouped fitting/threshold protocol reused.
- src/evaluation.py: existing alarm-gap convention now references shared configuration; no fitting or operating-point change.
- src/model.py and src/controller.py: persist target contract and gate onset inference for low/unknown/stale current SpO2, while retaining causal history.
- src/benchmark_latency.py: configured source paths and replay length, actual current-observation input to replay, repeated separate loading timings, explicit new-run output defaults; no training inside timers.
- src/replay.py: target-aware causal replay with timestamped current numeric observations and protected historical outputs.
- src/target_reporting.py (new): separate target reports, pooled/macro and recording-class metrics, matched frozen references, score/PR diagnostics and raw-event matching; historical reports are not rewritten.
- tests/test_target_hardening.py (new): onset boundaries/censoring, cache dependencies/corruption/reuse, training-only preprocessing, onset inference, artifact replay, matched rows/splits and historical preservation.
- README.md and .gitignore: current commands and contracts; generated large hardening datasets/OOF tables/binaries stay local.

## Executed validation

`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v`: **69 tests passed**, including all original tests. Initial focused suite also passed (63 tests); the final suite adds saved-target integration checks. The final log is results/hardened/test_results.txt. `python -m compileall -q src tests` and `git diff --check` succeeded. No skipped integration checks in the final run.

A clean-process `python -m src.replay --target new_onset --record 12` completed with 748 pulse outputs; five current-low outputs were explicitly ineligible and none alerted. This full-data replay is a demonstration, not held-out evaluation.

Both fixed-model experiments completed: 30 future-low outer recording folds and 26 onset folds, four unchanged models, three inner grouped OOF fits per outer model. No search. Every model uses identical eligible rows within its target, training-only fitted imputation/scaling and validation-only thresholds. Future-low saved labels, scores, thresholds and predictions match historical outputs within 1e-10. Current cache receipts were exercised without regeneration. Historical file fingerprints are recorded and verified in historical_preservation.json.

## Actual pooled before/after metrics

“Before” is the frozen historical future-low model evaluated on exactly the same eligible rows, relabelled for the selected target. It is NOT a previous onset-trained experiment. Future-low has 20,157 timestamps / 4,587 positives across 30 groups; onset has 16,050 / 501 across 26 groups, with only six positive-containing groups. The four omitted groups have no current-normal eligible rows, not poor-model-based exclusions.

| Target | Stage | Model | Recall | Precision | Specificity | PR_AUC_AP | FP | FN | False_Alerts_Per_Hour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| future_low | before_frozen_future_low | ELM | 0.9649 | 0.2390 | 0.0949 | 0.2559 | 14093 | 161 | 140.6868 |
| future_low | before_frozen_future_low | Logistic | 0.9320 | 0.2196 | 0.0243 | 0.2403 | 15192 | 312 | 27.6986 |
| future_low | before_frozen_future_low | GaussianNB | 0.9821 | 0.2257 | 0.0072 | 0.1963 | 15458 | 82 | 13.9864 |
| future_low | before_frozen_future_low | Tree | 0.8949 | 0.2146 | 0.0353 | 0.1895 | 15021 | 482 | 28.5213 |
| future_low | after_target_training | ELM | 0.9649 | 0.2390 | 0.0949 | 0.2559 | 14093 | 161 | 140.6868 |
| future_low | after_target_training | Logistic | 0.9320 | 0.2196 | 0.0243 | 0.2403 | 15192 | 312 | 27.6986 |
| future_low | after_target_training | GaussianNB | 0.9821 | 0.2257 | 0.0072 | 0.1963 | 15458 | 82 | 13.9864 |
| future_low | after_target_training | Tree | 0.8949 | 0.2146 | 0.0353 | 0.1895 | 15021 | 482 | 28.5213 |
| new_onset | before_frozen_future_low | ELM | 0.8623 | 0.0298 | 0.0949 | 0.0315 | 14074 | 69 | 179.8081 |
| new_onset | before_frozen_future_low | Logistic | 0.9002 | 0.0289 | 0.0240 | 0.0305 | 15176 | 50 | 35.1950 |
| new_onset | before_frozen_future_low | GaussianNB | 0.8882 | 0.0280 | 0.0071 | 0.0238 | 15439 | 56 | 18.4687 |
| new_onset | before_frozen_future_low | Tree | 1.0000 | 0.0323 | 0.0353 | 0.0299 | 15000 | 0 | 37.2858 |
| new_onset | after_target_training | ELM | 0.8902 | 0.0309 | 0.1010 | 0.0240 | 13979 | 55 | 182.9443 |
| new_onset | after_target_training | Logistic | 0.8862 | 0.0282 | 0.0161 | 0.0335 | 15299 | 57 | 21.6048 |
| new_onset | after_target_training | GaussianNB | 0.9002 | 0.0332 | 0.1563 | 0.0576 | 13118 | 50 | 60.6330 |
| new_onset | after_target_training | Tree | 1.0000 | 0.0312 | 0.0000 | 0.0396 | 15549 | 0 | 9.7570 |

## Macro metrics (separate from pooled metrics)

| Target | Model | Macro_Recall | Macro_Precision | Macro_F1 | Macro_Recall_Defined_Recordings | Negative_Only_Recordings |
| --- | --- | --- | --- | --- | --- | --- |
| future_low | ELM | 0.9371 | 0.2288 | 0.2396 | 10.0000 | 20.0000 |
| future_low | Logistic | 0.9266 | 0.2292 | 0.2344 | 10.0000 | 20.0000 |
| future_low | GaussianNB | 0.9586 | 0.2290 | 0.2419 | 10.0000 | 20.0000 |
| future_low | Tree | 0.9165 | 0.2308 | 0.2215 | 10.0000 | 20.0000 |
| new_onset | ELM | 0.9469 | 0.1061 | 0.1208 | 6.0000 | 20.0000 |
| new_onset | Logistic | 0.9269 | 0.1047 | 0.1194 | 6.0000 | 20.0000 |
| new_onset | GaussianNB | 0.8770 | 0.1094 | 0.1260 | 6.0000 | 20.0000 |
| new_onset | Tree | 1.0000 | 0.1053 | 0.1215 | 6.0000 | 20.0000 |

Undefined recording metrics remain NaN and are excluded from the corresponding macro denominator. Recording-class and positive-record detail tables are separate. Onset Records 17/29 retain only 43/11 all-positive eligible timestamps; they are not removed, but cannot demonstrate within-record discrimination and can distort simple macro averages.

## False alarms and ELM conclusion

Future-low pooled ELM recall remains 96.49%, precision 23.90%, but specificity is only 9.49%; chronic/ongoing lows dominate the positive population. Training for onset raises ELM recall from 86.23% to 89.02% and precision from 2.98% to 3.09%, yet onset AP decreases from 0.0315 to 0.0240 (prevalence 0.0312). Logistic onset recall changes from 90.02% to 88.62%; neither meets the 92% held-out timestamp goal. New-onset Gaussian NB has higher precision (3.32%), specificity (15.63%) and AP (0.0576) than ELM. The tree reaches 100% recall by alerting on every eligible timestamp. All validation operating points attain the predefined 92% OOF objective; held-out outcomes are reported independently.

ELM has descriptive pooled future-low advantages over logistic, but no practically useful superiority for impending detection is established. For onset, its AP is below logistic and Gaussian NB. ELM alerts on 89.88% of eligible timestamps and has about 182.94 false alarm episodes per evaluable hour, versus logistic 21.60; fewer episode starts for nearly sustained alerts is not better specificity. The inherited episode rule counts a positive run as false only if it contains no positive target. Runs split on negative predictions or >2s gaps; exposure sums eligible adjacent pulse intervals <=2s. This is approximate evaluable pulse exposure, not total recording duration; short unobserved/ineligible gaps can be bridged by that convention. FP timestamps and alert fraction must be read alongside episode rates. No alarm suppression policy or threshold was optimized after testing.

## Event sensitivity and lead time

| Target | Model | Subset | Eligible_Events | Detected | Sensitivity | Median_Lead_Time_s | Excluded_Events |
| --- | --- | --- | --- | --- | --- | --- | --- |
| future_low | ELM | all_qualifying_onsets | 14 | 14 | 1.0000 | 29.6840 | 8 |
| future_low | ELM | 91_92_episode_subset | 3 | 3 | 1.0000 | 9.7840 | 4 |
| future_low | Logistic | all_qualifying_onsets | 14 | 14 | 1.0000 | 29.6840 | 8 |
| future_low | Logistic | 91_92_episode_subset | 3 | 3 | 1.0000 | 9.7840 | 4 |
| future_low | GaussianNB | all_qualifying_onsets | 14 | 14 | 1.0000 | 29.6840 | 8 |
| future_low | GaussianNB | 91_92_episode_subset | 3 | 3 | 1.0000 | 9.7840 | 4 |
| future_low | Tree | all_qualifying_onsets | 14 | 14 | 1.0000 | 29.6840 | 8 |
| future_low | Tree | 91_92_episode_subset | 3 | 3 | 1.0000 | 9.7840 | 4 |
| new_onset | ELM | all_qualifying_onsets | 14 | 14 | 1.0000 | 29.3760 | 8 |
| new_onset | ELM | 91_92_episode_subset | 3 | 3 | 1.0000 | 9.7840 | 4 |
| new_onset | Logistic | all_qualifying_onsets | 14 | 14 | 1.0000 | 29.6840 | 8 |
| new_onset | Logistic | 91_92_episode_subset | 3 | 3 | 1.0000 | 9.7840 | 4 |
| new_onset | GaussianNB | all_qualifying_onsets | 14 | 14 | 1.0000 | 29.3880 | 8 |
| new_onset | GaussianNB | 91_92_episode_subset | 3 | 3 | 1.0000 | 9.7840 | 4 |
| new_onset | Tree | all_qualifying_onsets | 14 | 14 | 1.0000 | 29.6840 | 8 |
| new_onset | Tree | 91_92_episode_subset | 3 | 3 | 1.0000 | 9.7840 | 4 |

The event denominator changed from historical 13 to 14 because the new, explicitly defined rule requires a full valid RAW 30s lead-in and at least one eligible pre-onset prediction opportunity, rather than eligible pulse coverage throughout all 30s. This is an event-coverage definition change, not an improvement in performance. First eligible opportunity and available lead time are saved per event. Alerts only in [onset-horizon,onset) match; alerts at onset do not. Every model detects all 14 evaluable observed onsets and all three episodes containing an actual 91/92 reading, but an always-alert reference does too. Near-continuous alerting makes that sensitivity weak evidence of useful discrimination. The three severe-subset events come from three recordings; median observed lead is 9.784s, limited by available current-normal opportunities in brief recoveries. Nearby events can share an alert; event matching does not estimate one-to-one event precision.

## Measured computational latency

| Target | Model | Component | Median_ms | P95_ms | P99_ms | Repetitions |
| --- | --- | --- | --- | --- | --- | --- |
| future_low | ELM | model_loading | 0.7980 | 0.8716 | undefined | 100 |
| future_low | ELM | model_only | 0.0053 | 0.0107 | 0.0160 | 2000 |
| future_low | ELM | pulse_to_alert | 0.4919 | 0.8231 | 0.9214 | 2000 |
| future_low | GaussianNB | model_loading | 0.5137 | 0.5504 | undefined | 100 |
| future_low | GaussianNB | model_only | 0.2126 | 0.3859 | 0.4576 | 2000 |
| future_low | GaussianNB | pulse_to_alert | 0.7233 | 1.3655 | 2.2312 | 2000 |
| future_low | Logistic | model_loading | 0.5215 | 0.5748 | undefined | 100 |
| future_low | Logistic | model_only | 0.1330 | 0.2329 | 0.2801 | 2000 |
| future_low | Logistic | pulse_to_alert | 0.6268 | 1.0706 | 1.2878 | 2000 |
| future_low | Tree | model_loading | 0.6267 | 0.6611 | undefined | 100 |
| future_low | Tree | model_only | 0.1101 | 0.1461 | 0.1794 | 2000 |
| future_low | Tree | pulse_to_alert | 0.6196 | 1.4294 | 3.0360 | 2000 |
| new_onset | ELM | model_loading | 0.6542 | 1.4304 | undefined | 100 |
| new_onset | ELM | model_only | 0.0114 | 0.0125 | 0.0386 | 2000 |
| new_onset | ELM | pulse_to_alert | 0.5386 | 1.3905 | 3.7189 | 2000 |
| new_onset | GaussianNB | model_loading | 0.5031 | 0.5597 | undefined | 100 |
| new_onset | GaussianNB | model_only | 0.2089 | 0.3731 | 0.4433 | 2000 |
| new_onset | GaussianNB | pulse_to_alert | 0.7193 | 1.2099 | 1.3376 | 2000 |
| new_onset | Logistic | model_loading | 0.5056 | 0.5603 | undefined | 100 |
| new_onset | Logistic | model_only | 0.1308 | 0.1765 | 0.2366 | 2000 |
| new_onset | Logistic | pulse_to_alert | 0.6316 | 1.0384 | 1.1274 | 2000 |
| new_onset | Tree | model_loading | 0.6176 | 0.7306 | undefined | 100 |
| new_onset | Tree | model_only | 0.1094 | 0.1768 | 0.2567 | 2000 |
| new_onset | Tree | pulse_to_alert | 0.6158 | 1.0521 | 2.3430 | 2000 |

Full component timings include feature calculation, transformations/model, thresholding, raw waveform samples and pulse-producing waveform samples. Benchmarks were executed separately and sequentially after fitting completed, using perf_counter_ns, 100 warmups and 2,000 pulse measurements, one native thread. Loading has 100 separate warm-cache measurements; P99 is withheld below 1,000 observations, including the 236 pulse-producing replay samples. Environment fingerprints are saved per target (Python 3.14.2, AMD EPYC 7763). Timings vary on the shared host: ELM has lower median pulse latency, but onset ELM P95/P99 exceed logistic in this run. These are empirical compute measurements, not worst-case bounds. Acquisition, filter phase delay, monitor delay, communication and oxygen actuation are not measured. Pulse confirmation needs one additional 125Hz sample (8ms); 30 seconds is a label horizon, not guaranteed warning.

## Reproduce without changing historical outputs

```bash
python -m unittest discover -s tests -v
python -m src.run_experiments --target future_low
python -m src.run_experiments --target new_onset
python -m src.benchmark_latency --target new_onset
python -m src.target_reporting --output results/hardened/new_onset
python -m src.replay --target new_onset --record 12
```

Each run manifest records target semantics, features/configuration, seeds, fixed model settings, raw/data fingerprints, cache receipt and outer/inner recording splits. Evaluation source hashes are retained; benchmark_environment records the code used for timing. final_verification records the checked final source state. Original historical cache receipts remain untouched; newly generated version-2 receipts reflect current dependencies and detect source or artifact changes.

## Remaining limitations and recommendation

Only six recordings contribute prospective positives; all-positive eligible snippets and recurring threshold crossings limit generalization claims. Patient independence, sensor alignment/monitor delays and clinical validity of single-reading excursions are not established. Severe events are too few for reliable clinical confidence intervals. Existing configurations were fixed before this pass; repeated development on these recordings is not an external validation cohort. Onset eligibility also requires access to contemporaneous SpO2 at deployment.

Retain the frozen future-low experiment for reproducibility, use new-onset outputs for the impending-detection question, and retain all models as transparent research comparators. Gaussian NB shows better onset ranking here, but none demonstrates an acceptable alarm burden. The next valuable work is independent, longer recordings with verified timing and enough adjudicated onset events under a frozen protocol, not a broad search or more aggressive thresholds.
