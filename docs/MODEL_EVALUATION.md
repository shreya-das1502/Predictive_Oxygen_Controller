# Measured model evaluation

| Model | Recall | Precision | Specificity | F1 | PR_AUC_AP | Macro_Recall | Macro_F1 | FP | FN | False_Alerts_Per_Hour | Alert_Fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ELM | 0.9649 | 0.2390 | 0.0949 | 0.3831 | 0.2559 | 0.9371 | 0.2396 | 14093 | 161 | 140.6868 | 0.9187 |
| Logistic | 0.9320 | 0.2196 | 0.0243 | 0.3555 | 0.2403 | 0.9266 | 0.2344 | 15192 | 312 | 27.6986 | 0.9658 |
| GaussianNB | 0.9821 | 0.2257 | 0.0072 | 0.3670 | 0.1963 | 0.9586 | 0.2419 | 15458 | 82 | 13.9864 | 0.9904 |
| Tree | 0.8949 | 0.2146 | 0.0353 | 0.3462 | 0.1895 | 0.9165 | 0.2215 | 15021 | 482 | 28.5213 | 0.9489 |

All values above are held-out predictions under leave-one-recording-out evaluation on 20,157 verified pulse timestamps in 30 recordings. Positive prevalence is 22.7564%. 10 recordings contain positives; 6 are all positive. No difficult recording is removed. Raw monitor threshold crossings, rather than contiguous predictive labels, define events.

The target is min observed SpO2 in `(t,t+30]` <=94, with a complete observable future horizon. It includes ongoing low saturation. It is not an exact SpO2 regression model or proof of impending-event detection. Missing future measurements and incomplete recording tails have unknown labels and are explicitly excluded. Numeric/waveform timestamps retain their supplied shared coordinate, including numeric record 23 starting at -1s. Waveform export rounding is corrected with the verified 125Hz row clock, without altering waveform values.

Each model uses the same ten causal features and rows. Three inner GroupKFold fits generate OOF scores for every outer-training row. The largest unique threshold achieving >=92% validation recall maximizes specificity; tied score blocks are never split. Imputation/scaling are refitted on each training fold only. Fixed model configurations were chosen before evaluation; no search or test-based threshold tuning occurs. Full-data demonstration artifacts have separate full-data grouped OOF thresholds and are not used to report held-out metrics.

Undefined recall on negative-only records and specificity on all-positive records are blank (NaN). Macro metrics average only defined recording values, with denominators in pooled_metrics.csv. PR-AUC is average precision, not trapezoidal interpolation. Pooled ranking metrics mix uncalibrated scores from different refits; macro AP on mixed-class recordings is also provided. Weighted classifier outputs and ELM scores are not validated calibrated risks.

All 120 outer/model thresholds achieve 92% grouped validation recall: True. Held-out target outcomes are reported independently; target attainment on validation does not guarantee it on new recordings.

Always-alert/never-alert references (no learned model):

| Reference | Recall | Precision | Specificity | F1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| Always_Alert | 1.0000 | 0.2276 | 0.0000 | 0.3708 | 15570 | 0 |
| Never_Alert | 0.0000 | undefined | 1.0000 | 0.0000 | 0 | 4587 |

All-alert sensitivity is 100% on every eligible event by construction. Therefore high event sensitivity with high alert duty is not evidence of useful event discrimination. Alert_Fraction reports predicted-positive pulse fraction, not elapsed-time duty. Sustained always-on alarms can yield fewer counted episode starts than chattering models; episode rates must be read with timestamp FP burden and alert fraction.

## Genuine pre-onset detection

| Model | Recall | Precision | FP | FN |
| --- | --- | --- | --- | --- |
| ELM | 0.8623 | 0.0298 | 14074 | 69 |
| Logistic | 0.9002 | 0.0289 | 15176 | 50 |
| GaussianNB | 0.8882 | 0.0280 | 15439 | 56 |
| Tree | 1.0000 | 0.0323 | 15000 | 0 |

Current-observed-SpO2>94 subset, retaining identical models and operating points. This is a descriptive prospective-state subset; current SpO2 is never a predictor.

| Model | Subset | Eligible_Events | Detected | Sensitivity | Positive_Recordings | Median_Lead_Time_s |
| --- | --- | --- | --- | --- | --- | --- |
| ELM | all_94_events | 13 | 13 | 1.0000 | 6 | 29.7040 |
| ELM | 91_92_subset | 3 | 3 | 1.0000 | 3 | 29.6720 |
| Logistic | all_94_events | 13 | 13 | 1.0000 | 6 | 29.7040 |
| Logistic | 91_92_subset | 3 | 3 | 1.0000 | 3 | 29.6720 |
| GaussianNB | all_94_events | 13 | 13 | 1.0000 | 6 | 29.7040 |
| GaussianNB | 91_92_subset | 3 | 3 | 1.0000 | 3 | 29.6720 |
| Tree | all_94_events | 13 | 13 | 1.0000 | 6 | 29.7040 |
| Tree | 91_92_subset | 3 | 3 | 1.0000 | 3 | 29.6720 |

Episodes are contiguous observed SpO2<=94 at 1Hz; missing readings break episodes. Only crossings preceded by observed >94, with an observed 30s lead-in and continuous eligible pulse coverage (no >2s gap, first/last within 2s) count in sensitivity. An alert in [onset-30,onset) is a detection. A 91–92 subset requires an actual observed 91 or 92 within that episode; one-second monitor excursions are not clinically adjudicated events. Initial/chronic lows and unavailable lead-ins are separately enumerated. The same alert can anticipate nearby episodes, so event sensitivity does not imply independent alert-event precision.

False alerts/hour count predicted-positive runs split by negative predictions or >2s gaps; a false run contains no positive target timestamp. A sustained run overlapping a positive target is counted as true even if long. Exposure sums observed pulse intervals <=2s, excluding unavailable-target gaps; this is evaluable exposure, not total wall time. FP timestamps/hour is separately recorded and is not an episode alarm rate. No suppression/refractory alarm policy is retrospectively optimized.

## Records 07, 12, 26, 30

| Record | Model | Samples | Positives | Threshold | Recall | Precision | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | ELM | 682 | 49 | 0.2373 | 0.8980 | 0.0653 | 630 | 5 |
| 7 | Logistic | 682 | 49 | 0.1797 | 1.0000 | 0.0718 | 633 | 0 |
| 7 | GaussianNB | 682 | 49 | 0.0964 | 0.9796 | 0.0705 | 633 | 1 |
| 7 | Tree | 682 | 49 | 0.0650 | 1.0000 | 0.0724 | 628 | 0 |
| 12 | ELM | 702 | 50 | -0.0438 | 1.0000 | 0.0736 | 629 | 0 |
| 12 | Logistic | 702 | 50 | 0.1286 | 1.0000 | 0.0712 | 652 | 0 |
| 12 | GaussianNB | 702 | 50 | 0.0039 | 1.0000 | 0.0712 | 652 | 0 |
| 12 | Tree | 702 | 50 | 0.0091 | 1.0000 | 0.0712 | 652 | 0 |
| 26 | ELM | 660 | 344 | 0.0993 | 0.9535 | 0.5231 | 299 | 16 |
| 26 | Logistic | 660 | 344 | 0.1230 | 0.9971 | 0.5221 | 314 | 1 |
| 26 | GaussianNB | 660 | 344 | 0.0035 | 0.9971 | 0.5205 | 316 | 1 |
| 26 | Tree | 660 | 344 | 0.0685 | 1.0000 | 0.5212 | 316 | 0 |
| 30 | ELM | 787 | 203 | 0.1019 | 0.6158 | 0.2006 | 498 | 78 |
| 30 | Logistic | 787 | 203 | 0.1493 | 0.6601 | 0.2100 | 504 | 69 |
| 30 | GaussianNB | 787 | 203 | 0.0011 | 0.6108 | 0.2074 | 474 | 79 |
| 30 | Tree | 787 | 203 | 0.0012 | 1.0000 | 0.2579 | 584 | 0 |

Record 12 ELM recall is 100.0%, precision 7.4%, with 629 FP: higher sensitivity comes at high cost, not evidence of strong separation. Record 26 ELM AP=0.531 versus prevalence 0.521; positive/negative median scores are 0.481/0.455. Record 30 ELM recall=61.6%; positive/negative medians are 0.460/0.508, while 29.2% of amplitude-slope and 27.4% of rolling-rate-mean values lie outside outer-training 1–99% ranges. Record 07 AP=0.054 versus prevalence 0.072. These findings support weak/uneven feature discrimination plus distribution shift, alongside the verified historical leakage, target-selection and incomplete-window bugs. Class imbalance and poorly representative legacy validation compounded the problem. No threshold alone resolves it; no clinical causal mechanism is inferred.

Feature and score distributions, confusion counts, temporal/PR plots, missingness, and training percentile comparisons are saved per recording. Fraction outside training 1–99% ranges is a descriptive distribution-shift indicator, not a test of causation. Large differences by recording and weak/uneven within-record ranking support limited portability of pulse-only features. Correcting leakage and thresholds cannot create discriminative physiology. Target/population changes make direct numerical comparison with historical leaky/selective results invalid.

## ELM hypothesis

ELM recall=0.9649, precision=0.2390, AP=0.2559; logistic recall=0.9320, precision=0.2196, AP=0.2403. ELM has higher pooled recall, precision and AP and fewer FP timestamps than logistic at the common validation objective. These are descriptive advantages on this dataset. However, ELM produces more false alert episodes and its macro within-record AP is lower; practically useful ELM superiority is not established. Both largely alert across negative recordings. Retain ELM as the main research comparator and logistic as the prespecified transparent demonstration reference; neither is suitable for oxygen actuation from this evidence.

## Limitations and next step

Only 30 of the public 53 recordings are local. Recording IDs are used as groups; independent patient identity cannot be proven without source metadata. Few new-onset events, chronic-low records, rounded monitor readings, unknown sensor delay, and missing measurements limit interpretation. Exploratory 1,000-replicate recording bootstrap intervals are saved, but CV fits share training data and few positive-onset groups preclude reliable clinical confidence intervals. Thresholds pursue timestamp recall and can produce substantial alarms. The highest-value next step is acquiring independent, longer recordings with verified signal/SpO2 timing and sufficient adjudicated onset events; freeze this protocol for that evaluation instead of more threshold sweeps.
