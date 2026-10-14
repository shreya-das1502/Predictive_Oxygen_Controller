# Target-specific held-out evaluation

{
  "name": "new_onset",
  "horizon_s": 30.0,
  "threshold_spo2": 94.0,
  "window": "(t,t+horizon]",
  "complete_valid_future_required": true,
  "maximum_numeric_gap_s": 1.01,
  "current_spo2_eligibility": "latest valid reading within maximum_numeric_gap_s > threshold",
  "event_definition": "first valid <=threshold reading immediately following valid >threshold; adjacent times <=maximum_numeric_gap_s; one reading suffices; no persistence or washout requirement",
  "severe_subset": [
    91.0,
    92.0
  ],
  "interpretation": "observed monitor threshold crossing; not clinically adjudicated"
}

Fixed models and hyperparameters; no model search or threshold adjustment after testing. Pooled timestamp metrics and Macro_* recording metrics are distinct. Undefined metrics remain NaN. AP means average precision. Pooled scores combine refits and are uncalibrated.

| Model | Recall | Precision | Specificity | PR_AUC_AP | FP | FN | Macro_Recall | Macro_F1 | False_Alerts_Per_Hour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ELM | 0.8902 | 0.0309 | 0.1010 | 0.0240 | 13979 | 55 | 0.9469 | 0.1208 | 182.9443 |
| Logistic | 0.8862 | 0.0282 | 0.0161 | 0.0335 | 15299 | 57 | 0.9269 | 0.1194 | 21.6048 |
| GaussianNB | 0.9002 | 0.0332 | 0.1563 | 0.0576 | 13118 | 50 | 0.8770 | 0.1260 | 60.6330 |
| Tree | 1.0000 | 0.0312 | 0.0000 | 0.0396 | 15549 | 0 | 1.0000 | 0.1215 | 9.7570 |

| Model | Recall | Precision | Specificity | PR_AUC_AP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| ELM | 0.8623 | 0.0298 | 0.0949 | 0.0315 | 14074 | 69 |
| Logistic | 0.9002 | 0.0289 | 0.0240 | 0.0305 | 15176 | 50 |
| GaussianNB | 0.8882 | 0.0280 | 0.0071 | 0.0238 | 15439 | 56 |
| Tree | 1.0000 | 0.0323 | 0.0353 | 0.0299 | 15000 | 0 |

The second table uses frozen historical future-low models on the SAME eligible rows, relabelled for this target. It is a reference, not a prior new-onset-trained experiment. Primary future-low includes ongoing lows; onset requires current observed normal saturation. Targets and populations are not interchangeable.

| Model | Subset | Eligible_Events | Detected | Sensitivity | Median_Lead_Time_s | Excluded_Events |
| --- | --- | --- | --- | --- | --- | --- |
| ELM | all_qualifying_onsets | 14 | 14 | 1.0000 | 29.3760 | 8 |
| ELM | 91_92_episode_subset | 3 | 3 | 1.0000 | 9.7840 | 4 |
| Logistic | all_qualifying_onsets | 14 | 14 | 1.0000 | 29.6840 | 8 |
| Logistic | 91_92_episode_subset | 3 | 3 | 1.0000 | 9.7840 | 4 |
| GaussianNB | all_qualifying_onsets | 14 | 14 | 1.0000 | 29.3880 | 8 |
| GaussianNB | 91_92_episode_subset | 3 | 3 | 1.0000 | 9.7840 | 4 |
| Tree | all_qualifying_onsets | 14 | 14 | 1.0000 | 29.6840 | 8 |
| Tree | 91_92_episode_subset | 3 | 3 | 1.0000 | 9.7840 | 4 |

Events require one observed low reading; persistence is not presumed. Missing readings/gaps break episodes; left-censored onsets are excluded. A full valid raw lead-in plus at least one eligible prediction opportunity is required; opportunity coverage is saved. Alerts in [onset-horizon,onset) count. Nearby events can share an alert. The 91–92 subset refers to actual readings within the episode, not exact SpO2 prediction. Always-alert would detect every eligible event; event sensitivity must be read with FP burden.

False alarm episodes use the existing definition: positive runs split by negative decisions or >configured alert gap, false if the run contains no positive target. Evaluable exposure sums adjacent eligible pulse intervals <=that gap. It excludes current-low periods for onset; it is not total recording time. A sustained alarm can yield few episode starts yet high false-positive duty. No suppression policy has been tuned.

All-positive and short eligible exposures are explicitly flagged in positive_recording_detail.csv; they are retained, but do not establish within-record discrimination. Few independent positive recordings, unverified patient identity/monitor delay and one-second numeric precision limit inference. This is retrospective recording-wise CV, not an untouched external test set. Model parameters remain fixed from the earlier run.

| Model | Component | Median_ms | P95_ms | P99_ms | Repetitions |
| --- | --- | --- | --- | --- | --- |
| ELM | model_loading | 0.6542 | 1.4304 | undefined | 100 |
| ELM | feature_calculation | 0.2292 | 0.2900 | 0.7088 | 2000 |
| ELM | model_only | 0.0114 | 0.0125 | 0.0386 | 2000 |
| ELM | transforms_and_model | 0.5682 | 0.6646 | 2.0943 | 2000 |
| ELM | thresholding | 0.0003 | 0.0006 | 0.0008 | 2000 |
| ELM | pulse_to_alert | 0.5386 | 1.3905 | 3.7189 | 2000 |
| ELM | waveform_sample_to_alert | 0.0707 | 0.1361 | 0.6016 | 19901 |
| ELM | waveform_pulse_producing_sample | 0.6304 | 1.0339 | undefined | 236 |
| GaussianNB | model_loading | 0.5031 | 0.5597 | undefined | 100 |
| GaussianNB | feature_calculation | 0.1123 | 0.2166 | 0.2497 | 2000 |
| GaussianNB | model_only | 0.2089 | 0.3731 | 0.4433 | 2000 |
| GaussianNB | transforms_and_model | 0.5283 | 0.8711 | 1.0012 | 2000 |
| GaussianNB | thresholding | 0.0002 | 0.0002 | 0.0004 | 2000 |
| GaussianNB | pulse_to_alert | 0.7193 | 1.2099 | 1.3376 | 2000 |
| GaussianNB | waveform_sample_to_alert | 0.0702 | 0.1379 | 0.8420 | 19901 |
| GaussianNB | waveform_pulse_producing_sample | 0.8559 | 1.4562 | undefined | 236 |
| Logistic | model_loading | 0.5056 | 0.5603 | undefined | 100 |
| Logistic | feature_calculation | 0.1120 | 0.1517 | 0.2032 | 2000 |
| Logistic | model_only | 0.1308 | 0.1765 | 0.2366 | 2000 |
| Logistic | transforms_and_model | 0.4533 | 0.9185 | 1.4533 | 2000 |
| Logistic | thresholding | 0.0002 | 0.0002 | 0.0002 | 2000 |
| Logistic | pulse_to_alert | 0.6316 | 1.0384 | 1.1274 | 2000 |
| Logistic | waveform_sample_to_alert | 0.0708 | 0.1378 | 0.7514 | 19901 |
| Logistic | waveform_pulse_producing_sample | 0.7743 | 1.2053 | undefined | 236 |
| Tree | model_loading | 0.6176 | 0.7306 | undefined | 100 |
| Tree | feature_calculation | 0.1121 | 0.2334 | 0.2888 | 2000 |
| Tree | model_only | 0.1094 | 0.1768 | 0.2567 | 2000 |
| Tree | transforms_and_model | 0.4096 | 0.5426 | 0.7712 | 2000 |
| Tree | thresholding | 0.0002 | 0.0002 | 0.0003 | 2000 |
| Tree | pulse_to_alert | 0.6158 | 1.0521 | 2.3430 | 2000 |
| Tree | waveform_sample_to_alert | 0.0710 | 0.1352 | 0.7308 | 19901 |
| Tree | waveform_pulse_producing_sample | 0.7517 | 1.2753 | undefined | 236 |

Single native thread; warmed perf_counter_ns timings. Model loading measured separately over 100 repetitions; P99 withheld below configured minimum. Pulse components use 2,000 measurements; waveform replay is per incoming sample and separately per confirmed pulse. Synthetic repeat measurements characterize the API path, not physiological distributions. Numeric lookup outside replay timers simulates already-available monitor input. Computational timings exclude acquiring signals, filter phase delay, communication and actuation. Pulse confirmation adds one 125Hz sample; 30 seconds is a label horizon, not guaranteed lead time.
