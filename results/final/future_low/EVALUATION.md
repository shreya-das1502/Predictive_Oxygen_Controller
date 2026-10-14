# Target-specific held-out evaluation

{
  "name": "future_low",
  "horizon_s": 30.0,
  "threshold_spo2": 94.0,
  "window": "(t,t+horizon]",
  "complete_valid_future_required": true,
  "maximum_numeric_gap_s": 1.01,
  "current_spo2_eligibility": "unrestricted",
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
| ELM | 0.9649 | 0.2390 | 0.0949 | 0.2559 | 14093 | 161 | 0.9371 | 0.2396 | 140.6868 |
| Logistic | 0.9320 | 0.2196 | 0.0243 | 0.2403 | 15192 | 312 | 0.9266 | 0.2344 | 27.6986 |
| GaussianNB | 0.9821 | 0.2257 | 0.0072 | 0.1963 | 15458 | 82 | 0.9586 | 0.2419 | 13.9864 |
| Tree | 0.8949 | 0.2146 | 0.0353 | 0.1895 | 15021 | 482 | 0.9165 | 0.2215 | 28.5213 |

| Model | Recall | Precision | Specificity | PR_AUC_AP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| ELM | 0.9649 | 0.2390 | 0.0949 | 0.2559 | 14093 | 161 |
| Logistic | 0.9320 | 0.2196 | 0.0243 | 0.2403 | 15192 | 312 |
| GaussianNB | 0.9821 | 0.2257 | 0.0072 | 0.1963 | 15458 | 82 |
| Tree | 0.8949 | 0.2146 | 0.0353 | 0.1895 | 15021 | 482 |

The second table uses frozen historical future-low models on the SAME eligible rows, relabelled for this target. It is a reference, not a prior new-onset-trained experiment. Primary future-low includes ongoing lows; onset requires current observed normal saturation. Targets and populations are not interchangeable.

| Model | Subset | Eligible_Events | Detected | Sensitivity | Median_Lead_Time_s | Excluded_Events |
| --- | --- | --- | --- | --- | --- | --- |
| ELM | all_qualifying_onsets | 14 | 14 | 1.0000 | 29.6840 | 8 |
| ELM | 91_92_episode_subset | 3 | 3 | 1.0000 | 9.7840 | 4 |
| Logistic | all_qualifying_onsets | 14 | 14 | 1.0000 | 29.6840 | 8 |
| Logistic | 91_92_episode_subset | 3 | 3 | 1.0000 | 9.7840 | 4 |
| GaussianNB | all_qualifying_onsets | 14 | 14 | 1.0000 | 29.6840 | 8 |
| GaussianNB | 91_92_episode_subset | 3 | 3 | 1.0000 | 9.7840 | 4 |
| Tree | all_qualifying_onsets | 14 | 14 | 1.0000 | 29.6840 | 8 |
| Tree | 91_92_episode_subset | 3 | 3 | 1.0000 | 9.7840 | 4 |

Events require one observed low reading; persistence is not presumed. Missing readings/gaps break episodes; left-censored onsets are excluded. A full valid raw lead-in plus at least one eligible prediction opportunity is required; opportunity coverage is saved. Alerts in [onset-horizon,onset) count. Nearby events can share an alert. The 91–92 subset refers to actual readings within the episode, not exact SpO2 prediction. Always-alert would detect every eligible event; event sensitivity must be read with FP burden.

False alarm episodes use the existing definition: positive runs split by negative decisions or >configured alert gap, false if the run contains no positive target. Evaluable exposure sums adjacent eligible pulse intervals <=that gap. It excludes current-low periods for onset; it is not total recording time. A sustained alarm can yield few episode starts yet high false-positive duty. No suppression policy has been tuned.

All-positive and short eligible exposures are explicitly flagged in positive_recording_detail.csv; they are retained, but do not establish within-record discrimination. Few independent positive recordings, unverified patient identity/monitor delay and one-second numeric precision limit inference. This is retrospective recording-wise CV, not an untouched external test set. Model parameters remain fixed from the earlier run.

| Model | Component | Median_ms | P95_ms | P99_ms | Repetitions |
| --- | --- | --- | --- | --- | --- |
| ELM | model_loading | 0.8501 | 3.2040 | undefined | 100 |
| ELM | feature_calculation | 0.2304 | 0.3053 | 2.3128 | 2000 |
| ELM | model_only | 0.0094 | 0.0112 | 0.0281 | 2000 |
| ELM | transforms_and_model | 0.5461 | 3.5552 | 5.4721 | 2000 |
| ELM | thresholding | 0.0003 | 0.0004 | 0.0004 | 2000 |
| ELM | pulse_to_alert | 0.9032 | 3.8278 | 6.0531 | 2000 |
| ELM | waveform_sample_to_alert | 0.1334 | 0.2108 | 2.2288 | 19901 |
| ELM | waveform_pulse_producing_sample | 1.2866 | 5.0534 | undefined | 236 |
| GaussianNB | model_loading | 0.5011 | 0.5694 | undefined | 100 |
| GaussianNB | feature_calculation | 0.1129 | 0.1581 | 0.1927 | 2000 |
| GaussianNB | model_only | 0.2093 | 0.3838 | 0.4517 | 2000 |
| GaussianNB | transforms_and_model | 0.5314 | 1.0006 | 1.4531 | 2000 |
| GaussianNB | thresholding | 0.0002 | 0.0002 | 0.0002 | 2000 |
| GaussianNB | pulse_to_alert | 0.7025 | 0.9991 | 1.2391 | 2000 |
| GaussianNB | waveform_sample_to_alert | 0.0696 | 0.1478 | 0.9736 | 19901 |
| GaussianNB | waveform_pulse_producing_sample | 1.0263 | 1.9343 | undefined | 236 |
| Logistic | model_loading | 0.4999 | 0.6005 | undefined | 100 |
| Logistic | feature_calculation | 0.1134 | 0.2515 | 0.3202 | 2000 |
| Logistic | model_only | 0.1319 | 0.2641 | 0.4424 | 2000 |
| Logistic | transforms_and_model | 0.4383 | 0.7768 | 0.9636 | 2000 |
| Logistic | thresholding | 0.0002 | 0.0002 | 0.0002 | 2000 |
| Logistic | pulse_to_alert | 0.6119 | 1.0272 | 1.1413 | 2000 |
| Logistic | waveform_sample_to_alert | 0.0698 | 0.1298 | 0.8093 | 19901 |
| Logistic | waveform_pulse_producing_sample | 0.8572 | 1.1735 | undefined | 236 |
| Tree | model_loading | 0.6159 | 0.7009 | undefined | 100 |
| Tree | feature_calculation | 0.1121 | 0.2030 | 0.2622 | 2000 |
| Tree | model_only | 0.1087 | 0.1954 | 0.2335 | 2000 |
| Tree | transforms_and_model | 0.4177 | 0.7411 | 0.8507 | 2000 |
| Tree | thresholding | 0.0002 | 0.0002 | 0.0002 | 2000 |
| Tree | pulse_to_alert | 0.5959 | 1.0461 | 1.4154 | 2000 |
| Tree | waveform_sample_to_alert | 0.0695 | 0.1462 | 0.7687 | 19901 |
| Tree | waveform_pulse_producing_sample | 0.8044 | 1.5365 | undefined | 236 |

Single native thread; warmed perf_counter_ns timings. Model loading measured separately over 100 repetitions; P99 withheld below configured minimum. Pulse components use 2,000 measurements; waveform replay is per incoming sample and separately per confirmed pulse. Synthetic repeat measurements characterize the API path, not physiological distributions. Numeric lookup outside replay timers simulates already-available monitor input. Computational timings exclude acquiring signals, filter phase delay, communication and actuation. Pulse confirmation adds one 125Hz sample; 30 seconds is a label horizon, not guaranteed lead time.
