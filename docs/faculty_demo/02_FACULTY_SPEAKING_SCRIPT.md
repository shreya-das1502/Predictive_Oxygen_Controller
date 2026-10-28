# Faculty speaking script

Use this alongside the uploaded20-slide presentation. Speak in your own words. The purpose is a convincing explanation of the actual evidence, not a claim that the hypothesis succeeded. Code and image locations are in the companion guide.

## Slides1–4: problem statement and PID scope

“My project is Towards Predictive PID-Based Oxygen Control. The overall motivation is to investigate whether a feedback-control framework could benefit from advance notice of desaturation.

PID responds to the difference between a specified target and the measured output. Prediction has a different role: estimating a future event. Before connecting prediction to control, I need to establish whether the extra information produces useful warnings.

The specific problem I implemented is: Does adding PPG to historical SpO₂ provide earlier useful warnings than historical SpO₂ alone? I assess three requirements together: event detection, sufficient pre-onset lead, and manageable alarm burden. High timestamp accuracy alone cannot establish a useful warning system.

The completed work is the predictive subsystem and its retrospective evaluation. PID integration is proposed. I have not demonstrated closed-loop oxygen regulation or faster physical oxygen delivery.”

## Slides5–6: approach and why it matters

“I use a matched-input ablation study: history-only, PPG-only and combined inputs. SpO₂ history is a necessary strong baseline. Without it, I might attribute predictive information to PPG that is actually coming from saturation history.

I compare fixed logistic regression, Gaussian naïve Bayes and ELM models under the same recording framework. The final lead-time analysis uses existing saved predictions and thresholds. I do not tune the test results to obtain a favorable finding.

The contribution is not simply fitting a classifier. It is testing whether an additional input changes operational warning performance.”

## Slide7: dataset and data types

“I use30 existing local BIDMC recordings of approximately8min each, around4h total. PPG is a125Hz waveform; numeric SpO₂ is approximately1Hz. These are synchronized physiological time series. The classifiers receive extracted, time-indexed tabular feature vectors, not the entire raw waveform directly.

There are20150 common feature rows, including4587 positive future-low labels. These correlated rows are not independent events or patients. Recording IDs are not verified independent patient IDs.

At a decision time t, the label asks whether any valid SpO₂ observation after t up to t+30s is≤94%. Missing or incomplete reference windows are unknown, not negative. Future observations create labels, not inference inputs. This is an observed monitor threshold, not adjudicated arterial oxygen truth or a prescribed treatment target.

The training target includes ongoing low periods. To evaluate advance warning, I separately use current-normal opportunities before event onset. This distinction prevents strong recognition of existing lows from being presented as anticipation of new events.

There are22 observed episodes:7 left-censored/unknown-prior episodes,15 qualifying onsets, and one qualifying onset with insufficient30s lead-in. That leaves14 eligible original events across6 recordings. A sustained definition requires≥3 consecutive low readings spanning≥2s and leaves13 eligible events across5 recordings.”

## Slides8–9: pipeline and signal processing

“The pipeline is validated timestamps, causal PPG filtering, pulse extraction, rolling features, past-only SpO₂ history, classifier scores, fixed thresholds, actual alarm episodes, then event matching and evaluation.

The PPG filter is a causal0.5–5Hz Butterworth band-pass. Its prototype order is3 and digital band-pass order6. I do not use future samples through zero-phase filtering. A local peak is confirmed by the next sample. At125Hz that is8ms after the peak; it is not a claim about total sensor/filter/processing latency.

Amplitude is measured relative to a past trough. Accepted peak spacing and amplitude criteria are fixed historical settings. Pulse interval is the difference between accepted peak times; rate is60 divided by interval in seconds. Rolling PPG features use recent pulse counts, not fixed-second windows.

There are10 PPG features: amplitude, interval, rate, amplitude slope and rolling rate/amplitude/interval summaries. The7 SpO₂ features are current saturation,10/30s means,10/30s trends,30s sample standard deviation and age of the latest numeric reading. Combined uses17 features.

The30s horizon defines the target interval, not a guarantee of30s warning. Complete future coverage defines retrospective evaluability; it is not a deployable future-aware gate.”

## Slide10: what the models do and why

“Logistic regression forms a weighted sum of standardized features and applies a sigmoid. It is a transparent linear baseline, with fixed regularization and balanced class weights.

Gaussian naïve Bayes estimates class likelihoods using Gaussian feature distributions and class priors. It provides a simple contrasting model. Its conditional independence assumption is imperfect for correlated pulse features, but that is not a proven causal explanation of its failures.

ELM uses a fixed random nonlinear tanh hidden layer and learns a weighted ridge output. It does not train all hidden weights by backpropagation. It uses50 units, ridge1, positive weight5 and seed42. It supplies a nonlinear comparison, not a guaranteed better model or calibrated risk.

Median imputation and standardization are fitted on the appropriate training partition only. Logistic/GaussianNB probability-shaped outputs are not claimed to be calibrated clinical risks; ELM produces a score.”

## Slide11: fair validation and warning logic

“I hold out one entire recording. On the other recordings, grouped3-fold inner validation produces out-of-fold training scores. Threshold selection uses these training scores to maximize one-to-one event detections subject to≤5 off-event alarm starts/evaluable hour and≤5% active time. The threshold is fixed before held-out scoring.

Randomly splitting pulse rows would mix related observations across train and test. Recording grouping reduces that risk, although it does not establish patient-independent generalization.

Actual alarm episodes are formed from connected positive decisions. Chronological eligible events match the earliest unused actual episode start in[onset−30,onset). There is no alarm reuse and no artificial restart at an event-window edge.

Lead equals onset minus warning timestamp. Additional lead equals combined minus history lead, only where both detect. Misses remain undefined, not zero. Existing matching does not assess post-onset response latency.

Uncertainty uses2000 whole-recording bootstrap draws with shared multiplicities, seed20261027. Events within a recording travel together. These intervals condition on saved overlapping cross-validation fits, not refitting or verified independent patients.”

## Slides12–14: did results improve?

“The result is mixed and does not establish robust incremental PPG warning value.

For original events, logistic detection rises9/14 to10/14, but sustained detection is tied9/13. GaussianNB falls10/14 to6/14 and ELM6/14 to1/14. PPG-only detects0/14 in the preceding matched-input study under this policy; that does not mean PPG can never help.

Among7 shared logistic detections in4 recordings, combined is earlier for2, unchanged for3 and later for2. Median additional lead is0s, with95% recording-bootstrap interval−4 to+10.064s. Mean additional lead is+3.808s, with an interval also crossing zero. The paired subset excludes misses/disagreements, so it cannot establish overall superiority.

At the10s requirement, both logistic inputs cover6/14 original events. For sustained truth, history covers6/13 and combined5/13. Original missed-or-insufficient warnings remain8/14 each; sustained missed-or-insufficient increases7/13 to8/13. The small detection gain is therefore not a consistent useful-warning gain.”

## Slide15: alarm tradeoffs

“Original logistic off-event starts increase3.485 to4.879 per evaluable hour, while active duration decreases140.248 to98.568s. Alarm starts increase29 to33. This is more starts but less active time, without better original10s coverage.

The rate denominator is10329.032s, approximately2.869176 evaluable hours, not all4 source hours.26 recordings contribute exposure;4 have none. Both pooled original conditions pass the research budgets but violate constraints on individual recordings. Start-budget violations occur in3 records each; activity-cap violations in4 history and3 combined records. These counts overlap. Record29 has only6.264s exposure and100% activity in both. Combined sustained false starts rise to5.577/h and fail the pooled start budget. Research-budget adherence is not proof of safety.”

## Slide16: show a gain and a loss

“For record26/onset310, history warns2.368s early and combined24.312s early: a genuine observed21.944s improvement. But record30/onset72 has17.136s warning from history and no qualifying warning from combined.

Overall logistic gains3 original events and loses2. It is not the same9 detections plus1. Gains and losses must remain visible. Without human pulse references, I cannot attribute failures to detector error, poor signal quality or absent physiology merely from model scores.”

## Slides17–18: contribution and future scope

“The demonstrated contribution is a reproducible causal-feature and matched-input evaluation framework, with one-to-one event matching, useful-lead requirements, burden and recording-level uncertainty. The bounded finding is that PPG’s operational increment is not established here; it is not a new classifier or a globally novel PID algorithm.

Actual human pulse annotations and clinical review of event validity are important next steps. The60-window human-review workflow exists, but no human reference annotations have been supplied. Detector precision/recall is not established.

For the PID extension, I would define a justified plant/actuator model, discrete controller, constraints, anti-windup and derivative filtering, then specify exactly how prediction influences supervision. I would compare conventional PID, history-assisted PID and combined-assisted PID under matched disturbances, delays and constraints. Outcomes include tracking error, threshold-excursion duration/severity, control effort and saturation. Simulation evidence would not establish clinical benefit.

Six new focused tests passed, and12 analysis outputs reproduced byte-identically. Of67 relevant existing tests,66 passed and1 errored because an older historical artifact is missing. I do not claim the full suite passed. No models or thresholds changed in the final analysis.

My conclusion is: validate the predictive information first, then test whether it improves closed-loop control.”

## Short answers for questions

**Why30s/94%/10s?** Fixed research definitions/benchmark,not universally optimal parameters,clinical targets or established oxygen-response requirements.

**Why not random row splits?** Related pulse histories/events would appear in both partitions;grouped recordings reduce leakage.

**Why not deep learning?** More flexibility does not resolve few events,unknown patient linkage or imperfect references. First test added information against a strong baseline.

**Did PPG help?** Some individual/logistic original detections improve,but no robust overall≥10s timing gain;sustained gain disappears;other families lose coverage.

**Where is PID?** Proposed extension. `src/controller.py` is a research-alert wrapper,not PID and not17featurecombinedruntime.

**What is novel?** Auditable integration and bounded project-specific empirical evidence;no first-ever or patent claim. A stronger control contribution requires actual integration and closed-loop comparison.

**Can this be published?** Possibly as a carefully scoped retrospective/negative-results or methods study,subject to reference quality,validation,journal fit and review. Publication is not guaranteed.

**Did oxygen delivery become faster?** No timestamped intervention/actuator/flow evidence exists for that claim. Prediction lead,alarm response,physical delivery and clinical outcome are different quantities.

## Code demonstration: six stops

Open the companion line guide for exact ranges. Show labels→causal filter→past-only history→grouped training/threshold timing→one-to-one matching→saved results. Explain the purpose before reading individual lines. Do not run training during the demonstration.
