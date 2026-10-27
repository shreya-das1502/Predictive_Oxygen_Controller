# PPG pulse-detection quality review: completed automated analysis

## Conclusion

The existing detector is reproducible, and an outcome-masked reference-review kit is ready. **Detector accuracy and verified feature errors remain unknown because no manual reference annotations were available.** Automated candidate indicators do not justify changing the detector or blaming missed warnings on poor PPG.

This is an automated signal-processing review, not completed blinded human validation. The coordinator/assistant knew prior results. The packet and quality function were kept free of outcome metadata, and summaries were frozen before exploratory automated outcome associations. Future human reviewers must receive only the packet, not this coordinator report.

## Exact pipeline inspected

The source is exported `PLETH`, sampled at125Hz using the checked row clock. `src/preprocessing.py:WaveformState` uses a forward-only Butterworth prototype order3,0.5–5Hz bandpass, implemented as three SOS/digital order6 with zero state at original record start. Filtering is causal and phase-distorting; it is never restarted at review-window boundaries. No forward–backward filter or centered/future waveform window is introduced.

`PulseDetector.update` recognizes the previous sample as a local maximum when it exceeds its predecessor and is at least as high as the newly arrived sample. Peak-minus-past-trough amplitude must be≥0.03 exported filtered units; accepted peaks are separated by≥0.30s. A peak is confirmed one sample/8ms later. Amplitude uses the maintained past trough, not a future trough.

Accepted peaks and feature emissions differ. The first accepted peak has no paired interval; intervals>2s do not emit a feature row, though the detector updates its last accepted peak/amplitude/trough. Valid paired intervals produce `I`, rate`60/I` and across-pulse amplitude slope`ΔA/I`. `FeatureState` carries up to ten emitted pulses: means over up to5 pulses, sample SD over up to10, and rate change from five pulses earlier. These are pulse counts, not fixed-second feature windows. Singleton variability/change initialization and training-only model imputation remain historical behavior, not a new SQI or rejection rule.

The review instruments this exact detector without altering its state/rules. All30 records' emitted times, peak times, amplitudes, intervals, rates and amplitude slopes agree with `incremental_value_20261027/outputs/all_pulses.csv.gz`. The full-record detector has21,980 accepted maxima and21,948 emitted feature pulses:30 first-unpaired peaks and2>2s interval exclusions explain the difference. The two explicit long-interval exclusions occur in records04 and05, outside the six original eligible-event-bearing recordings. They therefore do not directly explain misses in other records; shorter missed/double detections remain possible.

## Windows and masking

134 windows were processed automatically from all30 recordings:120 randomized time-stratified20s windows (one per quarter per recording), plus14 eligible pre-event20s windows selected irrespective of model success. All windows use relative0–20s axes with2501 samples. Sampling/identifier seeds are preserved coordinator-side. Total window span is2680s; overlap reduces unique sampled span to2623.832s, about43.7 minutes. Windows are not independent observations.

Recording/window pseudonyms and randomized ordering hide source IDs/clocks and selection origin. The reviewer packet contains no saturation trajectories, event labels/onsets, model scores/cutoffs or warning status. Raw/filtered PPG, accepted/emitted locations and candidate indicators are available. Detector marks are off by default in the annotation tool to reduce anchoring; overlay use is recorded.

60 nonoverlapping windows, two per recording across candidate-flag extremes, are prioritized for future reference annotation. None has been manually labeled. These are not an unbiased population-accuracy sample: enriched and flag-stratified selection must be disclosed even after manual evaluation. The remaining windows provide additional coverage rather than assumed ground truth.

The coordinator is outcome-aware, and packet separation is not access-control enforcement. Genuine human blinding has not occurred. A future independent reviewer must not access root/coordinator files and must use a separate browser profile/device from another reviewer.

## Automated candidate findings

The134 windows contain4,147 accepted peak appearances, all emitted under the existing valid-interval rules in these sampled windows. Overlapping windows can repeat physical pulses; this is not a count of independently verified beats.

| Candidate indicator | Windows with indicator | Interpretation |
|---|---:|---|
| No accepted maxima |0|Does not establish absence of missed pulses|
| Accepted intervals<0.30s |0|Expected under the detector's refractory rule; not proof of correct pulse identity|
| Accepted internal gaps>2s |0|The two full-record long intervals fall outside sampled windows|
| Raw missing values |0|Finite samples can still contain artifacts|
| Constant exported segments≥100ms |28|Could reflect flat signal, normal diastole or quantization; not verified dropout|
|≥3 identical samples at window min/max |32|Candidate plateau; true sensor rails/clipping unverified|
| Relative abrupt-difference outliers |65|May include motion/contact steps or steep physiological upstrokes; not verified artifact|
| Window filtered variation<detector amplitude criterion |0|Not a validated perfusion/quality measurement|

The relative jump rule is median absolute first difference plus10×1.4826×MAD; when MAD is zero it is unavailable, not forced positive. Flat100ms and extrema-run criteria are documented exploratory signal indicators. They are not universal quality cutoffs.

Candidate-flag counts are0 in51 windows,1 in51,2 in22 and3 in10. Figures show low/mixed/high flag burden, **not human-verified good/ambiguous/poor categories**. Several plots contain variable/ambiguous morphology worth reference review; this visual observation is not a pulse-error annotation or validation.

## Associations with missed warnings

Pure-PPG summaries, packet artifacts, mapping and plan were hash-frozen before automated unblinding. Exact enriched windows were then compared to time-stratified controls from the same recording with no overlap with any eligible pre30s interval. This avoids treating every overlapping window/event pair as an independent event; repeated events still cluster by recording.

For combined logistic regression, the4 missed events come from2 recordings and the10 detected events from6. Median candidate flags are1 in missed versus2 in detected windows. Median case-minus-same-record-control flag differences are−0.75 versus+0.25. History LR shows the same broad direction (missed median1 versus detected2); PPG-only misses all14, so it has no detected subgroup for discrimination.

Thus these crude candidate flags do **not** support the simple explanation that missed windows consistently have greater flagged burden. They also cannot establish the opposite causal claim or prove signal quality adequate. Controls can differ in time/current physiological state, candidate thresholds are unvalidated, only six event-bearing recordings exist and outcome analyses are exploratory. No p-values, validated quality effect or detector precision/recall are inferred.

Full coordinator tables retain source-record clustering, control counts and window/event linkage. They must not be shared with future blinded reviewers.

## Reference evaluation and feature impact

Actual manually reviewed windows:0. Actual human reviewers:0. Adequate/partial/uninterpretable counts, missed reference pulses, extras, precision/recall, timing errors and inter-rater reliability are **not assessed**, not zero or perfect.

The packet supports reference annotation of plausible filtered systolic maxima corroborated with raw morphology, with uncertainty and exhaustive partial regions. Matching tolerance is fixed at80ms/ten samples, below half the source detector's minimum spacing. Ordered one-to-one matching maximizes count then minimizes absolute timing error; it compares peak timestamps, not8ms-later confirmations. No annotations were fabricated from detector outputs or synthetic test fixtures.

`evaluate_reference.py` is prepared for actual finalized reference files. It freezes references before metrics, keeps uninterpretable regions unavailable, excludes ambiguous-landmark bands, and specifies recording-cluster uncertainty. A second independent reviewer can provide quality-label agreement; none exists here. Bootstrap estimates would be conditional on the reviewed, enriched/flag-selected sample, not automatically all recording time.

No manually verified pulse corrections exist. Therefore `pulse_detection_metrics.csv` and `feature_impact_analysis.csv` are deliberately absent. `reference_availability.json` records this explicitly. The analysis cannot rank which features were most affected by actual detector errors. Algebraically, interval errors would alter rate/variability and pulse additions/removals would alter amplitude/trough and rolling history, but that dependency is not an observed error effect. Corrected-feature replay would need reliable pulse identities and full boundary/trough history; directly substituting clicked peaks into historical inference is not justified.

## Required conclusions

1. **How many windows/recordings?**134 automatically processed windows from30 recordings;60 prioritized for future reference review;0 manually reviewed.
2. **Genuinely blinded?**The reviewer packet is outcome-masked and quality computations are PPG-only. No genuinely blinded human validation occurred; the coordinator/assistant knows prior outcomes.
3. **Manual reference available?**No.
4. **Measured detector errors?**Unknown. Accepted pulse counts/candidate flags are not reference-matched errors.
5. **Features most affected by verified errors?**Unknown; no verified corrections and no corrected-feature comparison.
6. **Quality associated with missed warnings?**Validated quality/error association cannot be assessed. Automated candidates were not consistently worse in missed windows; small clustered descriptive results establish no cause.
7. **What remains unknown?**Actual missed/extra beats, landmark ambiguity, artifact validity, independent-rater agreement, corrected-feature magnitude, how quality affects warnings and patient-level/clinical generalization.
8. **Change the pipeline?**Not on this evidence alone. Preserve detector/features/models. The next step is independent annotation of the prepared subset, then at most one prespecified causal correction if reference evidence identifies a systematic defect, with original event coverage retained and no historical substitution.

## Tests and preservation

18 new tests pass: one-to-one/optimal pulse matching, tolerance, annotation finalization/ambiguity, empty/out-of-region-reference safeguards, pure-PPG flags, reproducible sampling, nonoverlapping priority selection, packet isolation and freeze integrity. The existing35 pipeline tests pass. JavaScript syntax checks pass; full interactive browser acceptance was not performed. No scientific models were fitted, no cutoffs/events changed and no data were downloaded. The full repository suite was not rerun; its previously documented missing historical artifacts are not repaired by this signal review.

All4,437 pre-existing files, including both reference experiments and their outputs, are preserved by fingerprints. Completion metadata and command logs record the final checks. This work establishes a reproducible review process and candidate diagnostics, not clinical validity or improved prediction.
