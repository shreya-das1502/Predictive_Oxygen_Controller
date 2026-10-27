# Blinding, sampling and unblinding protocol — coordinator-facing

**Do not distribute this file, the root report or `coordinator_only/` to reference reviewers. Share only `reviewer_packet.zip`.** The coordinator/assistant knows the previous study's outcomes. No blinded human review has been completed in this task; the completed review is automated PPG-only processing.

## Sampling and pseudonymization

The30 existing recordings remain unchanged. Sampling seed2026102701 selects one20-second interval wholly inside each120-second quarter of each480-second recording:120 time-stratified windows. Fourteen additional windows cover the20 seconds ending one125Hz sample before each eligible original onset, regardless of whether any warning was detected. Case enrichment therefore includes both detected and missed outcomes, not just failures. The14-case selection uses the already frozen original eligibility rule, not a new label or tuned model criterion.

Identifier seed2026102702 independently assigns randomly permuted recording codes and unique random window codes, then randomizes review order. `coordinator_only/source_window_mapping.csv` retains source IDs, absolute sample bounds, quarter/enrichment origin and event links. These fields and seeds are absent from the reviewer packet. The mapping is a blinding aid, not a guarantee that public waveforms cannot be reidentified. Recording pseudonyms do not establish unique patients.

All134 windows have20-second relative axes and2501 inclusive samples at125Hz. The window counts sum to2680 seconds; overlap reduces unique source span to2623.832 seconds. Windows from a recording are dependent. Full-record detector instrumentation covers all4 source hours, but detailed candidate quality is sampled rather than a claim about every second.

## Outcome-free processing

The quality function accepts only raw/filtered PPG, accepted peak times and sampling frequency. It receives no saturation, outcomes, prediction scores, model cutoffs or warning status. Existing historical pulse fields are read solely to verify detector parity. Full-record causal filter state is retained, rather than restarting at each review interval. Only detections confirmed by the end of the displayed interval are included.

The candidate rules,80ms reference tolerance, partial-region/ambiguity handling, sampling and subsequent analysis plan are specified in `protocol.json`. They are engineering diagnostics, not validated SQI or clinical thresholds. Source, plan, mapping and every reviewer asset are hash-sealed in `coordinator_only/automated_review_freeze.json` before outcome joins.

Two windows per recording are prioritized for manual review: lowest and highest candidate-flag count, deterministic window-ID ties, with the second required not to overlap the first. This gives60 windows across30 recordings. The packet marks priority membership but does not reveal the direction of the flag ranking or enrichment origin. Selection is conditional on automated indicators: any future accuracy estimate on this subset is not an unbiased estimate over all recording time.

## What a reviewer receives

Only the ZIP's anonymous manifest, PPG waveforms, accepted/emitted pulse locations, candidate indicators, pure-PPG figures, offline annotation tool and annotation guide. No source IDs/clocks, event labels/onsets, saturation trajectories, warning status, models or scores are present. Automated tests inspect columns, embedded waveform data and archive paths for prohibited metadata.

The annotation interface initially hides detector marks to reduce anchoring, records whether its overlay was shown, and stores draft work locally. Use separate browser profiles/devices for independent reviewers: local state is per packet, and a shared profile could expose earlier annotations. Exported reviewer codes alone do not prove reviewer independence. The UI's JavaScript syntax was checked; a complete browser-interaction acceptance test was not available.

Packet masking is enforced by which files are shared, not by filesystem access control. A reviewer who accesses the coordinator directory or root reports would be unblinded. The coordinator is already outcome-aware; this cannot be described as a completed genuinely blinded human validation.

## Automated unblinding performed

After source/packet/summary hashes were frozen and17 tests passed, the automated PPG summary was joined to event metadata in `coordinator_only/`. No manual annotations were present. These are **post-freeze automated candidate associations**, not unblinded human reference results. The reviewer packet remains unchanged and outcome-masked.

The within-record association plan uses the14 exact enriched intervals as cases. Controls are time-stratified windows from the same recording with zero overlap with any eligible30-second pre-event interval. Candidate count is compared to the median of available same-record controls. Multiple events still cluster within six recordings; state/time differences are not fully controlled. No significance or causal claim is made.

## Future manual freeze and release

An independent reviewer supplies actual landmarks and adequate/partial/uninterpretable labels using only PPG. Never turn empty templates into reference pulses. The coordinator validates and hash-seals final JSON, the80ms matching rule and scoring plan **before** any manual outcome association. `evaluate_reference.py` creates a NEW `manual_runs/` directory only for actual finalized input; it does not join event data.

Reference matching maximizes one-to-one count, then minimizes absolute filtered-peak timing error. Only exhaustively annotated regions are scored; ambiguity bands are excluded. Uninterpretable windows are counted as unavailable, not detector failures or perfect performance. Reference precision/recall use pulse denominators and recording-cluster uncertainty, not individual samples as independent observations. Each reviewer is evaluated separately; quality-label agreement is available only with at least two independent inputs.

No manual-reference unblinding or feature-correction experiment was possible here. Future revisions after exposure to outcomes must be labeled unblinded. Any diagnostic correction requires adequate reference confidence, causal full-history/trough handling and coverage reporting; it must never replace historical features/predictions or alter thresholds.
