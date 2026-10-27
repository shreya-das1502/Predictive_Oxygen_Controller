# Human pulse-reference workflow, revision 2

No human reference annotations have been supplied. Detector precision, recall and timing accuracy are unavailable. Automated waveform review is complete; human validation is pending. Synthetic software tests are not human review.

## Verified original artifacts

Repository-relative original directory: `experiments/pulse_quality_20261027T233919_IST/`.

- `reviewer_packet.zip`: 46,432,493 bytes, 411 archive entries; ZIP CRC and individual source hashes checked. It contains the offline interface and its supporting assets.
- `ANNOTATION_GUIDE.md`: original instructions; identical to the copy inside `reviewer_packet/`.
- `reviewer_packet/review.html`: original interface, loading `review_app.js` and `review_data.js`.
- Packet `PQ595c29cbb70b`: 134 anonymized 20-second windows from 30 pseudonymous recordings. Exactly 60 priority windows, two per recording, with no within-recording overlap.

All original artifacts are retained. SHA256 evidence is in SOURCE_ARTIFACTS.json; preservation_check.json checks every pre-existing fingerprint. The new reviewer ZIP is in this directory. Its waveforms, window IDs, packet ID and priority selection are unchanged. Candidate summaries and premarked detector figures are omitted from this first-pass packet; detector overlay remains optional and exposure is recorded.

## Corrections

Original source saved browser state by packet alone, allowing reviewer reassignment, and allowed incomplete final exports. Revision 2 isolates caches by reviewer, adds backup import, restricts navigation to the fixed priority set and requires all 60 valid reviews for final export. Invalid region text stays visible in local cache and prevents export for that window; correct malformed region text before navigating or exporting. Browser storage is a convenience, not a durable backup.

Server-side validation is authoritative: it rejects drafts, wrong packets, missing/unknown fields, nonfinite/boolean/string timestamps, duplicate JSON keys, duplicate windows/reviewers, unsorted/duplicate pulses, invalid/outside/overlapping/touching regions and outside-window pulses. Partial/ambiguous/uninterpretable reviews require notes. An uninterpretable window is unknown, never a zero-pulse reference. Final export validates clear references within scoring regions after ambiguity exclusions. Imported files are also checked; run Python validation before accepting any file.

## Human steps

1. Extract the NEW `reviewer_packet.zip` completely. Open `reviewer_packet/review.html` in Chrome/Chromium. If file URLs are blocked, from this repository run `python -m http.server 8765 --bind 127.0.0.1 --directory experiments/annotation_workflow_v2_20261028/reviewer_packet` and open http://127.0.0.1:8765/review.html. Stop the server with Ctrl-C afterwards.
2. Read packet `ANNOTATION_GUIDE.md`. Enter a pseudonym such as Reviewer_A and click **Start / switch reviewer**. Use separate browser profiles for independent reviewers. Do not consult detector figures, candidate flags, clinical outcomes or another reviewer's marks.
3. Review all 60 windows via Next/Previous. Inspect raw PPG and mark the corresponding primary systolic maxima on the causal filtered PPG; do not mark dicrotic peaks as separate beats. Zoom with start/span; click clear or ambiguous landmarks, right-click to remove. Time is seconds relative to the displayed window, sampled every 0.008 s, not an absolute recording timestamp. Do not force a reference where the morphology is uncertain.
4. Assign adequate/partial/uninterpretable quality. Exhaustively annotate all pulses in each declared scored region; default is `0.08:19.92`. For partial windows, enter only fully interpretable regions, e.g. `0.08:8;10:19.92`, and explain exclusions. For uninterpretable windows remove regions and clear marks, leave the no-pulses box unchecked, and explain why. Mark ambiguous landmarks and describe uncertainty. Their ±80 ms neighborhoods are excluded from scoring. Only confirm no reference pulses after actually inspecting the entire scored region.
5. Keep the detector overlay off for the initial independent pass. If used later, exposure is retained per window; report this as a blinding limitation. Export drafts frequently. To resume, activate the same reviewer and import that reviewer's backup. Drafts are not evaluable. Keep backup copies outside browser storage.
6. Export final only after all 60 are reviewed. The progress count is labelled windows; final export checks validity. Files download to your browser's configured Downloads location. Copy/upload the final JSON to `experiments/annotation_workflow_v2_20261028/incoming_annotations/Reviewer_A_final.json`. Do not place it in the historical packet. Keep reviewer identities in a separate private roster; notes must contain no identifying data.
7. Run validation, then evaluation commands below. Use a new output name on every run. Return independent reviewer files separately; do not merge or overwrite them. If adjudication is needed, preserve independent originals, document the process and retain a separately identified adjudicated file.

## Required final JSON contract

Top-level fields (all required, no extras):

| Field | Required value/type |
|---|---|
| packet_id | `PQ595c29cbb70b` |
| reviewer_id | Pseudonym: 1–40 ASCII letters, digits, `_` or `-` |
| finalized | JSON boolean `true` |
| windows | Array containing each of the 60 priority IDs exactly once; additional known windows permitted by coordinator validation |

Every window requires exactly these fields:

| Field | Format |
|---|---|
| window_id | Exact anonymized `W…` identifier in supplied manifest; no source recording names |
| quality | `adequately_interpretable`, `partially_interpretable`, or `uninterpretable` |
| scored_regions | Array of numeric `[start,end]` pairs; 0 ≤ start < end ≤ 20; increasing, disjoint and separated (merge touching regions) |
| pulses | Increasing unique objects with exactly numeric `time_s` in [0,20] and `certainty` equal to `clear` or `ambiguous` |
| notes | String; explanation required for partial, uninterpretable or ambiguous review |
| reviewed_no_reference_pulses | Boolean; explicit confirmed empty scored region, never a substitute for missing review |
| detector_overlay_ever_shown | Boolean; retained even after overlay is hidden |

`AnonRecordID` is supplied by the manifest and must not be added to the annotation JSON. No real reference example is provided because none exists. The interface generates this format. Full timestamp/type/semantic validation lives in validation.py.

## Coordinator commands

From `/workspace/Predictive_Oxygen_Controller`:

```bash
/workspace/.venvs/oxygen/bin/python -m experiments.annotation_workflow_v2_20261028.evaluate \
  --annotations experiments/annotation_workflow_v2_20261028/incoming_annotations/Reviewer_A_final.json \
  --validate-only

/workspace/.venvs/oxygen/bin/python -m experiments.annotation_workflow_v2_20261028.evaluate \
  --annotations experiments/annotation_workflow_v2_20261028/incoming_annotations/Reviewer_A_final.json \
  --out experiments/annotation_workflow_v2_20261028/manual_runs/reviewer_A_001
```

Repeat `--annotations` for independent reviewers; duplicate reviewer IDs are rejected. Default requires complete priority coverage. `--allow-partial` is explicitly exploratory and reports incomplete coverage; it must not be presented as the completed 60-window review. No actual evaluation output directory has been created yet.

## Prespecified evaluation and uncertainty

Files are validated and hash-sealed before calculation. Frozen original detector peak times are used, not confirmation or feature-emission times. Within each connected scored region, ordered one-to-one dynamic programming maximizes matches within inclusive 0.080 s tolerance, then minimizes total absolute timing error. It never bridges excluded gaps. Clear references and detections outside scoring regions or within ±80 ms of ambiguous marks are excluded and counted separately. No threshold/model fitting occurs and no event outcomes are joined.

For each reviewer/window/recording, report matches M, scored detections D, scored clear references R, missed references R−M and extras D−M. Precision=M/D and recall=M/R; zero denominators are undefined. Timing error is detection−reference, in seconds, with denominator M; the long timing table contains one row per match. Summaries report signed mean/median, absolute mean/median and absolute 95th percentile. Uninterpretable windows have null accuracy counts, not fabricated misses or true negatives.

Uncertainty uses 2,000 bootstrap draws of entire pseudonymous recordings, seed 2026102703; all their windows and timing pairs travel together. Percentile 95% intervals include defined-replicate counts. Recordings are not assumed to be independent patients; few clusters and selection limit inference. These are conditional intervals for reviewed regions, not design-corrected population intervals.

The 134-window pool consists of 120 quarter-stratified windows plus 14 pre-event enrichment windows. The priority 60 additionally select candidate-flag extremes, two per recording. This enriched selection cannot estimate prevalence or general recording/patient accuracy. Report selected/reviewed/scorable/uninterpretable coverage, excluded duration/marks and overlay exposure. Do not extrapolate pulse matching to oxygen-desaturation prediction or clinical validity.

## Verification

`logs/tests.txt`: seven Python tests covering malformed schema/timestamps/duplicates, 80 ms boundary, one-to-one matching, excluded-gap isolation, unknown reference semantics, explicit denominators and recording bootstrap outputs. All evaluation fixtures are labelled SYNTHETIC and generated in temporary folders, then deleted. `logs/browser_checks.json`: actual Chromium verifies reviewer isolation/resume, incomplete-export rejection, final export with synthetic windows, and real packet loading (134/60); no real landmarks were entered. Browser file navigation was blocked by this environment's policy, so tests used a temporary loopback HTTP server. Run Python tests with `python -m unittest discover -s experiments/annotation_workflow_v2_20261028/tests -v`.
