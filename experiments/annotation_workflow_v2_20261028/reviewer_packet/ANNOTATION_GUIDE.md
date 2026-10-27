# Independent PPG landmark annotation

No reference annotations exist yet. Review only the waveforms; do not consult project reports or other reviewers.

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

