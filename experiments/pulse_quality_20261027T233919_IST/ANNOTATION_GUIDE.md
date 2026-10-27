# PPG reference annotation guide — reviewer-facing

Open `review.html` locally with the other packet files in the same folder. This tool works offline. Use a nonidentifying reviewer code. Inspect only this packet; do not consult other project files or try to reconstruct source identities. Windows are randomized and recording codes are pseudonyms, not patient identifiers.

The60 windows marked with an asterisk are the prespecified priority subset. Review additional windows only if time permits. The asterisk is not a quality assessment. No reference annotation has been supplied in advance. Do not force a landmark to complete a window.

## First pass: reference without detector anchoring

Keep **Show algorithm detections** off while locating reference pulses. Inspect raw and causal filtered morphology together, zooming with view start/span. Mark one plausible systolic pulse per cycle at its corresponding maximum on the filtered waveform. Do not mark every small ringing/dicrotic maximum as a new pulse. The exact source detector is not the reference standard.

Click the filtered trace to add a landmark; times snap to the125Hz grid (8ms). Select clear or ambiguous certainty. Right-click near a marker to remove it. The automated overlay is optional for a second pass and its use is recorded; acknowledge potential anchoring if it was viewed before references were finalized.

The reference landmark is on the **filtered** trace to avoid confusing filter phase delay with detector error. It is not an ECG R peak or a validated arterial timing measurement. A source peak is confirmed one sample later; evaluation compares peak times, not confirmation times.

## Interpretability and exhaustive regions

- **Adequately interpretable:** plausible pulse landmarks can be exhaustively annotated over the stated scoring region.
- **Partially interpretable:** provide only regions that can be exhaustively annotated; explain excluded/ambiguous portions.
- **Uninterpretable:** no reliable region; leave scored regions empty. This is not an automatic detector failure.

Default scoring region is0.08–19.92s to reduce edge ambiguity. Regions must be ordered and nonoverlapping, such as `0.08:8;10:19.92`. Mark ambiguous landmarks separately; reference evaluation excludes an80ms band around them rather than treating uncertainty as an extra/missed pulse. Notes should describe morphology/ambiguity, not speculate about unavailable outcomes.

An empty annotation is not a zero-pulse ground truth. If an exhaustive region truly contains no reference pulses, explicitly confirm that judgment. Otherwise mark it uninterpretable or return a draft. Do not use unrelated signals to resolve ambiguity.

## Finalize and return

Export draft JSON while work is incomplete. Export final annotations only when each included window's labels, regions, references and notes are settled. Unreviewed windows are omitted, not silently assigned a quality category. Final export records reviewer code and any detector-overlay exposure.

Return only the annotation JSON to the coordinator. The coordinator must freeze its hash and scoring plan before joining any additional metadata. Do not revise references after receiving that metadata without creating a new explicitly unblinded revision. A second independent reviewer must work separately; one reviewer's labels do not establish inter-rater reliability.

The planned matching tolerance is80ms (ten125Hz samples), smaller than half the detector's300ms minimum accepted interval. It is a fixed annotation-comparison tolerance, not a validated physiological or clinical quality limit. Candidate flags and detector overlays are not validated signal-quality measurements.
