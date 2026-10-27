# PPG pulse-detection quality review

This isolated experiment completed **automated candidate-indicator processing** and prepared an outcome-masked human-review packet. It did not create manual ground truth, refit models, retune cutoffs or change historical features. Start with [REPORT.md](REPORT.md); coordinator procedures are in [BLINDING_PROTOCOL.md](BLINDING_PROTOCOL.md).

**For a future independent reviewer, share only [reviewer_packet.zip](reviewer_packet.zip).** Root reports and `coordinator_only/` contain unblinded information and must not accompany that packet.

## Files

| File/path | Contents |
|---|---|
| [window_review_manifest.csv](window_review_manifest.csv) |134 anonymous windows across30 pseudonymous recordings, randomized order, duration/fs and60-window priority subset; no outcome fields |
| [window_quality_summary.csv](window_quality_summary.csv) |Pure-PPG candidate indicators and accepted/emitted counts; not human quality labels or validated SQI |
| [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md) |Safe PPG-only instructions, filtered-peak reference, uncertainty/partial regions and finalization rules; copy included in packet |
| `reviewer_packet/review.html` |Offline canvas annotation interface: raw/filtered traces, zoom, reference clicks, quality labels, optional detector overlay and JSON export |
| `reviewer_packet/waveforms/`, `figures/`, `detected_pulses.csv` |Anonymous raw/filtered samples, accepted maxima, emitted pulse marks, relative timings and interval sequence |
| [reference_availability.json](reference_availability.json) |Actual availability:0 human-reviewed windows/reviewers; accuracy and corrected-feature effects unavailable |
| [representative_windows.json](representative_windows.json), `figures/` |Low/mixed/high candidate-flag examples, **not** verified good/ambiguous/poor quality categories |
| `coordinator_only/detector_parity.csv` |Full-record accepted/emitted counts and parity of historical pulse times/amplitude/interval/rate/slope |
| `coordinator_only/automated_review_freeze.json` |Frozen source/plan/packet/summary hashes before automated outcome association |
| `coordinator_only/source_window_mapping.csv` |Private source recording/sample bounds and enrichment links; never share with reviewers |
| `coordinator_only/recording_matched_candidate_comparison.csv` |Exploratory exact-case/same-record-control associations after freeze; no validated detector errors |
| `tests/`, `logs/`, `commands.json`, `COMPLETION.json` |Matching/annotation/reproducibility/blinding checks, command outcomes and preservation verification |

`pulse_detection_metrics.csv` is intentionally **not created** because no manual reference annotations exist. `feature_impact_analysis.csv` is likewise unavailable because no verified pulse corrections exist. Empty ground-truth templates are not interpreted as zero error. The required conditional outputs can only be populated after actual reference review, not fabricated to fill filenames.

## Completed reproduction commands

Use the existing pinned research environment from the repository root:

```bash
cd /workspace/Predictive_Oxygen_Controller
source /workspace/.venvs/oxygen/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR=/workspace/.cache/oxygen-mpl
export XDG_CACHE_HOME=/workspace/.cache PYTHONDONTWRITEBYTECODE=1
python -m experiments.pulse_quality_20261027T233919_IST.build --stage build
python -m unittest experiments.pulse_quality_20261027T233919_IST.tests.test_review -v
python -m experiments.pulse_quality_20261027T233919_IST.build --stage unblind
python -m experiments.pulse_quality_20261027T233919_IST.associations
python -m unittest discover -s tests -p test_pipeline.py -v
```

The first build deliberately refuses to overwrite a frozen review. Reproduce into a separately named copy with matching module imports, rather than deleting this freeze or editing historical outputs. Automated association also refuses to replace its frozen analysis plan. Packet integrity is checked before unblinding; source/event linkage stays coordinator-only.

`build.py` observes the unchanged historical detector on full-record causal filtered signals and verifies all emitted base pulse rows against the old cache. `quality.py` is the outcome-free candidate engine. `reference.py` supplies ordered one-to-one pulse matching and annotation validation. `associations.py` runs only post-freeze descriptive association, with no learning.

## Human-review use, not performed yet

Extract only the reviewer ZIP and open `review.html` with its companion files present. Use a fresh browser profile for each independent reviewer. Prioritize the60 marked windows; do not open coordinator reports. Review the guide and export actual finalized JSON when complete. Work is browser-local, with no external requests or server required.

After a real reviewer returns annotations, the coordinator may run the following **future command shape**, replacing placeholders with actual new paths:

```text
python -m experiments.pulse_quality_20261027T233919_IST.evaluate_reference \
  --annotations <actual-final-annotation.json> \
  --out <this-review-directory>/manual_runs/<NEW_RUN>
```

The evaluator rejects draft/blank/unknown/outcome-bearing schema fields, freezes references before metrics, and writes accuracy only for scored reference regions. Repeated `--annotations` inputs allow independently reviewed files with distinct reviewer codes; each reviewer retains separate results. Recording-cluster bootstrap is specified in advance. No reference evaluation was executed on real data in this task, and no manual feature correction/replacement is automatic.

The review instrument is a reproducible preparation tool, not a claim of completed human validation. Its JavaScript syntax and Python handling were checked; interactive browser behavior/inter-rater reliability remain unverified until actual use.
