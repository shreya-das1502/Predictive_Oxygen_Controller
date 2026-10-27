# Frozen warning lead-time analysis

Read [REPORT.md](REPORT.md) for results and [limitations_and_claims.md](limitations_and_claims.md) for claim boundaries. Input and output hashes, parameters, commands, verification status and preservation results are in manifest.json. No human annotations or intervention records were invented.

Primary tables: event_level_lead_times.csv (all22 episodes ×3 families ×2 definitions;132 rows), recording_level_summary.csv (30 recordings ×3 families ×2 modalities ×2 definitions;360 rows), paired_lead_time_summary.csv (6 contrasts), overall_summary.csv, timely_warning_summary.csv, paired_warning_rate_differences.csv. CSV blank means undefined/not applicable, never a missed-event zero lead. Event timestamps are seconds from recording start. Model identifies the classifier family; each event row contains BOTH modalities.

Alarm details: alarm_episodes.csv, plus exposure-weighted duration and burden in recording_level_summary.csv. Figures are warning_lead_time_comparison.png and timely_warning_comparison.png. Bootstrap weights are recording_bootstrap_weights.npy (2000×30, columns record1..30). source_hashes.json and preservation_before.json retain provenance.

Reproduce WITHOUT overwriting this directory, from repository root:

```bash
MPLCONFIGDIR=/tmp/ppg-leadtime-mpl XDG_CACHE_HOME=/tmp/ppg-leadtime-cache /workspace/.venvs/oxygen/bin/python experiments/final_warning_leadtime_20261028T005900_IST/analyze.py --out experiments/final_warning_leadtime_reproduction_NEW
```

The output directory must not already exist. Reproduction recomputes tables/figures/verification from frozen sources and does not fit models. This command does not regenerate narrative reports or the final archival manifest; those are preserved with the original run.

Focused tests:

```bash
MPLCONFIGDIR=/tmp/ppg-leadtime-mpl XDG_CACHE_HOME=/tmp/ppg-leadtime-cache /workspace/.venvs/oxygen/bin/python -m unittest discover -s experiments/final_warning_leadtime_20261028T005900_IST/tests -v
```

Existing focused suite: experiments.incremental_value_20261027.test_study, experiments.final_audit_20261027T215012_IST.tests.test_audit, tests.test_alarm_budget, tests.test_reliability. One older artifact test fails because results/final/future_low/causal_dataset.csv is absent; do not claim the repository suite is green.
