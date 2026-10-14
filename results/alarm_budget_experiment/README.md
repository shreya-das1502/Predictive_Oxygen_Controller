# Final planned alarm-budget experiment

Read [ALARM_BUDGET_EXPERIMENT.md](../../docs/ALARM_BUDGET_EXPERIMENT.md) for methods, negative results, actual commands and limitations.

- [Frozen protocol](protocol.json) and [Phase A audit](AUDIT.md).
- [Completed run02 summary](run02/summary.csv), [per-record results](run02/per_recording.csv), [event-level matches](run02/event_matches.csv), [training-selected thresholds](run02/selected_thresholds.csv) and [provenance](run02/manifest.json).
- [Verification](verification.json), [test log](test_results.txt), [protected historical fingerprints](preservation.json).

run01 contains the frozen training selections and stopped before producing outer result artifacts due to empty-schema handling. run02 revalidated and reused those selections unchanged, then completed evaluation. Use run02 for final results. No existing result directory was overwritten.
