# Reliability evaluation artifacts

Read the [audit report](../../docs/RELIABILITY_AUDIT.md) for definitions, results, limitations and exact reproduction commands.

- [Final fixed-window results](final/window_summary.csv), [individual event matches](final/window_event_matches.csv), [alarm starts](final/alarm_episodes.csv) and [timestamp/exposure verification](final/timestamp_burden.csv).
- [Final provenance receipt](final/manifest.json), [verification](verification.json), [full test log](test_results.txt) and [historical preservation hashes](preservation.json).
- [Original metric baseline comparison](original_baseline_comparison.csv) and [shared timestamp examples](shared_positive_timestamps.csv).

`verified/` is the completed initial recomputation with its exact evaluator snapshot; `final/` adds separately named off-event and unmatched alarm rates. `audit_v1/` is a partial failed first attempt with only a raw event table; use `final/` for results. Original `results/final`, `results/validated` and `results/hardened` artifacts were not modified.
