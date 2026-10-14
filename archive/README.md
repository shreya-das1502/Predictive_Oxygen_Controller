# Historical source only

`legacy_sources.zip` preserves 26 superseded experiment/diagnostic/demo source files plus the previous README at their original paths. `legacy_sources.json` records exact SHA-256 hashes. The source archive is retained because historical outputs and user edits must remain explainable. It is not an alternative maintained implementation. See ../docs/FILE_CLEANUP.csv and ../docs/CLEANUP_DEPENDENCIES.json for dependency decisions. Active models, labels, features and reports live in ../src/.

To inspect or restore without overwriting the working project:

```bash
python archive/restore_legacy.py --output /tmp/oxygen-legacy-inspection
```

Use a new destination each time. `--with-data` additionally copies local raw data and historical results; it never links them to a destination where legacy scripts could overwrite the working evidence. Run restored historical scripts from that destination only. Their row splits, lookahead and target limitations remain unchanged. Current evaluation is reproduced with the commands in ../docs/FULL_PIPELINE_GUIDE.md, not these historical scripts.

Restoration is tested for exact source recovery and safe destination handling. It is not certification that every retired command executes against today's dependencies: the superseded reporting module, in particular, originally depended on the matching active core version. Use the appropriate historical Git checkpoint for that complete old environment. Historical numeric outputs remain in results/ for comparison; the maintained reproduction route is the current guide.

`legacy_results.zip` and its SHA-256 JSON index retain historical-only root outputs formerly consumed by the archived scripts. Three root CSVs remain directly available (original derived data and the original v2 metrics/predictions). Two redundant checkpoint CSVs were deleted only after exact hash equality with those retained v2 CSVs was verified. `--with-data` restores the archived outputs into the isolated destination as well.
