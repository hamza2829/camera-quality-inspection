# Actual CI history

These are real hosted runs; failed runs have not been deleted or described as passes.

## First attempt: failed during test setup

[Run 41339919900](https://github.com/hamza2829/camera-quality-inspection/actions/runs/41339919900),
commit `8c12d031706ad672dd0ba14d87582485be55e916`: **31 passed, 11 setup errors**.
No assertion failures were reported, but the pytest step failed and training was
skipped, so this run produced no benchmark metrics.

The configured `--basetemp=artifacts/pytest-tmp` had a missing parent in a fresh
checkout. Pytest creates its base temporary directory without creating missing
parents. Local testing had an existing `artifacts/` directory, hiding that mistake.
The correction in `da9e7fd` uses `--basetemp=.pytest-tmp`, whose parent is the
repository root. Temporary files remain ignored. Tests were not removed or skipped.

An earlier local attempt also encountered Windows shared-temp permissions
(30 passed, 12 setup errors). Moving the temp directory inside the project resolved
that local environment issue; the complete corrected local suite passed 42 tests.

## Corrected benchmark: passed

[Run 41340065785](https://github.com/hamza2829/camera-quality-inspection/actions/runs/41340065785),
commit `da9e7fdd26c736f3b4db880549d8d17b685d3a1e`: **42 passed, 0 failures, 0 errors,
0 skipped**, in 2.433 seconds according to its JUnit XML. The ten-epoch CPU training
step completed; its measured training duration was 7.527 seconds.

The downloaded `synthetic-cpu-evaluation` artifact supplied `evaluation.md`,
`metrics.json`, and the three `examples/*.png` files checked into this docs folder.
The report contains source run/commit provenance. OpenCV precision/recall were
1.0000/0.5778; CNN precision/recall were 1.0000/0.8000. Full matrices and per-type
errors are in the report; no test images were excluded from those metrics.

Subsequent documentation pushes rerun all checks. The checked-in report is a
snapshot of this explicit successful run, rather than a claim that every future
CPU/platform run will have identical floating-point training results.
