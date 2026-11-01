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

## YOLO/C++ extension: passed

[Run 41341879887](https://github.com/hamza2829/camera-quality-inspection/actions/runs/41341879887),
commit `9f08c31be0328396f3b233efe0bc4c6391f5b296`: **55 Python tests passed**, with
0 failures, 0 errors, 0 skipped (3.918 seconds in JUnit), and **8/8 Google Tests
passed**. The Release CMake build, CPU training/evaluation, parity/benchmark,
schema-v2 mock replay, and artifact upload succeeded. Full job duration: 2m8s.
No new failed CI runs occurred before this extension snapshot.

YOLOv8n downloaded COCO-pretrained weights and completed five CPU epochs in
26.917 seconds. Held-out mAP50 was 0.8737724584; fixed-confidence class-aware
precision/recall were scratch 0.8889/0.8000, dent 0.9000/0.6000, hole 1.0000/1.0000.
74/77 boxes overlapped ground truth at IoU >=0.5, matching 72 unique defects.

The CNN retraining in this run had overall recall 0.7667 and dent recall 0.6000,
compared with the original snapshot's 0.8000 and 0.7000. YOLO dent image recall
0.6000 is lower than that original CNN baseline, and tied with this run's CNN.
The original PNG/CSV hash remains identical; floating-point training/validation
selection can vary across builds and hardware. No seed, split, or threshold was
tuned on test results to improve this comparison.

C++ stage parity on all 180 test images had 0 mask-pixel differences and 0 box
differences beyond one pixel. Original full-inspector boxes differed on 37 images.
Mean timings: OpenCV stage 0.293083 ms/image; C++ compute 1.527209 ms/image;
C++ subprocess/PGM/JSON/parse 4.645394 ms/image. The C++ tool was slower.

The actual artifact supplied `extended-metrics.json`, `extended-evaluation.md`,
and `yolo-examples/*.png`. Existing baseline snapshots remain intact. Documentation
adds a labeled historical-comparison note to the generated extension report.
