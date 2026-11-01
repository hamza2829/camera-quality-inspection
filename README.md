# camera-quality-inspection

[![CPU inspection checks](https://github.com/hamza2829/camera-quality-inspection/actions/workflows/ci.yml/badge.svg)](https://github.com/hamza2829/camera-quality-inspection/actions/workflows/ci.yml)

## What it is

A small Python/C++ example of a machine-to-cloud shaped inspection pipeline:
synthetic camera frames → OpenCV, a tiny CNN, and YOLOv8n → JSON/MQTT adapter →
validated SQLite records. **All fine-tuning and evaluation images are generated
by this code. No real machine data, real camera, or deployed cloud service.**
YOLO optionally starts from external COCO-pretrained weights; that distinction is
recorded, not hidden under the synthetic-data description. No Docker is needed.

## Results (real numbers from CI)

The extension's hosted benchmark has not run yet. No YOLO accuracy or C++ speed
numbers are claimed in this source commit; they will be copied from a real CI
artifact after the workflow completes. The original OpenCV/CNN report remains
available in [docs/evaluation.md](docs/evaluation.md), with its actual source run.
[CI history](docs/ci-history.md) preserves failed runs as well as successful ones.

## How it works

1. **Same data:** 96×96 grayscale metal-like textures, seed `20261101`, independent
   600/180/180 train/validation/test streams, 50% clean and 1/6 each scratch, dent,
   hole. Each defective image has one simulated defect. Noise and illumination
   vary. The original PNGs, IDs, CSV fields, seeds, and splits are preserved; a
   regression test checks the original labels.csv hash, which includes image IDs
   derived from pixels.
2. **YOLO labels:** every PNG now has a sibling `.txt`: class ID plus normalized
   center x/y/width/height, or an empty file for clean frames. IDs are scratch=0,
   dent=1, hole=2. `yolo.yaml` and split image lists sit next to `labels.csv`.
   For dents the truth box encloses mask values above 15% of peak, not a physical edge.
3. **Original methods:** OpenCV still uses median denoising, uint8 Gaussian
   background removal, thresholding, morphology and contours. Validation F1
   selects its settings. The 17,409-parameter CNN still trains ten CPU epochs;
   validation loss selects the checkpoint and validation F1 its threshold.
4. **YOLO:** pinned Ultralytics YOLOv8n, five CPU epochs, 96-pixel input, batch 16,
   no GPU, zero data-loader workers. It first attempts the official COCO weight
   download. If successful, the first ten backbone layers are frozen for this
   short fine-tuning run. If unavailable/unloadable, it builds from YAML and trains
   all layers from scratch. Both initialization and download errors/hashes are
   recorded. The test set does not select epochs, confidence, or other settings.
   Detection P/R uses confidence 0.25 and class-correct IoU >=0.5, one-to-one matches.
   mAP50 sweeps confidence; classification precision/recall is reported separately.
5. **C++17:** a CMake library and PGM-to-JSON CLI implement median denoising,
   float64 Gaussian background removal, residual thresholding and handwritten
   eight-connected BFS, with no OpenCV dependency in C++. The stage has an explicit
   Python/OpenCV reference. It omits the original inspector's morphology and
   contour-area rules, so full-inspector box differences are separately reported.
   [C++ details](cpp/README.md) define borders, precision, area and timing boundaries.
6. **Version 2 pipeline:** messages include actual YOLO classes, pixel boxes,
   confidence, initialization, and evaluated-weight hash. A missing YOLO run is
   explicitly `not_run`, never a fake pass. The combined rule fails if any evaluated
   method fails. The consumer validates consistency/bounds and stores detector
   boxes in `inspections_v2`, plus canonical JSON and a payload hash. Exact replay
   by image ID is a no-op; conflicting content is rejected. v1 payloads are rejected
   and old `inspections` tables are left intact. See [the contract](docs/pipeline.md).

The MQTT simulator publishes results/metadata, not image bytes or ground-truth
labels, and uses fixed simulated timestamps. The paho adapter uses QoS 1 and
commit-before-ack; CI uses an in-process transport double and mocked callbacks.
Invalid/conflicting messages are logged/discarded. SQLite errors disconnect
without ack; there is no automated recovery or durable producer outbox.

CI builds Release C++, runs Google Test and pytest (including generated-image
parity), trains/evaluates the CNN and YOLO on CPU, measures the C++ stage on the same
180 test images, replays results twice into SQLite, and uploads all metrics/data/
checkpoints/examples. The benchmark step has a six-minute time limit. Green CI
means execution/tests succeeded, not that an accuracy or real-time target was met.

## What is NOT done

- **Synthetic fine-tuning/evaluation data only:** no real factory images, camera,
  acquisition timing, camera calibration, or physical-unit measurements.
- **No GPU**; no real-time or production performance claim. A few YOLO epochs can
  underfit; a detector can perform worse than a binary classifier, especially on
  subtle scratches/dents. The report states actual per-type disadvantages.
- **MQTT remains mocked in CI:** no live broker, network protocol validation,
  broker durability/reconnect testing, TLS/authentication test, or cloud deployment.
- No production actuator/PLC, industrial throughput benchmark, drift monitoring,
  detection certification, or multi-seed uncertainty study.
- The **C++ module is small and not optimized**. It is a stage example, not a full
  port of the original inspector. CLI process/PGM/JSON overhead is measured separately.
- CNN produces no boxes or defect-type predictions. OpenCV/C++ have no class-scored
  detections, so their mAP is N/A. Image-level recall and class-aware IoU recall
  are different tasks; the tables identify each instead of treating them as interchangeable.
- External COCO pretraining, when used, was learned from real COCO images; this
  project's fine-tuning/test images remain fully synthetic. This is not training
  the entire pretrained model only on synthetic data.
- No end-to-end exactly-once guarantee, schema-v1 automatic migration, model-version
  history store, or hidden overwrite of the first record. **No Docker anywhere.**

## How to run locally

Use Python **3.11**, a native C++17 compiler, CMake **3.20+**, and Internet access
for dependencies, Google Test, and optional pretrained weights. From the repo root:

```sh
python -m venv .venv
# Linux/macOS:
. .venv/bin/activate
# PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,detector]"
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
python -m pytest -q
python -m camera_quality.extension_experiment --output artifacts/extended-local --yolo-epochs 5
```

Use a fresh output directory for each experiment. On multi-config Windows builds,
add `--config Release` to the build command and set `CPP_INSPECTOR` to the resulting
`build/Release/inspection_cli.exe`. Compiler runtime DLLs must be on PATH for MinGW.
The Python parity test intentionally fails if the C++ CLI has not been built.
The pinned Ultralytics dependency uses opencv-python 4.12.0.88; the original
headless build used the same OpenCV version. The pixel/label hash guards comparability.

The extension writes `metrics.json`, `extended-evaluation.md`, marked YOLO examples,
actual `predictions.json`, and `inspection.sqlite`. Its `baseline/` subfolder has
original-method metrics, generated PNGs/CSV/YOLO labels and CNN checkpoint; `yolo/`
has training output and the selected detector checkpoint. Full artifacts are
retained in GitHub Actions for 30 days; checked-in reports are labeled snapshots.

Replay the extended results without a broker:

```sh
python -m camera_quality.pipeline mock --predictions artifacts/extended-local/predictions.json --database artifacts/replay-v2.sqlite
```

Optional native Mosquitto on localhost, in two terminals:

```sh
python -m camera_quality.pipeline consume --database artifacts/live-v2.sqlite
python -m camera_quality.pipeline publish --predictions artifacts/extended-local/predictions.json
```

The optional local adapter defaults to port 1883 without TLS/credentials. It has
not been exercised against a live broker here; do not expose it as a production
service. No broker is needed for tests, training, evaluation, or mock replay.
