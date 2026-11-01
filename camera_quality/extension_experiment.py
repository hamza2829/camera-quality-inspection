"""CPU baseline + YOLO + native stage experiment. Test data never selects settings."""

import argparse
import csv
import importlib.metadata
import json
import sqlite3
from pathlib import Path

import cv2

from camera_quality import experiment
from camera_quality.cpp_bridge import benchmark, compare_boxes
from camera_quality.detection import localization, run_yolo
from camera_quality.inspection import metrics
from camera_quality.pipeline import MockMQTT, publish_camera


def annotated_examples(data, rows, predictions, output):
    output.mkdir(parents=True, exist_ok=True)
    selected = {}
    for kind in ("scratch", "dent", "hole"):
        indices = [i for i, row in enumerate(rows) if row["defect_type"] == kind]
        index = next((i for i in indices if predictions[i]["yolo"]["boxes"]), indices[0])
        row, result = rows[index], predictions[index]
        image = cv2.imread(str(data / row["image_path"]))
        image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_NEAREST)
        x, y, w, h = [int(row[key])*4 for key in ("x", "y", "width", "height")]
        cv2.rectangle(image, (x, y), (x+w-1, y+h-1), (255, 255, 0), 2)
        for box in result["yolo"]["boxes"]:
            x, y, w, h = [round(box[key]*4) for key in ("x", "y", "width", "height")]
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, f"{box['class_name']} {box['confidence']:.2f}", (max(0, x-10), max(12, y-4)), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1, cv2.LINE_AA)
        image = cv2.copyMakeBorder(image, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(image, f"True: {kind} | YOLO: {result['yolo']['decision']}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, "Cyan: truth | Green: predicted boxes", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
        if not cv2.imwrite(str(output / f"{kind}.png"), image):
            raise OSError("Cannot write YOLO example")
        selected[kind] = dict(image_id=row["image_id"], boxes=len(result["yolo"]["boxes"]))
    return selected


def report(output, result):
    baseline, yolo, native = result["baseline"], result["yolo"], result["cpp_benchmark"]
    lines = ["# CPU YOLO and C++ extension evaluation", "",
             "Synthetic fine-tuning/evaluation images only. No factory data, camera, GPU, live MQTT broker, or deployed cloud service.", "",
             f"Source: [actual CI run]({baseline['ci_run_url']}) at `{baseline['source_commit']}`." if baseline["ci_run_url"] else "Source: local CPU run (not a CI claim).", "",
             f"Seed 20261101; unchanged 600/180/180 splits. labels.csv SHA-256: `{baseline['labels_sha256']}`. Test: 90 clean frames plus 30 each scratch/dent/hole.", "",
             "## Methods on the same 180 test images", "",
             "Image P/R use any detected defect as positive, without checking location/type. Detection mAP50 is class-aware AP averaged over scratch/dent/hole. N/A means the method does not output class-scored detections; classification F1 is not mAP. Overlap counts below are class-agnostic at IoU >= 0.5, with unique matched truths shown separately.", "",
             "| Method | Image precision | Image recall | Detection mAP50 | Overlapping boxes / predicted boxes | Matched truths / 90 |",
             "| --- | ---: | ---: | ---: | --- | --- |"]
    for name in ("opencv", "cnn", "yolo", "cpp_stage"):
        item = result["image_metrics"][name]
        overlap = result["localization"].get(name)
        score = f"{yolo['map50']:.4f}" if name == "yolo" else "N/A"
        boxes = f"{overlap['overlapping_boxes_iou50']} / {overlap['predicted_boxes']}" if overlap else "N/A"
        matched = f"{overlap['matched_defects_iou50']} / 90" if overlap else "N/A"
        lines.append(f"| {name} | {item['precision']:.4f} | {item['recall']:.4f} | {score} | {boxes} | {matched} |")
    lines += ["", "CNN is classification-only. OpenCV is the unchanged full inspector. C++ is a preprocessing/connected-component stage with no morphology; it is not a replacement for the complete original inspector.", "",
              "## Per-defect performance", "",
              "OpenCV/CNN columns are image-level binary recall within each true-type subgroup. YOLO image recall is also location/type agnostic. YOLO detection P/R require the correct class AND IoU >= 0.5, at a confidence fixed to 0.25 before test evaluation. AP50 sweeps confidence using Ultralytics validation at conf=0.001. We do not report the library's test-selected best-F1 precision/recall.", "",
              "| True defect | OpenCV image recall | CNN image recall | YOLO image recall | YOLO detection precision | YOLO detection recall | AP50 | TP / FP / FN |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for kind in ("scratch", "dent", "hole"):
        errors = baseline["errors_by_type"][kind]
        item = yolo["per_class"][kind]
        recall = result["yolo_image_recall_by_type"][kind]
        lines.append(f"| {kind} | {1-errors['opencv_errors']/30:.4f} | {1-errors['cnn_errors']/30:.4f} | {recall:.4f} | {item['precision']:.4f} | {item['recall']:.4f} | {item['ap50']:.4f} | {item['tp']} / {item['fp']} / {item['fn']} |")
    lines += ["", "Lower YOLO subgroup image recall than a baseline is a direct pass/fail disadvantage. Class-aware detection recall is a stricter task; it must not be presented as the same quantity as a classifier's recall."]
    for kind, recall in result["yolo_image_recall_by_type"].items():
        errors = baseline["errors_by_type"][kind]
        worse = [name for name in ("opencv", "cnn") if recall < 1-errors[f"{name}_errors"]/30]
        lines += ["", f"- {kind}: YOLO image recall {recall:.4f}; worse than {', '.join(worse)} on this subgroup." if worse else f"- {kind}: YOLO image recall {recall:.4f}; not below either baseline's subgroup image recall in this run."]
    lines += ["", "## Training provenance and limits", "",
              f"YOLOv8n initialization: **{yolo['initialization']}**. {yolo['epochs']} epochs, 96-pixel input, CPU, frozen backbone layers: {yolo['freeze']}. Training time including per-epoch validation/final checkpoint validation: **{yolo['training_seconds']:.2f} s**. Confidence 0.25, NMS IoU 0.7, max 100 predictions/image; no test-driven tuning.", "",
              "COCO-pretrained initialization uses external pretrained weights learned from COCO, not weights learned only from synthetic images. All fine-tuning and reported evaluation images are synthetic. If the official weight download fails, the code constructs the architecture from YAML and trains all layers from scratch instead, recording the error.", "",
              f"Download outcome: `{yolo['download_error']}`. Initial weight hash: `{yolo['initial_weights_sha256']}`. Evaluated checkpoint hash: `{yolo['weights_sha256']}`.", "",
              f"Software: `{result['versions']}`. Hardware/OS: `{baseline['platform']}`. One seed/split only, uncalibrated scores, no production accuracy or real-time claim.", "",
              "## C++ parity and timing", "",
              "C++17 has no OpenCV dependency. It implements replicated-border median 3x3, separable double-precision Gaussian 31x31 sigma=5 with REFLECT_101, absolute residual >8, and handwritten eight-connected BFS with >=10 foreground pixels/component. The Python stage reference uses OpenCV with these exact settings, before morphology. The original inspector keeps its uint8 Gaussian, morphology, contour-area filtering, and centroid logic.", "",
              f"Across {native['images']} test images: {native['mask_pixels_different']} differing mask pixels; {native['images_with_box_difference']} images with stage boxes differing by more than one pixel. Compared with the original full inspector, {result['legacy_box_difference_images']} images have different boxes (different stages/area semantics, not an equivalence claim). Per-image differences are in metrics.json.", "",
              "Both references run one thread. Inputs are preloaded and PGM writes are outside the timing. Kernel timings average three repeats/image; OpenCV is warmed up. C++ CLI end-to-end timing uses a separate single-image process including PGM read, compute, full-mask JSON output and Python JSON parsing. It is not divided by the repeat count.", "",
              "| Implementation / boundary | Mean ms per image |",
              "| --- | ---: |",
              f"| Python/OpenCV stage in process | {native['opencv_ms_per_image']:.4f} |",
              f"| C++ compute only (Release) | {native['cpp_kernel_ms_per_image']:.4f} |",
              f"| C++ subprocess, input/output and parse | {native['cpp_subprocess_ms_per_image']:.4f} |", "",
              "This small C++ implementation is not optimized. No speedup is assumed; process/serialization overhead and different code paths are shown explicitly. Timings are runner-specific, not a latency SLA.", "",
              f"## Pipeline replay\n\nSchema/topic v2 publishes actual YOLO boxes plus confidence/class/weight provenance. Validated SQLite table inspections_v2 keeps boxes as explicit JSON and inside the canonical payload. {result['pipeline']['inserted']} inserted rows; {result['pipeline']['duplicates']} duplicate replays; {result['pipeline']['published']} mock publishes. v1 payloads are rejected; old v1 tables are not altered. No MQTT network/cloud boundary was tested.", "",
              "## Marked test examples", "",
              "For each true type, choose the first image with a YOLO prediction, otherwise its first image (showing a miss). Cyan is truth, green is actual predicted class/confidence/box. These are selected illustrations; every test image contributes to metrics."]
    for kind, item in result["examples"].items():
        lines += ["", f"### {kind}: `{item['image_id']}` ({item['boxes']} predicted boxes)", "", f"![YOLO {kind} example](yolo-examples/{kind}.png)"]
    (output / "extended-evaluation.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(output: Path, yolo_epochs=5):
    if (output / "metrics.json").exists():
        raise FileExistsError("Use a fresh experiment output directory")
    baseline = experiment.run(output / "baseline")
    data = output / "baseline/data"
    with (data / "labels.csv").open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["split"] == "test"]
    if baseline["labels_sha256"] != "408a456412ad255e32ae877a072106bb7c710c3977596170b101e0de07bee5cf":
        raise AssertionError("Original dataset changed; comparison would be invalid")
    yolo, detector_results = run_yolo(data, rows, output / "yolo", yolo_epochs)
    predictions = json.loads((output / "baseline/predictions.json").read_text())
    for item, detector in zip(predictions, detector_results, strict=True):
        item["yolo"] = detector
    (output / "predictions.json").write_text(json.dumps(predictions, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    images = experiment.load_images(data, rows)
    native, cpp_results = benchmark(images, output / "cpp-inputs")
    mock = MockMQTT(output / "inspection.sqlite")
    published = publish_camera(predictions, mock)+publish_camera(predictions, mock)
    with sqlite3.connect(mock.database) as connection:
        stored = connection.execute("SELECT COUNT(*) FROM inspections_v2").fetchone()[0]
    assert stored == mock.inserted == mock.duplicates == 180
    truth = [int(row["defect_type"] != "none") for row in rows]
    boxes = dict(opencv=[item["opencv"]["regions"] for item in predictions],
                 yolo=[item["yolo"]["boxes"] for item in predictions], cpp_stage=[item["boxes"] for item in cpp_results])
    result = dict(baseline=baseline, yolo=yolo, cpp_benchmark=native,
                  versions={name: importlib.metadata.version(name) for name in ("numpy", "opencv-python", "torch", "torchvision", "ultralytics")},
                  image_metrics=dict(opencv=baseline["opencv"], cnn=baseline["cnn"],
                                     yolo=metrics(truth, [int(bool(item)) for item in boxes["yolo"]]),
                                     cpp_stage=metrics(truth, [int(bool(item)) for item in boxes["cpp_stage"]])),
                  localization={name: localization(rows, items) for name, items in boxes.items()},
                  legacy_box_difference_images=sum(not compare_boxes(a, b) for a, b in zip(boxes["opencv"], boxes["cpp_stage"])),
                  yolo_image_recall_by_type={kind: sum(bool(boxes["yolo"][i]) for i, row in enumerate(rows) if row["defect_type"] == kind)/30 for kind in ("scratch", "dent", "hole")},
                  pipeline=dict(published=published, inserted=mock.inserted, duplicates=mock.duplicates, stored=stored))
    result["examples"] = annotated_examples(data, rows, predictions, output / "yolo-examples")
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    report(output, result)
    print(json.dumps({name: result[name] for name in ("yolo", "image_metrics", "pipeline")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/extended-local"))
    parser.add_argument("--yolo-epochs", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.yolo_epochs <= 10:
        parser.error("Use 1..10 YOLO epochs for this CPU example")
    run(args.output, args.yolo_epochs)


if __name__ == "__main__":
    main()
