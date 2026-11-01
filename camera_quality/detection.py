"""CPU-only YOLOv8n experiment and fixed-threshold, class-aware box counts."""

import hashlib
import time
import urllib.request
from pathlib import Path

import torch

from camera_quality.generator import DEFECT_CLASSES, SIZE

CONFIDENCE = .25
WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"


def iou(left: dict, right: dict) -> float:
    width = max(0., min(left["x"]+left["width"], right["x"]+right["width"])-max(left["x"], right["x"]))
    height = max(0., min(left["y"]+left["height"], right["y"]+right["height"])-max(left["y"], right["y"]))
    intersection = width*height
    union = left["width"]*left["height"] + right["width"]*right["height"]-intersection
    return intersection/union if union > 0 else 0.


def truth_box(row: dict) -> dict | None:
    if row["defect_type"] == "none":
        return None
    return {key: float(row[key]) for key in ("x", "y", "width", "height")}


def localization(rows: list[dict], boxes_per_image: list[list[dict]]) -> dict:
    """Class-agnostic overlap count; duplicate overlapping boxes are counted as boxes.

    matched_defects counts each of the generator's one-per-image truths once.
    """
    overlapping = matched = total = 0
    for row, boxes in zip(rows, boxes_per_image, strict=True):
        target = truth_box(row)
        hits = sum(target is not None and iou(box, target) >= .5 for box in boxes)
        overlapping += hits
        matched += int(hits > 0)
        total += len(boxes)
    return dict(predicted_boxes=total, overlapping_boxes_iou50=overlapping, matched_defects_iou50=matched,
                ground_truth_defects=sum(row["defect_type"] != "none" for row in rows))


def detection_counts(rows: list[dict], boxes_per_image: list[list[dict]]) -> dict:
    counts = {name: dict(tp=0, fp=0, fn=0, ground_truth=0) for name in DEFECT_CLASSES}
    for row, boxes in zip(rows, boxes_per_image, strict=True):
        target = truth_box(row)
        kind = row["defect_type"]
        if target:
            counts[kind]["ground_truth"] += 1
        matched = False
        for box in sorted(boxes, key=lambda item: -item["confidence"]):
            predicted = box["class_name"]
            if not matched and target and predicted == kind and iou(box, target) >= .5:
                counts[predicted]["tp"] += 1
                matched = True
            else:
                counts[predicted]["fp"] += 1
        if target and not matched:
            counts[kind]["fn"] += 1
    for item in counts.values():
        item["precision"] = item["tp"]/(item["tp"]+item["fp"]) if item["tp"]+item["fp"] else 0.
        item["recall"] = item["tp"]/(item["tp"]+item["fn"]) if item["tp"]+item["fn"] else 0.
    return counts


def run_yolo(data: Path, test_rows: list[dict], output: Path, epochs=5) -> tuple[dict, list[dict]]:
    from ultralytics import YOLO, settings
    output.mkdir(parents=True, exist_ok=True)
    settings.update({"sync": False})
    torch.manual_seed(20261101)
    initial_path = output / "yolov8n-coco.pt"
    download_error = None
    try:
        # This is the only attempt to obtain pretrained weights; failure is explicit.
        with urllib.request.urlopen(WEIGHTS_URL, timeout=30) as response:
            initial_path.write_bytes(response.read())
        model = YOLO(str(initial_path))
        initialization = "coco_pretrained"
    except Exception as error:
        download_error = f"{type(error).__name__}: {error}"
        model = YOLO("yolov8n.yaml")
        initialization = "scratch"
    for event in ("on_train_start", "on_val_start", "on_predict_start"):
        model.add_callback(event, lambda _: torch.set_num_threads(2))
    started = time.perf_counter()
    model.train(data=str(data / "yolo.yaml"), epochs=epochs, imgsz=96, batch=16,
                device="cpu", workers=0, seed=20261101, deterministic=True,
                pretrained=initialization == "coco_pretrained", freeze=10 if initialization == "coco_pretrained" else 0,
                optimizer="AdamW", lr0=.001, warmup_epochs=.5, amp=False,
                mosaic=0., close_mosaic=0, mixup=0., hsv_h=0., hsv_s=0., hsv_v=0.,
                translate=.05, scale=.1, fliplr=.5, plots=False, verbose=False,
                project=str(output.resolve()), name="train", exist_ok=False, save=True)
    duration = time.perf_counter()-started
    checkpoint = output / "train/weights/best.pt"
    detector = YOLO(str(checkpoint))
    for event in ("on_val_start", "on_predict_start"):
        detector.add_callback(event, lambda _: torch.set_num_threads(2))
    # AP sweeps confidence; do not report Ultralytics' test-selected best-F1 P/R.
    validation = detector.val(data=str(data / "yolo.yaml"), split="test", imgsz=96,
                              device="cpu", workers=0, batch=16, conf=.001, iou=.7,
                              max_det=100, plots=False, verbose=False,
                              project=str(output.resolve()), name="test", save_json=False)
    ap = {int(index): float(value) for index, value in zip(validation.box.ap_class_index, validation.box.ap50)}
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    paths = [str(data / row["image_path"]) for row in test_rows]
    results = detector.predict(paths, stream=True, imgsz=96, device="cpu", batch=16,
                               conf=CONFIDENCE, iou=.7, max_det=100, verbose=False, save=False)
    predictions = []
    for row, result in zip(test_rows, results, strict=True):
        boxes = []
        for coordinates, confidence, class_id in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
            x0, y0, x1, y1 = [max(0., min(float(SIZE), float(value))) for value in coordinates]
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append(dict(class_id=int(class_id), class_name=tuple(DEFECT_CLASSES)[int(class_id)],
                              confidence=float(confidence), x=x0, y=y0, width=x1-x0, height=y1-y0))
        predictions.append(dict(status="evaluated", model="yolov8n", initialization=initialization,
                                weights_sha256=digest, confidence_threshold=CONFIDENCE,
                                decision="fail" if boxes else "pass", boxes=boxes))
    per_class = detection_counts(test_rows, [item["boxes"] for item in predictions])
    for name, class_id in DEFECT_CLASSES.items():
        per_class[name]["ap50"] = ap.get(class_id, 0.)
    return dict(model="yolov8n", epochs=epochs, imgsz=96, initialization=initialization,
                initial_weights_url=WEIGHTS_URL if initialization == "coco_pretrained" else None,
                initial_weights_sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest() if initialization == "coco_pretrained" else None,
                download_error=download_error, weights_sha256=digest, training_seconds=duration,
                freeze=10 if initialization == "coco_pretrained" else 0,
                confidence_threshold=CONFIDENCE, nms_iou=.7, match_iou=.5,
                map50=float(validation.box.map50), per_class=per_class,
                localization=localization(test_rows, [item["boxes"] for item in predictions])), predictions
