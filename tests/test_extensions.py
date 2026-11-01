import hashlib
import json
import sqlite3

import cv2
import numpy as np
import pytest

from camera_quality.cpp_bridge import compare_boxes, executable, reference, run_cli
from camera_quality.detection import detection_counts, iou, localization
from camera_quality.generator import DEFECT_CLASSES, generate_dataset, synthesize
from camera_quality.messages import validate_message
from camera_quality.pipeline import load_message, make_message


def test_original_seed_splits_and_pixels_remain_unchanged(tmp_path):
    generate_dataset(tmp_path, dict(train=600, validation=180, test=180))
    assert hashlib.sha256((tmp_path / "labels.csv").read_bytes()).hexdigest() == "408a456412ad255e32ae877a072106bb7c710c3977596170b101e0de07bee5cf"


def test_yolo_export_classes_normalization_and_empty_negatives(tmp_path):
    rows = generate_dataset(tmp_path, dict(train=6, validation=6, test=6))
    assert (tmp_path / "yolo.yaml").is_file()
    for row in rows:
        text = (tmp_path / row["image_path"]).with_suffix(".txt").read_text()
        if row["defect_type"] == "none":
            assert text == ""
        else:
            cls, cx, cy, width, height = map(float, text.split())
            assert cls == DEFECT_CLASSES[row["defect_type"]]
            assert all(0 < value <= 1 for value in (cx, cy, width, height))
            assert cx*96 == pytest.approx(row["x"]+row["width"]/2, abs=1e-6)
            assert height*96 == pytest.approx(row["height"], abs=1e-6)


def test_iou_matching_does_not_count_duplicates_or_wrong_classes_as_tp():
    row = dict(defect_type="scratch", x=10, y=10, width=10, height=10)
    box = dict(x=10., y=10., width=10., height=10., class_name="scratch", confidence=.9)
    assert iou(box, box) == 1
    duplicate = dict(box, confidence=.8)
    wrong = dict(box, class_name="dent", confidence=.7)
    counts = detection_counts([row], [[box, duplicate, wrong]])
    assert counts["scratch"]["tp"] == 1 and counts["scratch"]["fp"] == 1
    assert counts["dent"]["tp"] == 0 and counts["dent"]["fp"] == 1
    assert localization([row], [[box, duplicate, wrong]])["overlapping_boxes_iou50"] == 3
    assert localization([row], [[box, duplicate]])["matched_defects_iou50"] == 1


def test_cpp_matches_explicit_opencv_stage_on_generated_images(tmp_path):
    # Intentionally fail if C++ was not built; no silent skip in CI or locally.
    executable()
    rng = np.random.default_rng(41)
    for index, kind in enumerate(["none", "scratch", "dent", "hole"]*3):
        image = synthesize(rng, kind)[0]
        path = tmp_path / f"{index}.pgm"
        assert cv2.imwrite(str(path), image)
        expected = reference(image)
        actual = run_cli(path)
        assert compare_boxes(expected["boxes"], actual["boxes"], tolerance=1), (kind, expected["boxes"], actual["boxes"])
        assert np.mean(expected["mask"] != actual["mask"]) <= .001


@pytest.fixture
def detector_prediction():
    box = dict(class_id=0, class_name="scratch", confidence=.8, x=10., y=10., width=12., height=3.)
    return dict(image_id="sim-"+"c"*24, opencv=dict(decision="pass", regions=[]),
                cnn=dict(decision="pass", probability=.2, threshold=.5),
                yolo=dict(status="evaluated", model="yolov8n", initialization="scratch", weights_sha256="b"*64,
                          confidence_threshold=.25, decision="fail", boxes=[box]))


def test_yolo_boxes_are_validated_stored_and_idempotent(tmp_path, detector_prediction):
    payload = make_message(detector_prediction, 0)
    message = validate_message(payload)
    assert message.schema_version == 2 and message.decision == "fail"
    database = tmp_path / "v2.sqlite"
    assert load_message(database, payload)
    assert not load_message(database, payload)
    with sqlite3.connect(database) as connection:
        decision, boxes = connection.execute("SELECT yolo_decision,yolo_boxes_json FROM inspections_v2").fetchone()
    assert decision == "fail" and json.loads(boxes) == detector_prediction["yolo"]["boxes"]


@pytest.mark.parametrize("field,value", [("class_id", 9), ("class_name", "hole"), ("confidence", .1), ("confidence", float("nan")), ("x", 95.), ("width", -1.)])
def test_yolo_box_contract_rejects_invalid_fields(detector_prediction, field, value):
    detector_prediction["yolo"]["boxes"][0][field] = value
    with pytest.raises(ValueError):
        make_message(detector_prediction, 0)


def test_unevaluated_yolo_cannot_be_reported_as_pass(detector_prediction):
    detector_prediction["yolo"]["status"] = "not_run"
    with pytest.raises(ValueError):
        make_message(detector_prediction, 0)


def test_v1_payload_is_explicitly_rejected(detector_prediction):
    payload = json.loads(make_message(detector_prediction, 0))
    payload["schema_version"] = 1
    with pytest.raises(ValueError):
        validate_message(json.dumps(payload))
