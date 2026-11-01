import hashlib

import cv2
import numpy as np
import pytest
import torch

from camera_quality.generator import generate_dataset, synthesize
from camera_quality.inspection import Config, inspect, metrics, preprocess
from camera_quality.model import TinyCNN, predict, train


def test_generator_is_reproducible_and_splits_are_disjoint(tmp_path):
    counts = dict(train=12, validation=6, test=6)
    first = generate_dataset(tmp_path / "a", counts, seed=11)
    second = generate_dataset(tmp_path / "b", counts, seed=11)
    assert first == second
    assert (tmp_path / "a/labels.csv").read_bytes() == (tmp_path / "b/labels.csv").read_bytes()
    assert len({row["image_id"] for row in first}) == 24
    for row in first:
        image = cv2.imread(str(tmp_path / "a" / row["image_path"]), cv2.IMREAD_GRAYSCALE)
        assert image.shape == (96, 96) and image.dtype == np.uint8
        assert (tmp_path / "a" / row["image_path"]).read_bytes() == (tmp_path / "b" / row["image_path"]).read_bytes()
        assert row["image_id"] == "sim-" + hashlib.sha256(image.tobytes()).hexdigest()[:24]
        if row["defect_type"] != "none":
            assert 0 <= row["x"] < row["x"]+row["width"] <= 96
            assert 0 <= row["y"] < row["y"]+row["height"] <= 96
        else:
            assert row["x"] == row["y"] == ""
    assert sum(row["defect_type"] == "none" for row in first) == 12


def test_training_count_does_not_change_test_data(tmp_path):
    a = generate_dataset(tmp_path / "a", dict(train=6, validation=6, test=6), seed=5)
    b = generate_dataset(tmp_path / "b", dict(train=18, validation=6, test=6), seed=5)
    assert [row for row in a if row["split"] == "test"] == [row for row in b if row["split"] == "test"]
    other = generate_dataset(tmp_path / "c", dict(train=6, validation=6, test=6), seed=6)
    assert a != other


@pytest.mark.parametrize("counts", [dict(train=5, validation=6, test=6), dict(train=6, test=6), dict(train=0, validation=6, test=6)])
def test_generator_rejects_invalid_counts(tmp_path, counts):
    with pytest.raises(ValueError):
        generate_dataset(tmp_path, counts)


def test_generator_rejects_unknown_defect():
    with pytest.raises(ValueError):
        synthesize(np.random.default_rng(1), "real-machine")


def test_preprocessing_reduces_noise_and_preserves_shape():
    image = np.clip(150 + np.random.default_rng(2).normal(0, 6, (96, 96)), 0, 255).astype(np.uint8)
    smooth, residual = preprocess(image)
    assert smooth.shape == residual.shape == image.shape
    assert smooth.std() < image.std()*.6
    assert residual.dtype == np.uint8


@pytest.mark.parametrize("image", [np.zeros((16, 16, 3), np.uint8), np.zeros((16, 16), float), np.zeros((4, 4), np.uint8)])
def test_preprocessing_rejects_wrong_input(image):
    with pytest.raises(ValueError):
        preprocess(image)


def test_clean_lighting_gradient_passes():
    image = np.tile(np.linspace(110, 180, 96, dtype=np.uint8), (96, 1))
    assert inspect(image)["decision"] == "pass"


@pytest.mark.parametrize("kind", ["hole", "scratch"])
def test_obvious_defects_fail_with_position_and_area(kind):
    image = np.full((96, 96), 150, np.uint8)
    if kind == "hole":
        cv2.circle(image, (48, 48), 6, 20, -1)
    else:
        cv2.line(image, (35, 48), (60, 48), 30, 3)
    result = inspect(image)
    assert result["decision"] == "fail"
    assert any(abs(region["center_x"]-48) < 5 and abs(region["center_y"]-48) < 5 for region in result["regions"])
    assert all(region["area_px"] >= 10 and region["width"] > 0 for region in result["regions"])


def test_minimum_area_rule_filters_defects():
    image = np.full((96, 96), 150, np.uint8)
    cv2.circle(image, (48, 48), 6, 20, -1)
    assert inspect(image, Config(min_area=10_000))["decision"] == "pass"
    with pytest.raises(ValueError):
        inspect(image, Config(threshold=0))


def test_confusion_matrix_and_zero_denominators():
    result = metrics([0, 0, 1, 1, 1], [0, 1, 0, 1, 1])
    assert result["confusion_matrix"] == [[1, 1], [1, 2]]
    assert result["precision"] == result["recall"] == pytest.approx(2/3)
    assert metrics([0, 1], [0, 0])["precision"] == 0
    assert metrics([0, 0], [0, 0])["recall"] == 0
    with pytest.raises(ValueError):
        metrics([1], [])


def test_cpu_cnn_learns_finite_loss_and_reloadable_weights():
    rng = np.random.default_rng(7)
    images = [synthesize(rng, kind)[0] for kind in ["none", "hole"]*6]
    labels = [0, 1]*6
    model, threshold, history, best_epoch = train(images[:8], labels[:8], images[8:], labels[8:], epochs=2, seed=7)
    assert len(history) == 2 and best_epoch in (1, 2)
    assert all(np.isfinite(item["train_loss"]) and np.isfinite(item["validation_loss"]) for item in history)
    assert .1 <= threshold <= .9
    initial = TinyCNN()
    initial.load_state_dict(model.state_dict())
    probabilities = predict(initial, images[8:])
    assert len(probabilities) == 4 and all(0 <= probability <= 1 for probability in probabilities)
    assert next(model.parameters()).device.type == "cpu"
