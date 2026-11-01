"""Classical pixel-based inspection; no ground truth is consumed by inspect()."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Config:
    threshold: int = 12
    min_area: float = 10.0


def preprocess(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if image.ndim != 2 or image.dtype != np.uint8 or min(image.shape) < 8:
        raise ValueError("Expected a grayscale uint8 image at least 8x8")
    smooth = cv2.medianBlur(image, 3)
    background = cv2.GaussianBlur(smooth, (0, 0), 5)
    residual = cv2.absdiff(smooth, background)
    return smooth, residual


def inspect(image: np.ndarray, config: Config = Config()) -> dict:
    if not 0 < config.threshold < 255 or config.min_area <= 0:
        raise ValueError("Invalid inspection configuration")
    _, residual = preprocess(image)
    _, mask = cv2.threshold(residual, config.threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < config.min_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        regions.append(dict(x=x, y=y, width=width, height=height, area_px=area,
                            center_x=round(moments["m10"]/moments["m00"], 3),
                            center_y=round(moments["m01"]/moments["m00"], 3)))
    regions.sort(key=lambda item: (item["x"], item["y"]))
    return {"decision": "fail" if regions else "pass", "regions": regions}


def metrics(truth: list[int], predictions: list[int]) -> dict:
    if len(truth) != len(predictions) or not truth:
        raise ValueError("Nonempty equal-length arrays required")
    if any(x not in (0, 1) for x in [*truth, *predictions]):
        raise ValueError("Binary labels required")
    tn = sum(t == 0 and p == 0 for t, p in zip(truth, predictions))
    fp = sum(t == 0 and p == 1 for t, p in zip(truth, predictions))
    fn = sum(t == 1 and p == 0 for t, p in zip(truth, predictions))
    tp = sum(t == 1 and p == 1 for t, p in zip(truth, predictions))
    precision = tp / (tp+fp) if tp+fp else 0.0
    recall = tp / (tp+fn) if tp+fn else 0.0
    return dict(precision=precision, recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.0,
                accuracy=(tp+tn)/len(truth), confusion_matrix=[[tn, fp], [fn, tp]])


def choose_classical(validation_images: list[np.ndarray], truth: list[int]) -> Config:
    """Tune only on validation, never on test images or labels."""
    candidates = [Config(threshold, area) for threshold in (8, 12, 16, 20) for area in (4, 10, 20)]
    return max(candidates, key=lambda c: (metrics(truth, [int(inspect(img, c)["decision"] == "fail") for img in validation_images])["f1"], c.threshold, c.min_area))
