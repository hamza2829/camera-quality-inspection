"""Generate independent, balanced splits of synthetic sheet-metal images."""

import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np

SEED = 20261101
SIZE = 96
FIELDS = ["image_id", "image_path", "split", "defect_type", "x", "y", "width", "height", "contrast", "noise_sigma"]


def synthesize(rng: np.random.Generator, defect_type: str) -> tuple[np.ndarray, dict]:
    """One defect at most; position/size are known simulated bounding boxes in pixels."""
    if defect_type not in {"none", "scratch", "dent", "hole"}:
        raise ValueError("Unknown defect type")
    yy, xx = np.mgrid[:SIZE, :SIZE].astype(np.float32)
    noise = float(rng.uniform(2, 8))
    image = (rng.uniform(125, 180) + rng.uniform(-30, 30) * (xx / SIZE - .5)
             + rng.uniform(-30, 30) * (yy / SIZE - .5)
             + 2 * np.sin(yy * rng.uniform(1, 2)))
    # Benign illumination variation can look like a shallow dent.
    lx, ly = rng.uniform(0, SIZE, 2)
    image += rng.uniform(-25, 25) * np.exp(-((xx-lx)**2 + (yy-ly)**2) / (2*rng.uniform(12, 35)**2))
    label = dict(x="", y="", width="", height="", contrast=0.0, noise_sigma=round(noise, 4))
    if defect_type != "none":
        cx, cy = rng.integers(24, SIZE-24, 2)
        mask = np.zeros((SIZE, SIZE), np.float32)
        contrast = float(rng.uniform(8, 65))
        if defect_type == "scratch":
            dx, dy = rng.integers(-22, 23, 2)
            if abs(dx) + abs(dy) < 12:
                dx = 20
            cv2.line(mask, (int(cx-dx//2), int(cy-dy//2)), (int(cx+dx//2), int(cy+dy//2)), 1.0, int(rng.integers(1, 4)))
            image -= mask * contrast
        elif defect_type == "hole":
            cv2.ellipse(mask, (int(cx), int(cy)), (int(rng.integers(4, 10)), int(rng.integers(4, 10))), float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
            contrast = float(rng.uniform(65, 120))
            image -= mask * contrast
        else:
            radius = float(rng.uniform(5, 11))
            mask = np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2*radius**2))
            contrast = float(rng.uniform(6, 32))
            highlight = np.exp(-((xx-cx-radius)**2 + (yy-cy+radius/2)**2) / (2*(radius*.75)**2))
            image += contrast * (-mask + .6*highlight)
        coords = np.argwhere(mask > .15)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        label.update(x=int(x0), y=int(y0), width=int(x1-x0+1), height=int(y1-y0+1), contrast=round(contrast, 4))
    image += rng.normal(0, noise, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8), label


def generate_dataset(output: Path, counts: dict[str, int], seed: int = SEED) -> list[dict]:
    """Every split is 50% clean and 1/6 each defect; counts must be multiples of six.

    Separate RNG streams keep validation/test unchanged when train count changes.
    Existing image files are overwritten deterministically; labels.csv is authoritative.
    """
    if set(counts) != {"train", "validation", "test"}:
        raise ValueError("Require train, validation and test counts")
    if any(type(n) is not int or n < 6 or n % 6 for n in counts.values()):
        raise ValueError("Split counts must be positive multiples of six")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for offset, split in enumerate(("train", "validation", "test")):
        rng = np.random.default_rng(np.random.SeedSequence([seed, offset]))
        count = counts[split]
        kinds = np.array(["none"]*(count//2) + ["scratch", "dent", "hole"]*(count//6))
        rng.shuffle(kinds)
        (output / split).mkdir(exist_ok=True)
        for index, kind in enumerate(kinds):
            image, label = synthesize(rng, str(kind))
            image_id = "sim-" + hashlib.sha256(image.tobytes()).hexdigest()[:24]
            relative = f"{split}/{index:05d}.png"
            if not cv2.imwrite(str(output / relative), image):
                raise OSError("Failed to save synthetic image")
            rows.append(dict(image_id=image_id, image_path=relative, split=split, defect_type=str(kind), **label))
    with (output / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows
