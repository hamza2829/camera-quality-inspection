"""Explicit floating-point stage reference and subprocess bridge; no full-inspector equivalence claim."""

import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


def executable() -> Path:
    path = Path(os.environ.get("CPP_INSPECTOR", "build/inspection_cli.exe" if os.name == "nt" else "build/inspection_cli"))
    if not path.is_file():
        raise FileNotFoundError(f"Build cpp/ first, or set CPP_INSPECTOR: {path}")
    return path.resolve()


def reference(image: np.ndarray, threshold=8., min_pixels=10) -> dict:
    """Stage before morphology, float64 Gaussian rather than old uint8 blur.

    The original inspector still uses its original contour/morphology pipeline.
    """
    smooth = cv2.medianBlur(image, 3).astype(np.float64)
    background = cv2.GaussianBlur(smooth, (31, 31), 5, borderType=cv2.BORDER_REFLECT_101)
    mask = np.where(np.abs(smooth-background) > threshold, 255, 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = [dict(x=int(x), y=int(y), width=int(w), height=int(h), pixels=int(area))
             for x, y, w, h, area in stats[1:] if area >= min_pixels]
    boxes.sort(key=lambda box: (box["x"], box["y"]))
    return dict(mask=mask, boxes=boxes)


def run_cli(path: Path, threshold=8., min_pixels=10, repeat=1) -> dict:
    completed = subprocess.run([str(executable()), str(path), str(threshold), str(min_pixels), str(repeat)],
                               check=True, capture_output=True, text=True, timeout=30)
    result = json.loads(completed.stdout)
    result["mask"] = np.array(result["mask"], np.uint8).reshape(result["height"], result["width"])
    return result


def compare_boxes(left, right, tolerance=1) -> bool:
    if len(left) != len(right):
        return False
    unused = list(right)
    for box in left:
        match = next((i for i, candidate in enumerate(unused) if all(abs(box[key]-candidate[key]) <= tolerance for key in ("x", "y", "width", "height"))), None)
        if match is None:
            return False
        unused.pop(match)
    return True


def benchmark(images: list[np.ndarray], output: Path, repeats=3) -> tuple[dict, list[dict]]:
    """Both timings use same preloaded images; CLI also reports its separate overhead."""
    cv2.setNumThreads(1)
    output.mkdir(parents=True, exist_ok=True)
    reference(images[0])  # Warm up OpenCV dispatch.
    python_ms, cpp_ms, cli_ms, differences, cpp_results = [], [], [], [], []
    for index, image in enumerate(images):
        path = output / f"{index:05d}.pgm"
        if not cv2.imwrite(str(path), image):
            raise OSError("Cannot write PGM benchmark input")
        start = time.perf_counter()
        for _ in range(repeats):
            expected = reference(image)
        python_ms.append((time.perf_counter()-start)*1000/repeats)
        start = time.perf_counter()
        actual = run_cli(path, repeat=repeats)
        # Measure a separate, genuine single-image invocation including JSON parsing.
        start = time.perf_counter()
        run_cli(path, repeat=1)
        cli_ms.append((time.perf_counter()-start)*1000)
        cpp_ms.append(actual["processing_ms"])
        cpp_results.append(actual)
        differing = int(np.count_nonzero(expected["mask"] != actual["mask"]))
        differences.append(dict(index=index, differing_mask_pixels=differing,
                                boxes_within_one_pixel=compare_boxes(expected["boxes"], actual["boxes"])))
    result = dict(images=len(images), repeats=repeats, gaussian_kernel=31, sigma=5,
                  threshold=8, min_pixels=10, python_threads=1, cpp_threads=1,
                  opencv_ms_per_image=float(np.mean(python_ms)), cpp_kernel_ms_per_image=float(np.mean(cpp_ms)),
                  cpp_subprocess_ms_per_image=float(np.mean(cli_ms)),
                  mask_pixels_different=sum(item["differing_mask_pixels"] for item in differences),
                  images_with_box_difference=sum(not item["boxes_within_one_pixel"] for item in differences),
                  differences=differences)
    return result, cpp_results
