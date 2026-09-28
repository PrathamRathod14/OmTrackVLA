#!/usr/bin/env python3
"""Measure whether moving the colour maths to CUDA is worth it.

Times the per-frame target-perception colour work three ways at the live frame
size: the current OpenCV/NumPy path, the same maths in torch on CPU, and on
CUDA.  The CUDA timing includes the host-to-device upload, because on this robot
the frame arrives as a decoded JPEG on the host and that transfer is a real cost
of the move, not an accounting trick.

    ./run_local.sh python real_robot/bench_gpu_ops.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import time
from typing import Callable, List, Sequence

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu_ops
from target_perception import COLOR_RANGES, appearance_descriptor, color_fraction


def make_boxes(count: int, width: int, height: int) -> List[Sequence[float]]:
    """Person-shaped boxes spread across the frame, as the detector would give."""
    boxes = []
    for index in range(count):
        left = 20.0 + index * (width - 160.0) / max(1, count)
        boxes.append((left, 40.0, left + 130.0, height - 40.0))
    return boxes


def timed(function: Callable[[], object], repeats: int, cuda: bool) -> float:
    """Median milliseconds per call, after a warmup."""
    for _ in range(3):
        function()
    if cuda:
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        if cuda:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--colour", default="red")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 256, size=(args.height, args.width, 3), dtype=np.uint8)
    ranges = COLOR_RANGES[args.colour]
    has_cuda = torch.cuda.is_available()

    print(f"frame {args.width}x{args.height}, median of {args.repeats}, CUDA {'yes' if has_cuda else 'no'}\n")
    print(f"{'boxes':>6}  {'opencv':>10}  {'torch cpu':>10}  {'torch cuda':>11}  {'verdict':>16}")
    print("-" * 62)

    for count in (1, 3, 5, 10):
        boxes = make_boxes(count, args.width, args.height)

        def opencv_path() -> None:
            for box in boxes:
                appearance_descriptor(bgr, box)
            color_fraction(bgr, boxes[0], args.colour)

        host = torch.from_numpy(bgr)

        def torch_cpu_path() -> None:
            hsv = gpu_ops.bgr_to_hsv_u8(host)
            gpu_ops.appearance_features(hsv, boxes)
            gpu_ops.color_fraction(hsv, boxes[0], ranges)

        opencv_ms = timed(opencv_path, args.repeats, cuda=False)
        torch_cpu_ms = timed(torch_cpu_path, args.repeats, cuda=False)

        if has_cuda:
            def torch_cuda_path() -> None:
                # The upload is counted: the frame is on the host when it arrives.
                device_frame = host.to("cuda", non_blocking=True)
                hsv = gpu_ops.bgr_to_hsv_u8(device_frame)
                features = gpu_ops.appearance_features(hsv, boxes)
                gpu_ops.color_fraction(hsv, boxes[0], ranges)
                features.cpu()

            cuda_ms = timed(torch_cuda_path, args.repeats, cuda=True)
            best = min(opencv_ms, torch_cpu_ms, cuda_ms)
            if best == cuda_ms:
                verdict = f"cuda {opencv_ms / cuda_ms:.2f}x"
            else:
                verdict = f"opencv {cuda_ms / opencv_ms:.2f}x faster"
            print(f"{count:>6}  {opencv_ms:>9.3f}ms  {torch_cpu_ms:>9.3f}ms  {cuda_ms:>10.3f}ms  {verdict:>16}")
        else:
            print(f"{count:>6}  {opencv_ms:>9.3f}ms  {torch_cpu_ms:>9.3f}ms  {'n/a':>11}  {'n/a':>16}")

    if has_cuda:
        upload_ms = timed(lambda: torch.from_numpy(bgr).to("cuda"), args.repeats, cuda=True)
        hsv_device = gpu_ops.bgr_to_hsv_u8(host.cuda())
        convert_ms = timed(lambda: gpu_ops.bgr_to_hsv_u8(hsv_device), args.repeats, cuda=True)
        print(f"\nupload only: {upload_ms:.3f} ms   hsv on an already-resident frame: {convert_ms:.3f} ms")
        print("At the 3 Hz detection rate, one millisecond saved per frame is 0.3% of one core.")


if __name__ == "__main__":
    main()
