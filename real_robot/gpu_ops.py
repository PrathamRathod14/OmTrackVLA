"""CUDA implementations of the target-perception colour maths.

These mirror the OpenCV/NumPy reference implementations in ``target_perception``
exactly enough to be swapped in: the same integer crop bounds, the same uniform
histogram bin edges, and OpenCV's 8-bit HSV convention (H in 0..179, S and V in
0..255).  The reference stays authoritative; ``test_gpu_ops.py`` asserts the two
agree, and ``bench_gpu_ops.py`` measures whether the swap is worth making.

Nothing here decides motion.  It produces the same descriptors and colour
fractions the CPU path produces, on a different device.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch


# Matches target_perception.APPEARANCE_STRIPES / APPEARANCE_BINS.
APPEARANCE_STRIPES: Tuple[Tuple[float, float], ...] = ((0.15, 0.45), (0.45, 0.75), (0.75, 1.00))
APPEARANCE_BINS: Tuple[int, int, int] = (8, 4, 4)
HIST_SIZE = APPEARANCE_BINS[0] * APPEARANCE_BINS[1] * APPEARANCE_BINS[2]


# OpenCV's 8-bit HSV path is fixed point, not float: it multiplies by a
# quantised reciprocal table and shifts.  Floating point cannot reproduce it --
# a saturation of 178.5 rounds to 179 in float but truncates to 178 through the
# table -- so the tables are rebuilt here rather than approximated.
_HSV_SHIFT = 12
_HSV_ROUND = 1 << (_HSV_SHIFT - 1)
_DIV_TABLES: dict = {}


def _div_tables(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """OpenCV's sdiv_table and hdiv_table, cached per device."""
    key = str(device)
    cached = _DIV_TABLES.get(key)
    if cached is None:
        index = torch.arange(1, 256, dtype=torch.float64)
        sdiv = torch.zeros(256, dtype=torch.int64)
        hdiv = torch.zeros(256, dtype=torch.int64)
        sdiv[1:] = torch.round((255 << _HSV_SHIFT) / index).to(torch.int64)
        hdiv[1:] = torch.round((180 << _HSV_SHIFT) / (6.0 * index)).to(torch.int64)
        cached = (sdiv.to(device), hdiv.to(device))
        _DIV_TABLES[key] = cached
    return cached


def bgr_to_hsv_u8(bgr: torch.Tensor) -> torch.Tensor:
    """OpenCV's 8-bit BGR->HSV, as tensor ops.

    ``bgr`` is (H, W, 3) uint8.  Returns (H, W, 3) uint8 with H in 0..179 and
    S, V in 0..255.  OpenCV resolves ties toward the red branch first, then
    green, so the ``where`` chain below is ordered to match.
    """
    if bgr.dtype != torch.uint8 or bgr.ndim != 3 or bgr.shape[-1] != 3:
        raise ValueError("expected a (H, W, 3) uint8 BGR tensor")
    x = bgr.to(torch.int64)
    b, g, r = x[..., 0], x[..., 1], x[..., 2]
    v = x.amax(dim=-1)
    diff = v - x.amin(dim=-1)
    sdiv, hdiv = _div_tables(bgr.device)

    saturation = torch.div(diff * sdiv[v] + _HSV_ROUND, 1 << _HSV_SHIFT, rounding_mode="floor")

    # The numerator is kept in units of diff, exactly as OpenCV does, so the one
    # reciprocal lookup covers all three branches.
    numerator = torch.where(v == r, g - b, torch.where(v == g, b - r + 2 * diff, r - g + 4 * diff))
    hue = torch.div(numerator * hdiv[diff] + _HSV_ROUND, 1 << _HSV_SHIFT, rounding_mode="floor")
    hue = torch.where(hue < 0, hue + 180, hue)

    return torch.stack([hue, saturation, v], dim=-1).clamp(0, 255).to(torch.uint8)


def _crop_bounds(box: Sequence[float], height: int, width: int, margin_fraction: float) -> Tuple[int, int, int, int]:
    """The integer crop target_perception.appearance_descriptor would take."""
    x1, y1, x2, y2 = [float(value) for value in box]
    margin = margin_fraction * max(0.0, x2 - x1)
    left = max(0, min(width - 1, int(math.floor(x1 + margin))))
    right = max(left + 1, min(width, int(math.ceil(x2 - margin))))
    top = max(0, min(height - 1, int(math.floor(y1))))
    bottom = max(top + 1, min(height, int(math.ceil(y2))))
    return left, top, right, bottom


def _histogram(hsv_patch: torch.Tensor) -> torch.Tensor:
    """Uniform 8x4x4 HSV histogram, flattened the way cv2.calcHist ravels it."""
    values = hsv_patch.reshape(-1, 3).to(torch.int64)
    h_bin = (values[:, 0] * APPEARANCE_BINS[0] // 180).clamp(0, APPEARANCE_BINS[0] - 1)
    s_bin = (values[:, 1] * APPEARANCE_BINS[1] // 256).clamp(0, APPEARANCE_BINS[1] - 1)
    v_bin = (values[:, 2] * APPEARANCE_BINS[2] // 256).clamp(0, APPEARANCE_BINS[2] - 1)
    flat = (h_bin * APPEARANCE_BINS[1] + s_bin) * APPEARANCE_BINS[2] + v_bin
    hist = torch.zeros(HIST_SIZE, dtype=torch.float32, device=hsv_patch.device)
    hist.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
    return hist


def appearance_descriptor(hsv: torch.Tensor, box: Sequence[float]) -> torch.Tensor:
    """One 384-d signature, matching target_perception.appearance_descriptor.

    ``hsv`` is a whole-frame (H, W, 3) uint8 tensor from :func:`bgr_to_hsv_u8`,
    so a batch of boxes shares one conversion.
    """
    height, width = hsv.shape[:2]
    left, top, right, bottom = _crop_bounds(box, height, width, margin_fraction=0.2)
    crop = hsv[top:bottom, left:right]
    crop_height = crop.shape[0]

    stripes = []
    for start, end in APPEARANCE_STRIPES:
        first = min(crop_height - 1, int(start * crop_height))
        last = max(first + 1, int(end * crop_height))
        hist = _histogram(crop[first:last])
        # sqrt(counts / total) already has unit L2 norm whenever total > 0, which
        # is why the reference's per-stripe normalise is a no-op there too.
        hist = torch.sqrt(hist / hist.sum().clamp(min=1.0))
        stripes.append(hist / hist.norm().clamp(min=1e-8))

    descriptor = torch.cat(stripes)
    return descriptor / descriptor.norm().clamp(min=1e-8)


def appearance_features(hsv: torch.Tensor, boxes: Sequence[Sequence[float]]) -> torch.Tensor:
    """Stacked signatures for every box, sharing one HSV conversion."""
    if len(boxes) == 0:
        return torch.empty((0, HIST_SIZE * len(APPEARANCE_STRIPES)), dtype=torch.float32, device=hsv.device)
    return torch.stack([appearance_descriptor(hsv, box) for box in boxes])


def color_fraction(hsv: torch.Tensor, box: Sequence[float], ranges: Sequence[Sequence[int]]) -> float:
    """Fraction of pixels in ``box`` inside any of ``ranges``.

    ``ranges`` is the COLOR_RANGES entry for a colour word: tuples of
    (h_low, h_high, s_min, s_max, v_min, v_max).
    """
    height, width = hsv.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in box]
    left, top = max(0, int(x1)), max(0, int(y1))
    right, bottom = min(width, int(math.ceil(x2))), min(height, int(math.ceil(y2)))
    if right <= left or bottom <= top:
        return 0.0
    crop = hsv[top:bottom, left:right].to(torch.int16)
    h, s, v = crop[..., 0], crop[..., 1], crop[..., 2]
    mask = torch.zeros(crop.shape[:2], dtype=torch.bool, device=hsv.device)
    for h_low, h_high, s_min, s_max, v_min, v_max in ranges:
        mask |= (h >= h_low) & (h <= h_high) & (s >= s_min) & (s <= s_max) & (v >= v_min) & (v <= v_max)
    return float(mask.to(torch.float32).mean())


def gallery_similarity(feature: torch.Tensor, gallery: torch.Tensor) -> float:
    """Best cosine similarity against a stacked gallery, in one matmul."""
    if gallery.numel() == 0:
        return -1.0
    feature = feature.reshape(1, -1)
    scores = (gallery @ feature.T).squeeze(1) / (
        gallery.norm(dim=1).clamp(min=1e-8) * feature.norm().clamp(min=1e-8)
    )
    return float(scores.max())
