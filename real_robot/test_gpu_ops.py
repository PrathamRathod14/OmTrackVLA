"""Equivalence tests: the CUDA colour maths must match the OpenCV reference.

The appearance threshold (0.72) and the colour-fraction threshold (0.15) were
tuned against the OpenCV implementation, so a GPU path is only safe to enable if
it reproduces those numbers.  These tests are the evidence for that.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu_ops
from target_perception import COLOR_RANGES, appearance_descriptor, color_fraction, cosine_similarity


def _frames():
    """A synthetic frame plus a structured one, both with saturated and grey regions."""
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, size=(480, 848, 3), dtype=np.uint8)

    structured = np.zeros((480, 848, 3), dtype=np.uint8)
    structured[:160] = (40, 40, 200)      # red-ish torso band
    structured[160:320] = (200, 180, 60)  # blue-ish
    structured[320:] = (25, 25, 25)       # near-black, where hue is undefined
    structured[100:140, 100:300] = 255    # pure white, diff == 0
    return {"noise": noise, "structured": structured}


BOXES = [
    (100.0, 50.0, 300.0, 400.0),
    (10.5, 12.25, 90.75, 330.5),
    (600.0, 0.0, 840.0, 478.0),
    (0.0, 0.0, 40.0, 40.0),
]


class TestHsvConversion(unittest.TestCase):
    def test_matches_opencv_exactly(self):
        for name, bgr in _frames().items():
            reference = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            produced = gpu_ops.bgr_to_hsv_u8(torch.from_numpy(bgr)).numpy()
            for channel, index in (("H", 0), ("S", 1), ("V", 2)):
                self.assertTrue(
                    np.array_equal(reference[..., index], produced[..., index]),
                    f"{name}: {channel} is not bit-identical to OpenCV",
                )


class TestAppearanceDescriptor(unittest.TestCase):
    def test_matches_reference(self):
        for name, bgr in _frames().items():
            hsv = gpu_ops.bgr_to_hsv_u8(torch.from_numpy(bgr))
            for box in BOXES:
                expected = appearance_descriptor(bgr, box)
                produced = gpu_ops.appearance_descriptor(hsv, box).numpy()
                self.assertEqual(produced.shape, expected.shape)
                np.testing.assert_allclose(np.linalg.norm(produced), 1.0, atol=1e-5)
                # What actually gates motion is the cosine, so bound that directly.
                self.assertGreater(cosine_similarity(produced, expected), 0.999, f"{name} {box}")

    def test_batch_matches_one_at_a_time(self):
        bgr = _frames()["noise"]
        hsv = gpu_ops.bgr_to_hsv_u8(torch.from_numpy(bgr))
        batched = gpu_ops.appearance_features(hsv, BOXES).numpy()
        self.assertEqual(batched.shape, (len(BOXES), 384))
        for index, box in enumerate(BOXES):
            single = gpu_ops.appearance_descriptor(hsv, box).numpy()
            np.testing.assert_allclose(batched[index], single, atol=1e-6)

    def test_empty_boxes(self):
        hsv = gpu_ops.bgr_to_hsv_u8(torch.from_numpy(_frames()["noise"]))
        self.assertEqual(gpu_ops.appearance_features(hsv, []).shape, (0, 384))


class TestColorFraction(unittest.TestCase):
    def test_matches_reference(self):
        for name, bgr in _frames().items():
            hsv = gpu_ops.bgr_to_hsv_u8(torch.from_numpy(bgr))
            for colour in ("red", "blue", "black", "white", "brown"):
                for box in BOXES:
                    expected = color_fraction(bgr, box, colour)
                    produced = gpu_ops.color_fraction(hsv, box, COLOR_RANGES[colour])
                    self.assertAlmostEqual(produced, expected, delta=0.01, msg=f"{name} {colour} {box}")

    def test_degenerate_box(self):
        hsv = gpu_ops.bgr_to_hsv_u8(torch.from_numpy(_frames()["noise"]))
        self.assertEqual(gpu_ops.color_fraction(hsv, (10.0, 10.0, 10.0, 10.0), COLOR_RANGES["red"]), 0.0)


class TestGallerySimilarity(unittest.TestCase):
    def test_matches_reference(self):
        rng = np.random.default_rng(3)
        gallery = rng.normal(size=(8, 384)).astype(np.float32)
        feature = rng.normal(size=384).astype(np.float32)
        expected = max(cosine_similarity(feature, row) for row in gallery)
        produced = gpu_ops.gallery_similarity(torch.from_numpy(feature), torch.from_numpy(gallery))
        self.assertAlmostEqual(produced, expected, places=5)

    def test_empty_gallery(self):
        self.assertEqual(gpu_ops.gallery_similarity(torch.zeros(384), torch.empty(0, 384)), -1.0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestCudaMatchesCpu(unittest.TestCase):
    def test_same_results_on_device(self):
        bgr = _frames()["noise"]
        host = torch.from_numpy(bgr)
        device = host.cuda()

        hsv_host = gpu_ops.bgr_to_hsv_u8(host)
        hsv_device = gpu_ops.bgr_to_hsv_u8(device)
        self.assertTrue(torch.equal(hsv_host, hsv_device.cpu()))

        features_host = gpu_ops.appearance_features(hsv_host, BOXES)
        features_device = gpu_ops.appearance_features(hsv_device, BOXES).cpu()
        torch.testing.assert_close(features_host, features_device, atol=1e-6, rtol=1e-5)

        for box in BOXES:
            self.assertAlmostEqual(
                gpu_ops.color_fraction(hsv_host, box, COLOR_RANGES["red"]),
                gpu_ops.color_fraction(hsv_device, box, COLOR_RANGES["red"]),
                places=6,
            )


if __name__ == "__main__":
    unittest.main()
