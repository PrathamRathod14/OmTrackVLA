"""Numerical checks for the shared CUDA vision preprocessing path."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as torch_functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cache_gridpool import resize_rgb_tensor_for_vision


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestVisionGpuPreprocess(unittest.TestCase):
    def test_cuda_resize_is_close_to_pil_reference(self):
        rng = np.random.default_rng(28)
        rgb = rng.integers(0, 256, size=(480, 848, 3), dtype=np.uint8)
        reference_image = Image.fromarray(rgb).resize((384, 384), Image.BICUBIC)
        reference = (
            torch.from_numpy(np.asarray(reference_image).copy())
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(torch.float32)
            .mul_(1.0 / 255.0)
        )

        produced = resize_rgb_tensor_for_vision(torch.from_numpy(rgb), 384, torch.device("cuda")).cpu()
        difference = (reference - produced).abs()
        cosine = torch_functional.cosine_similarity(reference.flatten(), produced.flatten(), dim=0)

        self.assertEqual(tuple(produced.shape), (1, 3, 384, 384))
        # Torchvision's antialiased CUDA bicubic kernel is not bit-identical to
        # Pillow's bicubic kernel. The bounds cover high-frequency random pixels;
        # natural images are closer and downstream trajectory equivalence is
        # measured separately by bench_vision_gpu_preprocess.py.
        self.assertLess(float(difference.mean()), 0.001)
        self.assertLess(float(difference.max()), 0.03)
        self.assertGreater(float(cosine), 0.99998)

    def test_rejects_wrong_shape_or_type(self):
        with self.assertRaises(ValueError):
            resize_rgb_tensor_for_vision(torch.zeros(10, 10, 3), 384, torch.device("cuda"))
        with self.assertRaises(ValueError):
            resize_rgb_tensor_for_vision(torch.zeros(3, 10, 10, dtype=torch.uint8), 384, torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()
