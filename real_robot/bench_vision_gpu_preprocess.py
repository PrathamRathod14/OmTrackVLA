#!/usr/bin/env python3
"""Compare legacy host preprocessing with shared CUDA vision preprocessing.

The benchmark reports isolated preprocessing, full DINOv3 plus SigLIP encoding,
feature similarity, and the resulting OmTrackVLA trajectory difference.

    ./run_local.sh python real_robot/bench_vision_gpu_preprocess.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as torch_functional

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from cache_gridpool import VisionCacheConfig, VisionFeatureCacher, grid_pool_tokens
from open_trackvla_hf import OpenTrackVLAConfig, OpenTrackVLAForWaypoint


def measure(function, repeats: int) -> tuple[float, float]:
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return statistics.median(samples), samples[max(0, int(0.95 * len(samples)) - 1)]


def pooled(dino_tokens: torch.Tensor, siglip_tokens: torch.Tensor, height: int, width: int):
    combined = torch.cat([dino_tokens, siglip_tokens], dim=-1)
    fine = grid_pool_tokens(combined, height, width, out_tokens=64)[0].float()
    coarse = grid_pool_tokens(combined, height, width, out_tokens=4)[0].float()
    return coarse, fine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default=str(PROJECT_DIR / "data/sample/frames/seed_100/17DRP5sb8fy/9/frame_00069.jpg"),
    )
    parser.add_argument("--preprocess-repeats", type=int, default=50)
    parser.add_argument("--encode-repeats", type=int, default=8)
    parser.add_argument("--compare-frames", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    os.environ["DINOV3_MODEL_PATH"] = str(PROJECT_DIR / "models/dinov3-vits16-pretrain-lvd1689m")
    os.environ["SIGLIP_MODEL_PATH"] = str(PROJECT_DIR / "models/siglip-so400m-patch14-384")

    source = Image.open(args.image).convert("RGB").resize((848, 480), Image.BICUBIC)
    rgb = np.asarray(source).copy()
    rgb_tensor = torch.from_numpy(rgb)
    vision = VisionFeatureCacher(VisionCacheConfig(image_size=384, batch_size=1, device="cuda")).eval()

    def legacy_preprocess():
        image = Image.fromarray(rgb).resize((384, 384), Image.BICUBIC)
        vision._preprocess_pils([image], which="dino")
        vision._preprocess_pils([image], which="siglip")

    def cuda_preprocess():
        vision._preprocess_rgb_tensor(rgb_tensor)

    def legacy_encode():
        image = Image.fromarray(rgb)
        dino, height, width = vision._encode_dino([image])
        siglip = vision._encode_siglip([image], out_hw=(height, width))
        return dino, siglip, height, width

    def cuda_encode():
        return vision.encode_rgb_tensor(rgb_tensor)

    preprocess_legacy = measure(legacy_preprocess, args.preprocess_repeats)
    preprocess_cuda = measure(cuda_preprocess, args.preprocess_repeats)
    encode_legacy = measure(legacy_encode, args.encode_repeats)
    encode_cuda = measure(cuda_encode, args.encode_repeats)

    dino_legacy, siglip_legacy, height, width = legacy_encode()
    dino_cuda, siglip_cuda, cuda_height, cuda_width = cuda_encode()
    if (height, width) != (cuda_height, cuda_width):
        raise RuntimeError("vision grid changed")

    def similarity(first: torch.Tensor, second: torch.Tensor) -> float:
        return float(torch_functional.cosine_similarity(first.flatten(), second.flatten(), dim=0))

    coarse_legacy, fine_legacy = pooled(dino_legacy, siglip_legacy, height, width)
    coarse_cuda, fine_cuda = pooled(dino_cuda, siglip_cuda, height, width)

    checkpoint = PROJECT_DIR / "models/OmTrackVLA-0.6B"
    qwen = PROJECT_DIR / "models/Qwen3-0.6B"
    config = OpenTrackVLAConfig.from_pretrained(str(checkpoint))
    config.llm_name = str(qwen)
    planner = OpenTrackVLAForWaypoint.from_pretrained(
        str(checkpoint), config=config, low_cpu_mem_usage=True
    ).eval().cuda()

    def trajectory(coarse: torch.Tensor, fine: torch.Tensor) -> torch.Tensor:
        history = torch.cat([coarse] * 31, dim=0).unsqueeze(0)
        coarse_tidx = torch.arange(31, device="cuda").repeat_interleave(4).unsqueeze(0)
        fine_tokens = fine.unsqueeze(0)
        fine_tidx = torch.full((1, fine_tokens.size(1)), 31, dtype=torch.long, device="cuda")
        return planner(
            history,
            coarse_tidx,
            fine_tokens,
            fine_tidx,
            ["Follow the person who is wearing dark red T-shirt."],
        )[0].detach().float()

    trajectory_legacy = trajectory(coarse_legacy, fine_legacy)
    trajectory_cuda = trajectory(coarse_cuda, fine_cuda)
    trajectory_difference = (trajectory_legacy - trajectory_cuda).abs()

    print(f"preprocess legacy median/p95: {preprocess_legacy[0]:.3f}/{preprocess_legacy[1]:.3f} ms")
    print(f"preprocess CUDA   median/p95: {preprocess_cuda[0]:.3f}/{preprocess_cuda[1]:.3f} ms")
    print(f"encode legacy     median/p95: {encode_legacy[0]:.3f}/{encode_legacy[1]:.3f} ms")
    print(f"encode CUDA       median/p95: {encode_cuda[0]:.3f}/{encode_cuda[1]:.3f} ms")
    print(f"DINO token cosine:   {similarity(dino_legacy, dino_cuda):.9f}")
    print(f"SigLIP token cosine: {similarity(siglip_legacy, siglip_cuda):.9f}")
    print(f"coarse token cosine: {similarity(coarse_legacy, coarse_cuda):.9f}")
    print(f"fine token cosine:   {similarity(fine_legacy, fine_cuda):.9f}")
    print(f"trajectory max abs difference:  {float(trajectory_difference.max()):.9f}")
    print(f"trajectory mean abs difference: {float(trajectory_difference.mean()):.9f}")
    print("legacy trajectory:", trajectory_legacy.cpu().tolist())
    print("CUDA trajectory:  ", trajectory_cuda.cpu().tolist())

    frame_paths = sorted(Path(args.image).parent.glob("*.jpg"))[: max(1, args.compare_frames)]
    frame_max_differences = []
    frame_mean_differences = []
    dino_cosines = []
    siglip_cosines = []
    for frame_path in frame_paths:
        frame = Image.open(frame_path).convert("RGB").resize((848, 480), Image.BICUBIC)
        frame_rgb = np.asarray(frame).copy()
        frame_pil = Image.fromarray(frame_rgb)
        old_dino, frame_height, frame_width = vision._encode_dino([frame_pil])
        old_siglip = vision._encode_siglip([frame_pil], out_hw=(frame_height, frame_width))
        new_dino, new_siglip, _, _ = vision.encode_rgb_tensor(torch.from_numpy(frame_rgb))
        old_coarse, old_fine = pooled(old_dino, old_siglip, frame_height, frame_width)
        new_coarse, new_fine = pooled(new_dino, new_siglip, frame_height, frame_width)
        old_trajectory = trajectory(old_coarse, old_fine)
        new_trajectory = trajectory(new_coarse, new_fine)
        difference = (old_trajectory - new_trajectory).abs()
        frame_max_differences.append(float(difference.max()))
        frame_mean_differences.append(float(difference.mean()))
        dino_cosines.append(similarity(old_dino, new_dino))
        siglip_cosines.append(similarity(old_siglip, new_siglip))

    print(f"comparison frames: {len(frame_paths)}")
    print(f"minimum DINO token cosine:   {min(dino_cosines):.9f}")
    print(f"minimum SigLIP token cosine: {min(siglip_cosines):.9f}")
    print(f"maximum trajectory difference across frames: {max(frame_max_differences):.9f}")
    print(f"mean trajectory difference across frames:    {statistics.mean(frame_mean_differences):.9f}")


if __name__ == "__main__":
    main()
