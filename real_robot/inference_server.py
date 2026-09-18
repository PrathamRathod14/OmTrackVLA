#!/usr/bin/env python3
"""Loopback-only OmTrackVLA inference service for the ROS 2 Ridgeback bridge."""

from __future__ import annotations

import argparse
import base64
from collections import deque
import json
import os
from pathlib import Path
import socket
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from protocol import receive_message, send_message


class OmTrackInference:
    def __init__(self, project_dir: Path, device: str, history: int, prediction_dt: float, waypoint_index: int):
        self.project_dir = project_dir
        self.device = torch.device(device)
        self.history = history
        self.prediction_dt = prediction_dt
        self.waypoint_index = waypoint_index
        self.coarse_history = deque(maxlen=history)
        self.last_prompt: Optional[str] = None

        checkpoint = project_dir / "models" / "OmTrackVLA-0.6B"
        qwen = project_dir / "models" / "Qwen3-0.6B"
        dino = project_dir / "models" / "dinov3-vits16-pretrain-lvd1689m"
        siglip = project_dir / "models" / "siglip-so400m-patch14-384"
        required = [
            checkpoint / "model.safetensors",
            qwen / "model.safetensors",
            dino / "model.safetensors",
            siglip / "model.safetensors",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing model files:\n  " + "\n  ".join(missing))

        os.environ["DINOV3_MODEL_PATH"] = str(dino)
        os.environ["SIGLIP_MODEL_PATH"] = str(siglip)
        from cache_gridpool import VisionCacheConfig, VisionFeatureCacher, grid_pool_tokens
        from open_trackvla_hf import OpenTrackVLAConfig, OpenTrackVLAForWaypoint

        config = OpenTrackVLAConfig.from_pretrained(str(checkpoint))
        config.llm_name = str(qwen)
        self.planner = OpenTrackVLAForWaypoint.from_pretrained(
            str(checkpoint), config=config, low_cpu_mem_usage=True
        ).eval().to(self.device)
        vision_config = VisionCacheConfig(image_size=384, batch_size=1, device=str(self.device))
        self.vision = VisionFeatureCacher(vision_config).eval()
        self.grid_pool_tokens = grid_pool_tokens

    def reset(self) -> None:
        self.coarse_history.clear()

    @torch.inference_mode()
    def infer(self, rgb: np.ndarray, prompt: str) -> Tuple[list, list]:
        if prompt != self.last_prompt:
            self.reset()
            self.last_prompt = prompt

        from PIL import Image

        image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        dino_tokens, height, width = self.vision._encode_dino([image])
        siglip_tokens = self.vision._encode_siglip([image], out_hw=(height, width))
        combined = torch.cat([dino_tokens, siglip_tokens], dim=-1)
        fine = self.grid_pool_tokens(combined, height, width, out_tokens=64)[0].float()
        coarse = self.grid_pool_tokens(combined, height, width, out_tokens=4)[0].float()
        self.coarse_history.append(coarse.cpu())

        history = list(self.coarse_history)
        if len(history) < self.history:
            history = [history[0]] * (self.history - len(history)) + history
        history = history[-self.history:]
        coarse_tokens = torch.cat([item.to(self.device) for item in history], dim=0).unsqueeze(0)
        coarse_tidx = torch.arange(self.history, device=self.device).repeat_interleave(4).unsqueeze(0)
        fine_tokens = fine.to(self.device).unsqueeze(0)
        fine_tidx = torch.full((1, fine_tokens.size(1)), self.history, dtype=torch.long, device=self.device)
        trajectory = self.planner(
            coarse_tokens,
            coarse_tidx,
            fine_tokens,
            fine_tidx,
            [prompt],
        )
        trajectory_cpu = trajectory[0].detach().float().cpu().tolist()
        index = min(max(0, self.waypoint_index), trajectory.shape[1] - 1)
        waypoint = trajectory[0, index].detach().float().cpu()
        command = [float(waypoint[0] / self.prediction_dt), float(waypoint[1] / self.prediction_dt)]
        command.append(float(waypoint[2] / self.prediction_dt) if waypoint.numel() >= 3 else 0.0)
        return command, trajectory_cpu


def decode_request_image(request: dict) -> np.ndarray:
    if request.get("encoding") != "jpeg":
        raise ValueError("only JPEG requests are supported")
    encoded = request.get("image_b64")
    if not isinstance(encoded, str):
        raise ValueError("image_b64 is missing")
    compressed = np.frombuffer(base64.b64decode(encoded, validate=True), dtype=np.uint8)
    bgr = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("JPEG decode failed")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def serve(args: argparse.Namespace) -> None:
    model = None if args.dummy else OmTrackInference(
        PROJECT_DIR, args.device, args.history, args.prediction_dt, args.waypoint_index
    )
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(json.dumps({"status": "ready", "host": args.host, "port": args.port, "dummy": args.dummy}), flush=True)
    while True:
        connection, address = server.accept()
        if address[0] not in ("127.0.0.1", "::1"):
            connection.close()
            continue
        connection.settimeout(args.socket_timeout)
        try:
            while True:
                request = receive_message(connection)
                started = time.monotonic()
                request_id = request.get("request_id")
                try:
                    prompt = str(request.get("prompt", "follow the person")).strip()[:256]
                    if not prompt:
                        raise ValueError("prompt is empty")
                    image = decode_request_image(request)
                    if model is None:
                        command, trajectory = [0.0, 0.0, 0.0], [[0.0, 0.0, 0.0]]
                    else:
                        command, trajectory = model.infer(image, prompt)
                    send_message(connection, {
                        "ok": True,
                        "request_id": request_id,
                        "command": command,
                        "trajectory": trajectory,
                        "inference_seconds": time.monotonic() - started,
                    })
                except Exception as exc:
                    send_message(connection, {
                        "ok": False,
                        "request_id": request_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
        except (ConnectionError, OSError, ValueError):
            pass
        finally:
            connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--history", type=int, default=31)
    parser.add_argument("--prediction-dt", type=float, default=0.1)
    parser.add_argument("--waypoint-index", type=int, default=1)
    parser.add_argument("--socket-timeout", type=float, default=10.0)
    parser.add_argument("--dummy", action="store_true", help="Protocol test mode; always returns zero velocity")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        serve(parse_args())
    except KeyboardInterrupt:
        pass
