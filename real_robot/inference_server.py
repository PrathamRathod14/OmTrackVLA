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
from typing import Optional

import cv2
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from protocol import receive_message, send_message
from target_perception import TargetPerception, trajectory_is_directionally_consistent


class OmTrackInference:
    def __init__(
        self,
        project_dir: Path,
        device: str,
        history: int,
        prediction_dt: float,
        waypoint_index: int,
        grounding_box_threshold: float,
        grounding_text_threshold: float,
        person_threshold: float,
        grounding_interval: int,
        target_acquire_frames: int,
        reid_match_threshold: float,
        planner_rate: float = 2.0,
        target_view_width: int = 512,
        cuda_vision_preprocess: bool = True,
    ):
        self.project_dir = project_dir
        self.device = torch.device(device)
        self.history = history
        self.prediction_dt = prediction_dt
        self.waypoint_index = waypoint_index
        self.coarse_history = deque(maxlen=history)
        self.last_prompt: Optional[str] = None
        self.target_was_valid = False
        self.planner_period = 1.0 / max(0.1, float(planner_rate))
        self.target_view_width = max(160, int(target_view_width))
        self.cuda_vision_preprocess = bool(cuda_vision_preprocess and self.device.type == "cuda")
        self.next_planner_due = 0.0
        self.last_planner_time: Optional[float] = None
        self.last_planner_command: Optional[list] = None
        self.last_planner_trajectory: Optional[list] = None

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
        self.target_perception = TargetPerception(
            project_dir,
            device=device,
            grounding_box_threshold=grounding_box_threshold,
            grounding_text_threshold=grounding_text_threshold,
            person_threshold=person_threshold,
            grounding_interval=grounding_interval,
            acquire_frames=target_acquire_frames,
            reid_match_threshold=reid_match_threshold,
        )

    def reset(self) -> None:
        self.coarse_history.clear()
        self.next_planner_due = 0.0
        self.last_planner_time = None
        self.last_planner_command = None
        self.last_planner_trajectory = None

    def _encode_target_view(self, annotated_bgr: np.ndarray) -> str:
        """Encode the target-manager view; model coordinates are displayed in RViz."""
        target_view = annotated_bgr
        if target_view.shape[1] > self.target_view_width:
            scale = self.target_view_width / float(target_view.shape[1])
            target_view = cv2.resize(
                target_view,
                (self.target_view_width, max(1, int(round(target_view.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        encoded_ok, encoded_jpeg = cv2.imencode(
            ".jpg", target_view, [cv2.IMWRITE_JPEG_QUALITY, 82]
        )
        if not encoded_ok:
            raise RuntimeError("failed to encode target visualization")
        return base64.b64encode(encoded_jpeg.tobytes()).decode("ascii")

    @torch.inference_mode()
    def infer(self, rgb: np.ndarray, prompt: str, reset_target: bool = False) -> dict:
        if reset_target:
            self.target_perception.reset()
            self.target_was_valid = False
        if prompt != self.last_prompt:
            self.reset()
            self.target_perception.reset()
            self.last_prompt = prompt
            self.target_was_valid = False

        target = self.target_perception.update(rgb, prompt)
        result = {
            "target_query": target.query,
            "target_state": target.state,
            "target_valid": target.valid,
            "target_reason": target.reason,
            "target_track_id": target.track_id,
            "target_bbox": target.bbox,
            "target_grounding_score": target.grounding_score,
            "target_person_score": target.person_score,
            "target_reid_similarity": target.reid_similarity,
            "target_people_count": target.people_count,
            "target_debug": target.debug,
        }
        if not target.valid:
            if self.target_was_valid:
                self.reset()
            self.target_was_valid = False
            result.update({
                "planner_ran": False,
                "planner_updated": False,
                "planner_age_seconds": None,
                "command": [0.0, 0.0, 0.0],
                "trajectory": [[0.0, 0.0, 0.0] for _ in range(8)],
                "trajectory_valid": False,
                "trajectory_reason": "target_not_locked",
                "target_image_b64": self._encode_target_view(target.annotated_bgr),
            })
            return result
        if not self.target_was_valid:
            self.reset()
        self.target_was_valid = True

        now = time.monotonic()
        planner_updated = self.last_planner_trajectory is None or now >= self.next_planner_due
        if planner_updated:
            if self.cuda_vision_preprocess:
                rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.uint8))
                dino_tokens, siglip_tokens, height, width = self.vision.encode_rgb_tensor(rgb_tensor)
            else:
                from PIL import Image

                image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
                dino_tokens, height, width = self.vision._encode_dino([image])
                siglip_tokens = self.vision._encode_siglip([image], out_hw=(height, width))
            combined = torch.cat([dino_tokens, siglip_tokens], dim=-1)
            fine = self.grid_pool_tokens(combined, height, width, out_tokens=64)[0].float()
            coarse = self.grid_pool_tokens(combined, height, width, out_tokens=4)[0].float()
            # Kept on the inference device: 31 x 4 x 1536 floats is about 762 KB, so
            # holding it in VRAM removes a device-to-host and host-to-device copy per frame.
            self.coarse_history.append(coarse.detach())

            history = list(self.coarse_history)
            if len(history) < self.history:
                history = [history[0]] * (self.history - len(history)) + history
            history = history[-self.history:]
            coarse_tokens = torch.cat(history, dim=0).unsqueeze(0)
            coarse_tidx = torch.arange(self.history, device=self.device).repeat_interleave(4).unsqueeze(0)
            fine_tokens = fine.unsqueeze(0)
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
            self.last_planner_command = command
            self.last_planner_trajectory = trajectory_cpu
            self.last_planner_time = time.monotonic()
            self.next_planner_due = self.last_planner_time + self.planner_period
        else:
            if self.last_planner_command is None or self.last_planner_trajectory is None:
                raise RuntimeError("planner cache is unexpectedly empty")
            command = list(self.last_planner_command)
            trajectory_cpu = list(self.last_planner_trajectory)
        if self.last_planner_time is None:
            raise RuntimeError("planner timestamp is unexpectedly empty")
        trajectory_valid, trajectory_reason = trajectory_is_directionally_consistent(
            target.bbox,
            rgb.shape[1],
            trajectory_cpu,
        )
        result.update({
            "planner_ran": True,
            "planner_updated": planner_updated,
            "planner_age_seconds": max(0.0, time.monotonic() - float(self.last_planner_time)),
            "command": command,
            "trajectory": trajectory_cpu,
            "trajectory_valid": bool(target.valid and trajectory_valid),
            "trajectory_reason": trajectory_reason if target.valid else "target_not_locked",
            "target_image_b64": self._encode_target_view(target.annotated_bgr),
        })
        return result


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
        PROJECT_DIR,
        args.device,
        args.history,
        args.prediction_dt,
        args.waypoint_index,
        args.grounding_box_threshold,
        args.grounding_text_threshold,
        args.person_threshold,
        args.grounding_interval,
        args.target_acquire_frames,
        args.reid_match_threshold,
        args.planner_rate,
        args.target_view_width,
        not args.legacy_vision_preprocess,
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
                        result = {
                            "planner_ran": False,
                            "planner_updated": False,
                            "planner_age_seconds": None,
                            "command": [0.0, 0.0, 0.0],
                            "trajectory": [[0.0, 0.0, 0.0]],
                            "target_valid": False,
                            "target_state": "DISABLED",
                            "target_reason": "dummy_inference",
                            "trajectory_valid": False,
                            "trajectory_reason": "dummy_inference",
                        }
                    else:
                        result = model.infer(image, prompt, bool(request.get("reset_target")))
                    response = {
                        "ok": True,
                        "request_id": request_id,
                        "inference_seconds": time.monotonic() - started,
                    }
                    response.update(result)
                    send_message(connection, response)
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
    parser.add_argument("--grounding-box-threshold", type=float, default=0.35)
    parser.add_argument("--grounding-text-threshold", type=float, default=0.25)
    parser.add_argument("--person-threshold", type=float, default=0.25)
    parser.add_argument("--grounding-interval", type=int, default=8)
    parser.add_argument("--target-acquire-frames", type=int, default=3)
    parser.add_argument("--reid-match-threshold", type=float, default=0.72)
    parser.add_argument("--planner-rate", type=float, default=2.0)
    parser.add_argument("--target-view-width", type=int, default=512)
    parser.add_argument(
        "--legacy-vision-preprocess",
        action="store_true",
        help="Use the former CPU/PIL preprocessing path for comparison or rollback",
    )
    parser.add_argument("--dummy", action="store_true", help="Protocol test mode; always returns zero velocity")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        serve(parse_args())
    except KeyboardInterrupt:
        pass
