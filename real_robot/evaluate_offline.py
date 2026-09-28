#!/usr/bin/env python3
"""Replay recorded frames through target perception (and optionally OmTrackVLA).

Runs in the OmTrackVLA environment and never touches ROS or the robot:

    ./run_local.sh python real_robot/evaluate_offline.py frames/run1 \
        --prompt "Follow the person wearing a black jacket." --out eval/run1 [--with-omtrack]

Input is a directory of frames (sorted by filename, as written by
export_bag_frames.py) or a video file. Frames should be sampled at the live
inference rate so tracker behaviour matches deployment. Outputs results.csv,
annotated.mp4, and summary.json.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path
import sys
import time
from typing import Iterator, Tuple

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def read_frames(source: Path, video_stride: int) -> Iterator[Tuple[str, np.ndarray]]:
    if source.is_dir():
        for path in sorted(p for p in source.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")):
            bgr = cv2.imread(str(path))
            if bgr is not None:
                yield path.name, bgr
        return
    capture = cv2.VideoCapture(str(source))
    index = 0
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        if index % video_stride == 0:
            yield f"frame_{index:06d}", bgr
        index += 1
    capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="Frame directory or video file")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--with-omtrack", action="store_true", help="Also run OmTrackVLA and the trajectory check")
    parser.add_argument("--video-stride", type=int, default=15, help="Use every Nth video frame (30 fps / 15 = 2 Hz)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-fps", type=float, default=2.0)
    parser.add_argument(
        "--planner-rate",
        type=float,
        default=2.0,
        help="Planner cap used during replay; use a high value only for throughput benchmarking",
    )
    parser.add_argument(
        "--legacy-vision-preprocess",
        action="store_true",
        help="Use the former CPU/PIL OmTrackVLA preprocessing path for comparison",
    )
    args = parser.parse_args()

    if args.with_omtrack:
        from inference_server import OmTrackInference

        model = OmTrackInference(
            PROJECT_DIR,
            args.device,
            31,
            0.1,
            1,
            0.35,
            0.25,
            0.25,
            2,
            3,
            0.72,
            planner_rate=args.planner_rate,
            cuda_vision_preprocess=not args.legacy_vision_preprocess,
        )

        def run(rgb: np.ndarray) -> Tuple[dict, np.ndarray]:
            result = model.infer(rgb, args.prompt)
            jpeg = np.frombuffer(base64.b64decode(result.pop("target_image_b64")), dtype=np.uint8)
            return result, cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    else:
        from target_perception import TargetPerception

        perception = TargetPerception(PROJECT_DIR, device=args.device)

        def run(rgb: np.ndarray) -> Tuple[dict, np.ndarray]:
            target = perception.update(rgb, args.prompt)
            return {
                "target_state": target.state,
                "target_valid": target.valid,
                "target_reason": target.reason,
                "target_track_id": target.track_id,
                "target_bbox": target.bbox,
                "target_reid_similarity": target.reid_similarity,
                "target_people_count": target.people_count,
                "target_debug": target.debug,
            }, target.annotated_bgr

    args.out.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame", "seconds", "target_state", "target_valid", "target_reason", "target_track_id",
        "target_bbox", "target_reid_similarity", "target_people_count",
        "trajectory_valid", "trajectory_reason", "command", "target_debug",
    ]
    writer_video = None
    rows = []
    with open(args.out / "results.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for name, bgr in read_frames(args.source, max(1, args.video_stride)):
            started = time.monotonic()
            result, annotated = run(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            result["seconds"] = round(time.monotonic() - started, 4)
            result["frame"] = name
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in result.items()})
            rows.append(result)
            if writer_video is None:
                height, width = annotated.shape[:2]
                writer_video = cv2.VideoWriter(
                    str(args.out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.output_fps, (width, height)
                )
            writer_video.write(annotated)
    if writer_video is not None:
        writer_video.release()

    locked_ids = [row["target_track_id"] for row in rows if row["target_valid"]]
    states = [row["target_state"] for row in rows]
    summary = {
        "prompt": args.prompt,
        "frames": len(rows),
        "locked_frames": len(locked_ids),
        "locked_fraction": round(len(locked_ids) / len(rows), 3) if rows else 0.0,
        "first_lock_frame": next((i for i, row in enumerate(rows) if row["target_valid"]), None),
        # Changes of the locked track ID; each needs manual review for a true identity switch.
        "locked_track_id_changes": sum(1 for a, b in zip(locked_ids, locked_ids[1:]) if a != b),
        "locked_track_ids": sorted(set(locked_ids)),
        "uncertain_entries": sum(1 for a, b in zip(states, states[1:]) if a == "LOCKED" and b != "LOCKED"),
        "lost_frames": states.count("LOST"),
        "recoveries": sum(1 for row in rows if row["target_reason"] == "identity_recovered"),
        "mean_seconds": round(float(np.mean([row["seconds"] for row in rows])), 4) if rows else None,
        "max_seconds": round(float(np.max([row["seconds"] for row in rows])), 4) if rows else None,
    }
    if args.with_omtrack:
        summary["trajectory_mismatch_frames"] = sum(
            1 for row in rows if row["target_valid"] and not row.get("trajectory_valid")
        )
        summary["motion_permitted_frames"] = sum(1 for row in rows if row.get("trajectory_valid"))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
