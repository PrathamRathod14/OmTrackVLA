#!/usr/bin/env python3
"""Estimate the RealSense height, pitch, and roll above base_link from the floor.

The camera is not in Ridgeback's TF tree, so ``camera_mount`` in ridgeback.yaml is
configured manually. With the robot on a flat floor and the lower part of the view
showing open floor, this fits the floor plane in one depth frame (read-only):

    source /opt/ros/jazzy/setup.bash
    /usr/bin/python3 real_robot/calibrate_camera_mount.py --namespace r100_0160

Forward/lateral offset and yaw cannot be observed from the floor; measure them from
base_link (the centre of the chassis at floor level) by tape.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


def fit_floor(points: np.ndarray, iterations: int = 400, tolerance: float = 0.02, seed: int = 0):
    """RANSAC plane fit in the optical frame. Returns (unit normal toward camera, offset, inliers)."""
    rng = np.random.default_rng(seed)
    best = (None, 0.0, np.zeros(len(points), dtype=bool))
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        offset = -float(normal @ sample[0])
        inliers = np.abs(points @ normal + offset) < tolerance
        if inliers.sum() > best[2].sum():
            best = (normal, offset, inliers)
    normal, offset, inliers = best
    # Refine with least squares on the inliers.
    subset = points[inliers]
    centroid = subset.mean(axis=0)
    normal = np.linalg.svd(subset - centroid)[2][-1]
    offset = -float(normal @ centroid)
    if offset < 0:  # orient the normal from the floor toward the camera
        normal, offset = -normal, -offset
    return normal, offset, inliers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", default="r100_0160")
    parser.add_argument("--depth-scale", type=float, default=0.001)
    args = parser.parse_args()

    rclpy.init()
    node = Node("omtrackvla_mount_calibration")
    received = {}
    prefix = f"/{args.namespace}/camera/depth"
    node.create_subscription(Image, f"{prefix}/image_rect_raw", lambda m: received.setdefault("depth", m), qos_profile_sensor_data)
    node.create_subscription(CameraInfo, f"{prefix}/camera_info", lambda m: received.setdefault("info", m), qos_profile_sensor_data)
    deadline = time.time() + 10.0
    while len(received) < 2 and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    if len(received) < 2:
        raise SystemExit("No depth image/camera_info received.")

    image, info = received["depth"], received["info"]
    depth = np.frombuffer(image.data, dtype=np.uint16).reshape(image.height, image.step // 2)[:, : image.width]
    depth = depth.astype(np.float64) * args.depth_scale
    k = np.asarray(info.k).reshape(3, 3)
    rows, cols = np.mgrid[image.height // 2 : image.height : 2, 0 : image.width : 2]
    z = depth[rows, cols]
    valid = (z > 0.2) & (z < 5.0)
    u, v, z = cols[valid], rows[valid], z[valid]
    points = np.stack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z], axis=1)
    if len(points) < 500:
        raise SystemExit("Too few valid depth points in the lower half of the image.")
    normal, height, inliers = fit_floor(points)

    # Floor normal in the optical frame (x right, y down, z forward) points "up" = -y when level.
    pitch = math.atan2(normal[2], -normal[1])  # positive: camera tilted down
    roll = math.atan2(-normal[0], -normal[1])
    print(f"Floor inliers: {int(inliers.sum())}/{len(points)} ({inliers.mean():.0%})")
    print(f"Camera optical centre height above floor: {height:.3f} m")
    print(f"Pitch (down positive): {math.degrees(pitch):.2f} deg = {pitch:.4f} rad")
    print(f"Roll: {math.degrees(roll):.2f} deg = {roll:.4f} rad")
    if inliers.mean() < 0.4:
        print("WARNING: the floor is not dominant in the lower image; clear the view and repeat.")
    print("\nSet in ridgeback.yaml (x, y, yaw measured by tape; base_link is at floor level):")
    print(f"    camera_mount: [X, Y, {height:.3f}, {roll:.4f}, {pitch:.4f}, YAW]")


if __name__ == "__main__":
    main()
