"""Metric target localization and trajectory consistency for the Ridgeback bridge.

Pure NumPy so it can be unit-tested without ROS or GPU models. The RealSense depth
stream is not aligned to color, so depth pixels are projected into the color image
with the published depth-to-color extrinsics before sampling the target's torso.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import numpy as np


# camera_link (x forward, y left, z up) <- camera optical frame (x right, y down, z forward)
OPTICAL_TO_LINK = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


@dataclass(frozen=True)
class CameraGeometry:
    depth_k: np.ndarray  # 3x3 depth intrinsics
    color_k: np.ndarray  # 3x3 color intrinsics
    depth_to_color_r: np.ndarray  # 3x3, p_color = R @ p_depth + t
    depth_to_color_t: np.ndarray  # 3
    mount: Sequence[float]  # camera_link pose in base_link: x, y, z, roll, pitch, yaw


def extrinsics_rotation(values: Sequence[float]) -> np.ndarray:
    """realsense2_camera_msgs/Extrinsics stores the rotation column-major."""
    return np.asarray(values, dtype=np.float64).reshape(3, 3, order="F")


def rotation_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def optical_to_base(point: np.ndarray, mount: Sequence[float]) -> np.ndarray:
    x, y, z, roll, pitch, yaw = [float(value) for value in mount]
    return rotation_rpy(roll, pitch, yaw) @ (OPTICAL_TO_LINK @ np.asarray(point, dtype=np.float64)) + np.array([x, y, z])


def locate_target(
    depth_m: np.ndarray,
    bbox: Sequence[float],
    geometry: CameraGeometry,
    stride: int = 2,
    min_points: int = 30,
    max_range: float = 6.0,
    image_size: Optional[Tuple[int, int]] = None,
    min_coverage: float = 0.4,
    min_consistency: float = 0.6,
) -> Tuple[Optional[np.ndarray], str]:
    """Return the target torso position in base_link, or None and a reason.

    Depth pixels are back-projected, moved into the color optical frame, and
    projected into the color image. Only points landing in the central torso
    region of the person box are used, and their median depth places the box
    center ray in 3D, which is robust to background pixels at the box edges.

    A person closer than the sensor's minimum range has no valid depth, and the
    depth-to-color baseline then projects background points into the person's
    box, which would place a too-close person far away. So the torso region must
    be densely covered (``min_coverage`` of the expected samples) by one
    consistent surface (``min_consistency`` within 0.25 m or 10% of the median),
    and a box spanning the full image height is reported as too close.
    """
    if bbox is None:
        return None, "target_bbox_missing"
    x1, y1, x2, y2 = [float(value) for value in bbox]
    if image_size is not None and y1 <= 2.0 and y2 >= image_size[1] - 2.0:
        return None, "target_too_close_for_depth"
    depth = np.asarray(depth_m, dtype=np.float32)[::stride, ::stride]
    rows, cols = np.nonzero((depth > 0.1) & (depth < max_range) & np.isfinite(depth))
    if rows.size == 0:
        return None, "depth_empty"
    z = depth[rows, cols].astype(np.float64)
    u = cols.astype(np.float64) * stride
    v = rows.astype(np.float64) * stride
    kd = geometry.depth_k
    points = np.stack([(u - kd[0, 2]) * z / kd[0, 0], (v - kd[1, 2]) * z / kd[1, 1], z])
    points = geometry.depth_to_color_r @ points + np.asarray(geometry.depth_to_color_t, dtype=np.float64).reshape(3, 1)
    in_front = points[2] > 0.1
    points = points[:, in_front]
    kc = geometry.color_k
    pu = kc[0, 0] * points[0] / points[2] + kc[0, 2]
    pv = kc[1, 1] * points[1] / points[2] + kc[1, 2]

    width, height = x2 - x1, y2 - y1
    torso = (
        (pu >= x1 + 0.3 * width) & (pu <= x2 - 0.3 * width)
        & (pv >= y1 + 0.2 * height) & (pv <= y1 + 0.6 * height)
    )
    count = int(torso.sum())
    if count < min_points:
        return None, "target_depth_insufficient"
    # Depth samples expected in the torso region if the whole surface had valid depth.
    density = (kd[0, 0] * kd[1, 1]) / (kc[0, 0] * kc[1, 1]) / float(stride * stride)
    expected = (0.4 * width) * (0.4 * height) * density
    if count < min_coverage * expected:
        return None, "target_depth_sparse"
    torso_z = points[2, torso]
    target_z = float(np.median(torso_z))
    consistent = np.abs(torso_z - target_z) <= max(0.25, 0.1 * target_z)
    if float(consistent.mean()) < min_consistency:
        return None, "target_depth_inconsistent"
    center_u = 0.5 * (x1 + x2)
    center_v = y1 + 0.4 * height
    optical = np.array([(center_u - kc[0, 2]) * target_z / kc[0, 0], (center_v - kc[1, 2]) * target_z / kc[1, 1], target_z])
    return optical_to_base(optical, geometry.mount), "target_located"


def trajectory_follows_target(
    trajectory: Sequence[Sequence[float]],
    target_xy: Sequence[float],
    min_follow_distance: float = 0.9,
    retreat_tolerance: float = 0.05,
    reject_retreat: bool = True,
) -> Tuple[bool, str, float, float]:
    """Check an OmTrackVLA path (base_link, metres) against the target position.

    The path end may not approach the target closer than ``min_follow_distance``,
    and, while the target is beyond that distance, may not increase the distance to
    the target by more than ``retreat_tolerance``. Lateral detours around obstacles
    barely change the distance and are therefore not rejected.

    Returns (ok, reason, current_distance, final_distance).
    """
    tx, ty = float(target_xy[0]), float(target_xy[1])
    current = math.hypot(tx, ty)
    if not trajectory:
        return False, "trajectory_missing", current, current
    end_x, end_y = float(trajectory[-1][0]), float(trajectory[-1][1])
    final = math.hypot(tx - end_x, ty - end_y)
    moved = math.hypot(end_x, end_y)
    if final < min_follow_distance and final < current - 1e-3:
        return False, "path_too_close_to_target", current, final
    if reject_retreat and current > min_follow_distance and final > current + retreat_tolerance and moved > retreat_tolerance:
        return False, "path_moves_away_from_target", current, final
    return True, "path_follows_target", current, final


def fuse_target_command(
    raw_command: Sequence[float],
    target_xy: Sequence[float],
    follow_distance: float = 0.9,
    target_weight: float = 0.75,
    linear_gain: float = 0.4,
    minimum_speed: float = 0.04,
    max_linear_speed: float = 0.2,
    yaw_gain: float = 0.8,
    max_angular_speed: float = 0.35,
) -> Tuple[Tuple[float, float, float], dict]:
    """Fuse an OmTrackVLA command with the depth-localized leader position.

    An OmTrackVLA proposal that has a positive projection toward the leader is
    retained as a detour component and blended toward the leader. An opposite or
    perpendicular proposal is replaced by the leader direction. Speed remains
    proportional to distance error and translation stops at ``follow_distance``.
    """
    if len(raw_command) != 3 or len(target_xy) < 2:
        raise ValueError("fusion requires a 3-axis command and 2D target")
    raw_x, raw_y, raw_yaw = [float(value) for value in raw_command]
    tx, ty = float(target_xy[0]), float(target_xy[1])
    values = (raw_x, raw_y, raw_yaw, tx, ty)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("fusion input is non-finite")
    distance = math.hypot(tx, ty)
    if distance <= 1e-6 or tx <= 0.0:
        return (0.0, 0.0, 0.0), {
            "reason": "target_not_in_front",
            "target_distance": round(distance, 3),
            "raw_aligned": False,
            "goal_xy": [0.0, 0.0],
        }

    ux, uy = tx / distance, ty / distance
    bearing = math.atan2(ty, tx)
    target_yaw = max(-max_angular_speed, min(max_angular_speed, yaw_gain * bearing))
    distance_error = distance - follow_distance
    goal_distance = max(0.0, distance_error)
    goal_xy = [ux * goal_distance, uy * goal_distance]
    if distance_error <= 0.0:
        return (0.0, 0.0, target_yaw), {
            "reason": "within_follow_distance",
            "target_distance": round(distance, 3),
            "distance_error": round(distance_error, 3),
            "target_bearing": round(bearing, 3),
            "raw_aligned": False,
            "goal_xy": goal_xy,
        }

    raw_speed = min(max_linear_speed, math.hypot(raw_x, raw_y))
    raw_aligned = False
    model_x = model_y = 0.0
    if raw_speed > 1e-6:
        model_x, model_y = raw_x / math.hypot(raw_x, raw_y), raw_y / math.hypot(raw_x, raw_y)
        raw_aligned = model_x * ux + model_y * uy > 0.0

    weight = max(0.0, min(1.0, target_weight))
    if raw_aligned:
        fused_x = weight * ux + (1.0 - weight) * model_x
        fused_y = weight * uy + (1.0 - weight) * model_y
        norm = math.hypot(fused_x, fused_y)
        fused_x, fused_y = fused_x / norm, fused_y / norm
        yaw = weight * target_yaw + (1.0 - weight) * max(
            -max_angular_speed, min(max_angular_speed, raw_yaw)
        )
        reason = "blended_omtrack_toward_target"
    else:
        fused_x, fused_y = ux, uy
        yaw = target_yaw
        reason = "replaced_misaligned_omtrack"

    target_speed = min(max_linear_speed, max(0.0, linear_gain) * distance_error)
    speed = min(target_speed, max(minimum_speed, raw_speed))
    command = (speed * fused_x, speed * fused_y, yaw)
    return command, {
        "reason": reason,
        "target_distance": round(distance, 3),
        "distance_error": round(distance_error, 3),
        "target_bearing": round(bearing, 3),
        "raw_aligned": raw_aligned,
        "raw_speed": round(raw_speed, 3),
        "fused_speed": round(speed, 3),
        "goal_xy": goal_xy,
    }
