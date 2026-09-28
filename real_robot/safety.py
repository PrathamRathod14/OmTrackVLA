"""Pure-Python safety gate shared by the Ridgeback ROS controller tests."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence, Tuple


Command = Tuple[float, float, float]


@dataclass(frozen=True)
class SafetyConfig:
    enable_timeout: float = 0.5
    camera_timeout: float = 0.75
    inference_timeout: float = 1.5
    scan_timeout: float = 0.5
    estop_timeout: float = 1.0
    obstacle_stop_distance: float = 0.70
    rotation_stop_distance: float = 0.67
    max_linear_x: float = 0.20
    max_linear_y: float = 0.20
    max_angular_z: float = 0.35
    require_scan: bool = True
    require_estop: bool = True
    require_target: bool = True
    require_trajectory_consistency: bool = True
    require_target_position: bool = True


@dataclass(frozen=True)
class SafetyState:
    enabled: bool
    enable_stamp: Optional[float]
    camera_stamp: Optional[float]
    inference_stamp: Optional[float]
    scan_stamp: Optional[float]
    obstacle_distance: Optional[float]
    rotation_obstacle_distance: Optional[float]
    estop_seen: bool
    estop_stamp: Optional[float]
    estop_active: bool
    inference_connected: bool
    target_valid: bool
    trajectory_valid: bool
    target_position_valid: bool
    candidate: Optional[Sequence[float]]


def is_fresh(now: float, stamp: Optional[float], timeout: float) -> bool:
    return stamp is not None and 0.0 <= now - stamp <= timeout


def evaluate(config: SafetyConfig, state: SafetyState, now: float) -> Tuple[bool, str, Command]:
    stop: Command = (0.0, 0.0, 0.0)
    if not state.enabled or not is_fresh(now, state.enable_stamp, config.enable_timeout):
        return False, "deadman_not_held", stop
    if config.require_estop and not state.estop_seen:
        return False, "estop_state_missing", stop
    if config.require_estop and not is_fresh(now, state.estop_stamp, config.estop_timeout):
        return False, "estop_state_stale", stop
    if state.estop_seen and state.estop_active:
        return False, "estop_active", stop
    if not is_fresh(now, state.camera_stamp, config.camera_timeout):
        return False, "camera_stale", stop
    if not state.inference_connected:
        return False, "inference_disconnected", stop
    if not is_fresh(now, state.inference_stamp, config.inference_timeout):
        return False, "inference_stale", stop
    if config.require_target and not state.target_valid:
        return False, "target_not_locked", stop
    if config.require_trajectory_consistency and not state.trajectory_valid:
        return False, "trajectory_target_mismatch", stop
    if config.require_target_position and not state.target_position_valid:
        return False, "target_position_inconsistent", stop
    if state.candidate is None or len(state.candidate) != 3:
        return False, "inference_invalid", stop
    command = tuple(float(value) for value in state.candidate)
    if not all(math.isfinite(value) for value in command):
        return False, "inference_nonfinite", stop
    clipped = (
        max(-config.max_linear_x, min(config.max_linear_x, command[0])),
        max(-config.max_linear_y, min(config.max_linear_y, command[1])),
        max(-config.max_angular_z, min(config.max_angular_z, command[2])),
    )
    reason = "ready"
    if config.require_scan:
        if not is_fresh(now, state.scan_stamp, config.scan_timeout):
            return False, "scan_stale", stop
        if state.obstacle_distance is None:
            return False, "scan_has_no_valid_ranges", stop
        if state.obstacle_distance < config.obstacle_stop_distance:
            return False, "obstacle_too_close", stop
        rotation_distance = state.rotation_obstacle_distance
        if abs(clipped[2]) > 1e-6 and (
            rotation_distance is None or rotation_distance < config.rotation_stop_distance
        ):
            clipped = (clipped[0], clipped[1], 0.0)
            if math.hypot(clipped[0], clipped[1]) <= 1e-6:
                return False, "obstacle_too_close_for_rotation", stop
            reason = "ready_rotation_suppressed"
    return True, reason, clipped


def directional_obstacle_distance(
    points_xy: Iterable[Sequence[float]],
    command: Sequence[float],
    half_length: float,
    half_width: float,
    lateral_margin: float,
    clear_distance: float,
) -> Optional[float]:
    """Nearest scan return in the rectangular robot's translation corridor."""
    points = [(float(point[0]), float(point[1])) for point in points_xy]
    if not points:
        return None
    vx, vy = float(command[0]), float(command[1])
    speed = math.hypot(vx, vy)
    if speed <= 1e-6:
        return min(math.hypot(x, y) for x, y in points)
    ux, uy = vx / speed, vy / speed
    corridor_half_width = abs(uy) * half_length + abs(ux) * half_width + lateral_margin
    distances = []
    for x, y in points:
        along = x * ux + y * uy
        across = -x * uy + y * ux
        if along > 0.0 and abs(across) <= corridor_half_width:
            distances.append(math.hypot(x, y))
    return min(distances) if distances else float(clear_distance)


def rate_limit(previous: Command, target: Command, dt: float, linear_accel: float, angular_accel: float) -> Command:
    dt = max(0.0, float(dt))

    def approach(old: float, new: float, step: float) -> float:
        return max(old - step, min(old + step, new))

    return (
        approach(previous[0], target[0], linear_accel * dt),
        approach(previous[1], target[1], linear_accel * dt),
        approach(previous[2], target[2], angular_accel * dt),
    )
