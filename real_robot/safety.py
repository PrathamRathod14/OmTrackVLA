"""Pure-Python safety gate shared by the Ridgeback ROS controller tests."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple


Command = Tuple[float, float, float]


@dataclass(frozen=True)
class SafetyConfig:
    enable_timeout: float = 0.5
    camera_timeout: float = 0.75
    inference_timeout: float = 1.5
    scan_timeout: float = 0.5
    estop_timeout: float = 1.0
    obstacle_stop_distance: float = 0.70
    max_linear_x: float = 0.20
    max_linear_y: float = 0.20
    max_angular_z: float = 0.35
    require_scan: bool = True
    require_estop: bool = True


@dataclass(frozen=True)
class SafetyState:
    enabled: bool
    enable_stamp: Optional[float]
    camera_stamp: Optional[float]
    inference_stamp: Optional[float]
    scan_stamp: Optional[float]
    obstacle_distance: Optional[float]
    estop_seen: bool
    estop_stamp: Optional[float]
    estop_active: bool
    inference_connected: bool
    candidate: Optional[Sequence[float]]


def _fresh(now: float, stamp: Optional[float], timeout: float) -> bool:
    return stamp is not None and 0.0 <= now - stamp <= timeout


def evaluate(config: SafetyConfig, state: SafetyState, now: float) -> Tuple[bool, str, Command]:
    stop: Command = (0.0, 0.0, 0.0)
    if not state.enabled or not _fresh(now, state.enable_stamp, config.enable_timeout):
        return False, "deadman_not_held", stop
    if config.require_estop and not state.estop_seen:
        return False, "estop_state_missing", stop
    if config.require_estop and not _fresh(now, state.estop_stamp, config.estop_timeout):
        return False, "estop_state_stale", stop
    if state.estop_seen and state.estop_active:
        return False, "estop_active", stop
    if not _fresh(now, state.camera_stamp, config.camera_timeout):
        return False, "camera_stale", stop
    if not state.inference_connected:
        return False, "inference_disconnected", stop
    if not _fresh(now, state.inference_stamp, config.inference_timeout):
        return False, "inference_stale", stop
    if config.require_scan:
        if not _fresh(now, state.scan_stamp, config.scan_timeout):
            return False, "scan_stale", stop
        if state.obstacle_distance is None:
            return False, "scan_has_no_valid_ranges", stop
        if state.obstacle_distance < config.obstacle_stop_distance:
            return False, "obstacle_too_close", stop
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
    return True, "ready", clipped


def rate_limit(previous: Command, target: Command, dt: float, linear_accel: float, angular_accel: float) -> Command:
    dt = max(0.0, float(dt))

    def approach(old: float, new: float, step: float) -> float:
        return max(old - step, min(old + step, new))

    return (
        approach(previous[0], target[0], linear_accel * dt),
        approach(previous[1], target[1], linear_accel * dt),
        approach(previous[2], target[2], angular_accel * dt),
    )
