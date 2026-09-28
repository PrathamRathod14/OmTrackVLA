#!/usr/bin/env python3
"""ROS 2 safety controller connecting OmTrackVLA to a Clearpath Ridgeback."""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path
import socket
import sys
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, Twist, TwistStamped
from nav_msgs.msg import Path as PathMessage
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, LaserScan
from std_msgs.msg import Bool, Empty, String

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import receive_message, send_message
from safety import Command, SafetyConfig, SafetyState, directional_obstacle_distance, evaluate, rate_limit
from target_geometry import CameraGeometry, extrinsics_rotation, fuse_target_command, locate_target, trajectory_follows_target

try:
    from realsense2_camera_msgs.msg import Extrinsics
except ImportError:  # the fixed-baseline fallback parameter is used instead
    Extrinsics = None


class RidgebackController(Node):
    def __init__(self) -> None:
        super().__init__("omtrackvla_ridgeback")
        self._declare_parameters()
        self._lock = threading.Lock()
        self._wake_worker = threading.Event()
        self._shutdown = threading.Event()
        self._socket: Optional[socket.socket] = None
        self._request_id = 0
        self._latest_jpeg: Optional[bytes] = None
        # (raw message bytes, receive time, is_compressed); decoded only when needed
        self._latest_frame_depth: Optional[Tuple[bytes, float, bool, Tuple[int, int, int]]] = None
        self._latest_depth: Optional[Tuple[bytes, float, bool, Tuple[int, int, int]]] = None
        self._depth_k: Optional[np.ndarray] = None
        self._color_k: Optional[np.ndarray] = None
        self._depth_to_color: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._target_position_valid = False
        self._target_position: Optional[List[float]] = None
        self._target_position_info: dict = {"reason": "no_inference_yet"}
        self._latest_frame_id = 0
        self._latest_frame_stamp = 0.0
        self._processed_frame_id = -1
        self._prompt = str(self.get_parameter("prompt").value).strip()
        self._enabled = False
        self._enable_stamp: Optional[float] = None
        self._camera_stamp: Optional[float] = None
        self._inference_stamp: Optional[float] = None
        self._scan_stamp: Optional[float] = None
        self._obstacle_distance: Optional[float] = None
        self._motion_obstacle_distance: Optional[float] = None
        self._scan_points: List[Tuple[float, float]] = []
        self._scan_clear_distance = 0.0
        self._estop_seen = False
        self._estop_stamp: Optional[float] = None
        self._estop_active = True
        self._inference_connected = False
        self._raw_candidate: Optional[Command] = None
        self._candidate: Optional[Command] = None
        self._trajectory: Optional[List[Command]] = None
        self._planner_ran = False
        self._planner_updated = False
        self._planner_age_seconds: Optional[float] = None
        self._target_valid = False
        self._raw_trajectory_valid = False
        self._trajectory_valid = False
        self._fusion_info: dict = {"reason": "no_inference_yet"}
        self._target: dict = {"target_state": "SEARCHING", "target_reason": "no_inference_yet"}
        self._target_image: Optional[bytes] = None
        self._inference_error = "not connected"
        self._last_inference_seconds: Optional[float] = None
        self._last_pipeline_seconds: Optional[float] = None
        self._next_frame_due = 0.0
        self._last_control_time = time.monotonic()
        self._last_command: Command = (0.0, 0.0, 0.0)
        self._last_reason = "starting"
        self._was_allowed = False
        self._stop_until = 0.0

        stamped = bool(self.get_parameter("cmd_vel_stamped").value)
        command_topic = str(self.get_parameter("cmd_vel_topic").value)
        self._stamped = stamped
        self._publisher = self.create_publisher(TwistStamped if stamped else Twist, command_topic, 10)
        self._status_publisher = self.create_publisher(String, "omtrackvla/status", 10)
        self._raw_proposed_publisher = self.create_publisher(TwistStamped, "omtrackvla/raw_cmd_vel", 10)
        self._proposed_publisher = self.create_publisher(TwistStamped, "omtrackvla/proposed_cmd_vel", 10)
        self._path_publisher = self.create_publisher(PathMessage, "omtrackvla/predicted_path", 10)
        self._fused_path_publisher = self.create_publisher(PathMessage, "omtrackvla/fused_path", 10)
        self._target_state_publisher = self.create_publisher(String, "omtrackvla/target_state", 10)
        self._target_image_publisher = self.create_publisher(Image, "omtrackvla/target_image", qos_profile_sensor_data)
        self._target_position_publisher = self.create_publisher(PointStamped, "omtrackvla/target_position", 10)
        if bool(self.get_parameter("require_target_position").value):
            # Compressed depth (~0.17 MB) is used by default: subscribing to raw depth
            # (~0.6 MB at 19 Hz) stalled delivery of the color stream on r100_0160.
            depth_compressed = bool(self.get_parameter("depth_compressed").value)
            self.create_subscription(
                CompressedImage if depth_compressed else Image,
                str(self.get_parameter("depth_topic").value),
                self._compressed_depth_callback if depth_compressed else self._depth_callback,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                CameraInfo,
                str(self.get_parameter("depth_info_topic").value),
                lambda message: self._camera_info_callback("depth", message),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                CameraInfo,
                str(self.get_parameter("color_info_topic").value),
                lambda message: self._camera_info_callback("color", message),
                qos_profile_sensor_data,
            )
            if Extrinsics is not None:
                self.create_subscription(
                    Extrinsics,
                    str(self.get_parameter("extrinsics_topic").value),
                    self._extrinsics_callback,
                    QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
                )
        self.create_subscription(String, "omtrackvla/prompt", self._prompt_callback, 10)
        self.create_subscription(Bool, "omtrackvla/enable", self._enable_callback, 10)
        self.create_subscription(Empty, "omtrackvla/reset_target", self._reset_target_callback, 10)
        self._reset_target_requested = False
        self.create_subscription(
            Bool,
            str(self.get_parameter("estop_topic").value),
            self._estop_callback,
            qos_profile_sensor_data,
        )
        if bool(self.get_parameter("require_scan").value):
            self.create_subscription(
                LaserScan,
                str(self.get_parameter("scan_topic").value),
                self._scan_callback,
                qos_profile_sensor_data,
            )
        camera_topic = str(self.get_parameter("camera_topic").value)
        if bool(self.get_parameter("camera_compressed").value):
            self.create_subscription(
                CompressedImage,
                camera_topic,
                self._compressed_image_callback,
                qos_profile_sensor_data,
            )
        else:
            self.create_subscription(
                Image,
                camera_topic,
                self._image_callback,
                qos_profile_sensor_data,
            )

        control_rate = float(self.get_parameter("control_rate").value)
        self.create_timer(1.0 / control_rate, self._control_tick)
        # Hand completed annotations to ROS promptly instead of waiting for the
        # slower JSON/status publication timer.
        self.create_timer(0.05, self._publish_target_image)
        self.create_timer(0.5, self._publish_status)
        self._worker_thread = threading.Thread(target=self._inference_loop, name="omtrack-inference-client", daemon=True)
        self._worker_thread.start()
        self.get_logger().warn(
            "Controller started in %s mode. Nonzero output also requires fresh deadman, camera, "
            "inference, E-stop, LiDAR data, a LOCKED target, and a path consistent with the "
            "target's depth-measured position. Target fusion corrects OmTrackVLA toward the locked leader."
            % ("DRY-RUN" if self.get_parameter("dry_run").value else "ARMABLE")
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "camera_topic": "camera/color/image_raw/compressed",
            "camera_compressed": True,
            "scan_topic": "sensors/scan",
            "estop_topic": "platform/emergency_stop",
            "cmd_vel_topic": "cmd_vel",
            "cmd_vel_stamped": True,
            "prompt": "Follow the person directly in front of you.",
            "inference_host": "127.0.0.1",
            "inference_port": 18765,
            "inference_rate": 3.0,
            "planner_cache_timeout": 0.75,
            "control_rate": 20.0,
            "dry_run": True,
            "require_scan": True,
            "require_estop": True,
            "enable_timeout": 0.5,
            "camera_timeout": 0.75,
            "inference_timeout": 1.5,
            "scan_timeout": 0.5,
            "estop_timeout": 1.0,
            "obstacle_stop_distance": 0.70,
            "rotation_stop_distance": 0.67,
            "robot_half_length": 0.48,
            "robot_half_width": 0.397,
            "obstacle_lateral_margin": 0.05,
            "max_linear_x": 0.20,
            "max_linear_y": 0.20,
            "max_angular_z": 0.35,
            "max_linear_accel": 0.30,
            "max_angular_accel": 0.50,
            "lateral_sign": 1.0,
            "yaw_sign": 1.0,
            "jpeg_quality": 85,
            "base_frame": "base_link",
            "socket_timeout": 5.0,
            "stop_burst_seconds": 0.5,
            "require_target": True,
            "require_trajectory_consistency": True,
            "require_target_position": True,
            "depth_topic": "camera/depth/image_rect_raw/compressedDepth",
            "depth_compressed": True,
            "depth_info_topic": "camera/depth/camera_info",
            "color_info_topic": "camera/color/camera_info",
            "extrinsics_topic": "camera/extrinsics/depth_to_color",
            "depth_scale": 0.001,
            "depth_max_age": 0.25,
            # camera_link pose in base_link: x, y, z, roll, pitch, yaw (m, rad)
            "camera_mount": [0.0, 0.0, 1.03, 0.0, 0.0, 0.0],
            "fallback_depth_to_color_translation": [-0.059, 0.0, 0.0],
            "max_target_range": 6.0,
            "min_follow_distance": 0.9,
            "retreat_tolerance": 0.05,
            "reject_path_retreat": False,
            "enable_target_fusion": True,
            "fusion_target_weight": 0.75,
            "fusion_linear_gain": 0.4,
            "fusion_minimum_speed": 0.04,
            "fusion_yaw_gain": 0.8,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _now(self) -> float:
        return time.monotonic()

    def _prompt_callback(self, message: String) -> None:
        prompt = message.data.strip()[:256]
        if not prompt:
            self.get_logger().error("Rejected an empty follow prompt")
            return
        with self._lock:
            self._prompt = prompt
            self._clear_prediction_locked()
            self._target = {"target_state": "SEARCHING", "target_reason": "prompt_changed"}
        self.get_logger().info(f"Follow prompt updated: {prompt}")

    def _reset_target_callback(self, _message: Empty) -> None:
        with self._lock:
            self._reset_target_requested = True
            self._clear_prediction_locked()
            self._target = {"target_state": "SEARCHING", "target_reason": "reset_requested"}
        self.get_logger().info("Target reset requested; searching for the prompted person again")

    def _enable_callback(self, message: Bool) -> None:
        now = self._now()
        with self._lock:
            self._enabled = bool(message.data)
            self._enable_stamp = now
            if not self._enabled:
                self._stop_until = now + float(self.get_parameter("stop_burst_seconds").value)

    def _estop_callback(self, message: Bool) -> None:
        with self._lock:
            self._estop_seen = True
            self._estop_stamp = self._now()
            self._estop_active = bool(message.data)
            if self._estop_active:
                self._stop_until = self._now() + float(self.get_parameter("stop_burst_seconds").value)

    def _scan_callback(self, message: LaserScan) -> None:
        valid = []
        points = []
        for index, value in enumerate(message.ranges):
            if not math.isfinite(value) or not message.range_min <= value <= message.range_max:
                continue
            valid.append(value)
            angle = message.angle_min + index * message.angle_increment
            points.append((value * math.cos(angle), value * math.sin(angle)))
        with self._lock:
            self._scan_stamp = self._now()
            self._obstacle_distance = min(valid) if valid else None
            self._scan_points = points
            self._scan_clear_distance = float(message.range_max)

    def _depth_callback(self, message: Image) -> None:
        if message.encoding.lower() not in ("16uc1", "mono16"):
            self.get_logger().error(f"Unsupported depth encoding: {message.encoding}", throttle_duration_sec=5.0)
            return
        with self._lock:
            self._latest_depth = (bytes(message.data), self._now(), False, (message.height, message.width, message.step))

    def _compressed_depth_callback(self, message: CompressedImage) -> None:
        if "16uc1" not in message.format.lower() or "png" not in message.format.lower():
            self.get_logger().error(f"Unsupported compressed depth format: {message.format}", throttle_duration_sec=5.0)
            return
        with self._lock:
            self._latest_depth = (bytes(message.data), self._now(), True, (0, 0, 0))

    @staticmethod
    def _decode_depth(frame_depth: Tuple[bytes, float, bool, Tuple[int, int, int]]) -> Optional[np.ndarray]:
        data, _, compressed, (height, width, step) = frame_depth
        if compressed:
            # compressed_depth_image_transport: 12-byte config header, then a 16-bit PNG.
            depth = cv2.imdecode(np.frombuffer(data[12:], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            return depth if depth is not None and depth.dtype == np.uint16 else None
        return np.frombuffer(data, dtype=np.uint16).reshape(height, step // 2)[:, :width]

    def _camera_info_callback(self, which: str, message: CameraInfo) -> None:
        k = np.asarray(message.k, dtype=np.float64).reshape(3, 3)
        with self._lock:
            if which == "depth":
                self._depth_k = k
            else:
                self._color_k = k

    def _extrinsics_callback(self, message) -> None:
        with self._lock:
            self._depth_to_color = (extrinsics_rotation(message.rotation), np.asarray(message.translation, dtype=np.float64))

    def _camera_geometry(self) -> Optional[CameraGeometry]:
        with self._lock:
            depth_k, color_k, extrinsics = self._depth_k, self._color_k, self._depth_to_color
        if depth_k is None or color_k is None:
            return None
        if extrinsics is None:
            translation = self.get_parameter("fallback_depth_to_color_translation").value
            extrinsics = (np.eye(3), np.asarray(translation, dtype=np.float64))
        mount = [float(value) for value in self.get_parameter("camera_mount").value]
        return CameraGeometry(depth_k, color_k, extrinsics[0], extrinsics[1], mount)

    def _check_target_position(
        self, bbox: object, trajectory: List[Command], frame_depth: Optional[tuple], frame_stamp: float
    ) -> Tuple[bool, Optional[List[float]], dict]:
        """Locate the locked target; retain raw path consistency as diagnostics."""
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False, None, {"reason": "target_bbox_missing"}
        geometry = self._camera_geometry()
        if geometry is None:
            return False, None, {"reason": "camera_info_missing"}
        if frame_depth is None or abs(frame_stamp - frame_depth[1]) > float(self.get_parameter("depth_max_age").value):
            return False, None, {"reason": "depth_not_synchronized"}
        depth = self._decode_depth(frame_depth)
        if depth is None:
            return False, None, {"reason": "depth_decode_failed"}
        depth_m = depth.astype(np.float32) * float(self.get_parameter("depth_scale").value)
        position, reason = locate_target(
            depth_m,
            [float(value) for value in bbox],
            geometry,
            max_range=float(self.get_parameter("max_target_range").value),
            image_size=(int(round(2 * geometry.color_k[0, 2])), int(round(2 * geometry.color_k[1, 2]))),
        )
        if position is None:
            return False, None, {"reason": reason}
        raw_path_ok, raw_path_reason, current, final = trajectory_follows_target(
            trajectory,
            position[:2],
            float(self.get_parameter("min_follow_distance").value),
            float(self.get_parameter("retreat_tolerance").value),
            bool(self.get_parameter("reject_path_retreat").value),
        )
        info = {
            "reason": "target_located",
            "target_distance": round(current, 3),
            "raw_path_end_distance": round(final, 3),
            "raw_path_consistent": raw_path_ok,
            "raw_path_reason": raw_path_reason,
        }
        return True, [float(value) for value in position], info

    def _compressed_image_callback(self, message: CompressedImage) -> None:
        now = self._now()
        if not self._reserve_inference_frame(now):
            return
        # The RealSense compressed transport already provides JPEG. Passing those
        # bytes through avoids decoding 30 FPS and re-encoding the selected frame in
        # this ROS process; inference_server.py performs the one required decode.
        if "jpeg" in message.format.lower() or "jpg" in message.format.lower():
            jpeg = bytes(message.data)
        else:
            array = np.frombuffer(message.data, dtype=np.uint8)
            bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
            jpeg = self._encode_bgr_frame(bgr) if bgr is not None else None
        if jpeg:
            self._store_inference_frame(jpeg, now)

    def _image_callback(self, message: Image) -> None:
        now = self._now()
        if not self._reserve_inference_frame(now):
            return
        channels_by_encoding = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}
        channels = channels_by_encoding.get(message.encoding.lower())
        if channels is None:
            self.get_logger().error(f"Unsupported camera encoding: {message.encoding}", throttle_duration_sec=5.0)
            return
        try:
            rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
            pixels = rows[:, : message.width * channels].reshape(message.height, message.width, channels)
            encoding = message.encoding.lower()
            if encoding == "rgb8":
                bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            elif encoding == "rgba8":
                bgr = cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
            elif encoding == "bgra8":
                bgr = cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
            elif encoding == "mono8":
                bgr = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
            else:
                bgr = pixels
            jpeg = self._encode_bgr_frame(bgr)
            if jpeg:
                self._store_inference_frame(jpeg, now)
        except (ValueError, cv2.error) as exc:
            self.get_logger().error(f"Camera conversion failed: {exc}", throttle_duration_sec=5.0)

    def _reserve_inference_frame(self, now: float) -> bool:
        """Update camera freshness and reserve only frames due for inference."""
        rate = float(self.get_parameter("inference_rate").value)
        period = 1.0 / rate
        with self._lock:
            self._camera_stamp = now
            if now < self._next_frame_due:
                return False
            # Advance an absolute deadline. Resetting from `now` on every sample
            # rounds every interval up to a camera-frame boundary.
            if self._next_frame_due <= 0.0 or now - self._next_frame_due > period:
                self._next_frame_due = now + period
            else:
                self._next_frame_due += period
        return True

    def _encode_bgr_frame(self, bgr: np.ndarray) -> Optional[bytes]:
        quality = int(self.get_parameter("jpeg_quality").value)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None
        return encoded.tobytes()

    def _store_inference_frame(self, jpeg: bytes, now: float) -> None:
        with self._lock:
            self._latest_jpeg = jpeg
            # Keep the depth frame that was current when this color frame was captured.
            self._latest_frame_depth = self._latest_depth
            self._latest_frame_stamp = now
            self._latest_frame_id += 1
        self._wake_worker.set()

    def _connect(self) -> socket.socket:
        host = str(self.get_parameter("inference_host").value)
        port = int(self.get_parameter("inference_port").value)
        timeout = float(self.get_parameter("socket_timeout").value)
        connection = socket.create_connection((host, port), timeout=timeout)
        connection.settimeout(timeout)
        return connection

    def _inference_loop(self) -> None:
        while not self._shutdown.is_set():
            self._wake_worker.wait(0.2)
            self._wake_worker.clear()
            with self._lock:
                frame_id = self._latest_frame_id
                jpeg = self._latest_jpeg
                frame_depth = self._latest_frame_depth
                frame_stamp = self._latest_frame_stamp
                prompt = self._prompt
                reset_target = self._reset_target_requested
            if jpeg is None or frame_id == self._processed_frame_id:
                continue
            try:
                if self._socket is None:
                    self._socket = self._connect()
                    self.get_logger().info("Connected to the local OmTrackVLA inference server")
                self._request_id += 1
                send_message(self._socket, {
                    "request_id": self._request_id,
                    "prompt": prompt,
                    "reset_target": reset_target,
                    "encoding": "jpeg",
                    "image_b64": base64.b64encode(jpeg).decode("ascii"),
                })
                response = receive_message(self._socket)
                if response.get("request_id") != self._request_id or not response.get("ok"):
                    raise RuntimeError(str(response.get("error", "invalid inference response")))
                if reset_target:
                    with self._lock:
                        self._reset_target_requested = False
                with self._lock:
                    if prompt != self._prompt:
                        continue
                command = response.get("command")
                if not isinstance(command, list) or len(command) != 3:
                    raise RuntimeError("inference response has no 3-axis command")
                lateral_sign = float(self.get_parameter("lateral_sign").value)
                yaw_sign = float(self.get_parameter("yaw_sign").value)
                raw_candidate = (float(command[0]), lateral_sign * float(command[1]), yaw_sign * float(command[2]))
                trajectory = self._parse_trajectory(response.get("trajectory"), lateral_sign, yaw_sign)
                planner_ran = response.get("planner_ran") is True
                planner_updated = response.get("planner_updated") is True
                planner_age = response.get("planner_age_seconds")
                if planner_ran:
                    if not isinstance(planner_age, (int, float)) or not math.isfinite(float(planner_age)):
                        raise RuntimeError("inference response has no finite planner age")
                    planner_age = float(planner_age)
                    if not 0.0 <= planner_age <= float(self.get_parameter("planner_cache_timeout").value):
                        raise RuntimeError(f"OmTrackVLA planner result is stale ({planner_age:.3f} s)")
                else:
                    planner_updated = False
                    planner_age = None
                target = {key: value for key, value in response.items() if key.startswith("target_") and key != "target_image_b64"}
                raw_trajectory_valid = planner_ran and response.get("trajectory_valid") is True
                target["raw_trajectory_valid"] = raw_trajectory_valid
                target["raw_trajectory_reason"] = str(response.get("trajectory_reason", "missing"))
                target_image = response.get("target_image_b64")
                position_valid, position, position_info = False, None, {"reason": "target_not_locked"}
                if response.get("target_valid") is True and bool(self.get_parameter("require_target_position").value):
                    position_valid, position, position_info = self._check_target_position(
                        response.get("target_bbox"), trajectory, frame_depth, frame_stamp
                    )
                candidate = raw_candidate
                fusion_info = {"reason": "target_fusion_disabled"}
                fusion_enabled = bool(self.get_parameter("enable_target_fusion").value)
                if (
                    fusion_enabled
                    and planner_ran
                    and response.get("target_valid") is True
                    and position_valid
                    and position is not None
                ):
                    candidate, fusion_info = fuse_target_command(
                        raw_candidate,
                        position[:2],
                        follow_distance=float(self.get_parameter("min_follow_distance").value),
                        target_weight=float(self.get_parameter("fusion_target_weight").value),
                        linear_gain=float(self.get_parameter("fusion_linear_gain").value),
                        minimum_speed=float(self.get_parameter("fusion_minimum_speed").value),
                        max_linear_speed=max(
                            float(self.get_parameter("max_linear_x").value),
                            float(self.get_parameter("max_linear_y").value),
                        ),
                        yaw_gain=float(self.get_parameter("fusion_yaw_gain").value),
                        max_angular_speed=float(self.get_parameter("max_angular_z").value),
                    )
                trajectory_valid = planner_ran and (
                    position_valid if fusion_enabled and response.get("target_valid") is True else raw_trajectory_valid
                )
                target["trajectory_valid"] = trajectory_valid
                target["trajectory_reason"] = (
                    str(fusion_info.get("reason")) if fusion_enabled else target["raw_trajectory_reason"]
                )
                target["fusion"] = fusion_info
                with self._lock:
                    self._raw_candidate = raw_candidate
                    self._candidate = candidate
                    self._trajectory = trajectory
                    self._planner_ran = planner_ran
                    self._planner_updated = planner_updated
                    self._planner_age_seconds = planner_age
                    # Target validity must be an explicit boolean from the same frame as the path.
                    self._target_valid = response.get("target_valid") is True
                    self._raw_trajectory_valid = raw_trajectory_valid
                    self._trajectory_valid = trajectory_valid
                    self._fusion_info = fusion_info
                    self._target = target
                    self._target_position_valid = position_valid
                    self._target_position = position
                    self._target_position_info = position_info
                    self._target_image = base64.b64decode(target_image) if isinstance(target_image, str) else None
                    self._inference_stamp = self._now()
                    self._inference_connected = True
                    self._inference_error = ""
                    self._last_inference_seconds = float(response.get("inference_seconds", 0.0))
                    self._last_pipeline_seconds = max(0.0, self._now() - frame_stamp)
                    self._processed_frame_id = frame_id
            except Exception as exc:
                with self._lock:
                    self._clear_prediction_locked()
                    self._inference_connected = False
                    self._inference_error = f"{type(exc).__name__}: {exc}"
                if self._socket is not None:
                    try:
                        self._socket.close()
                    except OSError:
                        pass
                    self._socket = None
                time.sleep(0.2)

    def _clear_prediction_locked(self) -> None:
        self._raw_candidate = None
        self._candidate = None
        self._trajectory = None
        self._planner_ran = False
        self._planner_updated = False
        self._planner_age_seconds = None
        self._inference_stamp = None
        self._last_pipeline_seconds = None
        self._target_valid = False
        self._raw_trajectory_valid = False
        self._trajectory_valid = False
        self._fusion_info = {"reason": "prediction_cleared"}
        self._target_image = None
        self._target_position_valid = False
        self._target_position = None

    @staticmethod
    def _parse_trajectory(value: object, lateral_sign: float, yaw_sign: float) -> List[Command]:
        if not isinstance(value, list) or not value:
            raise RuntimeError("inference response has no trajectory")
        trajectory: List[Command] = []
        for waypoint in value[:64]:
            if not isinstance(waypoint, (list, tuple)) or len(waypoint) < 2:
                raise RuntimeError("inference trajectory contains an invalid waypoint")
            x = float(waypoint[0])
            y = lateral_sign * float(waypoint[1])
            yaw = yaw_sign * float(waypoint[2]) if len(waypoint) >= 3 else 0.0
            if not all(math.isfinite(component) for component in (x, y, yaw)):
                raise RuntimeError("inference trajectory contains a non-finite waypoint")
            trajectory.append((x, y, yaw))
        return trajectory

    def _safety_config(self) -> SafetyConfig:
        value = lambda name: self.get_parameter(name).value
        return SafetyConfig(
            enable_timeout=float(value("enable_timeout")),
            camera_timeout=float(value("camera_timeout")),
            inference_timeout=float(value("inference_timeout")),
            scan_timeout=float(value("scan_timeout")),
            estop_timeout=float(value("estop_timeout")),
            obstacle_stop_distance=float(value("obstacle_stop_distance")),
            rotation_stop_distance=float(value("rotation_stop_distance")),
            max_linear_x=float(value("max_linear_x")),
            max_linear_y=float(value("max_linear_y")),
            max_angular_z=float(value("max_angular_z")),
            require_scan=bool(value("require_scan")),
            require_estop=bool(value("require_estop")),
            require_target=bool(value("require_target")),
            require_trajectory_consistency=bool(value("require_trajectory_consistency")),
            require_target_position=bool(value("require_target_position")),
        )

    def _control_tick(self) -> None:
        now = self._now()
        with self._lock:
            candidate = self._candidate
            motion_obstacle_distance = self._obstacle_distance
            if candidate is not None:
                motion_obstacle_distance = directional_obstacle_distance(
                    self._scan_points,
                    candidate,
                    float(self.get_parameter("robot_half_length").value),
                    float(self.get_parameter("robot_half_width").value),
                    float(self.get_parameter("obstacle_lateral_margin").value),
                    self._scan_clear_distance,
                )
            self._motion_obstacle_distance = motion_obstacle_distance
            state = SafetyState(
                enabled=self._enabled,
                enable_stamp=self._enable_stamp,
                camera_stamp=self._camera_stamp,
                inference_stamp=self._inference_stamp,
                scan_stamp=self._scan_stamp,
                obstacle_distance=motion_obstacle_distance,
                rotation_obstacle_distance=self._obstacle_distance,
                estop_seen=self._estop_seen,
                estop_stamp=self._estop_stamp,
                estop_active=self._estop_active,
                inference_connected=self._inference_connected,
                target_valid=self._target_valid,
                trajectory_valid=self._trajectory_valid,
                target_position_valid=self._target_position_valid,
                candidate=self._candidate,
            )
        allowed, reason, target = evaluate(self._safety_config(), state, now)
        if bool(self.get_parameter("dry_run").value):
            allowed, reason, target = False, f"dry_run:{reason}", (0.0, 0.0, 0.0)
        dt = min(0.2, max(0.0, now - self._last_control_time))
        self._last_control_time = now
        if allowed:
            target = rate_limit(
                self._last_command,
                target,
                dt,
                float(self.get_parameter("max_linear_accel").value),
                float(self.get_parameter("max_angular_accel").value),
            )
            self._publish_command(target)
            self._last_command = target
        else:
            if self._was_allowed:
                self._stop_until = now + float(self.get_parameter("stop_burst_seconds").value)
            self._last_command = (0.0, 0.0, 0.0)
            if now <= self._stop_until and not bool(self.get_parameter("dry_run").value):
                self._publish_command(self._last_command)
        if reason != self._last_reason:
            if allowed:
                self.get_logger().info(f"Motion gate: {reason}")
            else:
                self.get_logger().warning(f"Motion gate: {reason}")
            self._last_reason = reason
        self._was_allowed = allowed

    def _publish_command(self, command: Command) -> None:
        if self._stamped:
            message = TwistStamped()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = str(self.get_parameter("base_frame").value)
            twist = message.twist
        else:
            message = Twist()
            twist = message
        twist.linear.x, twist.linear.y, twist.angular.z = command
        self._publisher.publish(message)

    def _clip_proposed_command(self, command: Command) -> Command:
        config = self._safety_config()
        return (
            max(-config.max_linear_x, min(config.max_linear_x, command[0])),
            max(-config.max_linear_y, min(config.max_linear_y, command[1])),
            max(-config.max_angular_z, min(config.max_angular_z, command[2])),
        )

    def _publish_prediction(self) -> None:
        now = self._now()
        with self._lock:
            candidate = self._candidate
            raw_candidate = self._raw_candidate
            trajectory = list(self._trajectory) if self._trajectory is not None else None
            planner_ran = self._planner_ran
            fusion_info = dict(self._fusion_info)
            inference_stamp = self._inference_stamp
            connected = self._inference_connected
        timeout = float(self.get_parameter("inference_timeout").value)
        if (
            not connected
            or candidate is None
            or raw_candidate is None
            or trajectory is None
            or inference_stamp is None
            or not 0.0 <= now - inference_stamp <= timeout
        ):
            return

        ros_stamp = self.get_clock().now().to_msg()
        frame_id = str(self.get_parameter("base_frame").value)
        proposed = TwistStamped()
        proposed.header.stamp = ros_stamp
        proposed.header.frame_id = frame_id
        clipped = self._clip_proposed_command(candidate)
        proposed.twist.linear.x, proposed.twist.linear.y, proposed.twist.angular.z = clipped
        self._proposed_publisher.publish(proposed)

        raw = TwistStamped()
        raw.header = proposed.header
        raw_clipped = self._clip_proposed_command(raw_candidate)
        raw.twist.linear.x, raw.twist.linear.y, raw.twist.angular.z = raw_clipped
        self._raw_proposed_publisher.publish(raw)

        path = PathMessage()
        path.header.stamp = ros_stamp
        path.header.frame_id = frame_id
        if planner_ran:
            for x, y, yaw in trajectory:
                pose = PoseStamped()
                pose.header.stamp = ros_stamp
                pose.header.frame_id = frame_id
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.orientation.z = math.sin(yaw / 2.0)
                pose.pose.orientation.w = math.cos(yaw / 2.0)
                path.poses.append(pose)
        self._path_publisher.publish(path)

        goal = fusion_info.get("goal_xy")
        fused_path = PathMessage()
        fused_path.header = path.header
        if isinstance(goal, list) and len(goal) == 2:
            for x, y in ((0.0, 0.0), (float(goal[0]), float(goal[1]))):
                pose = PoseStamped()
                pose.header = path.header
                pose.pose.position.x = x
                pose.pose.position.y = y
                yaw = math.atan2(float(goal[1]), float(goal[0])) if math.hypot(*goal) > 1e-6 else 0.0
                pose.pose.orientation.z = math.sin(yaw / 2.0)
                pose.pose.orientation.w = math.cos(yaw / 2.0)
                fused_path.poses.append(pose)
        self._fused_path_publisher.publish(fused_path)

    def _publish_target(self) -> None:
        with self._lock:
            target = dict(self._target)
            target["target_valid"] = self._target_valid
            target["trajectory_valid"] = self._trajectory_valid
            target["target_position_valid"] = self._target_position_valid
            target["target_position_base"] = self._target_position
            target["target_position_check"] = self._target_position_info
            position = self._target_position
        message = String()
        message.data = json.dumps(target, allow_nan=False)
        self._target_state_publisher.publish(message)
        if position is not None:
            point = PointStamped()
            point.header.stamp = self.get_clock().now().to_msg()
            point.header.frame_id = str(self.get_parameter("base_frame").value)
            point.point.x, point.point.y, point.point.z = position
            self._target_position_publisher.publish(point)

    def _publish_target_image(self) -> None:
        with self._lock:
            jpeg = self._target_image
            self._target_image = None
        if jpeg is None:
            return
        bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return
        image = Image()
        image.header.stamp = self.get_clock().now().to_msg()
        image.header.frame_id = "omtrackvla_target_view"
        image.height, image.width = bgr.shape[:2]
        image.encoding = "bgr8"
        image.step = image.width * 3
        image.data = bgr.tobytes()
        self._target_image_publisher.publish(image)

    def _publish_status(self) -> None:
        self._publish_prediction()
        self._publish_target()
        with self._lock:
            raw_candidate = self._raw_candidate
            fused_candidate = self._candidate
            raw_proposed = self._clip_proposed_command(self._raw_candidate) if self._raw_candidate is not None else None
            fused_proposed = self._clip_proposed_command(self._candidate) if self._candidate is not None else None
            status = {
                "reason": self._last_reason,
                "dry_run": bool(self.get_parameter("dry_run").value),
                "prompt": self._prompt,
                "enabled": self._enabled,
                "estop_seen": self._estop_seen,
                "estop_active": self._estop_active,
                "obstacle_distance": self._obstacle_distance,
                "motion_obstacle_distance": self._motion_obstacle_distance,
                "inference_connected": self._inference_connected,
                "inference_error": self._inference_error,
                "inference_seconds": self._last_inference_seconds,
                "pipeline_seconds": self._last_pipeline_seconds,
                "planner_ran": self._planner_ran,
                "planner_updated": self._planner_updated,
                "planner_age_seconds": self._planner_age_seconds,
                "proposed_command_raw": raw_candidate,
                "proposed_command_raw_clipped": raw_proposed,
                "proposed_command_fused": fused_candidate,
                "proposed_command_clipped": fused_proposed,
                "fusion": self._fusion_info,
                "predicted_trajectory": self._trajectory,
                "target_valid": self._target_valid,
                "raw_trajectory_valid": self._raw_trajectory_valid,
                "trajectory_valid": self._trajectory_valid,
                "target_state": self._target.get("target_state"),
                "target_reason": self._target.get("target_reason"),
                "target_track_id": self._target.get("target_track_id"),
                "target_position_valid": self._target_position_valid,
                "target_position_base": self._target_position,
                "target_position_check": self._target_position_info,
                "prediction_frame": str(self.get_parameter("base_frame").value),
                "last_command": self._last_command,
            }
        message = String()
        message.data = json.dumps(status, allow_nan=False)
        self._status_publisher.publish(message)

    def close(self) -> None:
        self._shutdown.set()
        self._wake_worker.set()
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
        if not bool(self.get_parameter("dry_run").value) and rclpy.ok():
            for _ in range(5):
                self._publish_command((0.0, 0.0, 0.0))
                time.sleep(0.03)


def main() -> None:
    # Leave SIGINT to Python so the final zero-command burst is published before
    # the ROS context is shut down.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = RidgebackController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
