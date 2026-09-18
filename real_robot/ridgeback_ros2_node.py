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
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from nav_msgs.msg import Path as PathMessage
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from std_msgs.msg import Bool, String

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import receive_message, send_message
from safety import Command, SafetyConfig, SafetyState, evaluate, rate_limit


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
        self._latest_frame_id = 0
        self._processed_frame_id = -1
        self._prompt = str(self.get_parameter("prompt").value).strip()
        self._enabled = False
        self._enable_stamp: Optional[float] = None
        self._camera_stamp: Optional[float] = None
        self._inference_stamp: Optional[float] = None
        self._scan_stamp: Optional[float] = None
        self._obstacle_distance: Optional[float] = None
        self._estop_seen = False
        self._estop_stamp: Optional[float] = None
        self._estop_active = True
        self._inference_connected = False
        self._candidate: Optional[Command] = None
        self._trajectory: Optional[List[Command]] = None
        self._inference_error = "not connected"
        self._last_inference_seconds: Optional[float] = None
        self._last_frame_queued = 0.0
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
        self._proposed_publisher = self.create_publisher(TwistStamped, "omtrackvla/proposed_cmd_vel", 10)
        self._path_publisher = self.create_publisher(PathMessage, "omtrackvla/predicted_path", 10)
        self.create_subscription(String, "omtrackvla/prompt", self._prompt_callback, 10)
        self.create_subscription(Bool, "omtrackvla/enable", self._enable_callback, 10)
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
        self.create_timer(0.5, self._publish_status)
        self._worker_thread = threading.Thread(target=self._inference_loop, name="omtrack-inference-client", daemon=True)
        self._worker_thread.start()
        self.get_logger().warn(
            "Controller started in %s mode. Nonzero output also requires fresh deadman, camera, "
            "inference, E-stop, and LiDAR data."
            % ("DRY-RUN" if self.get_parameter("dry_run").value else "ARMABLE")
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "camera_topic": "camera/color/image_raw",
            "camera_compressed": False,
            "scan_topic": "sensors/scan",
            "estop_topic": "platform/emergency_stop",
            "cmd_vel_topic": "cmd_vel",
            "cmd_vel_stamped": True,
            "prompt": "Follow the person directly in front of you.",
            "inference_host": "127.0.0.1",
            "inference_port": 18765,
            "inference_rate": 2.0,
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
            self._candidate = None
            self._trajectory = None
            self._inference_stamp = None
        self.get_logger().info(f"Follow prompt updated: {prompt}")

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
        valid = [value for value in message.ranges if math.isfinite(value) and message.range_min <= value <= message.range_max]
        with self._lock:
            self._scan_stamp = self._now()
            self._obstacle_distance = min(valid) if valid else None

    def _compressed_image_callback(self, message: CompressedImage) -> None:
        array = np.frombuffer(message.data, dtype=np.uint8)
        bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if bgr is not None:
            self._queue_bgr_frame(bgr)

    def _image_callback(self, message: Image) -> None:
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
            self._queue_bgr_frame(bgr)
        except (ValueError, cv2.error) as exc:
            self.get_logger().error(f"Camera conversion failed: {exc}", throttle_duration_sec=5.0)

    def _queue_bgr_frame(self, bgr: np.ndarray) -> None:
        now = self._now()
        rate = float(self.get_parameter("inference_rate").value)
        with self._lock:
            self._camera_stamp = now
            if now - self._last_frame_queued < 1.0 / rate:
                return
            self._last_frame_queued = now
        quality = int(self.get_parameter("jpeg_quality").value)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return
        with self._lock:
            self._latest_jpeg = encoded.tobytes()
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
                prompt = self._prompt
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
                    "encoding": "jpeg",
                    "image_b64": base64.b64encode(jpeg).decode("ascii"),
                })
                response = receive_message(self._socket)
                if response.get("request_id") != self._request_id or not response.get("ok"):
                    raise RuntimeError(str(response.get("error", "invalid inference response")))
                command = response.get("command")
                if not isinstance(command, list) or len(command) != 3:
                    raise RuntimeError("inference response has no 3-axis command")
                lateral_sign = float(self.get_parameter("lateral_sign").value)
                yaw_sign = float(self.get_parameter("yaw_sign").value)
                candidate = (float(command[0]), lateral_sign * float(command[1]), yaw_sign * float(command[2]))
                trajectory = self._parse_trajectory(response.get("trajectory"), lateral_sign, yaw_sign)
                with self._lock:
                    self._candidate = candidate
                    self._trajectory = trajectory
                    self._inference_stamp = self._now()
                    self._inference_connected = True
                    self._inference_error = ""
                    self._last_inference_seconds = float(response.get("inference_seconds", 0.0))
                    self._processed_frame_id = frame_id
            except Exception as exc:
                with self._lock:
                    self._candidate = None
                    self._trajectory = None
                    self._inference_connected = False
                    self._inference_error = f"{type(exc).__name__}: {exc}"
                if self._socket is not None:
                    try:
                        self._socket.close()
                    except OSError:
                        pass
                    self._socket = None
                time.sleep(0.2)

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
            max_linear_x=float(value("max_linear_x")),
            max_linear_y=float(value("max_linear_y")),
            max_angular_z=float(value("max_angular_z")),
            require_scan=bool(value("require_scan")),
            require_estop=bool(value("require_estop")),
        )

    def _control_tick(self) -> None:
        now = self._now()
        with self._lock:
            state = SafetyState(
                enabled=self._enabled,
                enable_stamp=self._enable_stamp,
                camera_stamp=self._camera_stamp,
                inference_stamp=self._inference_stamp,
                scan_stamp=self._scan_stamp,
                obstacle_distance=self._obstacle_distance,
                estop_seen=self._estop_seen,
                estop_stamp=self._estop_stamp,
                estop_active=self._estop_active,
                inference_connected=self._inference_connected,
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
            trajectory = list(self._trajectory) if self._trajectory is not None else None
            inference_stamp = self._inference_stamp
            connected = self._inference_connected
        timeout = float(self.get_parameter("inference_timeout").value)
        if (
            not connected
            or candidate is None
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

        path = PathMessage()
        path.header.stamp = ros_stamp
        path.header.frame_id = frame_id
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

    def _publish_status(self) -> None:
        self._publish_prediction()
        with self._lock:
            proposed = self._clip_proposed_command(self._candidate) if self._candidate is not None else None
            status = {
                "reason": self._last_reason,
                "dry_run": bool(self.get_parameter("dry_run").value),
                "prompt": self._prompt,
                "enabled": self._enabled,
                "estop_seen": self._estop_seen,
                "estop_active": self._estop_active,
                "obstacle_distance": self._obstacle_distance,
                "inference_connected": self._inference_connected,
                "inference_error": self._inference_error,
                "inference_seconds": self._last_inference_seconds,
                "proposed_command_raw": self._candidate,
                "proposed_command_clipped": proposed,
                "predicted_trajectory": self._trajectory,
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
