#!/usr/bin/env python3
"""Export camera frames from a ROS 2 bag for offline target-perception evaluation.

Runs with the ROS 2 Jazzy system Python. Frames are written as JPEG files named
by their bag receive time in nanoseconds, so ordering and timing are preserved:

    source /opt/ros/jazzy/setup.bash
    /usr/bin/python3 real_robot/export_bag_frames.py BAG_DIR \
        --topic /r100_0160/sensors/camera_0/color/image --out frames/run1 --rate 2.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage, Image


def image_to_bgr(message: Image) -> np.ndarray:
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}[message.encoding.lower()]
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    pixels = rows[:, : message.width * channels].reshape(message.height, message.width, channels)
    conversion = {
        "rgb8": cv2.COLOR_RGB2BGR,
        "rgba8": cv2.COLOR_RGBA2BGR,
        "bgra8": cv2.COLOR_BGRA2BGR,
        "mono8": cv2.COLOR_GRAY2BGR,
    }.get(message.encoding.lower())
    return cv2.cvtColor(pixels, conversion) if conversion is not None else pixels.copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rate", type=float, default=2.0, help="Maximum export rate in Hz; match inference_rate")
    args = parser.parse_args()

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(args.bag)), rosbag2_py.ConverterOptions("", ""))
    types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    if args.topic not in types:
        raise SystemExit(f"{args.topic} is not in the bag. Topics: {sorted(types)}")
    message_type = CompressedImage if types[args.topic].endswith("CompressedImage") else Image
    reader.set_filter(rosbag2_py.StorageFilter(topics=[args.topic]))

    args.out.mkdir(parents=True, exist_ok=True)
    period_ns = int(1e9 / args.rate) if args.rate > 0 else 0
    last_ns = None
    count = 0
    while reader.has_next():
        _, data, stamp_ns = reader.read_next()
        if last_ns is not None and stamp_ns - last_ns < period_ns:
            continue
        message = deserialize_message(data, message_type)
        if message_type is CompressedImage:
            bgr = cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        else:
            bgr = image_to_bgr(message)
        if bgr is None:
            continue
        cv2.imwrite(str(args.out / f"{stamp_ns:019d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        last_ns = stamp_ns
        count += 1
    print(f"Exported {count} frames to {args.out}")


if __name__ == "__main__":
    main()
