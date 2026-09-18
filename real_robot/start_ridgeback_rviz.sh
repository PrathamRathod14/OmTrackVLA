#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
ROS_SETUP="/opt/ros/jazzy/setup.bash"
ROBOT_CONFIG="/etc/clearpath/robot.yaml"
ROBOT_NAMESPACE="${ROBOT_NAMESPACE:-}"

if [ ! -f "$ROS_SETUP" ]; then
    echo "ROS 2 Jazzy was not found at $ROS_SETUP" >&2
    exit 2
fi

if [ -z "$ROBOT_NAMESPACE" ] && [ -f "$ROBOT_CONFIG" ]; then
    ROBOT_NAMESPACE="$(awk '$1 == "namespace:" {print $2; exit}' "$ROBOT_CONFIG")"
fi
if [ -z "$ROBOT_NAMESPACE" ]; then
    echo "Set ROBOT_NAMESPACE to the Clearpath namespace, for example r100_0160." >&2
    exit 2
fi

set +u
source "$ROS_SETUP"
set -u

exec rviz2 \
    -d "$PROJECT_DIR/real_robot/ridgeback_omtrack.rviz" \
    --ros-args \
    -r "__ns:=/$ROBOT_NAMESPACE" \
    -r "__node:=omtrackvla_rviz" \
    -r "/tf:=tf" \
    -r "/tf_static:=tf_static"
