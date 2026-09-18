#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
ROS_SETUP="/opt/ros/jazzy/setup.bash"
ROBOT_NAMESPACE="${ROBOT_NAMESPACE:-r100_0160}"
CAMERA_TOPIC="${CAMERA_TOPIC:-camera/color/image_raw}"

if [ ! -t 0 ]; then
    echo "This launcher requires an interactive terminal." >&2
    exit 2
fi
if [ ! -f "$ROS_SETUP" ]; then
    echo "ROS 2 Jazzy was not found at $ROS_SETUP" >&2
    exit 2
fi
if pgrep -f "$PROJECT_DIR/real_robot/ridgeback_ros2_node.py" >/dev/null; then
    echo "An OmTrackVLA Ridgeback controller is already running." >&2
    echo "Stop the previous launcher with Ctrl+C, then run this command again." >&2
    exit 2
fi

set +u
source "$ROS_SETUP"
set -u

CONTROLLER_PID=""
RVIZ_PID=""

cleanup() {
    trap - EXIT INT TERM
    if [ -n "$RVIZ_PID" ]; then
        kill -TERM "$RVIZ_PID" 2>/dev/null || true
        wait "$RVIZ_PID" 2>/dev/null || true
    fi
    if [ -n "$CONTROLLER_PID" ]; then
        kill -TERM -- "-$CONTROLLER_PID" 2>/dev/null || true
        wait "$CONTROLLER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Starting OmTrackVLA in ARMABLE mode and opening RViz..."
ROBOT_NAMESPACE="$ROBOT_NAMESPACE" \
CAMERA_TOPIC="$CAMERA_TOPIC" \
OMTRACKVLA_ARM_OUTPUT=1 \
setsid "$PROJECT_DIR/real_robot/start_ridgeback.sh" &
CONTROLLER_PID=$!

ROBOT_NAMESPACE="$ROBOT_NAMESPACE" \
"$PROJECT_DIR/real_robot/start_ridgeback_rviz.sh" &
RVIZ_PID=$!

STATUS_TOPIC="/$ROBOT_NAMESPACE/omtrackvla/status"
ENABLE_TOPIC="/$ROBOT_NAMESPACE/omtrackvla/enable"
STATUS_READY=false
for _ in $(seq 1 60); do
    if ! kill -0 "$CONTROLLER_PID" 2>/dev/null; then
        wait "$CONTROLLER_PID"
        exit 1
    fi
    if ros2 topic list 2>/dev/null | grep -Fxq "$STATUS_TOPIC"; then
        STATUS_READY=true
        break
    fi
    sleep 0.25
done
if [ "$STATUS_READY" != true ]; then
    echo "The OmTrackVLA status topic did not become ready." >&2
    exit 1
fi
if ! kill -0 "$RVIZ_PID" 2>/dev/null; then
    echo "RViz exited before the controller became ready." >&2
    exit 1
fi

echo
echo "Verify in RViz that the magenta OmTrackVLA path follows the intended person."
echo "Keep the physical E-stop reachable and clear the robot's path."
echo "OmTrackVLA has no reliable target-lock signal and may propose motion without the target."
echo
read -r -p "Type ENABLE and press Enter to start the deadman: " confirmation
if [ "$confirmation" != "ENABLE" ]; then
    echo "Not enabled; shutting down."
    exit 1
fi

echo "Deadman active. Press Ctrl+C to stop the robot and shut everything down."
ros2 topic pub --rate 10 \
    "$ENABLE_TOPIC" \
    std_msgs/msg/Bool \
    "{data: true}"
