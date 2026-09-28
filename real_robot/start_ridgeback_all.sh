#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
ROS_SETUP="/opt/ros/jazzy/setup.bash"
ROBOT_NAMESPACE="${ROBOT_NAMESPACE:-r100_0160}"
CAMERA_TOPIC="${CAMERA_TOPIC:-camera/color/image_raw/compressed}"
CAMERA_COMPRESSED="${CAMERA_COMPRESSED:-true}"

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
LIDAR_PIDS=()
SENSOR_SERVICE_STOPPED=false
MJPEG_SERVICE_STOPPED=false

stop_stale_hokuyo_overlay() {
    local driver="$PROJECT_DIR/real_robot/urg_node_overlay/lib/urg_node/urg_node_driver"
    local deadline
    local pid
    local stale_pids=()

    mapfile -t stale_pids < <(pgrep -f "^${driver} " || true)
    if [ "${#stale_pids[@]}" -eq 0 ]; then
        return 0
    fi

    echo "Stopping stale OmTrackVLA Hokuyo drivers from an earlier interrupted run..."
    for pid in "${stale_pids[@]}"; do
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    done
    deadline=$((SECONDS + 5))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if ! pgrep -f "^${driver} " >/dev/null; then
            return 0
        fi
        sleep 0.1
    done

    echo "A stale OmTrackVLA Hokuyo driver did not stop; refusing to continue." >&2
    return 1
}

probe_hokuyo() {
    local ip_address="$1"
    local expected_serial="$2"
    local attempt

    for attempt in 1 2 3; do
        # This sends QT, verifies VV, and closes the TCP session cleanly.
        if /usr/bin/python3 "$PROJECT_DIR/real_robot/hokuyo_scip_reset.py" \
            "$ip_address" "$expected_serial"; then
            return 0
        fi
        sleep 1
    done

    echo "Hokuyo $expected_serial at $ip_address did not answer the read-only SCIP probe." >&2
    return 1
}

start_hokuyo() {
    local index="$1"
    local driver="$PROJECT_DIR/real_robot/urg_node_overlay/lib/urg_node/urg_node_driver"
    local config="/etc/clearpath/sensors/config/lidar2d_${index}.yaml"

    RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    setsid "$driver" --ros-args \
        -r "__ns:=/$ROBOT_NAMESPACE/sensors/lidar2d_${index}" \
        --params-file "$config" \
        -p error_limit:=20 \
        -r /diagnostics:=diagnostics &
    LIDAR_PIDS+=("$!")
}

require_scan_message() {
    local topic="$1"
    local description="$2"

    if ! timeout 30 ros2 topic echo --once --field header "$topic" >/dev/null 2>&1; then
        echo "No fresh $description arrived on $topic within 30 seconds." >&2
        return 1
    fi
    echo "Fresh $description received."
}

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
    for pid in "${LIDAR_PIDS[@]}"; do
        kill -TERM -- "-$pid" 2>/dev/null || true
    done
    for pid in "${LIDAR_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    if [ "$SENSOR_SERVICE_STOPPED" = true ]; then
        sudo -n systemctl start clearpath-sensors.service 2>/dev/null || \
            echo "Warning: run 'sudo systemctl start clearpath-sensors.service' to restore the LiDAR service." >&2
        SENSOR_SERVICE_STOPPED=false
    fi
    if [ "$MJPEG_SERVICE_STOPPED" = true ]; then
        sudo -n systemctl start ridgeback-camera-mjpeg.service 2>/dev/null || \
            echo "Warning: run 'sudo systemctl start ridgeback-camera-mjpeg.service' to restore the web camera stream." >&2
        MJPEG_SERVICE_STOPPED=false
    fi
}
trap cleanup EXIT INT TERM

echo "Taking supervised control of the two Hokuyo drivers for this run."
if ! sudo -v; then
    echo "Administrator authentication is required to supervise the LiDAR service." >&2
    exit 1
fi
stop_stale_hokuyo_overlay
if [ "${OMTRACKVLA_KEEP_MJPEG:-0}" != "1" ] && \
   systemctl is-active --quiet ridgeback-camera-mjpeg.service; then
    echo "Pausing the web MJPEG stream to free CPU for tracking; it will be restored on exit."
    sudo systemctl stop ridgeback-camera-mjpeg.service
    MJPEG_SERVICE_STOPPED=true
fi
sudo systemctl stop clearpath-sensors.service
SENSOR_SERVICE_STOPPED=true
echo "Standard LiDAR service paused; motor output is not enabled. Loading OmTrackVLA..."

echo "Starting OmTrackVLA in ARMABLE mode; RViz will open after LiDAR recovery..."
ROBOT_NAMESPACE="$ROBOT_NAMESPACE" \
CAMERA_TOPIC="$CAMERA_TOPIC" \
CAMERA_COMPRESSED="$CAMERA_COMPRESSED" \
OMTRACKVLA_ARM_OUTPUT=1 \
OMTRACKVLA_CPUSET="${OMTRACKVLA_CPUSET:-4-7}" \
setsid "$PROJECT_DIR/real_robot/start_ridgeback.sh" &
CONTROLLER_PID=$!

STATUS_TOPIC="/$ROBOT_NAMESPACE/omtrackvla/status"
ENABLE_TOPIC="/$ROBOT_NAMESPACE/omtrackvla/enable"
STATUS_READY=false
for _ in $(seq 1 180); do
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

PATCHED_OVERLAY="$PROJECT_DIR/real_robot/urg_node_overlay"
EXPECTED_URG_COMMIT="936abce8cd1282eb866c20676e0f34fd031386b9"
if [ ! -x "$PATCHED_OVERLAY/lib/urg_node/urg_node_driver" ] || \
   [ "$(tr -d '\n' < "$PATCHED_OVERLAY/OMTRACKVLA_PATCHED_COMMIT" 2>/dev/null || true)" != "$EXPECTED_URG_COMMIT" ]; then
    echo "The recovered urg_node driver is missing or has the wrong version." >&2
    echo "Run ./real_robot/build_urg_node_overlay.sh once, then launch again." >&2
    exit 1
fi
set +u
source "$PATCHED_OVERLAY/local_setup.bash"
set -u

echo "OmTrackVLA is resident. Recovering the front scanner..."
probe_hokuyo 192.168.131.21 L2215985
sleep 2
start_hokuyo 0
require_scan_message "/$ROBOT_NAMESPACE/sensors/lidar2d_0/scan" "front LiDAR scan"

echo "Recovering the rear scanner..."
probe_hokuyo 192.168.131.22 L2215963
sleep 2
start_hokuyo 1

if ! require_scan_message "/$ROBOT_NAMESPACE/sensors/lidar2d_1/scan" "rear LiDAR scan" || \
   ! require_scan_message "/$ROBOT_NAMESPACE/sensors/scan" "merged LiDAR scan"; then
    echo "The supervised LiDAR recovery failed; RViz and motor enable were not started." >&2
    exit 1
fi

# RViz is intentionally delayed until the model is resident so its rendering and
# subscriptions cannot compete with the Hokuyo drivers during the startup burst.
ROBOT_NAMESPACE="$ROBOT_NAMESPACE" \
"$PROJECT_DIR/real_robot/start_ridgeback_rviz.sh" &
RVIZ_PID=$!
sleep 0.5
if ! kill -0 "$RVIZ_PID" 2>/dev/null; then
    echo "RViz exited during startup." >&2
    exit 1
fi

# The web MJPEG service is normally paused by this launcher to save CPU. If the operator
# keeps it active, one raw subscriber is sustainable; a second raw subscription from
# RViz reproducibly starves both Hokuyo streams. The Target Tracker display remains
# available from the bridge's compressed RGB input.
sleep 2
RAW_RGB_TOPIC="/$ROBOT_NAMESPACE/camera/color/image_raw"
RAW_RGB_SUBSCRIBERS="$(ros2 topic info "$RAW_RGB_TOPIC" 2>/dev/null | awk '/Subscription count:/ {print $3; exit}')"
if [ -n "$RAW_RGB_SUBSCRIBERS" ] && [ "$RAW_RGB_SUBSCRIBERS" -gt 1 ]; then
    echo "Unsafe duplicate raw RGB subscription detected on $RAW_RGB_TOPIC." >&2
    echo "Keep the RViz raw RGB display disabled; use the Target Tracker panel." >&2
    exit 1
fi
if ! require_scan_message "/$ROBOT_NAMESPACE/sensors/lidar2d_0/scan" "post-RViz front LiDAR scan" || \
   ! require_scan_message "/$ROBOT_NAMESPACE/sensors/lidar2d_1/scan" "post-RViz rear LiDAR scan" || \
   ! require_scan_message "/$ROBOT_NAMESPACE/sensors/scan" "post-RViz merged LiDAR scan"; then
    echo "LiDAR became unhealthy after RViz startup; motor enable was not started." >&2
    exit 1
fi

echo
echo "Verify in RViz that the Target Tracker view shows TARGET LOCKED on the intended person"
echo "and that the magenta OmTrackVLA path heads toward that person."
echo "Keep the physical E-stop reachable and clear the robot's path."
echo "Motion is blocked unless the target is LOCKED and the path is left/right consistent with it."
echo "The RGB-only consistency check cannot verify following distance."
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
