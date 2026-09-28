#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
ROS_SETUP="/opt/ros/jazzy/setup.bash"
ROBOT_NAMESPACE="${ROBOT_NAMESPACE:-}"
CAMERA_TOPIC="${CAMERA_TOPIC:-}"

if [ -z "$ROBOT_NAMESPACE" ]; then
    echo "Set ROBOT_NAMESPACE to the Ridgeback namespace, for example r100_0001." >&2
    exit 2
fi
if [ -z "$CAMERA_TOPIC" ]; then
    echo "Set CAMERA_TOPIC to the RGB image topic before starting." >&2
    exit 2
fi
if [ -z "${CAMERA_COMPRESSED:-}" ]; then
    case "$CAMERA_TOPIC" in
        */compressed) CAMERA_COMPRESSED=true ;;
        *) CAMERA_COMPRESSED=false ;;
    esac
fi
if [ ! -f "$ROS_SETUP" ]; then
    echo "ROS 2 Jazzy was not found at $ROS_SETUP" >&2
    exit 2
fi
if pgrep -f "$PROJECT_DIR/real_robot/ridgeback_ros2_node.py" >/dev/null; then
    echo "An OmTrackVLA Ridgeback controller is already running." >&2
    echo "Stop its launcher with Ctrl+C before starting another controller." >&2
    exit 2
fi

DRY_RUN=true
DUMMY_ARG=""
if [ "${OMTRACKVLA_ARM_OUTPUT:-0}" = "1" ]; then
    DRY_RUN=false
fi
if [ "${OMTRACKVLA_DUMMY_INFERENCE:-0}" = "1" ]; then
    DUMMY_ARG="--dummy"
    if [ "${OMTRACKVLA_ARM_OUTPUT:-0}" = "1" ]; then
        echo "Refusing to arm motor output while dummy inference is enabled." >&2
        exit 2
    fi
elif [ ! -f "$PROJECT_DIR/models/dinov3-vits16-pretrain-lvd1689m/model.safetensors" ]; then
    echo "DINOv3 weights are missing. Download the licensed model before real inference." >&2
    echo "For a zero-velocity integration test, set OMTRACKVLA_DUMMY_INFERENCE=1." >&2
    exit 2
elif [ ! -f "$PROJECT_DIR/models/grounding-dino-tiny/model.safetensors" ] || [ ! -f "$PROJECT_DIR/models/yolo11n.pt" ]; then
    echo "Target-perception weights are missing (models/grounding-dino-tiny, models/yolo11n.pt)." >&2
    echo "Motion is gated on target lock, so real inference cannot start without them." >&2
    exit 2
fi
set +u
source "$ROS_SETUP"
set -u

cleanup() {
    if [ -n "${INFERENCE_PID:-}" ]; then
        kill "$INFERENCE_PID" 2>/dev/null || true
        wait "$INFERENCE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

cd "$PROJECT_DIR"
# Model construction briefly saturates CPU/I/O. Keep it below robot drivers and
# cap CPU thread pools; CUDA still performs the steady-state model computation.
INFERENCE_COMMAND=(nice -n "${OMTRACKVLA_NICE:-5}")
if [ -n "${OMTRACKVLA_CPUSET:-}" ]; then
    INFERENCE_COMMAND=(taskset -c "$OMTRACKVLA_CPUSET" "${INFERENCE_COMMAND[@]}")
fi
OMP_NUM_THREADS="${OMTRACKVLA_CPU_THREADS:-4}" \
MKL_NUM_THREADS="${OMTRACKVLA_CPU_THREADS:-4}" \
TOKENIZERS_PARALLELISM=false \
"${INFERENCE_COMMAND[@]}" \
"$PROJECT_DIR/run_local.sh" python "$PROJECT_DIR/real_robot/inference_server.py" $DUMMY_ARG &
INFERENCE_PID=$!

SERVER_READY=false
for _ in $(seq 1 300); do
    if ! kill -0 "$INFERENCE_PID" 2>/dev/null; then
        wait "$INFERENCE_PID"
        exit 1
    fi
    if /usr/bin/python3 -c 'import socket; s=socket.create_connection(("127.0.0.1",18765),0.1); s.close()' 2>/dev/null; then
        SERVER_READY=true
        break
    fi
    sleep 0.1
done
if [ "$SERVER_READY" != true ]; then
    echo "The OmTrackVLA inference server did not become ready." >&2
    exit 1
fi

echo "Namespace: /$ROBOT_NAMESPACE"
echo "Camera: $CAMERA_TOPIC"
echo "Output mode: $([ "$DRY_RUN" = true ] && echo DRY-RUN || echo ARMABLE)"
echo "Prompt topic: /$ROBOT_NAMESPACE/omtrackvla/prompt"
echo "Deadman topic: /$ROBOT_NAMESPACE/omtrackvla/enable"

/usr/bin/python3 "$PROJECT_DIR/real_robot/ridgeback_ros2_node.py" \
    --ros-args \
    -r "__ns:=/$ROBOT_NAMESPACE" \
    --params-file "$PROJECT_DIR/real_robot/ridgeback.yaml" \
    -p "camera_topic:=$CAMERA_TOPIC" \
    -p "scan_topic:=${SCAN_TOPIC:-sensors/scan}" \
    -p "estop_topic:=${ESTOP_TOPIC:-platform/emergency_stop}" \
    -p "cmd_vel_topic:=${CMD_VEL_TOPIC:-cmd_vel}" \
    -p "camera_compressed:=$CAMERA_COMPRESSED" \
    -p "dry_run:=$DRY_RUN" \
    "$@"
