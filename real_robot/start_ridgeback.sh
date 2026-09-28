#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
ROS_SETUP="/opt/ros/jazzy/setup.bash"
ROBOT_NAMESPACE="${ROBOT_NAMESPACE:-}"
CAMERA_TOPIC="${CAMERA_TOPIC:-}"
INFERENCE_HOST="${OMTRACKVLA_INFERENCE_HOST:-127.0.0.1}"
INFERENCE_PORT="${OMTRACKVLA_INFERENCE_PORT:-18765}"
REMOTE_INFERENCE=false
if [ "$INFERENCE_HOST" != "127.0.0.1" ]; then
    REMOTE_INFERENCE=true
fi
if [ "$REMOTE_INFERENCE" = true ] && [ "${OMTRACKVLA_ARM_OUTPUT:-0}" = "1" ] && \
   [ "${OMTRACKVLA_ALLOW_REMOTE_ARM:-0}" != "1" ]; then
    echo "Remote inference has only been verified in dry-run. Set OMTRACKVLA_ALLOW_REMOTE_ARM=1 only after the Thor replay, timing, and safety gates pass." >&2
    exit 2
fi

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
if [ "$REMOTE_INFERENCE" = true ] && [ "${OMTRACKVLA_DUMMY_INFERENCE:-0}" = "1" ]; then
    echo "Dummy inference must be started on Thor, not requested by the Ridgeback bridge." >&2
    exit 2
elif [ "$REMOTE_INFERENCE" = true ]; then
    DUMMY_ARG=""
elif [ "${OMTRACKVLA_DUMMY_INFERENCE:-0}" = "1" ]; then
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
if [ "$REMOTE_INFERENCE" = false ]; then
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
fi

SERVER_READY=false
for _ in $(seq 1 300); do
    if [ "$REMOTE_INFERENCE" = false ] && ! kill -0 "$INFERENCE_PID" 2>/dev/null; then
        wait "$INFERENCE_PID"
        exit 1
    fi
    if /usr/bin/python3 -c 'import socket,sys; from real_robot.protocol import send_message,receive_message; s=socket.create_connection((sys.argv[1],int(sys.argv[2])),0.2); s.settimeout(0.5); send_message(s,{"request_id":0,"prompt":"__probe__","encoding":"jpeg"}); r=receive_message(s); s.close(); sys.exit(0 if r.get("request_id")==0 and r.get("error")=="ValueError: image_b64 is missing" else 1)' \
        "$INFERENCE_HOST" "$INFERENCE_PORT" 2>/dev/null; then
        SERVER_READY=true
        break
    fi
    sleep 0.1
done
if [ "$SERVER_READY" != true ]; then
    echo "The OmTrackVLA inference server at $INFERENCE_HOST:$INFERENCE_PORT did not become ready." >&2
    exit 1
fi

echo "Namespace: /$ROBOT_NAMESPACE"
echo "Camera: $CAMERA_TOPIC"
echo "Inference: $INFERENCE_HOST:$INFERENCE_PORT"
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
    -p "inference_host:=$INFERENCE_HOST" \
    -p "inference_port:=$INFERENCE_PORT" \
    -p "dry_run:=$DRY_RUN" \
    "$@"
