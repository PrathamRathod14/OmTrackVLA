#!/usr/bin/env bash
# Run only the OmTrackVLA inference service on a Jetson AGX Thor.
set -euo pipefail

PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
THOR_BIND_IP="${OMTRACKVLA_THOR_BIND_IP:-192.168.131.51}"
RIDGEBACK_IP="${OMTRACKVLA_RIDGEBACK_IP:-192.168.131.1}"
INFERENCE_PYTHON="${OMTRACKVLA_INFERENCE_PYTHON:-$PROJECT_DIR/.conda-env/bin/python}"
INFERENCE_PORT="${OMTRACKVLA_INFERENCE_PORT:-18765}"

if [ ! -x "$INFERENCE_PYTHON" ]; then
    echo "Build an Arm64 CUDA Python environment and set OMTRACKVLA_INFERENCE_PYTHON to its interpreter." >&2
    exit 2
fi

"$INFERENCE_PYTHON" - "$THOR_BIND_IP" "$RIDGEBACK_IP" "$INFERENCE_PORT" <<'PY'
import ipaddress
import sys

bind, peer, port = sys.argv[1:]
for label, value in (("Thor bind", bind), ("Ridgeback", peer)):
    address = ipaddress.IPv4Address(value)
    if address.is_loopback or address.is_unspecified or address.is_multicast:
        raise SystemExit(f"{label} must be a concrete robot-network IPv4 address")
if not 1 <= int(port) <= 65535:
    raise SystemExit("OMTRACKVLA_INFERENCE_PORT must be between 1 and 65535")
PY

if [ "${OMTRACKVLA_DUMMY_INFERENCE:-0}" != "1" ]; then
    for model in \
        OmTrackVLA-0.6B/model.safetensors \
        Qwen3-0.6B/model.safetensors \
        dinov3-vits16-pretrain-lvd1689m/model.safetensors \
        siglip-so400m-patch14-384/model.safetensors \
        grounding-dino-tiny/model.safetensors \
        yolo11n.pt; do
        if [ ! -f "$PROJECT_DIR/models/$model" ]; then
            echo "Missing Thor model: $PROJECT_DIR/models/$model" >&2
            exit 2
        fi
    done
    DUMMY_ARG=()
else
    DUMMY_ARG=(--dummy)
fi

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${OMTRACKVLA_CPU_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export TOKENIZERS_PARALLELISM=false
echo "Starting Thor inference at $THOR_BIND_IP:$INFERENCE_PORT for Ridgeback $RIDGEBACK_IP"
exec "$PROJECT_DIR/run_local.sh" "$INFERENCE_PYTHON" \
    "$PROJECT_DIR/real_robot/inference_server.py" \
    --host "$THOR_BIND_IP" --port "$INFERENCE_PORT" --allowed-client "$RIDGEBACK_IP" \
    "${DUMMY_ARG[@]}" "$@"
