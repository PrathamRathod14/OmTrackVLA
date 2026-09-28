#!/usr/bin/env bash
# Keep the Ridgeback ROS/safety bridge local; connect to inference on Thor.
set -euo pipefail

PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
export OMTRACKVLA_INFERENCE_HOST="${OMTRACKVLA_INFERENCE_HOST:-192.168.131.51}"
exec "$PROJECT_DIR/real_robot/start_ridgeback.sh" "$@"
