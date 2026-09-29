#!/usr/bin/env bash
# Ridgeback RTX target perception plus Thor waypoint planning. Dry-run only.
set -euo pipefail
export OMTRACKVLA_INFERENCE_HOST="${OMTRACKVLA_INFERENCE_HOST:-192.168.131.51}"
export OMTRACKVLA_PERCEPTION_HOST=127.0.0.1
export OMTRACKVLA_PERCEPTION_PORT="${OMTRACKVLA_PERCEPTION_PORT:-18766}"
exec "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/start_ridgeback.sh" "$@"
