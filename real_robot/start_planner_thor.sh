#!/usr/bin/env bash
# Load only OmTrackVLA planning/vision models on Thor for the two-GPU dry-run.
set -euo pipefail
export OMTRACKVLA_INFERENCE_MODE=planner
exec "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/start_inference_thor.sh" "$@"
