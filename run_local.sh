#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export PATH="$PROJECT_DIR/.conda-env/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR/habitat-lab"
export HF_MODEL_DIR="${HF_MODEL_DIR:-$PROJECT_DIR/models/OmTrackVLA-0.6B}"
export DINOV3_MODEL_PATH="${DINOV3_MODEL_PATH:-$PROJECT_DIR/models/dinov3-vits16-pretrain-lvd1689m}"
export SIGLIP_MODEL_PATH="${SIGLIP_MODEL_PATH:-$PROJECT_DIR/models/siglip-so400m-patch14-384}"

if [ "$#" -eq 0 ]; then
    echo "Usage: ./run_local.sh <command> [args...]" >&2
    echo "Example: ./run_local.sh python train.py --help" >&2
    exit 2
fi

exec "$@"
