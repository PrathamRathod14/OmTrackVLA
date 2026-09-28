#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
ROS_SETUP="/opt/ros/jazzy/setup.bash"
OVERLAY_DIR="$PROJECT_DIR/real_robot/urg_node_overlay"
UPSTREAM_COMMIT="936abce8cd1282eb866c20676e0f34fd031386b9"

set +u
source "$ROS_SETUP"
set -u

BUILD_ROOT="$(mktemp -d -t omtrackvla-urg-node.XXXXXX)"
cleanup() {
    case "$BUILD_ROOT" in
        /tmp/omtrackvla-urg-node.*) rm -rf -- "$BUILD_ROOT" ;;
    esac
}
trap cleanup EXIT

git clone --quiet https://github.com/ros-drivers/urg_node.git "$BUILD_ROOT/src/urg_node"
git -C "$BUILD_ROOT/src/urg_node" checkout --quiet "$UPSTREAM_COMMIT"
patch -d "$BUILD_ROOT/src/urg_node" -p1 < "$PROJECT_DIR/real_robot/urg_node_recovery.patch"

colcon --log-base "$BUILD_ROOT/log" build \
    --base-paths "$BUILD_ROOT/src" \
    --packages-select urg_node \
    --allow-overriding urg_node \
    --build-base "$BUILD_ROOT/build" \
    --install-base "$OVERLAY_DIR" \
    --merge-install \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF

echo "$UPSTREAM_COMMIT" > "$OVERLAY_DIR/OMTRACKVLA_PATCHED_COMMIT"
echo "Patched urg_node overlay installed at $OVERLAY_DIR"
