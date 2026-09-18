# Clearpath Ridgeback deployment

This bridge keeps ROS 2 Jazzy (Python 3.12) separate from OmTrackVLA's Python 3.9
environment. `inference_server.py` performs GPU inference on loopback TCP only;
`ridgeback_ros2_node.py` owns ROS subscriptions and the safety-gated velocity output.

## Safety behavior

The controller starts in dry-run and publishes no velocity. Armed motion requires all
of the following at the same time:

- a continuously refreshed `omtrackvla/enable` Boolean deadman message;
- a fresh, inactive `platform/emergency_stop` state;
- fresh RGB camera, LiDAR scan, and model result;
- no valid LiDAR return inside the configured stop distance;
- finite model output, clipped to the configured speed limits.

Loss of any required input commands zero velocity briefly and then stops publishing.
Keep the physical E-stop in hand and test dry-run before allowing motor output.

## Start in dry-run

This machine's Clearpath configuration identifies the robot as `r100_0160`, with its
RealSense color stream at `camera/color/image_raw` and merged dual-LiDAR scan at
`sensors/scan`. You can verify those values without publishing anything:

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic list
ros2 topic info /r100_0160/camera/color/image_raw
ros2 topic info /r100_0160/sensors/scan
ros2 topic info /r100_0160/platform/emergency_stop
```

Start the bridge against those topics:

```bash
cd /home/robot/Desktop/omtrackvla
ROBOT_NAMESPACE=r100_0160 \
CAMERA_TOPIC=camera/color/image_raw \
./real_robot/start_ridgeback.sh
```

This loads the real model but remains motion-silent in dry-run. Dummy inference is
still available for protocol-only checks by adding `OMTRACKVLA_DUMMY_INFERENCE=1`.
Inspect status with:

```bash
ros2 topic echo /r100_0160/omtrackvla/status
```

## Observe OmTrackVLA's predictions

The bridge publishes OmTrackVLA's outputs even in dry-run. These are observation-only
topics and are never connected directly to Ridgeback's `cmd_vel` input:

```bash
# Speed-limited velocity proposed by OmTrackVLA (not sent to the wheels in dry-run)
ros2 topic echo /r100_0160/omtrackvla/proposed_cmd_vel

# All predicted local waypoints in the robot's base_link frame
ros2 topic echo /r100_0160/omtrackvla/predicted_path
```

Add `/r100_0160/omtrackvla/predicted_path` as a Path visualization in Foxglove or
RViz, using `base_link` as the fixed frame. The status JSON also contains
`proposed_command_raw`, `proposed_command_clipped`, and `predicted_trajectory`.
These values show what OmTrackVLA intends; they do not constitute a separate detector
or target-lock system.

### Complete Ridgeback RViz view

Close any existing RViz window, then start the repository-provided view:

```bash
cd /home/robot/Desktop/omtrackvla
./real_robot/start_ridgeback_rviz.sh
```

The launcher reads the namespace from `/etc/clearpath/robot.yaml` (or honors an
explicit `ROBOT_NAMESPACE`) and remaps the namespaced TF streams. Its configuration
shows the Ridgeback model, TF tree, merged dual-LiDAR scan, RealSense RGB image, and
OmTrackVLA predicted path with the QoS profile used by each live publisher. The
magenta line and yellow arrows are the eight raw OmTrackVLA waypoints, rendered
0.75 m above `base_link` for visibility; their X/Y coordinates are not scaled. Raw front
and rear LiDAR displays are included but disabled by default to avoid duplicating the
merged scan; enable them in the Displays panel when needed.

## Set the follow prompt

The startup prompt is configured in `real_robot/ridgeback.yaml`. Edit its `prompt:`
line and restart the bridge whenever you want a different target description. The
current configured prompt is:

```bash
prompt: Follow the person directly in front of you.
```

The ROS prompt topic remains available for temporary runtime overrides, but it is not
required when the desired prompt is stored in the YAML configuration.

This wording occurs directly in the included OmTrackVLA training data and is the
recommended real-camera trial prompt. Before enabling motion, place exactly one human
directly ahead of the camera and keep other people, humanoid robots, and person-shaped
objects outside the initial view. This improves the checkpoint's chances but is not a
target-lock guarantee.

The implementation history, upstream-versus-local component boundary, verified live
behavior, and current operating configuration are maintained in
[`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md).

## Deadman and controlled arming

For a supervised one-terminal run, use the interactive launcher. It opens RViz,
starts the controller in armable mode, and requires typing `ENABLE` before it starts
the foreground deadman. Pressing Ctrl+C stops the deadman and all launcher processes:

```bash
./real_robot/start_ridgeback_all.sh
```

The deadman must be published continuously; stopping this command stops the robot:

```bash
ros2 topic pub --rate 5 /r100_0160/omtrackvla/enable std_msgs/msg/Bool "{data: true}"
```

Only after validating camera, scan, E-stop, prompt, status, axis signs, and zero-command
behavior in dry-run should motor output be made armable:

```bash
ROBOT_NAMESPACE=r100_0160 \
CAMERA_TOPIC=camera/color/image_raw \
OMTRACKVLA_ARM_OUTPUT=1 \
./real_robot/start_ridgeback.sh
```

`OMTRACKVLA_ARM_OUTPUT=1` does not move the robot by itself. The continuously refreshed
deadman and all other safety gates are still required.

## Enable real model inference

The OmTrackVLA, Qwen, and SigLIP weights are already local. DINOv3 is the remaining
dependency and its Hugging Face repository requires accepting Meta's access terms.
After accepting them at
<https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m>, run:

```bash
cd /home/robot/Desktop/omtrackvla
./run_local.sh hf auth login
./run_local.sh hf download facebook/dinov3-vits16-pretrain-lvd1689m \
  --local-dir models/dinov3-vits16-pretrain-lvd1689m
```

Then repeat the dry-run command without `OMTRACKVLA_DUMMY_INFERENCE=1`. Do not arm
motor output until status, axes, stopping distance, and the physical E-stop have all
been validated in an open test area with a second person ready at the E-stop.
