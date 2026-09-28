# Clearpath Ridgeback deployment

This bridge keeps ROS 2 Jazzy (Python 3.12) separate from OmTrackVLA's Python 3.9
environment. `inference_server.py` performs OmTrackVLA waypoint inference and target
perception (`target_perception.py`) on loopback TCP only; `ridgeback_ros2_node.py` owns
ROS subscriptions and the safety-gated velocity output.

## Safety behavior

The controller starts in dry-run and publishes no velocity. Armed motion requires all
of the following at the same time:

- a continuously refreshed `omtrackvla/enable` Boolean deadman message;
- a fresh, inactive `platform/emergency_stop` state;
- fresh RGB camera, LiDAR scan, and model result;
- no valid LiDAR return inside the configured stop distance in the proposed
  translation corridor; close returns still block motion toward them and rotation;
- a `LOCKED` target from the target manager for the same frame as the path;
- a valid depth location for the locked target; target fusion corrects an opposite
  OmTrackVLA proposal toward that leader;
- target distance above `0.9 m` for translation; inside that distance translation stops;
- finite model output, clipped to the configured speed limits.

Loss of any required input commands zero velocity briefly and then stops publishing.
Keep the physical E-stop in hand and test dry-run before allowing motor output.

## Start in dry-run

This machine's Clearpath configuration identifies the robot as `r100_0160`, with its
RealSense compressed color stream at `camera/color/image_raw/compressed` and merged
dual-LiDAR scan at `sensors/scan`. The compressed transport is required here: adding a
second subscriber to the uncompressed RGB stream reproducibly stops both Hokuyo scan
streams on this computer. You can verify the selected topics without publishing:

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic list
ros2 topic info /r100_0160/camera/color/image_raw/compressed
ros2 topic info /r100_0160/sensors/scan
ros2 topic info /r100_0160/platform/emergency_stop
```

Start the bridge against those topics:

```bash
cd /home/robot/Desktop/omtrackvla
ROBOT_NAMESPACE=r100_0160 \
CAMERA_TOPIC=camera/color/image_raw/compressed \
CAMERA_COMPRESSED=true \
./real_robot/start_ridgeback.sh
```

This loads the real model but remains motion-silent in dry-run. Dummy inference is
still available for protocol-only checks by adding `OMTRACKVLA_DUMMY_INFERENCE=1`.
Inspect status with:

```bash
ros2 topic echo /r100_0160/omtrackvla/status
```

## Observe OmTrackVLA predictions

The bridge publishes OmTrackVLA's outputs even in dry-run. These are observation-only
topics and are never connected directly to Ridgeback's `cmd_vel` input:

```bash
# Target-fused velocity proposed to the safety gate (not sent in dry-run)
ros2 topic echo /r100_0160/omtrackvla/proposed_cmd_vel

# Raw speed-limited OmTrackVLA waypoint command before target fusion
ros2 topic echo /r100_0160/omtrackvla/raw_cmd_vel

# All predicted local waypoints in the robot's base_link frame
ros2 topic echo /r100_0160/omtrackvla/predicted_path

# Direct target-follow goal used by the fusion layer
ros2 topic echo /r100_0160/omtrackvla/fused_path
```

Add `/r100_0160/omtrackvla/predicted_path` as a Path visualization in Foxglove or
RViz, using `base_link` as the fixed frame. The status JSON also contains
`proposed_command_raw`, `proposed_command_fused`, `fusion`, and
`predicted_trajectory`. It also reports `planner_ran`, `planner_updated`,
`planner_age_seconds`, and `pipeline_seconds`, which distinguish a genuine/currently
updated planner result from target-search placeholders and measure both waypoint and
camera-response age. The raw command/path are OmTrackVLA outputs; the fused command
and cyan path are locally generated from OmTrackVLA plus the locked target position.
The bounding box is not an input to the released OmTrackVLA network: the local bridge
depth-localizes that selected box and fuses the resulting leader direction with
OmTrackVLA's raw command afterward.

### Complete Ridgeback RViz view

Close any existing RViz window, then start the repository-provided view:

```bash
cd /home/robot/Desktop/omtrackvla
./real_robot/start_ridgeback_rviz.sh
```

The launcher reads the namespace from `/etc/clearpath/robot.yaml` (or honors an
explicit `ROBOT_NAMESPACE`) and remaps the namespaced TF streams. Its configuration
shows the Ridgeback model, TF tree, merged dual-LiDAR scan, and annotated Target Tracker
image with the QoS profile used by each live publisher.
The separate raw RGB display is deliberately disabled: the system MJPEG service normally
occupies one raw subscription, and adding RViz as a second raw subscriber reproducibly
starves both Hokuyo streams. The supervised launcher pauses that optional service for
lower CPU load, while the Target Tracker uses the controller's compressed RGB input and
remains enabled. The camera panel shows target boxes only. OmTrackVLA's raw eight-point
trajectory is enabled in the 3D robot view as a green metric line with green waypoint
orientation arrows in `base_link`; X/Y coordinates are not scaled. The duplicate raw
front-LiDAR display is omitted; the raw rear display remains disabled by default. The
custom target-fused path remains
published but hidden by default. The hidden fused command still controls the Ridgeback
after safety checks; hiding it changes visualization only.
When `planner_ran` is false, the bridge publishes an empty raw path so RViz does not
misrepresent the eight zero-valued target-search placeholders as a model waypoint.

## Set the follow prompt

The startup prompt is configured in `real_robot/ridgeback.yaml`. Edit its `prompt:`
line and restart the bridge whenever you want a different target description. The
current configured prompt is:

```bash
prompt: Follow the person who is holding blue basket.
```

The ROS prompt topic remains available for temporary runtime overrides, but it is not
required when the desired prompt is stored in the YAML configuration.

The full line is sent directly to OmTrackVLA, and its target description (without the
leading `Follow`) is sent to Grounding DINO. Describe one visually distinctive leader
without alternatives, for example `Follow the person wearing a brown jacket.` Inspect
the Target Tracker panel and predicted path in dry-run before any supervised trial.

For a balanced response/load tradeoff, target detection and the annotated display run
at 3 Hz, while the heavier OmTrackVLA planner remains capped at 2 Hz. Between planner
updates the bridge may reuse the last genuine path for at most `0.75 s`; it still
rechecks the current target/depth geometry, and stale paths fail closed. The person
detector uses a 416-pixel input, Grounding DINO runs only every eighth target frame while
searching or recovering, and OmTrackVLA is skipped entirely until the target state is
`LOCKED`. The single
`start_ridgeback_all.sh` command launches the controller, perception, OmTrackVLA, RViz,
and deadman workflow from one terminal.

The raw RGB panel stays disabled to protect LiDAR; the annotated Target Tracker panel
follows the 3 Hz target rate. Completed annotations are published immediately and
image queues keep only the newest frame. The compressed RealSense JPEG is passed through
to the inference server without a decode/re-encode cycle in the ROS bridge. Annotated
frames are downscaled to at most 512 pixels wide before transfer back to ROS. The supplied
RViz view is capped at 10 FPS rather than rendering unnecessarily at 30 FPS.

The one-terminal launcher pauses `clearpath-sensors.service`, loads OmTrackVLA at
reduced CPU priority on CPUs 4-7, resets and verifies each scanner by fixed IP and
serial, and runs a locally patched build of the same official ROS `urg_node` driver.
The patch closes failed TCP sessions and avoids issuing a status command into a live
scan stream after a timeout. Front, rear, and merged scans must all be fresh before
RViz opens or `ENABLE` is offered. The standard Clearpath sensor service is restored
when the launcher exits. It also pauses `ridgeback-camera-mjpeg.service` during the run
because that optional web-camera encoder measured about 38% CPU; the RViz Target Tracker
does not depend on it, and cleanup restores it. Set `OMTRACKVLA_KEEP_MJPEG=1` only when
the separate web camera page is required during a trial. Build the pinned driver overlay again, if needed, with
`./real_robot/build_urg_node_overlay.sh`.

## Target identity lock

OmTrackVLA is unchanged and still plans from the full frame and prompt. In parallel,
the target manager grounds the prompt with Grounding DINO, accepts a match only when it
lies on a YOLO-detected person, tracks every person with BoT-SORT, and keeps an
appearance gallery for the selected identity. The state appears in the RViz *Target
Tracker* panel and on:

```bash
ros2 topic echo /r100_0160/omtrackvla/target_state
```

- `SEARCHING`: no unambiguous prompt match yet, or confirming one over 3 frames.
  `multiple_plausible_people` means two people match similarly; separate them or
  refine the prompt.
- `SEARCHING` with `no_person_with:red t shirt`: people are visible, but none has the
  prompted red shirt found on them in the right colour. Thin cyan boxes
  in the Target Tracker panel show what Grounding DINO found and its score.
- `LOCKED`: the only state that permits motion.
- `UNCERTAIN` / `LOST`: the target disappeared or its appearance changed. Motion stops,
  and only the original identity can be recovered; another person is never substituted.

`LOST` never times out. To start a fresh search without restarting:

```bash
ros2 topic pub --once /r100_0160/omtrackvla/reset_target std_msgs/msg/Empty {}
```

The depth check places the locked target in `base_link` (green sphere in RViz and
`omtrackvla/target_position`). Target fusion blends an aligned OmTrackVLA command toward
that position and replaces an opposite command with the target direction. Translation
slows as it approaches and stops at `min_follow_distance`; LiDAR can still reject the
fused direction. Similarity and distance thresholds are uncalibrated starting values.

The camera is not in TF, so its pose is `camera_mount` in `ridgeback.yaml`. Height,
roll, and pitch come from a floor fit (robot on flat floor, open floor in view):

```bash
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 real_robot/calibrate_camera_mount.py --namespace r100_0160
```

Measure the forward/lateral offset from the chassis centre and any yaw by tape.

Evaluate recorded runs offline before robot trials:

```bash
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 real_robot/export_bag_frames.py BAG_DIR \
    --topic /r100_0160/camera/color/image_raw --out frames/run1 --rate 2.0
./run_local.sh python real_robot/evaluate_offline.py frames/run1 \
    --prompt "Follow the person wearing a black jacket." --out eval/run1 --with-omtrack
```

Review `eval/run1/annotated.mp4`, `results.csv`, and `summary.json`; any
`locked_track_id_changes` must be checked by eye for a true identity switch.

Run the tests with:

```bash
./run_local.sh python -m unittest real_robot.test_safety real_robot.test_target_perception real_robot.test_target_geometry real_robot.test_gpu_ops real_robot.test_vision_gpu_preprocess
```

The implementation history, upstream-versus-local component boundary, verified live
behavior, and current operating configuration are maintained in
[`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md).

## Deadman and controlled arming

For a supervised one-terminal run, use the interactive launcher. It opens RViz,
starts the controller in armable mode, and requires `ENABLE` before it starts the
foreground deadman. Pressing Ctrl+C stops the deadman and all launcher processes:

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
CAMERA_TOPIC=camera/color/image_raw/compressed \
CAMERA_COMPRESSED=true \
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
