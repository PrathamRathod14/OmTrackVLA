# OmTrackVLA Ridgeback Project Context

Last updated: 2026-09-18

This is the living handoff record for the OmTrackVLA deployment on Clearpath Ridgeback
`r100-0160`. Update it in the same change whenever the real-robot implementation,
configuration, operating procedure, or verified behavior changes.

## Objective

Run the released OmTrackVLA 0.6B checkpoint on the Ridgeback's monocular RealSense
stream, convert its eight local waypoints into holonomic base commands, visualize the
raw model output, and retain fail-closed physical safety gates.

## Upstream baseline

- Repository: `https://github.com/om-ai-lab/OmTrackVLA.git`
- Checked-out upstream commit: `e9cb1fb`
- OmTrackVLA takes a natural-language instruction plus coarse video history and current
  fine visual features and always returns eight `[x, y, yaw]` waypoints.
- DINOv3 and SigLIP produce visual tokens; Qwen3-0.6B fuses text and vision; the
  OmTrackVLA waypoint head produces the trajectory.
- Upstream `trained_agent.py` converts waypoint index 1 to velocity using `dt=0.1`.
- Upstream inference has no target identity, target confidence, target-present flag, or
  target-lost output. The optional `bbox_feat` argument in `model.py` is not consumed by
  the current forward pass.
- The Habitat task guarantees a scripted humanoid target and uses semantic humanoid
  sensors for evaluation. Those ground-truth sensors are not used by the waypoint
  planner action.

## Locally downloaded models

- `models/OmTrackVLA-0.6B`
- `models/Qwen3-0.6B`
- `models/siglip-so400m-patch14-384`
- `models/dinov3-vits16-pretrain-lvd1689m`

`cache_gridpool.py` was minimally changed to allow DINOv3 and SigLIP to load from these
local directories. `run_local.sh` launches commands inside the Python 3.9 environment.

## Added real-robot stack

Everything under `real_robot/` is a local Ridgeback integration, not part of the
upstream repository at commit `e9cb1fb`:

- `inference_server.py`: Python 3.9/CUDA OmTrackVLA inference service on loopback port
  `18765`; maintains 31 coarse history frames and returns the raw eight-waypoint path.
- `ridgeback_ros2_node.py`: ROS 2 Jazzy bridge for camera, prompt, LiDAR, E-stop,
  deadman, status, visualization, and Ridgeback velocity commands.
- `safety.py`: fail-closed freshness, E-stop, obstacle, finite-value, speed, and
  acceleration gates.
- `ridgeback_omtrack.rviz`: namespaced TF, Ridgeback model, camera, merged LiDAR, and
  raw OmTrackVLA trajectory display.
- `start_ridgeback.sh`: controller/inference launcher; dry-run unless
  `OMTRACKVLA_ARM_OUTPUT=1`.
- `start_ridgeback_rviz.sh`: namespaced RViz launcher.
- `start_ridgeback_all.sh`: supervised one-terminal armable launcher. It requires the
  operator to type `ENABLE`, runs the deadman in the foreground, and cleans up on
  Ctrl+C.
- `protocol.py` and `test_safety.py`: loopback protocol and safety regression tests.

## ROS interfaces

- Camera input: `/r100_0160/camera/color/image_raw`
- Merged LiDAR: `/r100_0160/sensors/scan`
- E-stop: `/r100_0160/platform/emergency_stop` (`true` means active)
- Ridgeback command: `/r100_0160/cmd_vel` (`geometry_msgs/msg/TwistStamped`)
- Prompt: `/r100_0160/omtrackvla/prompt`
- Deadman: `/r100_0160/omtrackvla/enable`
- Status: `/r100_0160/omtrackvla/status`
- Proposed velocity: `/r100_0160/omtrackvla/proposed_cmd_vel`
- Raw model path: `/r100_0160/omtrackvla/predicted_path`

## Current prompt and trial arrangement

Configured in `ridgeback.yaml`:

```text
Follow the person directly in front of you.
```

This exact instruction appears 767 times in the included training data. It avoids the
ambiguous word `or` and does not ask the checkpoint to understand an unseen object such
as a carried box. For the best available trial condition:

1. Put exactly one human directly ahead of the camera before starting the history.
2. Keep the person fully visible and roughly centered.
3. Remove other people, the G1 humanoid, and person-shaped or dark objects from the
   initial view.
4. Validate that the magenta path/yellow arrows respond consistently before typing
   `ENABLE`.

This setup improves similarity to training; it does not prove target lock.

## Current safety configuration

- Dry-run is the default.
- Armable mode requires `OMTRACKVLA_ARM_OUTPUT=1`.
- Deadman messages must remain fresh within `0.5 s`.
- Camera timeout: `0.75 s`.
- Inference timeout: `1.5 s`.
- LiDAR timeout: `0.5 s`.
- E-stop timeout: `1.0 s`.
- Obstacle stop distance: `0.70 m` from the LiDAR origin. The R100 chassis is
  `0.960 x 0.793 m`, so this remains above its `0.48 m` front half-length.
- Maximum command: `0.20 m/s` forward, `0.20 m/s` lateral, `0.35 rad/s` yaw.
- Acceleration limits: `0.30 m/s^2` linear and `0.50 rad/s^2` angular.

No safety gate is a semantic target validator. LiDAR prevents close-obstacle motion but
does not determine whether OmTrackVLA selected the prompted person.

## Visualization meaning

- Magenta line: raw OmTrackVLA eight-waypoint trajectory.
- Yellow arrows: raw waypoint yaw orientations.
- Cyan points: merged Ridgeback LiDAR, not OmTrackVLA.
- Robot model and TF axes: Clearpath ROS data, not OmTrackVLA.
- RGB panel: raw RealSense stream.

The path is rendered `0.75 m` above `base_link`, with larger arrows and a thick
billboard line because the predicted horizon is often only about `0.14 m`. X/Y values
are not scaled or altered for visualization.

## Verified behavior and limitation

- Real inference runs at approximately `0.17-0.27 s` per frame on this machine.
- Camera, merged LiDAR, E-stop, namespaced TF, robot description, inference, and RViz
  topic connectivity have been verified live.
- The bridge publishes no wheel command in dry-run and stops on stale deadman, stale
  sensing, active/stale E-stop, close obstacle, stale inference, invalid values, or
  disconnection.
- Eight safety tests pass.
- The model path visibly changes with scene/person motion, demonstrating visual
  responsiveness.
- It does not demonstrate reliable identity selection. The robot has proposed motion
  toward other people, a G1 humanoid, and a black chair.
- An identical-frame prompt comparison produced similar forward commands for a black
  jacket, white lab coat, black chair, and even an instruction to move away. Thus text
  changes the numbers slightly but does not reliably control target selection on the
  real frame.
- The released checkpoint was trained on Habitat data. Real-camera domain shift and
  the absence of a target-presence output make unattended physical following unsafe.

## Recommended launch

Stop any previous controller, then use one interactive terminal:

```bash
cd /home/robot/Desktop/omtrackvla
./real_robot/start_ridgeback_all.sh
```

Inspect the raw path in RViz. Type `ENABLE` only during a supervised trial with the
physical E-stop reachable. Press Ctrl+C to stop the foreground deadman and shut down
the launcher processes.

## Change log

### 2026-09-17

- Cloned and installed the upstream repository and model dependencies.
- Downloaded OmTrackVLA, Qwen, SigLIP, and licensed DINOv3 weights.
- Added the split Python 3.9 inference/ROS 2 Jazzy architecture.
- Added Ridgeback topics, safety gates, dry-run/armable modes, status, tests, and
  documentation.
- Added predicted velocity/path publication and corrected ROS logger/shutdown issues.
- Added namespaced Ridgeback/TF/LiDAR/camera/trajectory RViz configuration.

### 2026-09-18

- Added the supervised one-terminal launcher.
- Reduced the LiDAR stop boundary from `0.75 m` to `0.70 m` after retaining clearance
  beyond the R100 front half-length.
- Enlarged and raised the RViz waypoint visualization without altering model values.
- Verified directly that real-frame outputs are only weakly sensitive to substantially
  different prompts.
- Changed the startup prompt to the training-matched instruction `Follow the person
  directly in front of you.`
- Added this living project context and linked it from the real-robot README.
