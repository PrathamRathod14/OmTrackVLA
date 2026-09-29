# OmTrackVLA Ridgeback Project Context

Last updated: 2026-09-29

This is the living handoff record for the OmTrackVLA deployment on Clearpath Ridgeback
`r100-0160`. Update it in the same change whenever the real-robot implementation,
configuration, operating procedure, or verified behavior changes.

## Objective

Run the released OmTrackVLA 0.6B checkpoint on the Ridgeback's monocular RealSense
stream, convert its eight local waypoints into holonomic base commands, visualize the
raw model output, and retain fail-closed physical safety gates. A locally added target
manager (Grounding DINO + YOLO person detection + BoT-SORT + appearance gallery) runs in
parallel and must verify the prompted person's identity before any motion is permitted;
OmTrackVLA itself is unchanged and remains the waypoint planner.

## Upstream baseline

- Repository: `https://github.com/om-ai-lab/OmTrackVLA.git`
- Fetched upstream baseline: `origin/main` at `e9cb1fb`. Local Ridgeback integration
  begins with downstream commit `8fedc69` and subsequent local changes. This is not a
  clean checkout of upstream `main`.
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
- `models/grounding-dino-tiny` (Hugging Face `IDEA-Research/grounding-dino-tiny`, local
  target perception only)
- `models/yolo11n.pt` (Ultralytics COCO detector, person class only, local target
  perception only)

`cache_gridpool.py` allows DINOv3 and SigLIP to load from these local directories and
provides the measured shared-CUDA preprocessing path used by the Ridgeback inference
server. `run_local.sh` launches commands inside the Python 3.9 environment.

## Added real-robot stack

The implementation under `real_robot/` and its documentation under `docs/` are local
Ridgeback integration, not part of the upstream repository at commit `e9cb1fb`.
Compatibility links keep the historical documentation paths under `real_robot/`
working:

- `inference_server.py`: Python 3.9/CUDA OmTrackVLA inference service on loopback port
  `18765`; uploads one RGB tensor for shared CUDA resize/normalization by DINOv3 and
  SigLIP, maintains 31 coarse history frames on CUDA, and returns the raw eight-waypoint path,
  plus the target-manager state and RGB trajectory-consistency result for the same frame.
  The locally added Thor split can run this process under Python 3.12/CUDA on Arm64,
  bound to `192.168.131.51` and allowing only the Ridgeback's `192.168.131.1` IP.
  The new `--mode perception` loads only YOLO/Grounding DINO and the target manager
  on Ridgeback; `--mode planner` loads only DINOv3, SigLIP, Qwen3 and OmTrackVLA on
  Thor. The existing `--mode full` remains the default for local and full-Thor runs.
  This is a deployment option for the local integration; upstream OmTrackVLA is
  unchanged. Loopback remains the default on the x86 host.
- `hybrid_protocol.py`: validates that local perception and Thor planning responses
  belong to the same request, camera frame and target identity before the ROS bridge
  accepts them. The Thor planner validates locked target boxes before inference;
  invalid-target frames carry metadata only, with no JPEG transfer to Thor.
- `perf_log.py`, `summarize_perf.py`: opt-in JSONL stage timing and read-only summary
  for dry-run baseline captures. Profiling synchronizes CUDA at stage boundaries and
  the ROS bridge refuses armable mode while it is enabled.
- `target_perception.py`: target manager. Grounding DINO grounds the prompt on
  initialization/recovery; all YOLO person detections (not only prompt matches) feed
  Ultralytics BoT-SORT with sparse-optical-flow camera-motion compensation; a part-based
  HSV clothing descriptor supplies BoT-SORT's appearance features and the target's
  identity gallery. States are `SEARCHING -> LOCKED -> UNCERTAIN -> LOST`. It never
  generates motion.
- `evaluate_offline.py`: replays a frame directory or video through target perception,
  optionally with OmTrackVLA, writing `results.csv`, `annotated.mp4`, `summary.json`.
- `target_geometry.py`: pure-NumPy depth localization of the locked target (unaligned
  RealSense depth projected into color with the published extrinsics, torso-region
  median, coverage/consistency checks) and the metric path-versus-target check.
- `calibrate_camera_mount.py`: read-only floor-plane fit from one depth frame giving
  the camera height, pitch, and roll for `camera_mount`.
- `export_bag_frames.py`: ROS 2 Python exporter from a bag's camera topic to
  timestamp-named JPEG frames for `evaluate_offline.py`.
- `ridgeback_ros2_node.py`: ROS 2 Jazzy bridge for camera, prompt, LiDAR, E-stop,
  deadman, status, visualization, and Ridgeback velocity commands.
- `safety.py`: fail-closed freshness, E-stop, obstacle, finite-value, speed, and
  acceleration gates.
- `ridgeback_omtrack.rviz`: namespaced TF, Ridgeback model, camera, merged LiDAR, raw
  OmTrackVLA trajectory, and annotated Target Tracker image display.
- `start_ridgeback.sh`: controller/inference launcher; dry-run unless
  `OMTRACKVLA_ARM_OUTPUT=1`; refuses to start a duplicate controller. With
  `OMTRACKVLA_INFERENCE_HOST` set to Thor, it starts only the local ROS bridge.
- `start_inference_thor.sh`: runs the project inference process on the attached Thor
  using its separate Arm64 CUDA environment and a single allowed Ridgeback IP.
  `start_planner_thor.sh` selects planner-only mode and requires only the four
  planner/vision model weights.
- `requirements_thor.txt`: direct Python dependency pins for this project's
  JetPack 7.0/Python 3.12 inference environment. It does not apply to other Thor
  projects or to the upstream Habitat training setup.
- `start_ridgeback_thor.sh`: dry-run by default, connects the host ROS bridge to the
  Thor server at `192.168.131.51` while retaining sensor and motor safety locally.
- `start_ridgeback_hybrid.sh`: starts Ridgeback loopback perception on RTX port
  `18766`, connects the same Ridgeback ROS bridge to Thor planner port `18765`, and
  rejects armable output until live locked-target and physical safety checks pass.
- `start_ridgeback_rviz.sh`: namespaced RViz launcher.
- `start_ridgeback_all.sh`: supervised one-terminal armable launcher. It requires the
  operator to type `ENABLE`, runs the deadman in the foreground, and cleans up on
  Ctrl+C.
- `hokuyo_scip_reset.py`: clean `QT`/`VV` SCIP reset and fixed-IP/serial verification
  for the two UST-20LX scanners.
- `urg_node_recovery.patch` and `build_urg_node_overlay.sh`: reproducible, pinned
  build of the official ROS driver with two connection-recovery fixes. The generated
  `urg_node_overlay/` install tree is local and is not committed; build it before a
  fresh deployment. This infrastructure does not modify OmTrackVLA.
- `protocol.py`, `test_safety.py`, `test_target_perception.py`, and
  `test_target_geometry.py`: loopback protocol, safety gate, target state-machine, and
  depth geometry regression tests.

## ROS interfaces

- Camera input: `/r100_0160/camera/color/image_raw/compressed` (JPEG). A second
  uncompressed RGB subscription reproducibly starves both scanner streams on this host;
  the bridge must use the compressed topic.
- Merged LiDAR: `/r100_0160/sensors/scan`
- E-stop: `/r100_0160/platform/emergency_stop` (`true` means active)
- Ridgeback command: `/r100_0160/cmd_vel` (`geometry_msgs/msg/TwistStamped`)
- Prompt: `/r100_0160/omtrackvla/prompt`
- Deadman: `/r100_0160/omtrackvla/enable`
- Status: `/r100_0160/omtrackvla/status`
- Proposed velocity: `/r100_0160/omtrackvla/proposed_cmd_vel`
- Raw model path: `/r100_0160/omtrackvla/predicted_path`
- Target state JSON: `/r100_0160/omtrackvla/target_state` (`std_msgs/String`; state,
  reason, track ID, bbox, grounding score, appearance similarity, validity flags)
- Target visualization: `/r100_0160/omtrackvla/target_image` (`sensor_msgs/Image`,
  best-effort QoS). It is published immediately when inference completes rather than
  waiting for the status timer; its expected rate is the configured 3 Hz target rate.
- Target reset: `/r100_0160/omtrackvla/reset_target` (`std_msgs/Empty`); clears the
  target identity and restarts the prompt search without restarting the models.
- Hybrid-only bridge parameters: `perception_host=127.0.0.1` and
  `perception_port=18766` from its launcher; an empty host in `ridgeback.yaml`
  selects the existing single inference server. Thor's planner remains a TCP
  process even though ROS 2 Jazzy is installed on both computers.
- `target_state` includes `target_debug`: every tracked person (score, box, gallery
  similarity), every Grounding DINO box (label, score, colour fraction), and the
  candidate ranking.
- Locked target position: `/r100_0160/omtrackvla/target_position`
  (`geometry_msgs/PointStamped`, `base_link`)
- Depth input: `/r100_0160/camera/depth/image_rect_raw/compressedDepth` (PNG 16UC1,
  ~0.17 MB, 30 Hz), `camera/depth/camera_info`, `camera/color/camera_info`,
  `camera/extrinsics/depth_to_color` (transient local). Subscribing to raw
  `image_rect_raw` (~0.6 MB, 19 Hz) stalled color delivery to the bridge after one
  frame, so raw depth must not be used.
- The RealSense is not in Ridgeback's TF tree or `/etc/clearpath/robot.yaml`; its pose
  is the `camera_mount` parameter.

A custom `TargetState` message was not added because this workspace has no ROS package
build; the JSON string carries the same fields.

## Current prompt and trial arrangement

Configured in `ridgeback.yaml`:

```text
Follow the person who is wearing black T-shirt.
```

The complete prompt is supplied unchanged to OmTrackVLA. The target manager strips a
leading `Follow` and trailing clauses such as `Maintain a safe distance.` to form the
Grounding DINO query (`the person who is wearing black T-shirt`).

The attribute phrase (`black T-shirt`) is extracted from the query by dropping the subject
word (`person`, `man`, ...) and fillers (`who is visibly holding a`, `wearing`, ...).
Grounding DINO is prompted with separate phrases, `person. black T-shirt.`.

Target selection rules:

1. An attribute box (labelled with the attribute words, score >= `0.35`) must lie at
   least 50% inside a YOLO person box and be smaller than 90% of that box, so an
   attribute on a chair or a spurious scene-sized box is rejected. Prompts without a
   visual attribute (`the person directly in front of you`) fall back to the Grounding
   DINO `person` box (IoU >= `0.50`).
2. If the attribute contains a colour word (red, orange, yellow, green, blue, purple,
   pink, brown, black, white, gray), at least 15% of the attribute box's pixels must be
   in that colour's HSV range.
3. If two people have prompt scores within `0.08`, selection is refused
   (`multiple_plausible_people`); the highest-scoring box is not picked automatically.
4. The selected track must persist in 3 consecutive target frames with matching
   appearance before `LOCKED`. Grounding runs every 8th frame while searching.
5. `LOCKED` is lost immediately (-> `UNCERTAIN`) if the track disappears or its
   appearance similarity falls below `0.72`; after 8 unverified frames it is `LOST`.
6. Recovery requires either the original track ID with similarity >= `0.72`, or a person
   who satisfies rules 1-3 in a fresh grounding pass and also has gallery similarity >=
   `0.72`; each for 3 consecutive frames. A lost target is never replaced by the next
   most similar person, and never by a prompt match whose appearance differs.
7. `LOST` does not time out. To select again (for example after the target changes
   clothes or a different person should be followed), publish
   `ros2 topic pub --once /r100_0160/omtrackvla/reset_target std_msgs/msg/Empty {}`.

For trials, still start with the leader fully visible and distinguishable from others.

## Current safety configuration

- Target command fusion blends 75% depth-localized locked-target direction and yaw
  with 25% OmTrackVLA direction and yaw when the model proposes motion generally
  toward the leader (`fusion_target_weight: 0.75`). An opposite or perpendicular
  model proposal is replaced with the target direction. A fresh planner result,
  target lock, depth and the `0.9 m` follow distance remain required; the model's
  raw command magnitude can cap translational speed.
- Dry-run is the default.
- Armable mode requires `OMTRACKVLA_ARM_OUTPUT=1`.
- Deadman messages must remain fresh within `0.5 s`.
- Camera timeout: `0.75 s`.
- Inference timeout: `1.5 s`.
- LiDAR timeout: `0.5 s`.
- E-stop timeout: `1.0 s`.
- Translation obstacle stop distance: `0.70 m` in a corridor aligned with the proposed
  x/y command and widened by the R100 footprint (`0.960 x 0.793 m`) plus `0.05 m`.
  A close side/rear return does not block travel away from it, but still blocks travel
  toward it. Rotation uses a `0.67 m` full-radius guard; yaw is suppressed while safe
  translation away remains available, and pure rotation is stopped.
- Maximum command: `0.20 m/s` forward, `0.20 m/s` lateral, `0.35 rad/s` yaw.
- Acceleration limits: `0.30 m/s^2` linear and `0.50 rad/s^2` angular.
- `require_target: true`: motion requires target state `LOCKED` from the same inference
  frame as the path (`target_not_locked` otherwise).
- `require_trajectory_consistency: true`: motion requires the RGB-only direction check to
  pass (`trajectory_target_mismatch` otherwise). It rejects only waypoint index 1 pointing
  clearly to the opposite side of an off-center (> 18% of half-width) target; it cannot
  verify metric distance or that the path ends behind the target.
- `require_target_position: true`: the locked target is located with depth in
  `base_link`; motion requires the path end to stay at least `min_follow_distance`
  (`0.9 m`) from the target (`target_position_inconsistent` otherwise). The active
  `reject_path_retreat` setting is false because the short OmTrackVLA turning horizon can
  briefly increase Euclidean target distance; same-frame target lock and the RGB
  direction check remain required. The target torso region must have >= 40% depth
  coverage and >= 60%
  of samples on one surface; a box spanning the full image height is
  `target_too_close_for_depth`. Depth must be within `0.25 s` of the color frame.
- `camera_mount: [0.0, 0.0, 1.03, 0.0, 0.0, 0.0]`: height/roll/pitch measured by
  `calibrate_camera_mount.py` on 2026-09-22 (optical centre 1.05 m above floor, level
  within 0.4 deg; `base_link` is 0.026 m above floor). Forward/lateral offset and yaw
  are not measured yet and are assumed zero.

The target gates are fail-closed but not proof of correct following: the appearance
descriptor is hand-crafted and uncalibrated, and two people in similar clothing (e.g.
two dark suits scored 0.84) may not be separable. LiDAR prevents close-obstacle motion
but does not determine target identity.

## Visualization meaning

- Green 3D metric line and arrows: raw OmTrackVLA eight-waypoint trajectory in
  `base_link`, enabled by default alongside the robot. X/Y coordinates are unscaled and
  each arrow uses the model-predicted waypoint yaw. It appears only when `planner_ran`
  is true.
- Cyan line/green arrows: locally generated target-fused follow goal. This display is
  hidden by default; its topic is still published and the fused command remains active
  internally.
- Cyan points: merged Ridgeback LiDAR, not OmTrackVLA.
- Robot model and TF axes: Clearpath ROS data, not OmTrackVLA.
- Raw RGB panel: disabled to protect LiDAR; the Target Tracker panel uses the selected
  compressed RealSense frame.
- Green sphere (Locked Target): depth-measured target torso position in `base_link`.
- Target Tracker panel: annotated inference frame without a trajectory overlay. Green
  box = `LOCKED` target, orange =
  candidate/uncertain target, grey = other tracked people, thin cyan = Grounding DINO
  boxes with label and score on grounding frames; header shows state and reason.

The 3D path is rendered `0.75 m` above `base_link`; its X/Y/yaw values are not scaled or
altered. The visualization does not change the controller command.

The detector bounding box does not enter the released OmTrackVLA neural network because
the checkpoint's exposed `bbox_feat` forward argument is unused. Coordination occurs in
the local Ridgeback bridge after inference: the box is depth-localized into a leader
position, OmTrackVLA independently proposes raw waypoints from the full RGB frame and
prompt, and `fuse_target_command` aligns/blends the raw command toward that leader before
the safety gates and `/cmd_vel`. Therefore target selection affects navigation directly,
but at the post-inference controller layer rather than inside upstream OmTrackVLA.

## Verified behavior and limitation

- The new two-GPU dry-run loads Ridgeback perception at about `2,390 MiB` RTX GPU
  memory while Thor runs planner mode. On 90 current-camera blue-basket frames, it
  matched the prior local RTX replay on all target state/reason/track-ID and
  trajectory-valid/reason decisions; 49 frames were `LOCKED`. Fast offline replay
  caused only four wall-clock-capped planner updates and is not a planner-cadence
  measurement. A 115-frame live no-person dry-run held `3.022 Hz` tracking,
  `32.237/34.995 ms` median/p95 bridge pipeline time, and
  `50.000/50.285/50.875 ms` median/p95/max 20 Hz control intervals. One first-frame
  response exceeded the camera freshness bound during model warm-up and was
  rejected. Stopping each service in separate dry-runs cleared predictions and
  produced connection errors; the deadman remained unheld, so this does not verify
  physical stopping. No live locked-target or physical-motion hybrid test has passed.
- The active dual-rate design targets 3 Hz target tracking/display and up to 2 Hz
  OmTrackVLA updates after `LOCKED`. The 20 Hz safety loop remains on the ROS/CPU side.
  One clean restarted run published Target Tracker images at about `3.08 Hz`. It used
  `5,403 MiB` GPU memory with `2,344 MiB` reported free.
- A 2026-09-28 profiled dry-run with no person in view captured 278 SEARCHING frames
  at `3.018 Hz`. Inference median/p95 was `22.666/24.730 ms`, and the 20 Hz control
  interval median/p95/max was `49.999/50.234/52.694 ms` with no interval over 55 ms.
  It did not exercise Grounding DINO model inference or OmTrackVLA planning. See the
  compute plan for stage timings and GPU sampling limits.
- A live Ridgeback-to-Thor dry-run with a blue-basket holder captured 187 frames:
  93 `LOCKED`, 7 `UNCERTAIN`, 86 `LOST`, and 1 `SEARCHING`. During a 30.241-second
  locked span, the planner updated 31 times (about 1.03 Hz, below its 2 Hz cap).
  Target, depth-position, and trajectory validity were observed together. One
  response was rejected for a stale source camera frame. The 20 Hz host control
  interval median/p95/max was 49.998/50.454/55.006 ms over 1,277 intervals.
  The physical E-stop stayed active, so this establishes dry-run behavior only.
- On 90 fresh current-camera frames showing the basket holder and departure, Thor
  and RTX matched every target state, track ID, target reason, and trajectory
  validity/reason with the active prompt. Both first locked ID 1 on frame 3 and had
  49 locked frames. With a planner update forced on each locked frame, the largest
  absolute command-component difference was 0.007871. This offline replay excludes
  ROS depth, LiDAR, motor safety, and network timing.
- In that run, one planner-update frame took `0.418 s`; one cached-planner response
  took `0.032 s` inference and `0.048 s` total pipeline time. These are individual
  observations, not median/p95 live latency or proof of sustained planner cadence.
- Two 12-frame offline comparisons of the shared CUDA vision preprocessing reduced
  locked-frame median latency from `303.8-319.1 ms` to `193.2-197.6 ms`, with unchanged
  target state, track ID, trajectory-valid decisions, and permitted-frame counts.
  Live before/after full-pipeline timing is still needed. See
  `docs/COMPUTE_PLACEMENT_PLAN.md` for measurements and next steps.
- The current prompt selected the person holding a blue basket and initially stayed
  `LOCKED` on ID 1 for 15/15 observed samples, with appearance similarity
  `0.968-0.993`. A later 20-status window had 19 `LOST` states. The same visible
  person was subsequently grounded and locked as ID 34, with valid depth at `1.177 m`.
  Detector fragments, occlusion, and appearance changes have not been separated by a
  saved replay. This is a target-stability limitation, not verified reliable following.
- In a later local RTX armable run, fresh camera and Target Tracker frames both showed
  a striped-shirt person holding the blue basket. YOLO tracked the person as ID 28
  (0.90 in one annotated frame), and grounding marked the blue basket, but the state
  stayed `LOST` with `prompt_match_appearance_differs` or
  `waiting_for_original_identity`. The stored gallery belonged to an earlier lock;
  the prompted candidate did not meet the 0.72 appearance-recovery threshold. Across
  24 status samples in 12 seconds, all commands stayed zero for `target_not_locked`.
  This is an identity-recovery refusal, not absence of camera frames or person boxes.
- Earlier live trials selected a red-shirt person over a white-shirt person, and
  grounded a maroon shirt while rejecting an incorrect brown colour description.
  These scene-specific observations do not establish general target-selection accuracy.
- RealSense compressed colour and depth, both Hokuyo scanners, merged LiDAR, E-stop,
  namespaced TF, inference, target visualization, and RViz connectivity have been
  verified live. The scanner overlay and launcher preflight recover the documented
  stale-driver condition; fresh LiDAR is still required before arming.
- Raw OmTrackVLA waypoints have been visualized on a live `LOCKED` frame. A supervised
  interval also observed brief `0.03-0.04 m/s` admitted commands, but most samples
  stopped for `scan_stale` and target state sometimes became `UNCERTAIN` or `LOST`.
  There is no verified sustained physical following run.
- The bridge is dry-run by default and stops on stale deadman, camera, LiDAR, inference,
  or E-stop data, an active E-stop, missing target lock, invalid target geometry,
  trajectory mismatch, obstacle, invalid values, or disconnection. The complete local
  unit suite contains 66 tests; it does not replace live safety validation.
- Same-frame prompt comparison produced similar forward commands for a black jacket,
  a white lab coat, a black chair, and an instruction to move away. The released
  OmTrackVLA checkpoint does not identify the prompted person or report target presence.
  The local target manager and controller fusion provide that separate connection,
  with hand-built appearance features that can confuse similar clothing.

## Recommended launch

Stop any previous controller, then use one interactive terminal:

```bash
cd /home/robot/Desktop/omtrackvla
./real_robot/start_ridgeback_all.sh
```

Inspect the Target Tracker panel and raw path in RViz. Type `ENABLE` only during a supervised trial with the
physical E-stop reachable. Press Ctrl+C to stop the foreground deadman and shut down
the launcher processes.

## Change log

### 2026-09-29 (technical review includes four local RTX trials)

- Updated the dated technical review PDF to distinguish the four supervised local
  RTX fusion trials from the two-GPU hybrid dry-runs. The review now records each
  weight, target-lock and gate counts, odometry endpoint change, and the unresolved
  Run D command-authority gap. It keeps the hybrid's live locked-target motion
  unverified and cites the detailed trial report. This is a documentation update;
  the upstream waypoint model and local controller behavior did not change.

### 2026-09-29 (restore 75% target / 25% OmTrackVLA setting)

- Set `fusion_target_weight` from `1.0` back to `0.75` for the next restarted
  controller run. Aligned OmTrackVLA direction and yaw again contribute 25% to
  the fused command; target geometry contributes 75%. The prompt, speed
  calculation, following distance and safety gates are unchanged. The prior
  four-run report remains historical evidence, not a verification of this
  newly restarted setting.

### 2026-09-29 (live target 100% direction trial observation)

- A detailed replay of the read-only capture found an unresolved command-authority
  discrepancy: during a 9.45-second gap after the OmTrackVLA controller's zero
  `/cmd_vel` burst for `obstacle_too_close`, wheel odometry changed by 0.63 m and
  reported up to 0.291 m/s. `twist_mux` has higher-priority joystick, RC, and
  interactive-marker inputs, but their historical messages, `platform/cmd_vel`,
  and motor feedback were not captured. The operator later recalled the joystick
  may have been used for much of the last run, without confirmed timing. Joystick
  override is plausible but unverified. The gate's zero command cannot be
  presented as a verified platform stop in this interval. Capture all command
  sources and mux output before another armable weight comparison.
- The running node confirmed `fusion_target_weight=1.0` and the black T-shirt
  prompt. In a 39.5-second read-only observation, all 80 status samples stayed
  `LOCKED` on track ID 20. Wheel odometry changed by 4.26 m between endpoints.
  The gate reported `ready` 24 times, `ready_rotation_suppressed` 30 times,
  and `obstacle_too_close` 26 times; all obstacle stops reported zero *controller*
  command.
  A sampled Target Tracker image showed the intended black-shirted person in
  the green box. Depth-estimated target distance ranged 1.31–2.53 m, without
  independent ground truth. No `scan_stale` status appeared. The model still
  affected the speed cap and its fresh planner result was required, but its
  route did not steer the robot. This is not a controlled comparison
  with prior weights; see `docs/RUN_REPORT_2026-09-29_OMTRACK75.md` for four runs.

### 2026-09-29 (target 100% command-direction trial setting)

- Changed `fusion_target_weight` from `0.0` to `1.0`, so the fused translation
  direction and yaw come entirely from the depth-localized locked leader. The
  OmTrackVLA proposal still enters the speed cap and a fresh planner result is
  required. The raw model path's target-consistency result is recorded as a
  diagnostic; with fusion enabled it does not gate the target-led command.
  The deadman, E-stop, LiDAR,
  freshness, speed limits and 0.9 m follow distance remain active. This setting
  has not yet been observed in a restarted live run.

### 2026-09-29 (live OmTrackVLA 100% trial observation)

- The running node confirmed `fusion_target_weight=0.0` and the black T-shirt
  prompt. In 80 status samples over 39.5 seconds, 68 were `LOCKED`, 7
  `UNCERTAIN`, and 5 `LOST`; 12 `target_not_locked` samples reported zero
  command. Wheel odometry changed by 2.17 m between endpoints. One sampled
  Target Tracker image showed the black-shirted person locked, and no
  `scan_stale` status appeared. In 41 locked samples the depth-estimated target
  distance was at or below 0.9 m and translation was zero; the minimum estimate
  was 0.506 m without independent distance ground truth. The 100% model
  direction applied in 26 aligned, outside-distance status samples. This is
  observed motion, not a controlled weight or following-accuracy comparison.
  See `docs/RUN_REPORT_2026-09-29_OMTRACK75.md` for the four-run comparison.

### 2026-09-29 (OmTrackVLA 100% aligned-command trial setting)

- Changed `fusion_target_weight` from `0.25` to `0.0`. When the model proposes
  motion with a positive component toward the locked leader, fused direction and
  yaw now follow OmTrackVLA entirely. The controller still requires target lock
  and valid depth, replaces opposite/perpendicular motion toward the leader, and
  stops translation within `0.9 m`. Deadman, E-stop, LiDAR, speed and freshness
  gates remain active. This setting has not yet been observed in a live run.

### 2026-09-29 (fusion trial report comparison)

- Expanded `docs/RUN_REPORT_2026-09-29_OMTRACK75.md` into a side-by-side report
  of the earlier 25% OmTrackVLA / 75% target blue-basket run and the later 75%
  OmTrackVLA / 25% target black-shirt run. Their prompts and scenes differed, so
  the report does not attribute the motion difference to the fusion weight.

### 2026-09-29 (live OmTrackVLA 75% trial observation)

- The running node confirmed `fusion_target_weight=0.25` and the black T-shirt
  prompt. In a 39.5-second supervised observation, 56/80 status samples were
  `LOST`, 20/80 `UNCERTAIN`, and only 4/80 `LOCKED`. The motion gate was
  `target_not_locked` in 76/80 samples. Wheel odometry changed by 0.109 m,
  primarily during short lock intervals. One annotated frame showed a black-shirt
  grounding box but `prompt_match_appearance_differs`; a sampled appearance score
  of 0.705 fell below the 0.72 recovery threshold. Track ID changed from 37 to 55.
  This trial does not establish a navigation improvement from the new weight because
  identity was unstable and the previous run used another prompt and scene. See
  `docs/RUN_REPORT_2026-09-29_OMTRACK75.md`.

### 2026-09-29 (OmTrackVLA-led fusion trial setting)

- Changed `fusion_target_weight` from `0.75` to `0.25`, giving aligned
  OmTrackVLA direction and yaw 75% of the blend and target geometry 25%.
  This is a configuration change for the next restarted controller run, not a
  verified improvement in following. The fail-closed target, LiDAR, E-stop and
  deadman gates, speed calculation and `0.9 m` follow distance remain in place.

### 2026-09-29 (active prompt changed to black T-shirt)

- `ridgeback.yaml` now selects `Follow the person who is wearing black T-shirt.`
  for the next launch; the blue-basket prompt remains a commented option. The
  supervised motion observation below used the earlier blue-basket prompt and does
  not validate target selection with the new prompt. Model weights, fusion, and
  safety configuration are unchanged.

### 2026-09-29 (supervised local motion observation)

- After the ROS argument fix, an operator ran the local RTX supervised launcher. A
  39.5-second read-only observation saw the controller, deadman, inference server,
  RViz and supervised scanners active. Wheel odometry showed 1.26 m endpoint motion.
  The prompted blue-basket holder was visibly boxed in one sampled Target Tracker
  image; 78/80 status samples were `LOCKED` on track ID 10.
- Two status samples became `UNCERTAIN` and stopped for `target_not_locked`; 21
  consecutive samples then stopped for `scan_stale` for about 10 seconds, despite
  target lock and connected inference. The merged-scan dropout cause is undetermined.
  No motion or identity accuracy percentage can be inferred from this short window.
  See `docs/RUN_REPORT_2026-09-29.md` for the timed evidence and limits.

### 2026-09-29 (local supervised launcher ROS argument fix)

- A supervised local launch loaded the full model and reported `ARMABLE`, but the
  Ridgeback ROS node exited before startup because ROS 2 rejected the empty
  `-p perception_host:=` override. `start_ridgeback.sh` now omits that override in
  local mode and uses the YAML default empty value. Hybrid mode still passes its
  nonempty perception host. No safety gate, motion limit, or arming condition changed.
  This fixes argument parsing; a subsequent full supervised launch is still needed
  to verify scanner recovery, RViz, and motion readiness.

### 2026-09-29 (deployment document diagrams)

- Updated the deployment guide to show all three implemented execution modes and
  the actual two-GPU hybrid request sequence. The diagrams distinguish Ridgeback
  RTX detection from Ridgeback CPU tracking, Thor GPU planning, and the independent
  Ridgeback CPU motion gate. The full-Thor diagram remains labelled as its separate
  deployment option. No runtime behavior or safety setting changed with this
  documentation update.
- Created the September 29 technical review PDF from a versioned ReportLab builder.
  It records the hybrid placement, sequential frame flow, identity and fusion rules,
  measured results, and dry-run limits. The September 28 editable presentation
  remains a historical snapshot.

### 2026-09-29 (two-GPU dry-run implementation)

- Added project-specific `perception` and `planner` modes. Ridgeback RTX now can
  run YOLO and Grounding DINO while Thor runs DINOv3, SigLIP and OmTrackVLA; the
  existing full-inference launch is retained. The bridge combines same-frame
  replies and keeps depth, fusion, safety and `/cmd_vel` on Ridgeback.
- Added dry-run hybrid launchers, target/result checks and failure reset. The
  local RTX process used about `2,390 MiB` during replay. All 90 blue-basket replay
  frames matched prior identity and trajectory validity decisions; the 115-frame
  live no-person run preserved 3 Hz tracking and the 20 Hz control timer. Stopping
  each service in a separate dry-run cleared predictions; a five-frame full-mode
  regression still locked target ID 1 on frame 3. Locked live cadence, disconnect
  recovery, and armable behavior remain unverified.
- On invalid-target frames the hybrid bridge now sends only state metadata to Thor.
  A 90-frame replay still matched all target and trajectory decisions and omitted
  Thor JPEG transfer on 41 frames. A later 46-frame live no-person dry-run held
  `3.054 Hz` target processing and a `51.455 ms` maximum control interval; its
  `25.564/27.673 ms` bridge pipeline median/p95 is from a separate recording and
  does not establish a speedup over the earlier full-JPEG run.

### 2026-09-28 (live host-placement audit)

- Confirmed that the active supervised run still sends Ridgeback camera frames to
  the **local** inference server over `127.0.0.1:18765`; the RTX 5060 held that
  process with `5,368 MiB` of GPU memory. Thor had a separate listening inference
  server but no connected client, so it was not doing the active run's model work.
  The sampled local status had E-stop active, target `LOST`, and zero command.
- Clarified the implemented alternative: with `start_ridgeback_thor.sh`, target
  perception and OmTrackVLA neural inference move to Thor, while the Ridgeback
  computer retains ROS sensors, depth, fusion, the 20 Hz safety gate, and `/cmd_vel`.
  A future all-on-Thor system remains a separate proposal.
- Rechecked both hosts after Thor printed `ready`: Thor had all six required model
  weight files (6.6 GB), a CUDA-capable PyTorch 2.10.0 environment, and a listener
  reachable by the Ridgeback. Both checkouts were at `3869f0c`. The active bridge
  remained connected to local loopback; Thor's listener had no active client. The
  Thor split uses CUDA on Thor and does not replace CUDA itself.

### 2026-09-28 (live blue-basket recovery diagnosis)

- Inspected only fresh `/r100_0160/camera/color/image_raw/compressed` and
  `/r100_0160/omtrackvla/target_image` frames during a local RTX armable run.
  The basket holder and basket were both annotated, but a new track ID failed the
  stored-gallery appearance check and `LOST` blocked output. The operator should stop
  the supervised launcher and begin a fresh target search with the intended person
  visible, then verify `LOCKED` in RViz before enabling the deadman; do not reset a
  target while an armable deadman is held. No threshold or motion gate was changed.

### 2026-09-28 (project-specific Thor inference split)

- Captured a live person-locked Thor split dry-run with the physical E-stop active:
  93 locked frames and 31 planner updates in a 30.241-second locked span. A stale
  camera response was rejected and the host safety timer continued at 20 Hz.
  Replayed 90 fresh basket-holder camera frames on Thor and RTX; all target and
  trajectory decisions matched. Sustained 2 Hz planning, reconnect recovery, and
  armable safety equivalence remain open; no end-to-end speedup is claimed.
- Updated the tracked technical review deck's process-split and verified-results
  slides with the Thor location, current camera evidence, 62-test count, and the
  dry-run limits; the earlier unrelated recording was excluded.
- Identified the attached `nvidia-thor-r100-0160.local` at `192.168.131.51` by mDNS
  and confirmed Arm64/JetPack 7.0 by SSH. The Ridgeback host routes to it from
  `192.168.131.1`.
- Copied this project's code and model weights to `/home/robot/dev/omtrackvla` on
  Thor; built a separate Python 3.12 environment with CUDA PyTorch 2.10.0+cu130.
  Model startup and an empty-frame inference request succeeded on Thor. The x86
  runtime and all other Thor projects were not changed.
- Added a single-client remote bind, separate Thor inference and host bridge launch
  paths, and stale-camera-response rejection. ROS, depth fusion, E-stop, LiDAR,
  deadman, and the 20 Hz motor safety gate remain on the Ridgeback computer. Remote
  arming requires a second explicit `OMTRACKVLA_ALLOW_REMOTE_ARM=1` opt-in after
  the replay, timing, and physical safety gates pass.
- A 39-second Thor split dry-run produced 117 SEARCHING frames at 3.007 Hz and
  777 control intervals (50.932 ms maximum); motor output stayed disabled. This
  first run did not exercise the person-locked planner.
- A second dry-run published the deadman at 10 Hz, then stopped the Thor server.
  The host bridge recorded 269 `dry_run:inference_disconnected` ticks with a
  52.744 ms maximum control interval. This checks the disconnect gate without
  commanding physical motion. The complete local test suite has 62 passing tests.
- A fixed 12-frame offline replay of one cropped real-person image reached target
  lock and the planner on Thor. Thor and RTX both locked track ID 1 on frame 3,
  produced 10 valid trajectories, and matched all states and decisions; maximum
  absolute command-component difference was 0.003344. This narrow replay does not
  establish live planner rate, moving-person identity, depth fusion, or motor safety.

### 2026-09-28 (opt-in dry-run pipeline instrumentation)

- Added `OMTRACKVLA_PROFILE_DIR` logging for inference and ROS bridge stages, including
  control tick intervals and stop reasons. GPU stage timing synchronizes CUDA, so
  profiled latency is diagnostic and must be compared with a similarly profiled run.
  Profiling is refused when `dry_run` is false.
- Added `summarize_perf.py` and two focused profiling tests. The complete local unit
  suite now has 61 passing tests. A no-person SEARCHING dry-run captured 278 frames
  and 1,856 control intervals with no interval over 55 ms. Locked-person and
  occlusion/recovery runs, CPU sampling, and Thor comparison remain pending; no motion
  threshold or runtime placement changed.

### 2026-09-28 (repository handoff and presentation alignment)

- Reconciled the current prompt, live observations, and test count in this handoff
  record with the active configuration and September 28 trials. Older trials remain
  in the dated entries below rather than being described as current behavior.
- Updated the technical review presentation to distinguish upstream waypoint output,
  local target identity and fusion, offline CUDA comparisons, and the limited live
  observations. The deck no longer presents unmeasured velocity placeholders or the
  earlier 48-test count as current evidence.

### 2026-09-28 (maintained CPU/GPU placement plan)

- Added `docs/COMPUTE_PLACEMENT_PLAN.md` as the maintained audit and implementation
  plan for CPU/GPU placement, pipeline scheduling, rate changes, and performance work.
  Workspace instructions now require it to be updated with each related discussion,
  measurement, or implementation change.
- Audited the current code path. OmTrackVLA, DINOv3, SigLIP, Grounding DINO, and YOLO
  inference run on CUDA; the coarse feature history remains in VRAM. ROS, JPEG decode
  and diagnostic encode, BoT-SORT association, OpenCV appearance features, depth and
  LiDAR geometry, fusion, and safety run on the CPU.
- Verified the installed hardware as an RTX 5060 with `8,151 MiB` VRAM and an
  eight-core i7-9700TE with 31 GiB system RAM. The inference stack was idle during the
  snapshot (`27 MiB` VRAM used), so the earlier approximately `5.4 GB` full-pipeline
  measurement remains the relevant loaded-memory observation.
- Re-ran all 57 tests successfully and repeated the CUDA colour benchmark. OpenCV CPU
  remained 2.16x to 4.51x faster than the CUDA implementation for 1 to 10 boxes, so no
  runtime placement was changed. The next planned step is opt-in full-pipeline stage
  instrumentation before further optimization.

### 2026-09-28 (shared CUDA vision preprocessing)

- Replaced the live planner's duplicate PIL/CPU DINOv3 and SigLIP preprocessing with
  one RGB tensor upload, one antialiased bicubic CUDA resize, and separate CUDA
  normalization for each vision tower. Neural encoder, token pooling, history, Qwen,
  and waypoint execution remain on CUDA. The former path is available through
  `--legacy-vision-preprocess` for rollback and comparison, and CPU inference devices
  automatically use the legacy processor.
- Added `test_vision_gpu_preprocess.py` and `bench_vision_gpu_preprocess.py`. The suite
  is now 59 tests and passes. The GPU resize is bounded against the PIL reference on a
  high-frequency synthetic image; the benchmark also compares vision tokens and final
  OmTrackVLA trajectories.
- Isolated preprocessing improved from `14.531/35.128 ms` median/p95 to
  `0.406/0.535 ms`; complete DINOv3 plus SigLIP encoding improved from
  `127.863/139.798 ms` to `114.493/114.897 ms`.
- Eight-frame comparison found minimum DINO/SigLIP token cosine similarities of
  `0.999914/0.995827` and maximum waypoint-component difference `0.006985`.
- Two reverse-order 12-frame offline pipeline comparisons forced a planner update on
  every locked frame. CUDA preprocessing reduced locked-frame median latency from
  `303.8-319.1 ms` to `193.2-197.6 ms` and p95 from `347.9-360.6 ms` to
  `202.5-220.5 ms`. Target state, track ID, trajectory-valid decisions, and permitted
  frame counts were identical; maximum command-component difference was `0.006375`.
- `evaluate_offline.py` now passes the current inference constructor arguments and
  exposes planner-rate and legacy-preprocessing options for repeatable comparison.
- This change requires an inference-server restart. It has offline evidence only and
  is not yet live-verified on the robot. ROS, depth/LiDAR geometry, fusion, safety,
  motion limits, and physical command behavior were not moved to CUDA or relaxed.

### 2026-09-28 (stale supervised Hokuyo process preflight)

- The first restart with shared CUDA preprocessing loaded all models and reported the
  inference server ready, but front-scanner recovery timed out before RViz or motor
  enable. The cause was two orphaned `urg_node_overlay` drivers from an interrupted
  earlier run. They still owned both scanner TCP sessions while the restored standard
  service attempted to reconnect. This was unrelated to model loading or CUDA.
- Terminated the two stale overlay drivers. The standard front and rear drivers then
  connected to serials `L2215985` and `L2215963`, and both raw scan topics were measured
  at approximately 40 Hz.
- `start_ridgeback_all.sh` now detects the exact custom overlay executable before it
  stops the standard sensor service, sends its process groups `SIGTERM`, waits up to
  five seconds, and refuses to continue if any stale driver remains. It does not match
  or terminate the standard `/opt/ros/jazzy` drivers. Motion and safety thresholds are
  unchanged. CUDA runtime cadence is still awaiting a clean restarted observation.

### 2026-09-28 (live target-stability inspection after CUDA preprocessing)

- A clean restarted run loaded the models with `5,403 MiB` VRAM used and `2,344 MiB`
  reported free. The Target Tracker image published at approximately `3.08 Hz`. One
  planner-update frame measured `0.418 s`; one cached-planner response measured
  `0.032 s` inference and `0.048 s` total pipeline time. These are individual live
  observations, not a complete latency distribution.
- The prompt was `Follow the person who is holding blue basket.` The current images
  showed the intended basket holder in a green box and another real seated person in a
  grey box. The target was initially LOCKED on ID 1 for 15/15 samples, with person
  confidence `0.686-0.897` and appearance similarity `0.968-0.993`.
- Tracking was not consistently persistent. A following 20-status sample contained
  19 LOST states and one LOCKED state. The intended visible person was subsequently
  grounded and LOCKED as ID 34 with grounding score `0.663`, person confidence `0.879`,
  appearance similarity `0.994`, valid depth at `1.177 m`, and a valid fused trajectory.
  Low-confidence small or partial YOLO person boxes appeared as extra grey track IDs.
- This instability is in the existing target-perception path. YOLO, Grounding DINO,
  BoT-SORT, and the HSV gallery consume the original frame before the newly added
  DINOv3/SigLIP CUDA planner preprocessing is invoked. No target threshold was changed
  from this observation. A recorded replay is needed to determine whether the loss was
  caused by detector fragmentation, pose/occlusion, or HSV appearance change.

### 2026-09-24 (documentation organization)

- Moved the living project context and Ridgeback deployment guide into the top-level
  `docs/` directory and added a documentation index. Compatibility links remain at
  `real_robot/PROJECT_CONTEXT.md` and `real_robot/README.md` so existing maintenance
  instructions and operational references continue to resolve.

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

### 2026-09-22

- Removed the experimental MVLM dependency, adapter, weights, target-lock image,
  target-confirmation interface, and target-dependent motion gates.
- Restored direct full-frame camera plus prompt inference through OmTrackVLA only.
- Retained independent deadman, E-stop, LiDAR, freshness, speed, and acceleration
  safety gates.
- Kept the configured blue-basket prompt while documenting that upstream OmTrackVLA
  provides no target identity or target-presence signal.

### 2026-09-22 (target perception)

- Added `target_perception.py`: Grounding DINO prompt grounding validated against YOLO
  person boxes, BoT-SORT tracking of all people, part-based HSV appearance gallery, and
  a fail-closed `SEARCHING/LOCKED/UNCERTAIN/LOST` target manager.
- Replaced the initially planned `yolo11n-cls` embeddings after measuring that they do
  not separate different people; fixed BoT-SORT input to CPU numpy detections.
- Confirmation and recovery now require consecutive frames; a new track ID can only
  recover the target with appearance match plus independent prompt agreement.
- Added `require_target` and `require_trajectory_consistency` gates, the
  `omtrackvla/target_state` and `omtrackvla/target_image` topics, the RViz Target
  Tracker panel, launcher weight checks, and updated operator prompts.
- Added offline evaluation (`evaluate_offline.py`) and bag export
  (`export_bag_frames.py`) tools and target state-machine tests.

### 2026-09-22 (depth target check)

- Added `target_geometry.py` and the `require_target_position` gate: the locked target
  is located with RealSense depth in `base_link` and the path is checked against a
  minimum following distance and retreat tolerance.
- Switched depth input to `compressedDepth` after raw depth stalled the color stream.
- Added coverage, surface-consistency, and full-height-box rejection after a live
  too-close person was mislocated at 3.59 m.
- Added `calibrate_camera_mount.py`, the `camera_mount` parameter (z/roll/pitch
  measured), the `omtrackvla/target_position` topic, and the RViz Locked Target display.

### 2026-09-22 (prompt attribute fix)

- Diagnosed the live report that a person holding a blue basket was not detected: the
  running controller was `LOST` with an earlier grounding score of `0.731`. The
  full-sentence caption let Grounding DINO's word "person" match any human (0.50 and 0.39
  on two basket-less people in a test image), so whoever appeared first was locked, and
  strict appearance-only-gated recovery then refused the real target.
- Grounding now uses separate `person. <attribute>.` phrases and requires the attribute
  box on the person, smaller than the person, and in the named colour. Verified on test
  images: no lock for "blue basket" when none is present; correct lock for "black
  jacket"; a black tie that Grounding DINO labelled "red tie" (0.66) was rejected.
- Recovery accepts a fresh full-prompt match with appearance >= 0.72; added the
  `reset_target` topic, grounding debug output/overlay, and grounding every 2nd frame.
- Not yet verified live with the blue basket.

### 2026-09-23 (compute reduction)

- Reduced the live inference rate from 2.0 Hz to 1.5 Hz and the YOLO person detector
  input to 416 pixels.
- Reduced Grounding DINO execution to every fourth inference frame while searching or
  recovering; it remains disabled during stable `LOCKED` tracking.
- OmTrackVLA now skips visual-token and waypoint computation until the target manager
  reaches `LOCKED`, and its history is reset on target loss/reacquisition.
- The one-terminal launcher remains the complete runtime entry point; model components
  do not print per-frame detections.
- Live no-target dry-run verification reduced inference from approximately `0.213 s`
  to `0.023 s`; GPU utilization was mostly `0-1%` between brief detector passes instead
  of repeatedly spiking near `84%`. Resident model memory remains about `5.36 GB` so a
  target can be acquired without reloading checkpoints.

### 2026-09-23 (live detection and display validation)

- Verified the color camera at approximately `10.75 Hz`. Before this change, inference
  output measured approximately `1.35 Hz` because the 1.5 Hz limiter rounded each sample
  upward to a camera-frame boundary. The limiter now advances an absolute deadline so
  its long-term rate remains 1.5 Hz.
- Verified a two-person live scene: the red-shirt prompt selected the red-shirt person
  and rejected a white-shirt person. Observed grounding confidence was `0.695-0.808`,
  YOLO person confidence `0.920-0.956`, and appearance similarity `0.776-0.998` during
  loss/recovery testing. This is one trial, not a general accuracy guarantee.
- Annotated images are now handed to ROS on a 20 Hz publication timer as soon as an
  inference finishes, rather than waiting up to 0.5 seconds for status publication.
- RViz was consuming approximately 64% CPU while configured for 30 FPS. Its supplied
  configuration now renders at 15 FPS, updates TF/RobotModel at 10 Hz, and keeps only
  the newest camera/target image. This does not reduce the raw camera topic rate or alter
  target selection, OmTrackVLA waypoints, or any motion-safety gate.

### 2026-09-23 (LiDAR startup contention)

- Both UST-20LX units were verified directly with SCIP `VV`, then streamed at about
  `40 Hz` each with a `39.9 Hz` merged scan. When the OmTrackVLA launcher began at
  `10:21:12`, both drivers logged `Could not grab single echo scan` starting at
  `10:21:13`; the configured `error_limit: 4` triggered reconnects that did not recover.
- Temporarily increasing both live Hokuyo `error_limit` parameters to `100` was tested
  but did not prevent the disconnect. `start_ridgeback_all.sh` now obtains one `sudo`
  authentication, pauses only `clearpath-sensors.service` while models load, restarts
  it afterward, and refuses to open RViz or show the enable prompt unless a fresh merged
  scan arrives within 30 seconds. Cleanup restores the service if startup is interrupted.
  The controller's scan freshness remains `0.5 s`; no motion-safety threshold is relaxed.
- The inference process now starts at niceness `5` with OpenMP/MKL limited to four CPU
  threads. RViz starts only after inference and the scan preflight are ready, avoiding
  simultaneous model-loading and rendering load on the sensor drivers.

### 2026-09-24 (Hokuyo reconnect recovery)

- Pausing both drivers for model loading was verified, but merely restarting
  `clearpath-sensors.service` still left each `urg_node` with an established TCP socket
  containing unread data and a second socket stuck reconnecting; no raw or merged scan
  was published within the 30-second preflight.
- Isolation testing disproved model loading as the direct runtime cause: both scanners
  stayed at approximately 40 Hz while the complete inference server loaded and idled.
  Starting only a second subscriber to uncompressed
  `/r100_0160/camera/color/image_raw` then stopped both scan streams immediately, even
  with the inference socket deliberately unavailable. Subscribing to
  `camera/color/image_raw/compressed` at 30 Hz instead preserved front, rear, and merged
  scan rates at approximately 40 Hz while the real inference server and ROS bridge ran.
- `hokuyo_scip_reset.py` now sends `QT`, verifies `VV`, checks the fixed UST-20LX serials
  (`L2215985` at `192.168.131.21`, `L2215963` at `192.168.131.22`), and closes each SCIP
  session cleanly. This replaces the timeout-killed `nc` probe.
- The installed official `urg_node` has two recovery defects exposed by the failure: a
  missed scan is followed by a status query that desynchronizes the streaming protocol,
  and a failed Ethernet constructor leaks the scanner's single TCP session. A pinned
  build of official commit `936abce8cd1282eb866c20676e0f34fd031386b9` with only those
  two recovery fixes is installed locally in `real_robot/urg_node_overlay`; the
  committed reproducible sources are `urg_node_recovery.patch` and
  `build_urg_node_overlay.sh`.
- The one-terminal launcher keeps `clearpath-sensors.service` stopped during its run,
  starts the patched front and rear drivers sequentially with `error_limit=20`, and
  restores the standard service on exit. OmTrackVLA is constrained to CPUs 4-7. Failure
  remains fail-closed: RViz, deadman, and motor enable are not started without fresh
  front, rear, and merged scans. No obstacle, freshness, target, E-stop, speed, or
  acceleration safety threshold was relaxed.

### 2026-09-24 (follow-gate diagnosis and rollback)

- Live inspection of the previously working detector showed intermittent target locks,
  but the controller alternated between `obstacle_too_close` and
  `target_position_inconsistent`, so no nonzero wheel command was admitted. The merged
  scan minimum was approximately `0.55 m` at `-85 degrees`, while the front sector was
  approximately `2.41 m` clear.
- A directional LiDAR experiment and relaxed path-retreat check were implemented but
  explicitly rolled back at the operator's request after the next trial did not acquire
  a target. The detection/tracking implementation itself was not changed by that
  experiment. The active behavior is again the earlier global `0.70 m` obstacle gate,
  original depth/path consistency gate, and PointStamped green target display.
- The latest trial log contained only `target_not_locked`; the controller was stopped
  before further live diagnosis. Wheel-motion diagnosis must therefore be repeated with
  the restored version while the Target Tracker visibly reports `TARGET LOCKED`.

### 2026-09-24 (combined detector and directional motion gate)

- At the operator's request, retained the earlier prompt-grounded target detector and
  tracker unchanged, together with the proven compressed RGB transport, patched Hokuyo
  drivers, and one-terminal LiDAR recovery launcher.
- Re-enabled command-direction LiDAR clearance after confirming the room geometry:
  approximately `2.41 m` ahead and a `0.55 m` return at about `-85 degrees`. The numeric
  translation threshold remains `0.70 m`; it was not reduced below the chassis safety
  envelope. Movement toward the close return remains blocked, and rotation retains a
  conservative full-radius guard.
- Disabled only the uncalibrated path-retreat veto. Depth localization and the `0.9 m`
  minimum leader distance remain required, as do target `LOCKED`, RGB path-direction
  consistency, fresh sensing/deadman, inactive E-stop, and speed/acceleration limits.
- The released OmTrackVLA model still receives the full camera frame and text prompt.
  The local detected bounding box gates motion and checks path direction, but cannot be
  injected into the released model because its bbox forward argument is unused. This is
  a limitation of the upstream checkpoint interface, not a detector regression.
- All 43 safety, geometry, and target-perception tests pass. Live wheel motion has not
  yet been verified after this combined change.

### 2026-09-24 (RViz raw-camera LiDAR regression)

- A combined live run used the patched Hokuyo drivers but both immediately entered
  repeated `Could not grab single echo scan` reconnect loops. Topic endpoint inspection
  found two subscribers to uncompressed `camera/color/image_raw`: the always-running
  `ridgeback_camera_mjpeg_server` and the enabled RViz RGB Camera display.
- Stopping only RViz reduced raw subscribers from two to one; without restarting either
  scanner, front, rear, and merged LiDAR all recovered to approximately `40 Hz`. This
  confirms the duplicate raw RGB subscription as the trigger in this run, not the
  detector, directional safety logic, or patched-driver selection.
- The supplied RViz raw RGB display is now disabled and clearly labelled. The annotated
  Target Tracker display remains enabled and is generated from the controller's
  compressed RGB subscription. The launcher checks for more than one raw RGB subscriber
  after RViz starts and rechecks fresh front, rear, and merged scans before presenting
  the `ENABLE` prompt. A regression therefore fails closed before deadman activation.
- Verified after the change with RViz running: raw RGB remained at one subscriber and
  the merged scan remained approximately `39.7-39.9 Hz`; both individual scanner topics
  were also approximately `40 Hz` immediately beforehand.

### 2026-09-24 (target-conditioned command fusion and RViz provenance)

- Added a post-inference fusion layer in the local Ridgeback bridge. Grounding/tracking
  selects a leader bounding box, depth localization converts it to a `base_link` target,
  and the bridge aligns/blends OmTrackVLA's raw waypoint command toward that position.
  Safety gates still determine whether the resulting command can reach `/cmd_vel`.
- This is coordinated with OmTrackVLA but is not a bounding-box input to the upstream
  neural network: the released model leaves `bbox_feat` unused. Raw model output remains
  separately available on `omtrackvla/raw_cmd_vel` and `omtrackvla/predicted_path`; the
  custom result is available on `omtrackvla/proposed_cmd_vel` and
  `omtrackvla/fused_path`.
- RViz now enables only the raw magenta OmTrackVLA path with yellow arrows by default.
  The custom cyan path with green arrows remains publishable and can be enabled for
  diagnosis, but is hidden by default. This display-only change does not disable fusion.

### 2026-09-24 (camera latency and compute pass)

- Measured the live RealSense compressed RGB topic at `29.97 Hz` with approximately
  `0.09-0.10 MB` JPEG frames and low timing jitter. There was one uncompressed subscriber
  (the web MJPEG service), no compressed subscriber, and no `target_image` publisher,
  proving the OmTrackVLA controller was stopped during the report; no live inference
  latency could be sampled in that state.
- Removed an avoidable ROS-bridge conversion: compressed JPEG frames are now rate-gated
  before any decode and the selected original JPEG is sent directly to the inference
  server. The previous path decoded all 30 incoming FPS, discarded most frames, and
  re-encoded selected frames. Latest-only replacement and safety freshness behavior are
  unchanged.
- Increased inference/annotated-display rate from `1.5 Hz` to `2.0 Hz`. To avoid raising
  average Grounding DINO search load, its interval changed from every fourth frame to
  every fifth frame (both configurations are approximately `0.4` grounding runs/sec).
  YOLO remains at 416 pixels and OmTrackVLA remains skipped until target lock.
- The optional `ridgeback-camera-mjpeg.service` measured approximately `38%` CPU while
  idle from OmTrackVLA. The supervised launcher now pauses it during a trial and restores
  it on every cleanup path; `OMTRACKVLA_KEEP_MJPEG=1` retains it when the separate web
  camera page is needed. The RViz Target Tracker uses compressed RGB and is unaffected.
- Static checks and all 48 safety/target/geometry tests pass. The revised 2 Hz runtime
  still requires a fresh supervised live measurement after restart; it is not claimed
  verified from the stopped-controller observation.
- Independent health checks during this pass showed the merged LiDAR stable at
  approximately `39.9-40.0 Hz`, E-stop state publishing inactive (`false`), 25 GiB RAM
  available, 98 GiB disk free, zero swap use, and CPU temperatures around `44-47 C`.
  A stale read-only `tf2_echo` diagnostic left by an earlier investigation was also
  found running for over an hour and terminated; it was not a controller process.

### 2026-09-24 (live waypoint/provenance audit)

- A subsequent armable live run was inspected without altering the running process.
  RGB inference, target images, status, and raw path all averaged `2.0 Hz`; the annotated
  image initially varied from `0.146-0.847 s` between updates and later stabilized to
  `0.345-0.654 s`. This is bounded synchronous Grounding DINO timing, not an accumulating
  ROS image queue. Front, rear, and merged LiDAR remained approximately `40 Hz`.
- The captured Target Tracker frame showed one centered person at YOLO confidence
  `0.91`. Grounding DINO correctly localized the shirt (`0.833-0.840`) and person
  (`0.807-0.828`), but the configured `dark brown t shirt` colour gate rejected it:
  brown fraction was only `0.055-0.081` versus the fail-closed `0.15` threshold. Direct
  HSV inspection found `0.678` of the shirt ROI in the red/maroon ranges, so relaxing the
  brown gate would create a false semantic match; the live garment is maroon/dark red,
  not brown under this camera.
- Because target state remained `SEARCHING`, the server intentionally skipped
  OmTrackVLA and returned zero placeholders. No genuine nonzero waypoint was produced,
  fusion stayed disabled, and `/cmd_vel` remained zero. A full locked-target real-world
  waypoint/motion validation therefore did not occur in this run.
- Audited official upstream commit `e9cb1fb`: the released agent builds 31-frame coarse
  history plus current fine tokens, predicts eight `[x,y,yaw]` local waypoints, selects
  waypoint index 1, and divides by `dt=0.1` to obtain base velocity. The Ridgeback bridge
  uses the same history, index, and time conversion, then clips/fuses/safety-gates it.
  RViz publishes the model's metric x/y unchanged in `base_link`; the yellow arrow
  graphic is local RViz presentation of the model's yaw quaternion. Upstream's own
  video overlay draws only the x/y curve, not RViz arrows.
- The upstream forward signature contains `bbox_feat` and the checkpoint contains
  `bbox_proj` weights, but `OpenTrackVLA.forward` leaves `extra=[]` and never consumes
  `bbox_feat`. This confirms that direct selected-box conditioning is unavailable in the
  released computation graph despite the exposed argument.
- Fixed a local provenance bug for the next restart: responses now expose `planner_ran`,
  status exposes end-to-end `pipeline_seconds`, and an empty Path is published whenever
  the planner was skipped. RViz will no longer draw stacked yellow arrows from eight
  zero-valued search placeholders; yellow arrows will mean OmTrackVLA actually ran.
- `/r100_0160/cmd_vel` has both the webserver and OmTrackVLA as potential publishers on
  twist_mux's priority-1 external input (`0.5 s` timeout), but the webserver emitted no
  messages during the observation window. No active command collision was measured.
  The MJPEG process stopped as intended, although systemd marked its slow shutdown as a
  timeout after ultimately killing it; it was no longer consuming CPU.

### 2026-09-24 (dual-rate target-view smoothing)

- The live audit showed that the RealSense JPEG stream itself was healthy at about
  `29.97 Hz` and that annotations did not accumulate in a ROS queue. Visible pauses
  came from the old synchronous `2 Hz` target/planner cadence, especially on sparse
  Grounding DINO frames.
- Split the response cadence without increasing heavy OmTrackVLA load: YOLO/BoT-SORT
  target tracking and the annotated Target Tracker view now run at `3 Hz`, while
  OmTrackVLA visual encoding and waypoint inference remain capped at `2 Hz`. On the
  intermediate target frame, the most recent genuine model trajectory is reused and
  fused against the current frame's target/depth location.
- Added explicit waypoint-cache provenance (`planner_updated` and
  `planner_age_seconds`). The ROS bridge rejects a reused result older than `0.75 s`;
  target loss, prompt change, reset, invalid trajectory, or any existing freshness and
  safety failure still clears/stops motion. This is bounded reuse, not an image queue.
- Downscaled only the diagnostic annotated image to at most `512` pixels wide at JPEG
  quality `82`; detector and OmTrackVLA inputs still use the original selected camera
  frame. Reduced RViz's render cap from `15` to `10 FPS` because its fastest changing
  input is now `3 Hz`.
- Grounding DINO now runs every eighth target frame while searching/recovering, about
  `0.375` runs/s, preserving approximately the old `0.4` runs/s compute budget. The
  configured prompt was synchronized with the live garment description as
  `Follow the person who is wearing dark red T-shirt.`
- These changes apply only after the controller/inference launcher is restarted. A
  final read-only check found no controller, inference-server, or RViz process and no
  `target_image` publisher; the source camera remained stable at `29.96-29.99 Hz`.
  Therefore the new `3/2 Hz` behavior requires post-restart measurement and is not yet
  claimed as live-verified.

### 2026-09-24 (upstream-style trajectory overlay)

- Replaced the default yellow-arrow/magenta-line RViz presentation with the official
  OmTrackVLA camera-overlay convention from upstream `trained_agent.py`. The Target
  Tracker image now draws the raw model's X/Y trajectory as a four-pixel
  `(0, 255, 180)` green line over an eight-pixel black outline, anchored at 86% image
  height and scaled at `120 px/m`, with a green four-pixel start marker. Yaw is omitted,
  as it is upstream.
- The overlay consumes the same raw `trajectory_cpu` returned directly by the loaded
  OmTrackVLA planner. It is not based on the custom fused target goal. If the target is
  not locked and the planner is skipped, no curve is drawn; a cached genuine path may
  remain visible only within the existing `0.75 s` bounded reuse window.
- The metric `/omtrackvla/predicted_path` ROS topic remains unchanged for diagnostics
  and control provenance. Its RViz display is now hidden by default, colored green if
  manually enabled, and has pose arrows disabled. The custom fused path also remains
  hidden. These are visualization-only changes and do not alter inference, fusion,
  safety gates, or `/cmd_vel`.
- Static/unit verification passed, but this overlay still requires a restarted live
  controller and a `LOCKED` target for visual confirmation.
- The restarted stack was subsequently inspected live. The updated Target Tracker was
  publishing steadily at approximately `3.08-3.33 Hz`, but the sampled state remained
  `LOST` with `planner_ran=false`, `planner_updated=false`, eight zero placeholders,
  and `last_command=[0,0,0]`. The captured frame showed a gray-shirted person while the
  configured target is a dark-red T-shirt; therefore no green trajectory was expected
  or drawn. Controller logs show earlier brief `ready` intervals, but no frame/status
  from those locked intervals was retained, so a live nonzero overlay is not yet
  visually verified. The pure renderer test confirms the upstream pixel mapping and
  confirms that no curve is drawn for a missing trajectory.
- A second supervised observation with the prompted leader in view did reach
  `LOCKED/identity_verified` and `planner_ran=true`. The captured published Target
  Tracker frame visibly contained the upstream-style green/black curve at the bottom
  centre, confirming the live overlay path. It was easy to overlook: the captured raw
  OmTrackVLA trajectory had a maximum waypoint radius of only `0.1513 m`, which is about
  `18 px` at the official `120 px/m` scale, and it lay between the leader's legs. Across
  30 seconds, raw trajectory radii ranged from `0.0376 m` to `0.3325 m` (roughly
  `4.5-40 px`), explaining why the official-scale curve often looks like a dot or short
  stroke rather than the long curve seen in upstream Habitat demos. Ninety annotated
  frames were received and the target view remained near `3 Hz`.
- During the same interval the controller intermittently issued approximately
  `0.03-0.04 m/s` when the gate was ready, but most samples were stopped by
  `scan_stale`; there were also brief target `UNCERTAIN/LOST` transitions. This is a
  separate live LiDAR-freshness/target-stability issue, not an overlay failure. No
  visualization scale was changed: the displayed curve remains the exact upstream
  metric-to-pixel mapping.

### 2026-09-24 (metric OmTrackVLA arrows restored in RViz)

- At the operator's request, removed the raw OmTrackVLA curve from the Target Tracker
  camera image and restored the model trajectory in RViz beside the robot. The enabled
  green `Path` display uses `base_link`, unmodified metric X/Y coordinates, and green
  pose arrows from each predicted yaw. It remains raised `0.75 m` only in Z for
  visibility. The custom target-fused path remains hidden.
- This is visualization-only: the published `/omtrackvla/predicted_path`, fusion,
  safety gates, and motor command are unchanged. When `planner_ran=false`, the metric
  path remains empty rather than displaying zero placeholders.
- Clarified close-target behavior: `fuse_target_command` stops X/Y translation when
  target distance is at or below the configured `0.9 m` follow distance. It can still
  rotate toward the target, but it does not command reverse motion merely because the
  leader approaches the Ridgeback.
- Corrected the front raw-LiDAR RViz display to have both `Enabled: false` and
  `Value: false`. An inconsistent setting had made its thin green points appear over
  the cyan merged dual-LiDAR scan. After restart, cyan is the only default scan colour;
  the remaining green elements are the locked-target sphere and genuine OmTrackVLA
  metric path/arrows.
- The front raw-LiDAR display was then removed from the RViz configuration entirely at
  the operator's request, preventing it from being accidentally re-enabled. The front
  scanner remains included in the cyan merged `sensors/scan` data used for safety.

### 2026-09-28 (CUDA placement measured, mostly rejected)

- Moved the 31-frame coarse history buffer to the inference device. It was previously
  copied to host on every frame and copied back on every planner run; at
  `31 x 4 x 1536` floats it is about `762 KB` of VRAM against `2.6 GB` free, so the
  two transfers per inference were removed for no meaningful memory cost. This is the
  only CUDA placement change that was kept.
- Added `gpu_ops.py`: torch implementations of the target-perception colour maths
  (BGR to HSV, the part-based appearance descriptor, the colour-fraction test, and
  gallery cosine similarity). `bgr_to_hsv_u8` reproduces OpenCV's 8-bit conversion
  bit-for-bit by rebuilding its fixed-point `sdiv`/`hdiv` reciprocal tables; a float
  approximation is not sufficient, because a saturation of `178.5` rounds to `179` in
  float and truncates to `178` through OpenCV's table.
- Added `test_gpu_ops.py` (9 tests): HSV is asserted bit-identical to OpenCV, the
  descriptor is asserted within `0.999` cosine of the reference, the colour fraction
  within `0.01`, and the CUDA results identical to the same code on CPU. The suite is
  now 57 tests and passes.
- Added `bench_gpu_ops.py` and measured the placement rather than assuming it. At the
  live `848 x 480` frame size, median of 50:

  | boxes | OpenCV | torch CPU | torch CUDA |
  | --- | --- | --- | --- |
  | 1 | `1.137 ms` | `92.065 ms` | `2.567 ms` |
  | 3 | `1.669 ms` | `87.253 ms` | `5.276 ms` |
  | 5 | `2.173 ms` | `83.721 ms` | `8.031 ms` |
  | 10 | `3.486 ms` | `100.250 ms` | `15.828 ms` |

- **The colour path was therefore NOT wired into `target_perception.py`.** CUDA is
  2.3x to 4.5x slower than the existing OpenCV path, and the gap widens with the
  number of people, because each box costs three separate small kernels (one per
  appearance stripe) and launch overhead dominates the arithmetic. Upload alone is
  `0.274 ms` and the whole-frame conversion `0.537 ms`, so the fixed cost already
  exceeds OpenCV's total for a single box. The CPU path remains authoritative.
- `gpu_ops.py` is retained unused: it is the correct foundation if the hand-crafted
  descriptor is later replaced by a learned re-identification model, where the work
  becomes one batched forward pass instead of thirty small kernels and the placement
  argument reverses. No safety, identity, geometry or motion threshold was changed.

### 2026-09-28 (architecture diagrams)

- Added a `System architecture` section to `docs/RIDGEBACK_DEPLOYMENT.md` (linked as
  `real_robot/README.md`), ahead of the Thor split section. It documents current
  behavior only; no code, threshold, rate, or safety configuration was changed.
- Four Mermaid diagrams: the two-machine deployment map with the `18765` socket
  between Thor and the Ridgeback computer; one frame's journey through the 3 Hz
  detection gate, the 2 Hz planner and the 20 Hz safety gate; the target state
  machine; and the ordered motion gate.
- Added a module map table naming, for each runtime module, which machine it runs on
  and whether it can stop the robot. It records explicitly that `gpu_ops.py` is not
  wired into the runtime path, and separates the trial-time modules from tooling
  (`evaluate_offline.py`, `export_bag_frames.py`, `calibrate_camera_mount.py`,
  `hokuyo_scip_reset.py`, the benchmarks and `summarize_perf.py`).
- The diagrams state the invariant the design depends on: the safety gate samples the
  planner rather than being called by it, so a hung or disconnected inference server
  cannot hold the robot in motion.
