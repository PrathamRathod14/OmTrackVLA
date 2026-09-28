# OmTrackVLA Ridgeback Compute Placement Plan

Last updated: 2026-09-28

This document records where each part of the live Ridgeback pipeline executes, which
placement changes have been measured, and what should be tested next. Update it with
every CPU/GPU placement, rate, or pipeline-scheduling discussion and implementation.

## Goal and constraints

Use the RTX 5060 in the active local run, or Thor CUDA in the inference split, for
work that benefits from wide parallel execution. Keep the Ridgeback computer's 20 Hz
motion gate independent, predictable, and able to stop the robot when inference is
late or unavailable.

Placement changes must satisfy all of these constraints:

- Preserve fail-closed E-stop, deadman, LiDAR, target, trajectory, freshness, speed,
  and acceleration gates.
- Keep ROS callbacks and motor safety responsive while CUDA is fully occupied.
- Preserve target-selection and identity behavior on the same replay inputs.
- Measure end-to-end latency and cadence. A faster isolated kernel is insufficient.
- Keep enough VRAM headroom for transient allocations and Grounding DINO frames.
- Separate proposals from verified runtime behavior in this document.

## Hardware and audit snapshot

Audit time: `2026-09-28T09:20:04Z`. The OmTrackVLA controller and inference server
were not running during this snapshot, so the live numbers below describe the idle GPU
and the robot's background CPU load rather than a full inference run.

| Resource | Verified state |
| --- | --- |
| CPU | Intel Core i7-9700TE, 8 physical cores, 1.80 GHz base, 3.80 GHz maximum |
| System RAM | 31 GiB total, 24 GiB available during the audit |
| GPU | NVIDIA GeForce RTX 5060 |
| GPU memory | 8,151 MiB reported by `nvidia-smi`, approximately 7.96 GiB |
| Idle GPU state | 27 MiB used, 0% compute utilization, 33 C |
| Recorded full pipeline | Approximately 5.4 GB resident during the earlier live trial |

The point-in-time background CPU snapshot showed the camera MJPEG server at about
45% of one core, the robot web server at 24%, the RealSense process at 20%, and the
depth conversion process at 13%. These values are diagnostic observations, not stable
benchmarks. Full-pipeline profiling must be performed with the same background services
enabled because they compete for the eight CPU cores.

## Current placement

The CPU/CUDA entries below describe stages of the active local RTX run. In the
implemented Thor split, the inference-server stages move together to Thor as shown in
the host-placement table below; the ROS bridge stages remain on Ridgeback.

| Pipeline stage | Current execution | Evidence and decision |
| --- | --- | --- |
| RealSense, LiDAR, E-stop and ROS 2 callbacks | CPU | Linux device drivers, DDS, ROS executors and message handling require host execution. Keep on CPU. |
| Camera rate gate and loopback protocol | CPU | The ROS bridge selects a compressed frame and sends base64 JPEG in length-prefixed JSON over loopback. Measure protocol cost before changing it. |
| JPEG decode and RGB/BGR conversion | CPU | `cv2.imdecode` and `cv2.cvtColor` run before inference. Candidate for copy and conversion reduction, subject to end-to-end measurement. |
| YOLO11n person detector | CUDA | Ultralytics receives `device="cuda"`. Detection boxes are copied back to CPU for tracking. Keep neural inference on CUDA. |
| BoT-SORT association and sparse optical flow | CPU | Track management, matching, Kalman state and OpenCV camera-motion compensation operate on host arrays. Keep unless profiling identifies a material bottleneck. |
| HSV appearance descriptor and colour fraction | CPU OpenCV | CUDA implementation exists but is intentionally unused because it is slower at the live frame and person counts. Keep the OpenCV reference path. |
| Grounding DINO preprocessing | CPU then CUDA upload | PIL and the Hugging Face processor build input tensors on the host. Benchmark a tensor-native path only as part of the full grounding stage. |
| Grounding DINO model | CUDA FP16 autocast | The model and inputs are on CUDA, with FP16 autocast enabled. Keep on CUDA. |
| Grounding DINO postprocessing and candidate ranking | CPU | Boxes and scores are copied to host for geometry, colour checks and target selection. This workload is small and safety-relevant. Keep on CPU. |
| DINOv3 and SigLIP preprocessing | CUDA | One uint8 RGB tensor is uploaded, resized once on CUDA, then normalized separately for each tower. The former PIL/CPU path remains available through `--legacy-vision-preprocess` for comparison and rollback. |
| DINOv3 and SigLIP encoders | CUDA | Both models are explicitly moved to the inference device. Keep on CUDA. |
| Token concatenation and grid pooling | CUDA | The encoder outputs remain on the device for concatenation and pooling. Keep on CUDA. |
| Thirty-one-frame coarse feature history | CUDA | The retained change removes one device-to-host and one host-to-device transfer per planner update for about 762 KB of VRAM. Keep on CUDA. |
| Qwen3, visual projection and waypoint head | CUDA | The planner is moved to CUDA and Qwen loads as BF16 when CUDA is available. Keep on CUDA. |
| Eight-waypoint result conversion | CPU | Only the small final trajectory is copied to host for JSON, geometry and control. Keep on CPU. |
| Target annotation, resize and JPEG encoding | CPU OpenCV | This produces the diagnostic target view. Profile separately because it does not affect model correctness. |
| Depth projection and target geometry | CPU NumPy | Small deterministic calculations using ROS depth data and calibration. Keep on CPU. |
| Command fusion and rate limiting | CPU | Small control calculations at the ROS layer. Keep on CPU. |
| Directional LiDAR, freshness, E-stop and motion gate | CPU at 20 Hz | This must remain independent of CUDA load and failure. Keep on CPU. |
| Velocity publication and status/visualization topics | CPU | ROS and operating-system I/O. Keep on CPU. |

### Verified active host placement at 2026-09-28T14:54Z

The running `start_ridgeback_all.sh` has an armable Ridgeback ROS bridge pointed at
`127.0.0.1:18765`, a local `inference_server.py`, and local RViz. `ss` showed an
established loopback connection between that bridge and server. The RTX 5060 listed
the inference Python process with `5,368 MiB` of GPU memory. At the sampled status,
the physical E-stop was active, the target state was `LOST`, and the last command was
zero. This is the **local RTX run**, not a Thor inference run.

Thor independently had an `inference_server.py` listening on
`192.168.131.51:18765`, with no established client socket at that snapshot. It was
loaded but was not supplying frames to the running ROS bridge. Running a server on
Thor alone does not switch the bridge; the bridge's `inference_host` determines which
server processes camera frames. No matched live RTX-versus-Thor performance comparison
was measured in this placement audit.

A second check at `2026-09-28T14:57Z` found the same loopback connection and a single
Thor listener with no established client. Both checkouts were at commit `3869f0c`.
Thor had all six required model weight files (`6.6 GB` total) and PyTorch
`2.10.0+cu130` reported CUDA available on `NVIDIA Thor`. A protocol probe from
`192.168.131.1` reached Thor and received the expected missing-image error. The
`{"status":"ready"}` line is emitted after model construction and socket bind; it
confirms a loaded server, not that the Ridgeback is sending it images. Thor itself
uses CUDA for these models; moving inference to Thor changes the CUDA device from
the RTX 5060 to Thor's integrated GPU, rather than replacing CUDA.

| Work | Active local run | Implemented Thor split when selected |
| --- | --- | --- |
| Camera, compressed depth, LiDAR, E-stop and deadman ROS inputs | Ridgeback computer | Ridgeback computer |
| JPEG selection and request transport | Ridgeback bridge to loopback | Ridgeback bridge across robot network to Thor |
| JPEG decode, YOLO and Grounding DINO models, DINOv3, SigLIP, Qwen/OmTrackVLA planner | Ridgeback inference process; neural work on RTX CUDA | Thor inference process; neural work on Thor CUDA |
| BoT-SORT, optical flow, HSV appearance, target identity, diagnostic image encoding | Ridgeback inference CPU | Thor inference CPU |
| Depth localization, leader/model fusion, LiDAR checks, 20 Hz safety gate and `/cmd_vel` | Ridgeback ROS bridge CPU | Ridgeback ROS bridge CPU |
| RViz and target/status/path ROS topics | Ridgeback computer | Ridgeback computer; Thor returns the annotated image and proposed path data |

Next action for a Thor comparison: stop the local armable launcher, connect the
Ridgeback bridge to the Thor server with `start_ridgeback_thor.sh` in dry-run, and
record a matched local-versus-Thor locked-target run. Keep the current armable path
local until the remote cadence, reconnection, and safety gates pass.

## Verified placement experiment

The CUDA colour path was rechecked on 2026-09-28 at `848 x 480`, median of 50 runs.
The timing includes frame upload and the result copy because both occur in the real
pipeline.

| Person boxes | OpenCV CPU | Torch CPU | Torch CUDA | Decision |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1.169 ms | 95.565 ms | 2.531 ms | Keep OpenCV CPU |
| 3 | 1.661 ms | 88.914 ms | 5.194 ms | Keep OpenCV CPU |
| 5 | 2.197 ms | 103.523 ms | 8.116 ms | Keep OpenCV CPU |
| 10 | 3.530 ms | 113.430 ms | 15.910 ms | Keep OpenCV CPU |

The isolated upload was `0.278 ms` and CUDA HSV conversion on an already resident
frame was `0.533 ms`. Small per-box kernels and synchronization dominate this workload.
All 57 safety, geometry, target-perception and GPU-equivalence tests passed after the
audit.
After the shared vision-preprocessing tests were added, the complete local suite has
59 passing tests. This unit result does not measure full-pipeline timing.

### Shared DINOv3 and SigLIP preprocessing

Implemented and measured on 2026-09-28. The retained path uploads one RGB frame,
performs one antialiased bicubic resize on CUDA, and applies the two model-specific
normalizations on CUDA. The former path resized a PIL image and invoked two separate
host processors before two uploads.

| Measurement | Legacy CPU/PIL | Shared CUDA |
| --- | ---: | ---: |
| Preprocessing median | 14.531 ms | 0.406 ms |
| Preprocessing p95 | 35.128 ms | 0.535 ms |
| DINOv3 plus SigLIP encoding median | 127.863 ms | 114.493 ms |
| DINOv3 plus SigLIP encoding p95 | 139.798 ms | 114.897 ms |

Across eight sample frames, minimum DINO token cosine similarity was `0.999914`,
minimum SigLIP token cosine similarity was `0.995827`, and the maximum absolute
difference in an OmTrackVLA waypoint component was `0.006985`.

Two reverse-order 12-frame offline comparisons forced a planner update on every locked
frame. The shared CUDA path reduced locked-frame median time from `303.8-319.1 ms` to
`193.2-197.6 ms`, and p95 from `347.9-360.6 ms` to `202.5-220.5 ms`. Target states,
track IDs, trajectory-valid decisions, and the 10 motion-permitted frames were
identical. Maximum command-component difference was `0.006375`; mean absolute command
difference was `0.001036`.

This is offline evidence. A restarted dry-run stack still needs live cadence, peak
VRAM, and safety-timer verification before the change is called live-verified.
Selecting a CPU inference device automatically retains the legacy PIL/CPU processor.

## Main performance questions

1. How much time is spent in JPEG/base64 transport, decode, colour conversion, and
   diagnostic JPEG encoding now that duplicate DINOv3/SigLIP host preprocessing is
   removed?
2. What are the median, p95, and maximum stage times on tracking-only, grounding, and
   planner frames during a full supervised run?
3. Does Grounding DINO's synchronous frame block cause the observed tracking gaps, or
   is ROS/camera/LiDAR contention the larger source of missed cadence?
4. Can DINOv3 and SigLIP use mixed precision or a tensor-native preprocessing path
   without materially changing the eight predicted waypoints?
5. Is 2 Hz planner output the actual limiting factor at the current `0.20 m/s` speed,
   or are target and LiDAR freshness failures more important?

## Implementation plan

### Phase 0: full-pipeline measurement

Status: opt-in stage logging implemented and a SEARCHING dry-run baseline captured.
A locked-person planner capture and occlusion/recovery capture are still required.

Execution order for the Thor comparison: instrument and record the RTX 5060 baseline
first, freeze a replay scene and configuration, then run that same workload on Thor
after its Arm64 environment and robot interfaces are validated. A Thor-only timing
number without this baseline will not establish a speedup.

- Implemented: `OMTRACKVLA_PROFILE_DIR` writes separate inference and bridge JSONL
  files with transport/decode, YOLO, association, Grounding DINO, vision
  preprocessing, DINOv3, SigLIP, token/history, planner, annotation/encode, depth
  geometry, fusion, and control tick timings. CUDA stage boundaries synchronize the
  device while profiling. This changes timing and must be used only in dry-run.
- Implemented: `real_robot/summarize_perf.py` reports count, median, p95, maximum,
  observed planner/tracking cadence, and motion-gate reasons from the JSONL files.
- Still needed: collect GPU utilization/VRAM and CPU utilization alongside the JSONL
  capture; inspect control intervals and stale-sensor stop reasons.
- Record GPU utilization, VRAM, CPU utilization, target-view rate, planner update rate,
  and stale-sensor stop reasons during fixed supervised scenarios.
- Measure three cases for at least 60 seconds each: searching with grounding, locked
  tracking with the planner, and temporary occlusion/recovery.
- Save raw measurements in a timestamped file. Do not tune from console impressions.

First profiled live capture: `log/profile-20260928-baseline/` (local, not committed).
The controller ran in dry-run with the configured blue-basket prompt and the standard
Hokuyo drivers for about 92 seconds after startup. All 278 inference frames remained
`SEARCHING`; no person was detected, so Grounding DINO returned before model inference
and OmTrackVLA never ran. The target-view cadence was `3.018 Hz`. Inference median/p95
was `22.666/24.730 ms`; bridge pipeline median/p95 was `24.641/27.394 ms`. Median
YOLO and BoT-SORT/appearance times were `11.106/5.161 ms`; JPEG decode and target-view
encode were `2.427/3.169 ms`. The first YOLO frame reached `886.577 ms`, so the
`911.644 ms` maximum inference time reflects warmup and is not steady-state p95.

The dry-run ROS control interval had median `49.999 ms`, p95 `50.234 ms`, and maximum
`52.694 ms` over 1,856 intervals; none exceeded 55 ms. Control processing median/p95
was `0.236/0.286 ms`. Every motion-gate reason was `dry_run:deadman_not_held`; this
run did not exercise a held deadman or physical motion. One-second GPU samples showed
`3,907 MiB` median loaded memory, `2%` median utilization, and `13%` maximum sampled
utilization. Coarse one-second samples may miss short GPU peaks. CPU utilization was
not recorded in this first capture. These numbers describe only the no-person
SEARCHING case with profiling enabled and cannot establish planner speed or a Thor
comparison.

Exit gate: a stage-level median/p95/max report exists and identifies a measured
bottleneck. No motion behavior or threshold changes are part of this phase.

### Phase 1: remove avoidable host work and copies

Status: in progress. Shared CUDA DINOv3/SigLIP preprocessing is implemented and passed
offline comparison. The remaining items depend on live Phase 0 measurements.

- Benchmark replacing base64 JSON image payloads with a binary payload or shared-memory
  handoff while preserving the Python 3.12/3.9 process split.
- Avoid the current RGB-to-BGR round trip by maintaining clearly named RGB and BGR
  views or converting only for the consumer that needs it.
- Completed: replace duplicate host preprocessing with one uploaded RGB tensor, one
  CUDA resize, and separate DINOv3/SigLIP CUDA normalization. A legacy switch is kept
  for rollback and comparison.
- Measure diagnostic JPEG encoding separately. Reduce its cost only if it affects the
  3 Hz target cadence.

Exit gate: retain a change only if it improves the affected full-pipeline p95 by at
least 10%, preserves replay outputs within defined tolerances, and does not increase
stale safety events.

### Phase 2: isolate slow grounding frames from steady tracking

Status: proposed, higher implementation risk.

- Prototype asynchronous Grounding DINO execution with a frame identifier and prompt
  generation attached to every result.
- Discard stale grounding results after prompt changes, reset, or incompatible track
  state. Never allow an asynchronous result to bypass the three-frame confirmation.
- Compare a separate worker, CUDA stream scheduling, and the existing synchronous
  design under identical scenes. The shared GPU may make concurrency slower.

Exit gate: tracking remains at or above its configured cadence on grounding frames,
identity state transitions match the synchronous reference, and GPU memory retains a
measured safety margin.

### Phase 3: optimize the heavy planner path

Status: proposed, dependent on Phase 0 numerical baselines.

- Benchmark FP16 or BF16 autocast for DINOv3 and SigLIP, which currently execute on
  CUDA without an explicit autocast region in the Ridgeback inference path.
- Compare predicted trajectories on a fixed replay before considering TF32,
  compilation, or lower precision.
- Test a 3 Hz planner only after the full pipeline sustains a p95 planner latency below
  its required period with stable target and LiDAR freshness. Keep the 20 Hz safety
  loop unchanged.

Exit gate: no meaningful target/path regression, no out-of-memory event, at least
1.5 GiB measured VRAM headroom at peak, and no increase in safety-loop deadline misses.

### Phase 4: consider learned GPU ReID only for identity quality

Status: deferred.

- Do not replace the HSV descriptor merely to move work to CUDA. It currently costs
  about 1 to 4 ms and beats the tested CUDA implementation.
- Consider a batched learned ReID model only if real replay data shows that clothing
  histograms cannot maintain identity. Evaluate accuracy, latency, and VRAM together.

Exit gate: a labelled multi-person replay shows a material identity improvement with
acceptable full-pipeline latency and memory.

### Later all-on-Thor deployment option

Status: architectural proposal only. The inference-only Thor split described below
has been implemented and dry-run tested; moving the ROS bridge and motor safety to
Thor has not been performed.

- A full Thor deployment would replace the current x86 host and discrete RTX 5060 platform.
  It would not replace CUDA: the neural-network workload would still run through the
  CUDA, cuDNN, and TensorRT stack supplied by JetPack.
- In that later design, ROS communication, depth geometry, command fusion, LiDAR
  checks, timeouts, and motor safety would run on Thor CPU. In the implemented split,
  these remain on the Ridgeback CPU; target identity runs in the Thor inference process.
- Before choosing Thor, build the Python and ROS dependencies for ARM64, confirm the
  camera and Hokuyo drivers, measure sustained power and thermals, and run the same
  replay-equivalence, latency, freshness, and physical safety checks used here.
- Thor's larger unified memory could remove the current 8 GB VRAM constraint, but
  there is no matched live RTX-versus-Thor full-pipeline speed comparison yet.

Proposed placement on one Thor device:

- Thor GPU through CUDA: OmTrackVLA/Qwen waypoint inference, DINOv3, SigLIP,
  Grounding DINO, YOLO, token pooling, and heavy image tensor preprocessing.
- Thor CPU: ROS 2/DDS, camera and LiDAR callbacks, networking, JPEG handling,
  BoT-SORT association, sparse optical flow unless a measured accelerated replacement
  is adopted, depth geometry, target state, command fusion, timeouts, and motor safety.
- Thor shared system memory would hold both CPU and GPU allocations. That can reduce
  explicit host/device copies, but placement and synchronization still require
  measurement; shared memory does not mean every task should execute on the GPU.
- The cameras, Hokuyo scanners, E-stop, Ridgeback motor controller, and other sensors
  remain external hardware connected to Thor through their normal interfaces.

## Acceptance and rollback rules

- Run the complete real-robot unit suite after every implementation change.
- Compare state, selected track, scores, path, and command provenance on a fixed replay.
- Verify the control timer remains responsive under peak CUDA load and that inference
  failure still produces a stop without waiting for the GPU.
- Keep a change only with measured end-to-end benefit or a demonstrated correctness
  improvement. Revert placements that merely shift load or make timing less stable.
- Perform physical motion trials only after dry-run timing, replay equivalence, and
  safety-gate checks pass.

## Jetson AGX Thor migration assessment

Status: split inference port implemented, tested in a live locked-target dry-run,
and checked on fresh blue-basket Ridgeback frames replayed on both computers.
Network-delay/reconnect, sustained 2 Hz planner cadence, and armable equivalence
are not yet verified. The chosen
first port retains the x86/RTX host for ROS and motor safety,
while Thor runs model inference over the dedicated robot network. A later full move
would replace the present x86 host plus discrete RTX 5060 with an Arm64 system that
has an integrated NVIDIA GPU. It does not replace CUDA. The neural models,
shared DINOv3/SigLIP preprocessing, and feature tensors would still use CUDA. ROS 2,
tracking association, target/depth geometry, command fusion, and the independent 20 Hz
safety gate remain CPU tasks unless a separately measured change justifies moving them.

Feasibility conclusion: the project-specific split port loads the real model stack
on the attached Thor and processes Ridgeback camera frames. Thor is
`nvidia-thor-r100-0160.local` (`192.168.131.51`, JetPack 7.0/Arm64) and the
Ridgeback source address is `192.168.131.1`. A separate Python 3.12 environment
with CUDA PyTorch 2.10.0+cu130, torchvision 0.25.0+cu130, and matching project
dependencies was built at `/home/robot/dev/omtrackvla/.conda-env`. The weights
were copied separately from Git. This establishes a working SEARCHING path and a
live locked-target planner path in dry-run, not an armable Thor deployment. The patched Hokuyo
driver and ROS interfaces remain on the existing computer.

### What is installed and where the split executes

The project checkout at `/home/robot/dev/omtrackvla` on Thor contains this project's
inference server and model code, six model weight sets (`models/`, about 6.6 GB),
and a separate Arm64 Python 3.12 environment with CUDA PyTorch. Copying the code
and weights did not move the robot's sensor or motor interfaces. In the selected
split, the process placement is:

| Machine and processor | Project work |
| --- | --- |
| Thor GPU, through CUDA | YOLO11n person model; Grounding DINO model; DINOv3 and SigLIP encoders and their shared CUDA image preprocessing; OmTrackVLA/Qwen3 visual projection and waypoint planner; token pooling and retained feature tensors. |
| Thor CPU | Inference socket and JSON/JPEG handling; BoT-SORT association, optical flow, HSV clothing features and colour checks; grounding-box postprocessing, target identity state machine, diagnostic image drawing and JPEG encoding. |
| Ridgeback computer CPU | ROS 2 subscriptions for camera, depth, LiDAR, E-stop and deadman; camera frame selection; depth localization; 75/25 leader/model command fusion; freshness, obstacle, speed and 20 Hz motor safety checks; `/cmd_vel`, status/path/image ROS publication and RViz. |
| Ridgeback RTX 5060 | No project neural inference when the bridge is connected to Thor. RViz may still use GPU graphics rendering, and unrelated system processes may use the GPU. The local RTX inference path remains available as a separate launch mode. |

`protocol.py` is used on both computers. The Ridgeback bridge sends a selected
compressed RGB frame and prompt to Thor; Thor returns target state, annotated image,
raw waypoints and provenance. Ridgeback combines that result with its local depth
and safety data. An all-on-Thor ROS/motor deployment is not implemented.

The Thor CPU/GPU split is **within one `inference_server.py` Python process**, not
two services or a hardware partition. The CPU receives JSON, decodes JPEG with
OpenCV, and runs the tracking/identity logic. PyTorch models are placed on `cuda`
(`.to(self.device)` or Ultralytics `device=self.device`), so their tensor operations
execute on Thor's GPU. Detector boxes return through `.cpu().numpy()` for BoT-SORT
and HSV checks. For a locked target, `torch.from_numpy` starts with a CPU RGB frame;
`resize_rgb_tensor_for_vision` explicitly uploads it with `.to(device)`, then CUDA
does shared resize, vision encoding, token pooling and waypoint planning. The small
trajectory returns through `.detach().float().cpu()` for JSON serialization. There
is no custom CPU/GPU scheduler, dedicated CPU core assignment, zero-copy path, or
separate CPU/GPU worker service. The Thor launcher caps OpenMP/MKL threads at four;
the source code and framework decide the individual tensor placements.

### How this repository would run on Thor

The implemented first port moves only `real_robot/inference_server.py` and its CUDA
models to Thor. Keep `real_robot/ridgeback_ros2_node.py`, the 20 Hz motion gate, ROS
sensor drivers, E-stop, deadman and `/cmd_vel` on the present Ridgeback computer. The
bridge already has an `inference_host` parameter, so it can send its JPEG request to
Thor and receive the target and waypoint response. Depth geometry and command fusion
then remain next to the sensors and motor safety gate. This topology has passed the
live locked-target dry-run, while the current armable deployment remains on the x86 host.

First-port inventory:

| Move to Thor | Keep on the Ridgeback computer |
| --- | --- |
| `real_robot/inference_server.py` and its local helpers: `target_perception.py`, `protocol.py`, and `perf_log.py` | `real_robot/ridgeback_ros2_node.py` and the 20 Hz safety/motion gate |
| Model Python code used by that process, including `open_trackvla_hf/` and `cache_gridpool.py` | ROS 2 Jazzy sensor/robot interfaces, camera/depth, Hokuyo LiDAR, E-stop, deadman and `/cmd_vel` |
| `models/` weights: YOLO11n, Grounding DINO, DINOv3, SigLIP, Qwen3-0.6B and OmTrackVLA-0.6B (about 6.6 GB here) | `real_robot/ridgeback.yaml` safety and ROS settings, with `inference_host` pointed to Thor |
| A newly built Arm64 Python/CUDA runtime for the inference process | The existing x86 `.conda-env` stays on its present host; its binaries are not portable to Arm64 |

The inference server's CPU work (JPEG decode, BoT-SORT/optical flow, target
selection and response encoding) also moves because it runs inside that process.
Its neural stages still run on Thor's CUDA GPU. Thor receives selected camera JPEGs
and the active prompt; it returns target state, box, waypoint/command proposal and
diagnostic target image. The existing ROS bridge keeps depth localization, target
fusion, obstacle checks and final velocity publication.

The split is now project-specific launch configuration. `start_inference_thor.sh`
binds only to Thor's robot-network address and accepts only the Ridgeback source IP.
`start_ridgeback_thor.sh` starts just the existing ROS bridge in dry-run and sets
`inference_host` to Thor. `start_ridgeback.sh` continues to default to local inference
for rollback. Remote arming is rejected unless the operator explicitly sets
`OMTRACKVLA_ALLOW_REMOTE_ARM=1` after the remaining verification gates pass. The
length-prefixed JSON/base64-JPEG socket has no authentication or
encryption, so port 18765 must remain on the dedicated robot network. The bridge now
rejects a response if the source camera frame is older than `camera_timeout`; its
existing safety gate also checks inference freshness. A dry-run with a held deadman
confirmed that stopping Thor changed the gate reason to
`dry_run:inference_disconnected` while the 20 Hz host timer continued. Delay and
reconnect tests remain.

Build the inference environment natively on Arm64. `run_local.sh` points at the local
`.conda-env`, whose current binaries are x86_64, and upstream's Python 3.9/Habitat
setup is not a Thor deployment recipe. The attached Thor's Python 3.12 environment
has imported CUDA PyTorch, torchvision, Transformers, Ultralytics, and OpenCV; it
loaded all real model files and answered an empty-frame request. Numerical behavior
on locked-person frames, BF16/FP16 planner outputs, and custom CUDA preprocessing
were checked on a short fixed replay and on current camera frames. Broader scenes
and end-to-end live timing still require a matched RTX baseline.

The fixed offline replay repeated one cropped, real-person image for 12 frames with
the prompt `Follow the person.`. Both Thor and RTX locked track ID 1 on frame 3,
kept it locked for 10 frames, and marked the same 10 trajectories valid. All 12
target states, target reasons, track IDs, trajectory-valid decisions, and trajectory
reasons matched. Maximum absolute command-component difference was `0.003344`;
mean absolute difference was `0.001672`. This confirms a basic Thor planner path
and close numerical agreement for this one input, not identity behavior in a moving
scene. The replay runs model code without ROS depth, LiDAR, the network bridge, or
motor control. Its mean frame times are not a full-pipeline speed comparison.

A fresh 90-frame capture from `/r100_0160/camera/color/image_raw/compressed` included
the person holding the blue basket and then leaving it. Replaying those frames with
the active blue-basket prompt on Thor and RTX gave identical target state, reason,
track ID, trajectory-valid decision, and trajectory reason on all 90 frames. Both
locked track ID 1 on frame 3 for 49 frames, entered `UNCERTAIN` once, and recorded
31 `LOST` frames after the target was no longer verified. With a planner update
forced on every locked frame, maximum absolute command-component difference was
`0.007871` and mean absolute difference was `0.000597`. At the normal wall-clock
planner cap, maximum difference was `0.043096`; the two machines can update/cache
on different frames, so that number is not an isolated model-precision comparison.
These are offline model-path results on current robot imagery, not end-to-end robot
motion or a timing speedup.

The live Thor split dry-run with a blue-basket holder captured 187 frames over about
65 seconds: 93 `LOCKED`, 7 `UNCERTAIN`, 86 `LOST`, and 1 `SEARCHING`. There were
31 planner updates during the 30.241-second locked span, about `1.03 Hz` observed,
below the configured 2 Hz cap. Live status confirmed `target_valid`,
`target_position_valid`, `trajectory_valid`, and `planner_ran` together during lock.
Bridge pipeline median/p95 was `63.034/369.874 ms`; reported inference median/p95
was `44.930/349.571 ms`. One response was rejected because its source camera frame
exceeded the `0.75 s` freshness limit. The 20 Hz host control interval had
median/p95/max `49.998/50.454/55.006 ms` over 1,277 intervals. The operator kept
the physical E-stop active, so every control decision remained blocked and no
physical motion was tested. The observed planner rate and isolated stale response
need investigation before considering remote arming.

### Would using the Ridgeback RTX alongside Thor help?

Current Thor split: the Ridgeback RTX does no project neural inference. Thor's GPU
runs the detector, vision encoders, and planner; its CPU runs tracking and image
handling. This is a placement choice, not evidence that Thor has unlimited capacity.
In the live blue-basket dry-run, the 31 frames that actually updated the planner had
median `346.890 ms` inference and `366.603 ms` bridge pipeline time; their maximum
pipeline time was `495.263 ms`. Across the locked span, observed planner cadence was
`1.03 Hz` against a configured `2 Hz` *cap*. The cap is not a guaranteed output
rate, and this log alone cannot attribute the gap to Thor GPU load: target state,
request scheduling, CPU work, and transport also affect cadence. There is no
matched live local-RTX-versus-Thor baseline or GPU-utilization/thermal trace yet.

A possible **unimplemented** two-GPU variant would run YOLO/Grounding DINO person
perception on the Ridgeback RTX and DINOv3/SigLIP/OmTrackVLA waypoint planning on
Thor. Ridgeback would have to send the selected image and target identity/state to
Thor, and coordinate two inference services and their frame timestamps. Since the
planner depends on the verified target, splitting one frame across GPUs adds a
network handoff and does not automatically shorten its latency; throughput could
improve only if stages for different frames can overlap without delaying target
freshness or the 20 Hz safety gate. Keeping both services synchronized and handling
one side's failure would add operational and safety complexity.

Decision for now: keep the tested single-server Thor split as the experimental
remote mode, and retain the local RTX path as the armable mode. Before implementing
the two-GPU variant, capture matched live RTX and Thor runs with the same scene,
prompt, rate settings, and full stage profiling; record both GPU utilization, memory,
power/thermal throttling, transport time, planner cadence, p95 latency, and stale
responses. If a measurable Thor stage or resource bottleneck remains, prototype the
two-GPU placement in dry-run and compare full-pipeline latency and control timing.

The first live split dry-run on 2026-09-28 captured 117 no-person `SEARCHING`
frames over approximately 39 seconds: target-view cadence `3.007 Hz`, reported
inference median/p95 `43.133/44.577 ms`, bridge pipeline median/p95
`47.117/48.755 ms`, and 20 Hz control interval median/p95/max
`50.001/50.255/50.932 ms` over 777 intervals. Every motion-gate reason was
`dry_run:deadman_not_held`. The prior x86 SEARCHING capture had inference
`22.666/24.730 ms` and pipeline `24.641/27.394 ms`, but the recordings were made
at different moments and Thor stage profiling was not enabled. These values do not
show a speedup; they also do not measure target lock, grounding, or planner work.

In a second dry-run, the deadman was published at 10 Hz and the Thor server was
stopped. The bridge recorded 269 `dry_run:inference_disconnected` control ticks;
the maximum observed control interval was `52.744 ms`. This verifies a disconnect
stop reason with motor output disabled. It does not verify a physical stop distance
or recovery after Thor restarts.

Bring-up order: (1) load all models and run a fixed offline frame replay on Thor;
(2) connect the existing bridge to Thor in `dry_run: true` and compare target IDs,
trajectories, and motion-gate reasons against the x86/RTX baseline; (3) measure
transport, full-pipeline p50/p95/max, 3 Hz target view, 2 Hz planner, GPU memory,
thermal/power state and the host's 20 Hz control interval; (4) verify E-stop,
deadman, LiDAR and network-failure stops before any armable trial. The first profiled
baseline contains only no-person SEARCHING frames, so locked-target and grounding
baselines are still needed for a valid performance comparison.

A later all-on-Thor deployment is also possible, but it requires ROS 2 Jazzy and the
robot topics to reach Thor, and a separate launch/service design. The current
`start_ridgeback_all.sh` controls the x86 host's Clearpath service, scanner processes,
and CPU cores 4-7; it cannot be treated as a Thor launcher. Rebuild the patched
`urg_node` overlay for Arm64 only if Thor will own the Hokuyo drivers. Preserve the
existing camera calibration if the camera stays mounted in the same physical pose.

Potential benefits for this Ridgeback deployment, subject to a working port:

| Benefit | Practical meaning | Current evidence limit |
| --- | --- | --- |
| Onboard integration | One compact module can host the Arm CPU and CUDA GPU instead of the present x86 host and discrete GPU. This may simplify packaging on the robot. | Power wiring, cooling, mounting, and scanner/camera connectivity still need design. |
| Memory headroom | The 128 GB shared LPDDR5X pool can accommodate larger models, longer histories, or future learned re-identification work. | The current stack already fits in 8 GB GPU memory, so this is expansion capacity rather than a measured speed gain. |
| Power envelope | Thor's published 40-130 W operating range may allow a smaller compute power and cooling budget. | Current complete-computer power was not measured; module ratings are not a system energy comparison. |
| Data sharing and accelerators | Thor's coherent CPU/GPU memory and vision accelerator create options for reducing copies or moving image processing away from the CPU. | The current Python/ROS pipeline does not use Thor-specific zero-copy or PVA paths. They require code changes and end-to-end measurement. |

These benefits are deployment options, not promises of faster OmTrackVLA inference or
more reliable target identity. In the profiled SEARCHING run, JPEG decode was about
`2.4 ms` median and the model never ran; there is no evidence yet that removing a
CPU-to-GPU copy would materially change the full locked-person pipeline.

NVIDIA lists the Jetson AGX Thor Developer Kit with a 14-core Arm CPU, Blackwell GPU,
128 GB shared LPDDR5X, and a 40-130 W operating range. Its shared system memory is not
equivalent to 128 GB of dedicated GPU VRAM. JetPack 7.2.1 lists Ubuntu 24.04,
CUDA 13.2.2, and TensorRT 10.16.2; ROS 2 Jazzy supports Ubuntu 24.04 Arm64.
These are platform specifications, not OmTrackVLA performance results.

The current `5,403 MiB` loaded stack fits in the RTX 5060's 8,151 MiB, so Thor's
larger memory pool alone does not remove a measured capacity bottleneck. The RTX 5060
has a discrete Blackwell GPU rated at 3,840 CUDA cores and 145 W graphics power;
Thor's integrated Blackwell GPU has 2,560 CUDA cores in a 40-130 W system power
range. Core count, advertised AI throughput, and power ratings cannot predict this
pipeline's latency because the CPU stages, memory system, kernel mix, and thermal
state differ. Thor may improve an optimized model stage yet be slower on another
stage. Faster end-to-end following remains a hypothesis, not an expected result.

Migration gates:

1. Build the inference environment for Arm64; verify PyTorch/CUDA, Transformers,
   Ultralytics, OpenCV, model loading, and every native extension at compatible
   versions. Do not copy the x86 install tree or binary wheels.
2. For the first split trial, verify the current host still receives camera, both
   Hokuyo streams, E-stop, deadman and TF; that its ROS bridge reaches Thor over the
   restricted socket; and that `/cmd_vel` and prompt/status topics behave as before.
   If later moving ROS and scanner drivers too, build ROS 2 Jazzy and the patched
   `urg_node` overlay on Arm64 and reverify every interface there.
3. Run the same replay and safety suite, then measure full-pipeline median/p95/max
   latency, target-view and planner cadence, memory pressure, power mode, thermal
   throttling, and 20 Hz control deadlines on Thor. Use the same scene and settings
   as the RTX 5060 baseline. Shared memory may reduce transfer overhead, but this
   must be measured end to end.
4. Keep the current system until Thor meets target identity equivalence, stale-stop
   behavior, planner cadence, and safety timing with measured margin. A larger memory
   pool or advertised AI throughput alone is not evidence of faster following.

Platform sources: [NVIDIA Thor specifications](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/),
[NVIDIA RTX 5060 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/),
[JetPack releases](https://developer.nvidia.com/embedded/jetpack/downloads),
[PyTorch Arm support for Thor](https://discuss.pytorch.org/t/does-torch-now-offically-supported-nvidia-jetson/224665/2),
[CUDA for Tegra memory behavior](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/),
[NVIDIA PVA SDK](https://developer.nvidia.com/embedded/pva),
[NVIDIA on Thor shared memory](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html),
and [ROS 2 Jazzy supported platforms](https://www.openrobotics.org/blog/2024/5/ros-jazzy-jalisco-released).

## Decision log

### 2026-09-28

- Assessed using the idle Ridgeback RTX concurrently with Thor. The live Thor log's
  31 planner-update frames had `346.890 ms` median inference and `366.603 ms` median
  bridge time, while locked-span cadence was `1.03 Hz` under a `2 Hz` cap. Neither
  these timings nor Thor's hardware rating establish a GPU bottleneck or capacity
  guarantee. A proposed perception-on-RTX/planner-on-Thor split remains unimplemented;
  require matched full-pipeline and resource measurements before adding it.
- Made the implemented split inventory explicit: Thor stores the project code,
  CUDA environment and models; its GPU runs neural stages while its CPU runs the
  inference service and tracking logic. Ridgeback retains ROS, depth/fusion and
  motor safety. This changes documentation only; the next action remains a matched
  live RTX/Thor dry-run comparison before considering remote arming.
- Traced the in-process handoffs on Thor: model `.to(cuda)` placement, explicit RGB
  upload, detector/result copies back to CPU, and CPU JSON/OpenCV/tracker work. The
  implementation uses one inference process and no custom CPU/GPU scheduler or
  zero-copy mechanism; the measured full-pipeline gate remains unchanged.
- Clarified that switching inference to Thor removes this project's neural workload
  from the Ridgeback RTX, not every possible use of that GPU; RViz graphics can
  still use it, and the separate local inference launch retains the RTX path.
- Audited the live host placement: the armable bridge was connected to the local
  loopback inference server, whose RTX process occupied `5,368 MiB`; Thor's separate
  server was listening without a client. Clarified that the implemented Thor split
  moves the inference process, not the ROS bridge or motor safety. No new speed
  comparison or armable Thor result was obtained.
- Rechecked Thor model files, CUDA availability, matching code commit, and a network
  protocol probe. `ready` means loaded and listening; the running bridge still used
  `127.0.0.1`. The switch to Thor requires a separate Ridgeback bridge launch with
  `inference_host=192.168.131.51`, in dry-run under the current validation status.
- Implemented the Ridgeback-to-Thor inference split with a single allowed client,
  separate server and bridge launchers, and rejection of late camera-frame responses.
  Confirmed the attached Thor IP via mDNS/SSH and its JetPack 7.0 Arm64 platform;
  installed a dedicated CUDA PyTorch environment and copied project weights.
- Verified model startup, a real empty-frame request, and a 117-frame ROS dry-run
  across the network. SEARCHING held 3.007 Hz and the host control timer stayed
  within 50.932 ms; median inference and bridge pipeline times were 43.133 and
  47.117 ms. Live planner and armable behavior remain unverified. No Thor speedup claimed.
- Ran the same 12-frame offline person replay on Thor and RTX. Both locked the same
  track on frame 3 with identical state and trajectory-valid decisions on all frames;
  maximum command-component difference was 0.003344. This is a narrow model-path
  comparison, not live planner timing or a moving-scene identity test.
- Captured a live Thor blue-basket dry-run: 93 locked frames, 31 planner updates in
  30.241 seconds of lock, valid depth and trajectory status, one stale-camera reply,
  and a 55.006 ms maximum host control interval. The physical E-stop stayed active.
- Replayed 90 fresh current-camera frames with the active prompt on Thor and RTX.
  Identity and trajectory decisions matched on every frame, including loss; forced
  per-frame planner outputs differed by at most 0.007871 per command component.
  Earlier annotated footage was excluded from this current-project comparison.
- Stopped Thor during a second dry-run with a held deadman. The bridge recorded
  `dry_run:inference_disconnected` and retained 20 Hz timing (52.744 ms maximum).
  Recovery and physical stopping still need verification.
- Listed the exact first-port files, model weights, and CPU/GPU work that move to
  Thor versus the ROS, sensor and motor-safety work retained on the current host.
- Traced the actual launch path for Thor: first move the CUDA inference process,
  retain ROS sensors and the 20 Hz safety gate on the current Ridgeback computer,
  and use the bridge's `inference_host` setting after adding a trusted remote peer
  and separate launch modes. This is proposed; the current server rejects remote
  clients and neither launch script supports a split deployment yet.
- Documented the concrete potential benefits of a Thor port: onboard integration,
  memory headroom, a bounded module power range, and possible copy/vision-accelerator
  options. None is a measured improvement for the present OmTrackVLA pipeline.
- Confirmed the Thor feasibility boundary: its CUDA/Arm64/ROS/PyTorch platform stack
  exists, but this repository's pinned inference dependencies, ROS interfaces, and
  patched scanner driver still require an on-device build and validation.
- Added opt-in dry-run JSONL stage profiling in the inference server and ROS bridge,
  plus a summary command. Existing 59 tests and two profiling tests pass. CUDA
  synchronization makes profiled timing diagnostic rather than directly comparable
  with the earlier unprofiled offline numbers. No placement or motion threshold changed.
- Captured 278 no-person SEARCHING frames at `3.018 Hz` and 1,856 control intervals
  without a >55 ms gap in dry-run. The locked-person planner and grounding-model cases
  remain unmeasured because no person appeared during this capture.
- Set the next action for the Thor question: produce a repeatable RTX 5060
  full-pipeline baseline before attempting an Arm64 port or comparing runtimes.
- Expanded the Thor option into explicit Arm64 build, device-interface, replay,
  performance, power, and safety gates. No Thor hardware is attached and no speedup
  is claimed. The current RTX 5060 measurements remain the baseline.
- Revisited the expectation that Thor will run faster. The existing models already
  fit in RTX 5060 memory, and vendor peak specifications do not determine the
  measured 3 Hz target and 2 Hz planner pipeline. The comparison remains open until
  matched full-pipeline runs on Thor establish latency and control timing.

- Confirmed the hardware as an RTX 5060 with 8,151 MiB VRAM and an eight-core
  i7-9700TE host.
- Confirmed that the live inference process was not running during the audit. Idle GPU
  usage must not be confused with full-pipeline VRAM or utilization.
- Reaffirmed the heterogeneous design: neural networks and feature tensors on CUDA;
  ROS, geometry, state, fusion, and safety on CPU.
- Re-ran all 57 tests successfully.
- Re-ran the colour-placement benchmark and rejected CUDA integration because OpenCV
  CPU remains 2.16x to 4.51x faster at the tested person counts.
- Chose full-pipeline stage instrumentation as the next implementation step before any
  additional placement or rate change.
- Implemented shared CUDA preprocessing for the two OmTrackVLA vision encoders. The
  isolated visual-encoding stage improved by about 10%, and two offline full-pipeline
  comparisons reduced locked-frame median latency by about 36-38% while preserving
  target and trajectory-valid decisions.
- Kept `--legacy-vision-preprocess` as an explicit rollback and comparison path. The
  next gate is a restarted dry-run measurement of cadence, peak VRAM, control timing,
  and stale-stop reasons. No physical motion threshold was changed.
- The first live restart loaded the CUDA models and reached an inference-server-ready
  state, but LiDAR recovery stopped the launch before runtime performance could be
  measured. Two orphaned custom Hokuyo drivers from an interrupted earlier run still
  owned the scanner connections. Both standard scanners recovered at about 40 Hz after
  those processes were terminated. The launcher now clears this exact stale-driver
  condition before taking control of the scanner service. This was a LiDAR process
  lifecycle failure, not a CUDA preprocessing failure.
- A subsequent clean live run used `5,403 MiB` VRAM with `2,344 MiB` reported free and
  published the Target Tracker image at approximately `3.08 Hz`. One observed
  planner-update frame took `0.418 s`; a cached-planner sample took `0.032 s` inference
  and `0.048 s` end-to-end pipeline time. These isolated samples do not replace the
  planned median/p95 instrumentation.
- Live target inspection found a tracking-stability issue rather than a CUDA placement
  issue. The intended person holding a blue basket was initially LOCKED for 15/15
  samples on track ID 1, with appearance similarity `0.968-0.993`. A later 20-sample
  window contained 19 LOST states, after which the same visible person was grounded
  and LOCKED as track ID 34. Low-confidence partial-person boxes also appeared as
  separate grey tracks. Target perception consumes the original RGB frame before the
  new planner preprocessing runs, so this evidence does not implicate the CUDA resize.
  The next identity work should use a saved replay to distinguish detector fragments,
  pose/occlusion effects, and HSV-gallery mismatch before thresholds are changed.
- Recorded Jetson AGX Thor as a possible future full-platform replacement. Thor still
  uses CUDA through JetPack. This earlier option is separate from the subsequently
  implemented inference-only split; no speedup or armable Thor validation is claimed.
- Considered Jetson AGX Thor as a future onboard-compute target. This would be a
  hardware and `x86_64`-to-`arm64` deployment migration, not a replacement for CUDA:
  Thor runs the neural stages through CUDA under JetPack while the ROS callbacks,
  state, fusion, LiDAR safety, and motor-control work remains on its CPU. Before any
  migration, verify ROS 2 and Python dependency availability on JetPack, rebuild or
  replace architecture-specific extensions, and benchmark the complete pipeline on
  the actual Thor power profile. Keep the current RTX 5060 deployment until those
  compatibility, latency, VRAM, and safety-timing gates pass.

### 2026-09-28 (placement documented, not changed)

- The current placement is now drawn rather than only described: see `System
  architecture` in `docs/RIDGEBACK_DEPLOYMENT.md`. The deployment diagram shows
  inference on Thor (`192.168.131.51`) and camera, depth, LiDAR, E-stop, deadman,
  depth localisation, command fusion, the 20 Hz gate and `/cmd_vel` on the Ridgeback
  computer (`192.168.131.1`), with the `18765` JSON socket between them.
- The module map names, per module, the machine it runs on and whether it can stop the
  robot. `gpu_ops.py` is recorded there as not wired into the runtime path, consistent
  with the measurement in this plan that the CUDA colour path is slower than OpenCV.
- No placement decision changed and no new measurement was taken in this entry. It is
  documentation of the placement already in effect.
