# OmTrackVLA Ridgeback Compute Placement Plan

Last updated: 2026-09-28

This document records where each part of the live Ridgeback pipeline executes, which
placement changes have been measured, and what should be tested next. Update it with
every CPU/GPU placement, rate, or pipeline-scheduling discussion and implementation.

## Goal and constraints

Use the RTX 5060 for work that benefits from wide parallel execution while keeping the
20 Hz motion gate independent, predictable, and able to stop the robot when inference
is late or unavailable.

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

Status: next action. Offline preprocessing timing is available, but live stage timing
is still required.

- Add opt-in per-stage timings for transport/decode, target tracking, Grounding DINO,
  visual preprocessing, DINOv3, SigLIP, Qwen/waypoint inference, annotation/encode,
  ROS-side target geometry, fusion, and the 20 Hz control interval.
- Record GPU utilization, VRAM, CPU utilization, target-view rate, planner update rate,
  and stale-sensor stop reasons during fixed supervised scenarios.
- Measure three cases for at least 60 seconds each: searching with grounding, locked
  tracking with the planner, and temporary occlusion/recovery.
- Save raw measurements in a timestamped file. Do not tune from console impressions.

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

## Acceptance and rollback rules

- Run the complete real-robot unit suite after every implementation change.
- Compare state, selected track, scores, path, and command provenance on a fixed replay.
- Verify the control timer remains responsive under peak CUDA load and that inference
  failure still produces a stop without waiting for the GPU.
- Keep a change only with measured end-to-end benefit or a demonstrated correctness
  improvement. Revert placements that merely shift load or make timing less stable.
- Perform physical motion trials only after dry-run timing, replay equivalence, and
  safety-gate checks pass.

## Decision log

### 2026-09-28

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
- Considered Jetson AGX Thor as a future onboard-compute target. This would be a
  hardware and `x86_64`-to-`arm64` deployment migration, not a replacement for CUDA:
  Thor runs the neural stages through CUDA under JetPack while the ROS callbacks,
  state, fusion, LiDAR safety, and motor-control work remains on its CPU. Before any
  migration, verify ROS 2 and Python dependency availability on JetPack, rebuild or
  replace architecture-specific extensions, and benchmark the complete pipeline on
  the actual Thor power profile. Keep the current RTX 5060 deployment until those
  compatibility, latency, VRAM, and safety-timing gates pass.
