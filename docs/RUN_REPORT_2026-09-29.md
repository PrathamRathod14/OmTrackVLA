# Supervised Ridgeback run observation — 2026-09-29

## Scope and method

Read-only ROS subscriptions observed the local RTX, armable run from **09:55:09.833
to 09:55:49.336 UTC** (39.5 seconds): 80 controller status samples at 2 Hz, 80 target
state samples, 590 `/r100_0160/cmd_vel` messages, and 800 wheel-odometry samples.
One existing Target Tracker image was saved and visually inspected. The controller,
deadman, inference server, and supervised scanner drivers were running during this
window; the operator subsequently stopped the run. No robot command was sent by this
observation.

## What happened

| Observation | Result |
| --- | --- |
| Physical movement estimate | Wheel odometry changed by **1.26 m** between endpoints; integrated odometry path was **1.45 m**. This is wheel odometry, not surveyed ground truth. |
| Target selection | `LOCKED` in **78/80** status samples. The same track ID, **10**, appeared in all 80 status samples. One Target Tracker frame showed the green box on the visible person holding the blue basket. One frame does not establish sustained identity accuracy. |
| Identity interruption | `UNCERTAIN` in **2/80** samples around 09:55:31.834–32.333, with `appearance_mismatch` then `verifying_recovery`. The gate reported `target_not_locked` and `last_command=[0,0,0]`; the original identity recovered by 09:55:32.834. |
| LiDAR interruption | `scan_stale` in **21 consecutive** status samples around 09:55:37.836–47.833, about 10 seconds. Each reported zero `last_command` despite `LOCKED` target and connected inference. The gate returned to `ready` by 09:55:48.338. The cause of the stale merged scan was not established. |
| Motion gate | `ready` in 54/80 samples, `ready_rotation_suppressed` in 3, `target_not_locked` in 2, `scan_stale` in 21. Deadman enabled, E-stop inactive, and inference connected in all 80 samples. |
| Commands | 566/590 observed `/cmd_vel` messages were nonzero. Median commanded planar speed was **0.049 m/s**, p95 **0.102 m/s**, maximum **0.113 m/s**. A zero-command burst was observed around stops; the controller normally publishes no continuous command while blocked. |
| Target range estimate | Depth-derived planar distance was available in 78/80 samples, median **1.06 m**, range **0.80–1.26 m**. These are controller estimates, not measured true person–robot separation. |
| Pipeline timing | Status-sampled inference median/p95 **35/214 ms** and bridge pipeline median/p95 **53/231 ms**. Status is only 2 Hz and may repeat or miss inference frames; these figures are diagnostic, not a full frame-by-frame latency distribution or a before/after speedup. |

## Assessment and limits

The run demonstrated commanded motion and odometry movement while the intended
basket holder was visibly selected in one frame. The fail-closed behavior was visible:
the controller reported zero command when identity became uncertain and while the
scan was stale. The **10-second scan interruption** is the main reliability issue in
this sample. It prevented sustained following even though target lock and inference
continued. The observation does not determine whether the scanner drivers, merged
pointcloud-to-laserscan node, ROS delivery, or host scheduling caused it.

This window cannot give a percentage for target-selection or following accuracy.
There is no frame-by-frame human ground truth, surveyed path, or independent distance
measurement. A longer annotated recording with explicit target identity and measured
person/robot positions would be needed to quantify those errors. In the next supervised
dry-run, log timestamped front, rear, and merged scan arrivals alongside the control
gate to isolate the stale-scan source before another sustained-motion trial.
