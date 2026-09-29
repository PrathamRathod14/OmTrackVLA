# Fusion weight trial comparison — 2026-09-29

## Four observed runs

All four runs used the local RTX supervised launcher. `fusion_target_weight` is the
locked-target share when the raw OmTrackVLA proposal points generally toward the
target; the remaining share belongs to OmTrackVLA. If the proposal points away,
the controller replaces it with the target direction. The weight does not directly
change commanded speed or the safety gates.

| | Run A: 25% OmTrackVLA | Run B: 75% OmTrackVLA | Run C: 100% OmTrackVLA | Run D: 100% target |
| --- | --- | --- | --- | --- |
| Configured blend | 25% OmTrackVLA / 75% target (`fusion_target_weight=0.75`) | 75% OmTrackVLA / 25% target (`fusion_target_weight=0.25`; read back from the running node) | 100% OmTrackVLA / 0% target for aligned proposals (`fusion_target_weight=0.0`; read back from the running node) | 0% OmTrackVLA / 100% target for direction/yaw (`fusion_target_weight=1.0`; read back from the running node) |
| Prompt | Follow the person who is holding blue basket. | Follow the person who is wearing black T-shirt. | Follow the person who is wearing black T-shirt. | Follow the person who is wearing black T-shirt. |
| Observation window, UTC | 09:55:09.833–09:55:49.336 (39.5 s) | 10:26:03.733–10:26:43.232 (39.5 s) | 12:35:12.308–12:35:51.788 (39.5 s) | 12:46:21.075–12:47:00.574 (39.5 s) |
| Target state in 80 status samples | `LOCKED` 78; `UNCERTAIN` 2; `LOST` 0 | `LOCKED` 4; `UNCERTAIN` 20; `LOST` 56 | `LOCKED` 68; `UNCERTAIN` 7; `LOST` 5 | `LOCKED` 80; `UNCERTAIN` 0; `LOST` 0 |
| Motion gate in 80 status samples | `ready` 54; `ready_rotation_suppressed` 3; `target_not_locked` 2; `scan_stale` 21 | `ready` 4; `target_not_locked` 76 | `ready` 64; `ready_rotation_suppressed` 4; `target_not_locked` 12 | `ready` 24; `ready_rotation_suppressed` 30; `obstacle_too_close` 26 |
| Observed nonzero `/cmd_vel` messages | 566 of 590 | 29 of 72 | 585 of 700, including rotation-only commands | 549 of 599 |
| Wheel-odometry endpoint change | 1.26 m (1.45 m integrated path) | 0.109 m (0.115 m integrated path) | 2.17 m (2.29 m integrated path) | 4.26 m (5.62 m integrated path) |
| Sampled Target Tracker image | Green box on the person holding the blue basket; `TARGET LOCKED`. | Black-shirted person and a `black t shirt` grounding box visible; `TARGET LOST: prompt_match_appearance_differs`. | Green box on a black-shirted person viewed from behind; `TARGET LOCKED`. | Green box on a black-shirted person; `TARGET LOCKED`. |
| Main interruption | About 10 s of `scan_stale`, with zero reported command despite target lock. | Unstable target identity; 76/80 status samples blocked for `target_not_locked`. | Two shorter identity-loss intervals; 12/80 samples blocked for `target_not_locked`. No `scan_stale` was observed. | Nearby obstacle blocked OmTrackVLA controller commands in 26/80 samples and suppressed rotation while permitting translation in 30/80; platform movement during the first block is unresolved. |

The status topic was sampled at 2 Hz. All four captures also included target messages
and at least 800 wheel-odometry samples. The observations subscribed to existing ROS topics
and sent no robot commands. Odometry is an estimate, not surveyed travel distance.

### Diagnostic timing from the sampled status topic

| Metric | Run A | Run B | Run C | Run D |
| --- | ---: | ---: | ---: | ---: |
| Inference median / p95 | 35.5 / 214 ms | 257 / 318 ms | 36.1 / 262 ms | 34.2 / 69.0 ms |
| Bridge pipeline median / p95 | 52.9 / 232 ms | 259 / 321 ms | 54.9 / 264 ms | 55.4 / 96.1 ms |
| Depth-estimated target range, min–max | 0.80–1.26 m | 1.43–1.85 m on 4 valid-position samples | 0.51–2.10 m | 1.31–2.53 m |

These values come from 2 Hz status samples, which can repeat a result or miss
inference frames. Run B mostly searched or recovered identity, while the other
runs had longer locks. This timing table cannot rank the fusion weights or infer
a pipeline speedup. No full 20 Hz gate trace was captured for these runs.

## Run A: 25% OmTrackVLA / 75% target

The target remained on track ID 10 in all 80 status samples. At about 09:55:31.834,
appearance mismatch caused two `UNCERTAIN` samples and a zero command; identity
recovered by 09:55:32.834. From about 09:55:37.836 to 09:55:47.833, 21 consecutive
samples reported `scan_stale` and zero command while target lock and inference
continued. During that interval odometry changed by less than 0.002 m and no
nonzero controller command was captured. The merged-scan interruption's cause
was not established. During observed
motion, median commanded planar speed was 0.049 m/s and p95 was 0.102 m/s. The
depth-derived target range median was 1.06 m, but it was not checked against an
independent distance measurement. The [original Run A report](RUN_REPORT_2026-09-29.md)
contains its full measurements and limits.

## Run B: 75% OmTrackVLA / 25% target

The operator changed the configuration and the running node reported
`fusion_target_weight=0.25` and the black T-shirt prompt. This run had only brief
lock intervals, so the new blend rarely reached the motion gate as a valid target
command. Two long blocked intervals lasted about 16.5 and 15.5 seconds in the
status sample sequence, with negligible odometry movement. The observed details are:

| Signal | Observation |
| --- | --- |
| Target state | `LOST` in 56/80, `UNCERTAIN` in 20/80, and `LOCKED` in 4/80 status samples. |
| Motion gate | `target_not_locked` in 76/80 samples; `ready` in 4/80. Zero `last_command` was reported in the blocked samples. |
| Motion | 29/72 observed `/cmd_vel` messages were nonzero during brief lock intervals. Wheel odometry changed by 0.109 m between endpoints; integrated path was 0.115 m. This is wheel odometry, not surveyed displacement. |
| Target perception | The sampled Target Tracker image showed a black-shirted person and Grounding DINO's `black t shirt` box, but displayed `TARGET LOST: prompt_match_appearance_differs`. One target-debug sample gave garment grounding score 0.837 and color fraction 0.597, while gallery similarity was 0.705, below the configured 0.72 recovery threshold. |
| Identity continuity | The reported target track ID changed from 37 to 55 during the window. Short `LOCKED` intervals occurred near 10:26:20.734, 10:26:37.234–37.732, and 10:26:40.233 UTC, followed by `UNCERTAIN`/`LOST`. A track-ID change alone does not prove the tracker changed physical people. |
| Other gates | Deadman enabled, E-stop inactive, and inference connected in all 80 status samples. No `scan_stale` status was observed in this window. |

## Run C: 100% OmTrackVLA / 0% target for aligned proposals

The running node reported `fusion_target_weight=0.0` and the same black T-shirt
prompt as Run B. Target track ID 24 remained the reported identity across the
sample. Lock was lost around 12:35:28–33 and 12:35:36–37 UTC; `target_not_locked`
reported zero `last_command` during both intervals. Deadman was enabled, E-stop
inactive, and inference connected in all 80 status samples. A sampled Target
Tracker frame showed the black-shirted person in a green `TARGET LOCKED` box.

The robot received nonzero commands and wheel odometry changed by 2.17 m between
the window endpoints. The maximum observed commanded planar speed was 0.20 m/s,
the configured limit. For 41 of the 68 `LOCKED` status samples, the depth-derived
target distance was at or below the 0.9 m follow setting; every one of those
samples reported zero translational `last_command`. The smallest target distance
estimate was 0.506 m, but there was no independent distance measurement or record
of the person's motion. The controller does not command reverse motion when the
target approaches within the follow distance. In the 68 locked samples, the
fusion reason was `blended_omtrack_toward_target` 26 times,
`replaced_misaligned_omtrack` once, and `within_follow_distance` 41 times. Thus
the 100% model direction actually applied only to the 26 aligned, outside-distance
status samples. The first identity-loss interval lasted about 4.5 seconds in the
sampled status sequence; odometry integrated about 0.03 m during that interval,
which may include stopping dynamics and measurement noise. No controller command
was captured after its initial zero burst within that interval.

## Run D: 0% OmTrackVLA / 100% target direction and yaw

The running node reported `fusion_target_weight=1.0` and the same black T-shirt
prompt as Runs B and C. Target state stayed `LOCKED` on track ID 20 in all 80 status
samples. A sampled Target Tracker frame showed a black-shirted person in the green
box. Deadman was enabled, E-stop inactive, and inference connected throughout.
All 80 fusion results reported an aligned OmTrackVLA proposal, so the fused
direction and yaw followed the depth-localized target. OmTrackVLA's route did not
steer the robot. The system still required a fresh planner result and used the
raw command magnitude in the speed cap; raw path consistency was diagnostic
under target fusion, not the motion gate. Operationally this was direct target
following with obstacle stops, not an OmTrackVLA navigation comparison.

Wheel odometry changed by 4.26 m between endpoints and integrated 5.62 m of path.
The median observed commanded planar speed was 0.20 m/s, at the configured axis
limit. Depth-derived target range stayed between 1.31 and 2.53 m in the status
samples; there was no independent distance measurement. The directional LiDAR
clearance fell below the configured 0.70 m translation stop boundary in 26
`obstacle_too_close` samples, and all 26 reported zero controller `last_command`. In another
30 `ready_rotation_suppressed` samples, a nearer side obstacle was within the
0.67 m rotation guard while directional travel remained clear enough for
translation. No `scan_stale` or identity-loss status appeared.

**Unresolved movement during an obstacle stop.** In the first roughly 10-second
`obstacle_too_close` interval (12:46:32–42 UTC), the last captured controller
zero command was at 12:46:32.222. It published no more `/cmd_vel` messages until
12:46:41.674, when nonzero commands resumed. During that **9.45-second command
gap**, wheel odometry changed by **0.63 m**, with a maximum reported planar
speed of **0.291 m/s**. That is far too much to describe as a confirmed stop
of the *platform*. The second, roughly 3-second obstacle-stop interval had only
about 0.02 m odometry change. The controller's zero `last_command` is therefore
evidence of its own gate decision, not proof that the robot was stationary.
The Ridgeback's `twist_mux` has joystick, RC and interactive-marker input topics
with higher priorities than this controller's `cmd_vel` input. The historical
capture did not record those inputs, the mux output (`platform/cmd_vel`), or
motor feedback. On follow-up, the operator recalled that the **joystick may have
been used for much of the last run**, but could not confirm its timing. Joystick
override is a plausible explanation for this movement; the source remains
unverified. The movement should not be attributed to the fusion weight or to a
failure of the OmTrackVLA gate.
Before another armable comparison, capture all mux inputs, its output, motor
feedback, odometry and gate decisions on one clock, and verify exclusive
command authority.

On these same sampled raw commands and target positions, recalculating with 100%
OmTrackVLA direction instead of 100% target direction changed heading by a median
13.9 degrees (p95 26.2 degrees), but left the instantaneous fused speed unchanged.
This is a command-level counterfactual, not a prediction of the robot's path under
a different weight.

## What this comparison can establish

Run A demonstrated sustained target lock and movement, interrupted mainly by a
stale scan. Run B demonstrated that the controller stopped when the black-shirt
identity was lost; its sampled garment grounding succeeded, but appearance recovery
was unstable. Run C demonstrated movement with longer black-shirt lock and brief
identity stops. Run D demonstrated continuous lock and odometry movement with
obstacle gating and rotation suppression, but movement during one controller
command gap has an unresolved source. None of the runs provides a target-selection or
following accuracy percentage because there was no continuous human ground truth
or surveyed path.

Run A used a different prompt from Runs B–D. Runs B–D used the same prompt,
but target-lock duration, starting pose, scene and person's motion were not held
constant. Their different travel distances therefore cannot be attributed to
the fusion weight. A matched trial with the same prompt, person, start pose and
route, plus continuous target and scan recording, is needed to compare actual
25%, 75%, and 100% OmTrackVLA against 100% target-direction navigation.
