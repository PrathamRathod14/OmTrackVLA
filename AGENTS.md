# Workspace maintenance

Whenever changing the Ridgeback deployment, any file under `real_robot/`, the active
prompt, model-loading behavior, launch procedure, safety configuration, ROS interfaces,
or verified runtime behavior, update `real_robot/PROJECT_CONTEXT.md` in the same patch.
Keep its `Last updated` date and change log current, and preserve the distinction
between upstream OmTrackVLA behavior and locally added real-robot integration.

Whenever discussing, measuring, or changing CPU/GPU placement, pipeline scheduling,
inference rates, or compute-performance work, also update
`docs/COMPUTE_PLACEMENT_PLAN.md` in the same patch. Keep its current/proposed
distinction, measurements, next actions, and decision log current. Do not present a
microbenchmark as an end-to-end improvement without a full-pipeline measurement.
