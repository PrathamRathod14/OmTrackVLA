# Workspace maintenance

Whenever changing the Ridgeback deployment, any file under `real_robot/`, the active
prompt, model-loading behavior, launch procedure, safety configuration, ROS interfaces,
or verified runtime behavior, update `real_robot/PROJECT_CONTEXT.md` in the same patch.
Keep its `Last updated` date and change log current, and preserve the distinction
between upstream OmTrackVLA behavior and locally added real-robot integration.
