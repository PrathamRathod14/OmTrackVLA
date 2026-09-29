# Documentation

This directory contains documentation for the local Clearpath Ridgeback deployment.
The repository's root `README.md` remains the upstream OmTrackVLA introduction and
setup guide.

- [`RIDGEBACK_DEPLOYMENT.md`](RIDGEBACK_DEPLOYMENT.md): diagrams of the local, full-Thor,
  and two-GPU hybrid options; the hybrid frame sequence; the target state machine and
  motion gate; and the operator guide for startup, observation, safety, and testing.
- [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md): living engineering context, component
  boundaries, current configuration, verified behavior, limitations, and change log.
- [`COMPUTE_PLACEMENT_PLAN.md`](COMPUTE_PLACEMENT_PLAN.md): maintained CPU/GPU
  placement audit, measurements, optimization phases, acceptance gates, and decisions.
- [`OmTrackVLA Ridgeback technical review 2026-09-28.pptx`](OmTrackVLA%20Ridgeback%20technical%20review%202026-09-28.pptx):
  September 28 editable technical review; it predates the September 29 hybrid split.
- [`OmTrackVLA Ridgeback technical review 2026-09-29.pdf`](../output/pdf/OmTrackVLA_Ridgeback_technical_review_2026-09-29.pdf):
  current six-page review of the hybrid architecture, frame sequence, identity checks,
  command fusion, measured evidence, and remaining validation gates. Regenerate it
  with [`build_ridgeback_review_pdf.py`](build_ridgeback_review_pdf.py).

Compatibility links remain under `real_robot/` for older commands and workspace
maintenance instructions.
