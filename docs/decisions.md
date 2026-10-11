# Design decisions

## D1. Input: plain video primary, iPhone LiDAR as reference (2026-10-11)
- Capture with an app that saves RGB video plus LiDAR depth and ARKit poses (Stray Scanner or Record3D).
- Main pipeline uses plain RGB video only (learned reconstruction, e.g. VGGT / MapAnything), so any phone works.
- LiDAR depth and poses are a measured reference: report reconstruction error and its effect on change detection.
- `--use-sensor-depth` flag switches to sensor data for LiDAR users.
- Why: the brief allows any tools, but reviewers must be able to run it on a plain video, and video-only 3D is the harder, recognised problem.
