# Design decisions

## D1. Input: plain video primary, iPhone LiDAR as reference (2026-10-11)
- Capture with an app that saves RGB video plus LiDAR depth and ARKit poses (Stray Scanner or Record3D).
- Main pipeline uses plain RGB video only (learned reconstruction, e.g. VGGT / MapAnything), so any phone works.
- LiDAR depth and poses are a measured reference: report reconstruction error and its effect on change detection.
- `--use-sensor-depth` flag switches to sensor data for LiDAR users.
- Why: the brief allows any tools, but reviewers must be able to run it on a plain video, and video-only 3D is the harder, recognised problem.

## D2. Compute: Modal cloud GPU, ~$30 cap (2026-10-11)
- Heavy reconstruction models on Modal; local RTX 4070 (8 GB) for small models and dev.

## D3. Language: keyless default, Claude optional (2026-10-11)
- Default: local open-vocab models (SAM 3 / Grounding DINO / OWLv2 / CLIP) for labels, template sentences from 3D facts.
- `--describe claude` flag adds a richer scene description via the Claude API.
- Why: a reviewer must run the full pipeline with no API key or account.

## D4. Roles: user decides, Claude builds (2026-10-11)
- User makes design calls and reviews until he can explain each part. Claude writes code, tests, runs jobs.
- Real data: 6-10 labelled phone pairs, train/test split by room; demo pairs in test split only (from I2).
