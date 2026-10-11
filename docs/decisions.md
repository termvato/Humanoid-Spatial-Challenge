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

## D5. Learned components: A (change verifier) and B (object matcher); C (dense change map) only if time (2026-10-11)
- Both are small heads on frozen foundation-model features; trained on Isaac sim + real pairs; ratio sweep evaluated on held-out real rooms.
- C only after A and B are evaluated end to end.

## D6. Compute: cloud for heavy models, no Jetson (2026-10-11)
- A Jetson is an on-robot inference device, slower than the RTX 4070 laptop, so it is no help for training or offline reconstruction. Spend goes to Modal.
- Big models (reconstruction, SAM 3, large VLMs) run on Modal A100/H100; laptop runs Isaac Sim rendering and small-head training.

## D7. RTX 3080 Windows PC: deferred, add only if rendering is the bottleneck (2026-10-11)
- Measure Isaac render throughput on the laptop first. If the dataset needs > ~6 h of rendering, add the PC as a render worker.
- How: keep Windows (Isaac Sim 5.1 supports it; installing Linux plus the driver pinning costs hours), OpenSSH server + Tailscale so it is reachable off the home network.
- Don't install Linux on it for this project.

## D8. Accept deep-research revisions, provisionally (2026-10-11)
- Path: fork SceneDiff, extend with 3D boxes, 3-state (changed/unchanged/unobserved) map, learned verifier that must beat SceneDiff thresholds on its test split, costmap delta.
- Reconstruction default pi3X or DA3 (not VGGT). Keyless default = SAM 2 + DINOv2 unless S4 says otherwise.
- Isaac role (dataset vs robot demo) and co-training decided by spikes S1-S4 in docs/spikes.md.
- User: "what matters is the true best decision", not his earlier choices.

## D9. Use case: a robot consumes the output (2026-10-12)
- Consumer: a mobile robot. Outputs serve navigation first: 3D change list, costmap delta, suggested viewpoints. Language is the simplest layer (templates).
- Input: plain video primary (confirms D1); LiDAR + ARKit as an add-on, since S7 showed it is the easier problem.
- Uncertainty: report "unobserved" AND suggest a viewpoint that would resolve it.
- Scale: big (navigation-relevant, > shoebox) changes first; small objects are a bonus at lower confidence.
- Timing: offline, between two visits. Stated in the design note as a concession to the 3-4 day budget and compute; online is the product version.
- Memory: two-visit diff only.
- Robot demo: simple diff-drive robot in Isaac Sim that uses the change map.
- Priority (user): "making the thing exist" with a sound architecture before extras.

## D10. Architecture question: 3D-first vs SceneDiff fork (2026-10-12, open, decided by S9)
- User asked whether to switch to "build a 3D model of each visit, then compare the 3D models".
- Evidence so far: S7 (3D-first with LiDAR) found the big change cleanly, was immune to the light change, and produced "unobserved" natively. SceneDiff (2D masks on a joint reconstruction) produced most false alarms from lighting (S2, S6) and outputs 2D masks we would have to lift to 3D anyway.
- D9 needs per-voxel visibility (unobserved, viewpoints, costmap), which is a 3D-first output.
- Proposed: 3D-first core, SceneDiff's appearance cues demoted to a small-object bonus that geometry verifies. Pending S9: does plain-video depth (feed-forward model) give a usable map?
