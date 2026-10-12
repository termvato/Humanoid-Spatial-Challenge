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

## D11. Reconstruction model and pipeline shape (2026-10-12, PROPOSED, user to decide)
Evidence: docs/spikes.md S5 + S9.
- Model: pi3 gives the best change detection on the benchmark at every setting; Pi3X gives the best geometry on pair01 (ATE 3.3 cm, depth 2.2%) and near-metric scale (1.016) but worse benchmark AP (11.4 vs 15.3, geometry only). VGGT and DA3 are far behind on change detection; MapAnything far behind on geometry (focal 22% off, possibly our integration).
  Proposal: pi3 for change detection; Pi3X only if metric scale is needed without the phone (else ARKit gives scale).
- Pipeline: 3D-first core (pi3 -> robust ICP -> voxel carving -> SAM object masks -> DINOv2 moved matching) as the primary system, because it is the only one that yields 3D boxes, unobserved space and a costmap (D9), and it found 5 of 6 changes on pair01 vs SceneDiff's 3. SceneDiff stays as a second (appearance) channel through fusion.
- Not yet proven: that 3D-first or fusion beats SceneDiff on held-out benchmark pairs (5 pairs is too few). Next test: 30 more SD-V val pairs, ~$6.5.

## D12. Added / removed first, "moved" is a bonus (2026-10-12, decided by user)
- For a robot, what matters is space that became blocked (added) or free (removed). "Moved" is a nicer description of a removed + added pair, not a separate need.
- Headline metric: obj/sc AP without change type. Typed AP is reported but not optimised.
- Keep the DINOv2 matcher (it also raised untyped AP, 25.5 -> 34.3, by merging duplicate pairs), but no further work on moved matching.
- Occupancy / costmap grid: not built until D11 is settled (user).
