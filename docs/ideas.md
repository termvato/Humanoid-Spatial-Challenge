# Ideas under discussion (not decided)

## I1. Isaac Sim procedural room pairs (proposed by user, 2026-10-11)
- Auto-generate rooms, apply known move/add/remove edits, render two camera trajectories per pair. Exact ground truth.
- Uses: (a) benchmark with precision/recall at scale, (b) training data, (c) generalisation measurement, (d) robot nav demo in the same env.
- Curriculum learning suggested.
- Open risks: sim-to-real gap vs the judged real phone videos; asset variety; Isaac render throughput on 8 GB; 3-4 day budget.
- Open question to user: what exactly would the model learn (input, output)?

## I2. Sim+real co-training (user, 2026-10-11)
- Evidence: Maddukuri et al., RSS 2025, arXiv 2503.24361: sim data improves real manipulation by 38% on average; best sim share ~99% on CupPnP, extreme shares (99.9%) hurt. More sim data helps. RoboMIND 2.0 (arXiv 2512.24653): 1:5 real:sim beat 1:1 on some tasks.
- Caveat: those are manipulation policies, not change detection. Transfer of the finding is a hypothesis to test, not a given.
- Requirement: several labelled real pairs, split train/test by room. The demo videos must be in the real TEST split.
- Experiment: sim-only vs real-only vs sim+real (ratio sweep), scored on held-out real pairs.
- User agreed: use state-of-the-art pretrained models, no reinventing.

## I3. 3RScan as backup source of real 3D ground truth (2026-10-11)
- 1,482 RGB-D rescans of 478 indoor scenes; per-object rigid transforms for changed objects, removed lists, consistent instance IDs, camera poses.
- Access GRANTED 2026-10-11. Official download script: gist WaldJohannaU/55f5e35992ea91157b789b15eac4d432 (copy in third_party/3rscan_dl/). Licence CC BY-NC-SA 4.0 (derived data must share alike). Full release ~94 GB; per-scan download possible with --id. Server redirects http -> https.
- Metadata (data/raw/3rscan/3RScan.json, 3 MB): 478 reference scenes, 1,004 rescans (982 with >= 1 rigid change), 3,323 rigid object moves with 3D transforms, 542 removed, 558 nonrigid. Splits: 385 train / 47 val / 46 test scenes.
- Versus SceneDiff val (100 pairs, 2D masks): ~10x more pairs, with real 3D motion ground truth. Main candidate if the heads need more or 3D-labelled real data.
- Use if: we need real 3D localisation GT, or SceneDiff val is too small for the heads. Not phone video (Tango RGB-D), so a domain gap of its own.

## Mirror and glass handling (later project, user 2026-10-12)
Mirrors make every depth source (pi3, Pi3X, LiDAR) see a fake room behind the glass; reflections change with viewpoint, so they cause phantom and doubled changes.
- Detect: a learned mirror segmenter on frames (mirror detection is an existing 2D task), or geometric cues: a planar frame (border) with "room" geometry behind a wall, LiDAR depth jumping past known walls, or points that are a reflection of real points across a plane.
- Use: treat the mirror plane as a surface, drop voxels behind it, and mark the reflected region "unobserved / unreliable" rather than changed.
- Test case: pair02 (corridor with a mirror), once recorded.
