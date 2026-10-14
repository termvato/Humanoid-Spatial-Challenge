# Design note: choices and tradeoffs

Short version of `decisions.md` (D1-D17) and `spikes.md` (S1-S23). Every choice below was settled by a test whose pass/fail rule was written before it ran; several of our own ideas failed their rule and were dropped.

## The user is a robot

The output is meant for a mobile robot that returns to a room (D9). That decided most of the design:

- **Changes are 3D boxes in a floor-aligned map frame, in metres**, not 2D masks. A robot plans in 3D.
- **Three states, not two.** A voxel is occupied, free, or *unobserved*. "I did not see it" is different from "it is gone". Unobserved regions get a suggested viewpoint ("stand here, look there").
- **ROS map_server maps** (before, after, change layer) so a navigation stack can load them directly.
- **Added / removed matter most** (space became blocked or free); "moved" is a removed + added pair that looks like the same object (D12, D13).

## Pipeline

1. **One reconstruction from both videos.** pi3 (feed-forward, no SLAM) runs on frames of both videos at once, so both visits land in one coordinate frame without a separate registration step. A robust ICP then removes the small offset left between the visits. Chosen over Pi3X, VGGT, VGGT-Omega, DA3 and MapAnything by benchmark change-detection AP, not by geometry scores (S5, S10, S12): Pi3X had the best geometry but worse change detection.
2. **Voxel carving.** A camera ray that passes through a voxel proves it empty. Comparing occupied / free / unobserved per visit gives added (free before, occupied after), removed (the reverse) and unobserved. Lighting changes do not create geometry, so the big false-alarm source of 2D methods (S2) mostly disappears. Conservative carving (a voxel is free only if the whole voxel was seen past, with a margin along the surface normal) halves false alarms on our rooms (D16).
3. **SAM outlines.** Projected voxel blobs are blobby; SAM, prompted by each blob, gives clean object masks. SAM may only sharpen a detection, never create one.
4. **Moved = removed + added that look alike.** DINOv2 features of the two blobs, matched one-to-one. Fragments of one object are merged first (S18).
5. **Names and a confidence check.** Qwen2.5-VL-7B names each change from its sharpest crops (S19). It is also shown the same 3D spot from the other video and asked whether something is missing; a majority "no" over three view pairs marks the change low-confidence. It is listed last, never deleted (S16, S17, S20).
6. **Robot maps and viewpoints** from the same voxel map.

## Guarding the input

Two videos of different rooms would still produce a "change list". The run first compares DINOv2 image features of both videos and refuses below a threshold that every same-room pair cleared with margin (S22). After reconstruction, a low 3D overlap between the visits adds a warning (S23). Neither check passed its own strict rule on its own, so the first only refuses clear cases and the second only warns.

## Results, honestly

- **Our rooms** (phone video, strong lighting change in pair01): pair01 finds 5 of 6 changes (the pen is never detected) with 1 confident false alarm; pair02 detects 4 of 4, 3 confident (the white shoe is judged unchanged by the VLM), 2 confident false alarms. SceneDiff, the ECCV 2026 baseline, found 3 of 6 and 3 of 4.
- **SceneDiff benchmark** (official evaluator, object-level AP, 5 held-out pairs, 13 objects): ours 23.1 untyped / 21.4 typed vs SceneDiff 32.0 / 15.8. Better at saying *what kind* of change, worse at finding small objects. 5 pairs is too few to rank either.
- Our rooms were used to develop the false-alarm steps, so they are development data, not a clean test.

## Tradeoffs we chose

- **Offline, two visits.** Minutes per pair on an A100. Real time and long-term memory are product work, not a 4-day challenge.
- **Scale from phone height.** Plain video has no metric scale; we assume the phone is held 1.5 m above the floor (1.9% too large on pair01, checked against ARKit; one room). A robot with depth or odometry would give exact scale.
- **The VLM is the weak link.** Its yes/no answer flips with the wording or a few pixels of crop (S19-S21). We stopped tuning it on 2 rooms (D17); it ranks changes but never deletes one.
- **Mirrors** fool every depth source, including LiDAR (S11). Not handled; see `ideas.md`.
- **Frame budget.** pi3 memory grows with frames (16 GB for 243). The public demo caps 150 frames per video.

## What we tried and dropped

Keyframe selection (S6), grazing-angle cut (S13), minimum object size (S15), VLM as a hard filter (S16), a named-object before/after question (S21), synthetic Isaac Sim training data (I1, D14), a robot demo in Isaac Sim (D14).
