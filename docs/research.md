# Research notes (2026-10-11)

## Closest prior work: SceneDiff (ECCV 2026, arXiv 2512.16908)
- Task = ours: two videos of a scene, find added/removed/moved objects, ignore viewpoint/occlusion/lighting.
- Benchmark: 350 video pairs, 50 scenes, 1,009 changed objects. SD-V (200 pairs, 7 people, varied scenes), SD-K (150 egocentric kitchen pairs from HD-Epic). Dense 2D instance masks in all visible frames + change type. Splits: val 50/subset, test 250 total. No train split.
- Method (training-free): pi3 (depth+pose+intrinsics) -> co-visible frame pairs -> change score = depth reprojection + DINOv3 feature reprojection + DINOv3 region matching over SAM masks -> max-entropy threshold -> merge to instances -> match across sequences -> Added/Removed/Moved.
- Main metric obj/sc AP: SD-V 22.8, SD-K 10.6 (best baselines 5.3, 2.1). Low absolute numbers = room to improve.
- Limitations they state: identical-object swaps, pose failure (low light, textureless, low overlap), no semantic state changes, people must be masked manually.
- Gaps vs the Humanoid brief: output is 2D masks (brief wants a shared 3D representation + 3D viz), no language descriptions, hand-set thresholds (no learning), no explicit "unobserved" state, no navigation output.
- Code: github.com/yuqunw/scene_diff, MIT, pushed 2026-08-29. Data: huggingface.co/datasets/yuqun/SceneDiff, MIT, not gated, single zip.

## Other change-detection work
- "Has Anything Changed?" (arXiv 2312.01148, 2023): two posed 3D scans, 2D masks lifted to 3D.
- 3DGS-CD (RA-L 2025, arXiv 2411.03706): Gaussian-splat change detection for rearrangement.
- Not yet read: arXiv 2512.22830 (3D scene change modelling), 2512.00534 (cross-temporal 3DGS update).

## Reconstruction models (self-reported numbers; no neutral leaderboard)
- Depth Anything 3 (arXiv 2511.10647, ICLR 2026): claims +35.7% pose / +23.6% geometry over VGGT (ICLR version). DA3-Large ~1/3 VGGT params.
- pi3: permutation-equivariant (no reference frame). Used by SceneDiff. Close to VGGT, wins indoor pointmaps in one study.
- VGGT: common baseline, first frame is reference.
- MapAnything (arXiv 2509.13414): accepts optional poses/intrinsics; best fit for the LiDAR/ARKit path (D1).
- Plan: run 2-3 on our own captures, score against ARKit poses (D1). Don't trust paper numbers.

## Segmentation
- SAM 3 (arXiv 2511.16719, Meta, 2025-11-19): text-prompted open-vocab instance segmentation in images and video. SAM License (research-oriented; read before use). PyTorch >= 2.7, Python >= 3.12. SAM 3.1 noted 2026-03-27, details unknown.

## Co-training evidence: see docs/ideas.md I2.
