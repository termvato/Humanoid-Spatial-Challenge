# Reconstruction, Cross-Session Alignment, and Visibility Reasoning for Two-Video Room Change Detection (as of October 2026)

Scope note: about 20 tool calls were used. Several claims could only be confirmed from the model authors' own repos or papers; these are flagged "(self-reported)". Items drawn from general background knowledge rather than a fetched source sit under Inferences or Gaps, never under Cited Findings.

## Q1. Feed-forward reconstruction models: frame limits, memory, metric scale, independent accuracy, licence, code

### Takeaway
For a commercially clean build, the best options are MapAnything's Apache weights (metric, accepts ARKit poses, intrinsics and depth as inputs) and DA3's Apache-licensed variants (DA3-BASE/SMALL, DA3METRIC-LARGE). For research-only use, DA3-Giant/Large and Pi3/Pi3X are the strongest in independent tests. In those tests VGGT-1B had weak pose accuracy on hard data, and all of these models need chunking past about 100-400 frames per pass.

### Cited Findings
**Depth Anything 3 (DA3, ByteDance Seed, ICLR 2026)**
- Variants and licences: DA3NESTED-GIANT-LARGE (1.40B), DA3-GIANT (1.15B) and DA3-LARGE (0.35B) are CC BY-NC 4.0. DA3-BASE (0.12B), DA3-SMALL (0.08B), DA3METRIC-LARGE and DA3MONO-LARGE are Apache 2.0. "-1.1" checkpoints were retrained after a training bug fix, and the originals are deprecated ([DA3 GitHub](https://github.com/ByteDance-Seed/Depth-Anything-3)).
- Metric scale: DA3NESTED-GIANT-LARGE outputs meters directly. DA3METRIC-LARGE gives metric depth via `metric_depth = focal * net_output / 300` with focal in pixels ([DA3 GitHub](https://github.com/ByteDance-Seed/Depth-Anything-3)).
- Long video: DA3-Streaming (released 11 Dec 2025) does sliding-window inference on ultra-long video in under 12 GB GPU memory. The model supports pose-conditioned depth (camera poses as input) and `use_ray_pose` (slower, more accurate). It exports glb, npz, ply and depth images ([DA3 GitHub](https://github.com/ByteDance-Seed/Depth-Anything-3)).
- Self-reported: DA3-Giant beats VGGT on HiRoom pose Auc3 (80.3 vs VGGT 49.1 vs Pi3 67.0). The claimed average gain over VGGT is 35.7% (pose) / 23.6% (geometry) in one paper version, versus 44.3% / 25.1% on the HF card, so the figures vary by revision ([DA3 arXiv 2511.10647](https://arxiv.org/pdf/2511.10647); [DA3-BASE model card](https://huggingface.co/depth-anything/DA3-BASE)).

**VGGT (Meta, CVPR 2025 Best Paper)**
- Licence: since 29 Jul 2025 the code allows commercial use (excluding military). Only the "VGGT-1B-Commercial" checkpoint (gated application form) is commercial, and the original checkpoint stays non-commercial ([VGGT GitHub](https://github.com/facebookresearch/vggt)).
- A 15 May 2026 memory fix removed redundant intermediate tensors and allows "roughly 2-3x more input frames" under the same budget. The README states no absolute frame cap ([VGGT GitHub](https://github.com/facebookresearch/vggt)).
- VGGT-Omega (arXiv 2605.15195, CVPR 2026 Oral; README update 18 May 2026) replaces some global attention with "register attention" and uses about 30% of VGGT's training memory. It claims a 77% camera-accuracy improvement on Sintel (self-reported). The abstract states no frame limit, metric-scale behaviour or licence ([arXiv abs](https://arxiv.org/abs/2605.15195); [VGGT GitHub](https://github.com/facebookresearch/vggt)).

**Pi3 / Pi3X (permutation-equivariant, no reference frame)**
- Code is BSD-3-Clause. Pi3 and Pi3X weights are CC BY-NC 4.0, "strictly non-commercial" ([Pi3 GitHub](https://github.com/yyfz/Pi3)).
- Original Pi3 is scale-invariant. Pi3X (28 Dec 2025, now recommended) supports approximate metric scale, optional conditioning on poses, intrinsics and depth, and gives more reliable confidence. Video input defaults to `--interval 10`. The README gives no memory figures ([Pi3 GitHub](https://github.com/yyfz/Pi3)).

**MapAnything (Meta + CMU, 3DV 2026, arXiv 2509.13414)**
- Code is Apache 2.0. Weights `facebook/map-anything` are CC-BY-NC 4.0 (best performance), and `facebook/map-anything-apache` is Apache 2.0 for commercial use ([MapAnything GitHub](https://github.com/facebookresearch/map-anything)).
- Metric outputs include `metric_scaling_factor`. Optional inputs: intrinsics or ray directions, `depth_z`, and `camera_poses` (OpenCV cam2world, 4x4 or quat+trans), with an `is_metric_scale` flag. If any view has a pose, the first view must too ([MapAnything GitHub](https://github.com/facebookresearch/map-anything)).
- Memory: with `memory_efficient_inference=True` it handles "up to 2000 views on 140 GB" (self-reported), with `minibatch_size=1` for the smallest footprint and bf16 AMP ([MapAnything GitHub](https://github.com/facebookresearch/map-anything)).

**MASt3R / MASt3R-SfM (Naver)**
- Code is CC BY-NC-SA 4.0. Checkpoints also inherit the training datasets' licences, and the mapfree dataset licence is flagged as very restrictive. Treat both as non-commercial ([MASt3R HF card](https://huggingface.co/camenduru/mast3r); [MASt3R-SfM arXiv](https://arxiv.org/abs/2409.19152v1)).

**CUT3R**: a Trendshift listing shows a "Custom license" for the repo. Terms are not verified ([Trendshift](https://trendshift.io/repositories/15936)).

**Independent comparisons**
- Asset-inspection benchmark (Gray et al., Univ. of Sydney, arXiv 2603.17358; not the model developers): batch limits used were DA3 400, VGGT 200 and Pi3 100 images per pass. "Current state-of-the-art end-to-end methods ... cannot process the full trajectories ... due to large VRAM requirements." They ran sequential batches of N in [100, 400] and Sim(3)-aligned each batch to GT with evo ([arXiv 2603.17358](https://arxiv.org/pdf/2603.17358)).
- Same benchmark, pose AUC30 (Office scene, 4 conditions): DA3 0.549-0.560, Pi3 0.537-0.538, VGGT 0.121-0.137, COLMAP 0.901-0.951, GLOMAP 0.709-0.849. On Crane: DA3 0.223, Pi3 0.272, VGGT 0.043, COLMAP 0.736. "VGGT performed poorly", while DA3 and Pi3 "performed relatively well". COLMAP and GLOMAP benefited from the continuous-video prior ([arXiv 2603.17358](https://arxiv.org/pdf/2603.17358)).
- Same benchmark: the authors say the feed-forward models do well where each view has long sight lines covering much of the scene, and poorly where each view sees only a small part. Note: a room walk-through is closer to the favourable case ([arXiv 2603.17358](https://arxiv.org/pdf/2603.17358)).
- UAVFF3D (arXiv 2605.17942) benchmarks VGGT, Pi3, Pi3X, MapAnything and DA3 on UAV imagery. Zero-shot ATE: Pi3 60.39, VGGT 54.29. A fine-tuned Pi3 reaches 14.52. DA3 was only run zero-shot because its training code is not public. Full tables were not read; numbers are from a search summary ([UAVFF3D](https://arxiv.org/html/2605.17942v1)).
- DejaView (arXiv 2605.30215), a method paper so not neutral, uses official checkpoints of VGGT, Pi3, MapAnything, DA3, MASt3R and MASt3R-SfM. It calls Pi3 the strongest prior on indoor pointmap accuracy, with VGGT best on DTU pose at the tightest threshold ([DejaView](https://arxiv.org/pdf/2605.30215)).

### Inferences
- A 3-4 day build that might go commercial should use MapAnything-apache or DA3 Apache variants. If research-only, DA3-Giant or Pi3X are the strongest picks on independent evidence.
- Because ARKit gives poses and intrinsics, MapAnything or Pi3X with pose and intrinsics conditioning is the most direct way to get dense metric depth that already agrees with ARKit's frame. DA3 pose-conditioned depth is the alternative.
- The VGGT-1B pose weakness in the independent benchmark agrees with the DA3 self-reported gap, which makes VGGT-1B a poor default for pose.

### Gaps
- No single independent indoor benchmark (e.g., ScanNet++ under one protocol) covering DA3, Pi3X, MapAnything, VGGT-Omega and MASt3R-SfM was found.
- Concrete GPU memory per frame count for DA3, Pi3 and VGGT (after the May 2026 fix) was not found in the READMEs.
- VGGT-Omega code and licence status were not confirmed.
- MapAnything's own pose numbers and SpatialBench (arXiv 2605.27367) rankings were not read.

## Q2. Handling hundreds of frames: keyframes, chunking, global alignment

### Takeaway
Subsample to about 40-150 sharp keyframes per video. If more frames are needed, chunk with overlap and Sim(3)-align the chunks (VGGT-Long style) or use DA3-Streaming. For a room-scale video of 30-90 s, one pass of up to about 200-400 frames per model fits the independent benchmark's batch sizes.

### Cited Findings
- VGGT-Long splits sequences into overlapping chunks (about 60-75 frames), aligns adjacent chunks with IRLS, detects loops, and runs a global Sim(3) optimisation. Plain VGGT runs out of memory on every vKITTI scene, while VGGT-Long completes them ([VGGT-Long arXiv 2507.16443](https://arxiv.org/pdf/2507.16443)).
- CUT3R-style recurrent methods are memory-bounded but drift on long sequences. StreamVGGT memory grows with length and hits OOM on extended sequences ([VGGT-Long](https://arxiv.org/pdf/2507.16443); [LongStream arXiv 2602.13172](https://arxiv.org/pdf/2602.13172)).
- On a 10,000+ frame benchmark (HorizonStream), VGGT-Long scores about 80.43 average error, VGGT-SLAM 78.20 and LoGeR 28.02. These numbers come from different protocols across papers ([HorizonStream arXiv 2605.23889](https://arxiv.org/pdf/2605.23889)).
- Other 2026 long-sequence work: LongStream, RetrieveVGGT (training-free KV retrieval), InfiniteVGGT, VGGT-SLAM 2.0, and FastVGGT (training-free token-merging acceleration) ([RetrieveVGGT](https://arxiv.org/pdf/2605.09644); [VGGT-SLAM 2.0](https://arxiv.org/pdf/2601.19887); [FastVGGT](https://arxiv.org/pdf/2509.02560)).
- DA3-Streaming: sliding window, under 12 GB ([DA3 GitHub](https://github.com/ByteDance-Seed/Depth-Anything-3)).
- Practitioner keyframe recipe (room-diff): split each video into time bins and keep the sharpest frame by Laplacian variance. Defaults are 40 keyframes per video, or 60 for large rooms ([room-diff](https://github.com/viv344/room-diff)).
- Independent benchmark recipe: sequential batches of 100-400 frames, each Sim(3)-aligned to a global frame ([arXiv 2603.17358](https://arxiv.org/pdf/2603.17358)).

### Inferences
- With ARKit poses available, the cleanest chunk alignment uses ARKit as the global frame: Sim(3)-align each chunk's predicted camera centres to the ARKit camera centres (Umeyama). This avoids drift accumulation entirely.
- Recommended budget: about 1-2 keyframes per second, filtered by blur score and a minimum ARKit baseline or rotation delta between keyframes.

### Gaps
- No independent study of keyframe density vs accuracy for these models on indoor phone video was found.

## Q3. Cross-session alignment: joint vs separate, registration methods, robustness to change

### Takeaway
The recent change-detection papers align sessions jointly or by relocalisation against the static majority: SceneDiff runs Pi3 on both videos together, room-diff runs one COLMAP SfM over both, and 3DGS-CD localises post-change images in the pre-change SfM map. All of these depend on RANSAC or robust losses to reject changed regions. Classical 3D descriptor registration is known to degrade under object changes.

### Cited Findings
- SceneDiff (UIUC + Meta, arXiv 2512.16908): runs Pi3 jointly on the before and after videos in one pass to get depth, pose and intrinsics for every frame, then normalises to a unit cube. It pairs frames by co-visibility: over 50% bidirectional co-visibility, or the single best after-frame. It is training-free. Benchmark: 350 video pairs with instance-level annotations ([SceneDiff arXiv v1](https://arxiv.org/html/2512.16908v1)).
- SceneDiff results: SD-V per-view AP 49.6 vs CYWS-3D 26.9. RC-3D two-view AP50 68.7 vs CYWS-3D 41.0 (50.0 with sensor depth). ChangeSim F1 57.6 vs GeSCF 54.8. 3DGS-CD scored near zero on the SceneDiff benchmark. Relative AP improvement figures differ between v1 and v2 of the paper ([SceneDiff v1](https://arxiv.org/html/2512.16908v1); [arXiv v2](https://arxiv.org/pdf/2512.16908v2)).
- 3DGS-CD (arXiv 2411.03706): trains 3DGS on the pre-change images (SfM frame), then localises post-change images with HLoc against the pre-change SfM points, assuming most of the scene is unchanged. Poses are refined by render-and-compare with changed regions masked out of the loss. It works with as few as 1 post-change image (about 8 used). It takes about 18-58 s per scene excluding training, with F1 96.4-98.8 on its own dataset (self-reported) ([3DGS-CD](https://arxiv.org/html/2411.03706)).
- room-diff (practitioner, not peer reviewed): one COLMAP SfM over both videos' keyframes (SIFT, exhaustive matching, one camera model per video). The static majority links the sessions, and moved objects fall out as RANSAC outliers. Scale comes from monocular metric depth (Depth Anything V2) and is about 17% small on the synthetic test. Floor plane sets gravity and z=0. It stops if the videos fail to register ([room-diff](https://github.com/viv344/room-diff)).
- RIO / 3RScan (ICCV 2019): 1482 RGB-D scans of 478 environments with object changes between rescans. Hand-crafted descriptors (FPFH, SHOT) and 3DMatch-style learned descriptors trained on static scenes are "very susceptible to geometry changes". RIO's instance re-localisation reached 30.58% accuracy, which shows how hard the problem is ([RIO paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wald_RIO_3D_Object_Instance_Re-Localization_in_Changing_Indoor_Environments_ICCV_2019_paper.pdf)).
- Living Scenes (MORE2): failures on 3RScan come from many similar objects (pillows) and low rescan coverage. Instances not matched in the target are treated as removed ([Living Scenes arXiv 2312.09138](https://arxiv.org/pdf/2312.09138)).

### Inferences
- Recommended pipeline: (1) reconstruct each session in its own ARKit frame, which is metric and gravity-aligned since ARKit's y axis follows gravity. (2) Get a coarse cross-session transform with yaw + translation only, using 4-DoF RANSAC on 2D-2D matches lifted to 3D with LiDAR depth (e.g., LightGlue or MASt3R matches between co-visible frames), or via a joint feed-forward pass on a subset of keyframes from both sessions. (3) Refine with robust point-to-plane ICP (Tukey/Huber, tight correspondence distance) restricted to structural surfaces: floor, walls, ceiling, large furniture. Changed objects are then outliers.
- The gravity constraint shrinks the problem to 4 DoF, which makes RANSAC very robust even at 30% change. This rests on background knowledge of ARKit's gravity-aligned world frame and is not verified in a fetched source.
- Joint pass vs separate: SceneDiff's joint Pi3 pass shows that joint inference works for the two videos together. However, it yields non-metric, normalised geometry, and Pi3 weights are non-commercial. Separate ARKit-framed reconstructions plus registration keep metric scale for free.
- GeoTransformer and similar learned registration are trained on static 3DMatch-like data. Following RIO's argument, expect degradation under change, though this was not verified for GeoTransformer specifically.

### Gaps
- No source quantified alignment error as a function of the percentage of scene changed (10-30%). No GeoTransformer evaluation on changed scenes was found.
- SceneDiff's GPU and runtime section (8.2) and frames per joint run were not read.

## Q4. Visibility and free space: separating "removed" from "unobserved"

### Takeaway
The robust rule across the literature is: a surface counts as removed only if the other session positively observed free space through it (ray passes beyond the old surface). Occluded or out-of-frustum points are "unknown", not changed. This gives a natural 3-state output (changed / unchanged / unobserved).

### Cited Findings
- Underwood et al. (ICRA 2013): earlier methods "implicitly assume that unmeasured space is unoccupied". Ray tracing the beam's free-space path, not just endpoints, reasons explicitly about occlusion and incomplete coverage ([Underwood ICRA13](https://www.acfr.usyd.edu.au/papers/pdfs/Underwood-ICRA13-ChangeDetection.pdf)).
- LiDAR change mapping finds disappearances where ray casting shows a map voxel was observed empty, and appearances from new points against occupied voxels, with temporal filtering to remove ray-cast errors ([ResearchGate: Reliable real-time change detection](https://www.researchgate.net/publication/320747288_Reliable_real-time_change_detection_and_mapping_for_3D_LiDARs)).
- SceneDiff uses an asymmetric depth rule: it keeps only pixels where the before point is in front of the after surface (after depth larger), which suggests an object present before and absent after. It treats the opposite sign as ambiguous because it may come from occlusion. Regions more than 60% masked by visibility are excluded, and view order is swapped to detect additions ([SceneDiff v1](https://arxiv.org/html/2512.16908v1)).
- room-diff 4-way per-frame vote. Consistent: surface seen at that depth. Free: frame sees through the point, the only change evidence. Occluded: something in front, no evidence. Unobserved: outside the frustum, no evidence. A point is changed only with at least 3 free votes, at least 70% free ratio, and no nearby point in the other cloud. Depth tolerance grows with distance. Fusion uses 2 cm voxels kept if at least 2 frames agree. Vetoes include an SfM-anchor veto and a photometric veto ([room-diff](https://github.com/viv344/room-diff)).
- POCD (probabilistic object-level change detection) gives objects that should be visible but are not a large pseudo-change, and models viewing angle, occlusion and partial observation ([POCD arXiv 2205.01202](https://arxiv.org/pdf/2205.01202)).
- 3DGS-CD fuses object masks only from views where at least 80% of projected object points are visible, checks pixel-wise against rendered depth, and may fail if changes are heavily occluded in all post-change views ([3DGS-CD](https://arxiv.org/html/2411.03706)).
- "Has Anything Changed?" detects moving points via depth render-and-compare ([arXiv 2312.01148](https://arxiv.org/html/2312.01148)).
- nvblox (NVIDIA): a CUDA TSDF/ESDF mapping library with `nvblox_torch` Python bindings. The README fetched did not describe unknown/free/occupied exposure or the licence ([nvblox GitHub](https://github.com/nvidia-isaac/nvblox)).

### Inferences
- Concrete 3-state formulation for the build: voxelise each session at 2-5 cm with log-odds occupancy or TSDF weight. Mark a voxel "observed" if any ray in that session passed through or ended in it. For a surface voxel from session A: if session B observed it occupied, label unchanged. If B marked it free with enough rays (e.g., at least 3, with free ratio at least 0.7 as in room-diff), label removed. If B never observed it, or it was occluded, label unknown. Symmetric for added.
- With LiDAR depth (192x256, see Q5), free-space carving by ray marching is cheap. Open3D VoxelBlockGrid or a simple numpy depth-reprojection test (project A's points into B's depth maps, compare depths with a distance-dependent tolerance) gives the same answer as ray casting for the per-frame tests above. This is background knowledge, not from a fetched source.
- Masking low-confidence LiDAR pixels (confidence < 2) and depth discontinuities ("flying pixels") is important, since false "free" votes come from depth edge errors.

### Gaps
- No 2025-2026 peer-reviewed paper was found that formally outputs a 3-state changed / unchanged / unknown map for phone video. room-diff and POCD are the closest. OctoMap/voxblox specifics were not fetched.

## Q5. Metric scale recovery and ARKit/LiDAR as ground truth

### Takeaway
ARKit gives metric, gravity-aligned poses, and LiDAR gives metric depth. Use them either as the reference to Sim(3)-align and evaluate model output (ATE/RPE via evo, AbsRel on LiDAR depth) or as conditioning inputs (MapAnything, Pi3X, DA3 pose-conditioned). Monocular-only metric scale has errors of order 10-20% (room-diff observed about 17%).

### Cited Findings
- Stray Scanner format (unofficial data-model page): per-frame depth `.npy`, 192x256 uint16 in millimetres. `camera_matrix.csv` holds the 3x3 intrinsics. `odometry.csv` has one row per frame with position in metres plus quaternion (qx, qy, qz, qw). Depth confidence maps are included ([Stray data model](https://excellent-legal-071.notion.site/Stray-Scanner-Data-Model-aea95b702b484d41b2de12d2e90781ce); [App Store](https://apps.apple.com/app/id1557051662)).
- StrayVisualizer: confidence levels 0, 1 and 2, with a `--confidence` filter. `--integrate` runs Open3D RGB-D integration with `--voxel-size` ([StrayVisualizer](https://github.com/kekeblom/StrayVisualizer)).
- The independent benchmark aligned each batch to GT with Sim(3) using the evo toolbox, and evaluated depth with scale-invariant MSE and poses with AUC ([arXiv 2603.17358](https://arxiv.org/pdf/2603.17358)).
- MapAnything accepts metric depth, intrinsics and poses as inputs and outputs a metric scaling factor. DA3NESTED outputs meters, and DA3METRIC needs focal/300 scaling. Pi3X gives approximate metric scale ([MapAnything](https://github.com/facebookresearch/map-anything); [DA3](https://github.com/ByteDance-Seed/Depth-Anything-3); [Pi3](https://github.com/yyfz/Pi3)).
- room-diff's monocular-depth scale was about 17% small on its synthetic scene ([room-diff](https://github.com/viv344/room-diff)).

### Inferences
- Evaluation protocol: (a) pose: Sim(3) Umeyama of predicted camera centres to ARKit, then report ATE RMSE and RPE, plus the recovered scale factor (scale error = |s-1|) for "metric" models. (b) depth: AbsRel and delta<1.25 against LiDAR depth at confidence 2, after resizing predictions to 256x192. (c) cross-session: if both sessions start from ARKit, compare the registration transform with one from a manually picked set of correspondences.
- Caveat to verify: the unofficial page says z points forward. ARKit camera coordinates natively follow an OpenGL-style convention (z backward, y up), whereas MapAnything and Pi3 expect OpenCV cam2world. Checking a flip like diag(1,-1,-1) on sample data is a must-do. This is background knowledge, not verified in a fetched source.
- ARKit odometry is not ground truth in a strict sense: VIO drifts slowly over a 1-2 minute room walk. Treat it as a strong reference rather than truth. Record3D (.r3d) exports similar per-frame depth and pose data, but its format was not fetched.

### Gaps
- Record3D file format details and Stray Scanner's exact odometry.csv header (timestamp/frame columns) and camera axis convention were not confirmed from the official sources.
- No published numbers on ARKit VIO drift for room-scale captures were gathered.
