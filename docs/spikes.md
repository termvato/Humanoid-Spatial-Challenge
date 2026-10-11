# Spikes: cheap tests before committing (2026-10-11)

Each spike has a pass/fail rule written BEFORE running it, so the result decides, not our hopes.

| ID | Question | Test | Decision rule | Cost |
|---|---|---|---|---|
| S1 | Is the SceneDiff baseline real and runnable? | Run SceneDiff on Modal on 5 SD-V val pairs | Runs end to end and per-pair scores are in the paper's ballpark -> fork it. Fails after 3 h of effort -> build our own lean pipeline (report option 2) | ~$2, 2-3 h |
| S2 | Does it work on OUR phone data? | Record 1 pair with Stray Scanner (3 changes), run SceneDiff | Finds >= 2 of 3 changes -> phone pipeline viable as is. Finds < 2 -> diagnose reconstruction vs segmentation first | 20 min capture + 30 min |
| S3 | What is Isaac Sim's role? | Render one furnished room on the laptop, headless: fps, VRAM, segmentation quality | >= 5 fps clean instance masks at 640x480 -> synthetic pairs feasible overnight. Else Isaac = robot demo only | 1-2 h |
| S4 | Keyless cost | Same 5 val pairs with SAM 2 + DINOv2 vs SAM 3 + DINOv3 | Gap < 10% relative -> keyless default. Larger -> gated models default, keyless as fallback | ~$2 |
| S5 | Which reconstruction model? | pi3X vs DA3 poses on S2 pair, scored against ARKit (ATE after Sim(3)) | Lowest ATE wins default | ~$1, day 2 |

## Results

### S3: PASS (2026-10-11)
- Office scene, 640x480, RaytracedLighting, headless, RTX 4070 laptop: **9.4 fps mean** (p50 103 ms, p95 159 ms), **peak VRAM 4.4 GB**, 86 instances per frame, masks clean (chairs, plants, rope posts separated). First load 181 s (asset streaming; cached afterwards).
- Throughput estimate: a pair = 2 x ~150 frames = ~32 s render, so ~100 pairs/hour; 300 pairs overnight on the laptop alone. The 3080 PC is not needed (D7 stays deferred).
- The deep-research "8 GB is below the 16 GB minimum" concern does not hold for this workload.
- Caveats: one stock scene, orbit camera, stock lighting. The real cost is the room generator (placing, moving, adding, removing objects plausibly), not rendering. Whether synthetic data is worth it at all is still decided after S1 (real labelled SceneDiff val pairs exist).
- Artifacts: outputs/s3/ (result_office.json, rgb_*.png, inst_*.png).

### S1: PASS (2026-10-11)
- SceneDiff (commit 62dcbd2, default config, DINOv3 ViT-H+) on the first 5 SD-V val pairs, current evaluator (mask IoU matching, dup thresholds 1):

| Metric | Ours, 5 val pairs | Paper, SD-V test |
|---|---|---|
| px/im IoU | 18.7 | 39.1 |
| obj/im AP | 44.6 | 43.8 |
| obj/sc AP, no change type | 32.0 | 22.8 (older protocol) |
| obj/sc AP, with change type | **15.8** | n/a |

- Detections: **7 TP, 30 FP, 6 FN** over 13 changed objects. 5 pairs is a tiny sample; treat numbers as ballpark only.
- Rule met: runs end to end, numbers in the paper's ballpark -> **fork SceneDiff**.
- Failure profile (what our heads must fix): false positives outnumber true positives 4:1 (target of head A, verifier); requiring the right change type halves AP (target of head B, matcher); about half the changes are missed (recall, not fixable by A or B).
- Cost: ~8.5 min/pair on A100-40GB. SAM = 79% (2010 of 2540 s), pi3 6%, cost map 6%, voxelisation 8%. SAM output is cached per pair (`sam_masks.pkl`), so it is paid once per pair.
- Artifacts: outputs/s1/ (log, metrics), Modal volume scenediff:/outputs/s1/pred.
- Follow-up (s1_cpu8, pair 6 `bedroom_2_bedroom_3`, cpu=8): **259 s/pair vs 508 s/pair** average before (SAM 206 s, pi3 12 s, cost map 23 s, voxels 12 s). Different pair, so roughly 2x, not exact. The default fraction-of-a-core CPU was throttling every stage.
- Cost estimate at ~4.3 min/pair on A100-40GB: ~$0.18/pair. SD-V val 50 ~ $9, SD-V test 150 ~ $27, all 350 ~ $63.
- Baseline must run at SceneDiff's default config (no SAM shortcuts) to be a fair comparison.

### S2: borderline FAIL on the written rule, pipeline viable (2026-10-11)
- pair01 (user's room, iPhone, Stray Scanner; strong light change warm -> daylight). SceneDiff default config, resample 90 (14 frames per video, benchmark-matched; 40 frames OOMed at stage 7).
- Rule was ">= 2 of 3 changes"; scaled to 6 changes = >= 4. Found 3, of which 2 fully correct:

| Change | Result |
|---|---|
| little suitcase, moved | found in both videos and matched (correct type) |
| black pouch, added | found (correct) |
| blue tool, moved | found only in after -> reported as added (wrong type) |
| hat, removed (wardrobe top) | missed. No sampled "before" frame shows the wardrobe top clearly: an UNOBSERVED case, not a detector failure |
| pen, moved (floor) | missed. A few pixels at 518 px input |
| blanket, removed (on suitcase) | missed. Visible in both videos (v1 frames 1-2, v2 frames 6-7): a genuine miss |

- False alarms: 4. P2 = the user's finger over the lens in before frames 7-9 (capture artifact); P0, P3, P6 = wall and curtain edges (light change).
- Fail-branch diagnosis (as the rule required): reconstruction and alignment are fine (suitcase matched across sessions). Failures are coverage (hat), object size vs input resolution (pen), appearance under a light change (blanket), and capture hygiene (finger).
- Implications: (1) uniform 1-in-90 sampling wastes frames; choose keyframes by sharpness and coverage. (2) The hat is the poster case for an explicit "unobserved" output. (3) The verifier (head A) would target the edge and finger false alarms. (4) Capture guide: keep fingers off the lens; pause on each changed area.
- Artifacts: outputs/s2/pair01.png, outputs/s2/sampled_frames.jpg, outputs/s2/photos_grid.jpg. Cost ~$0.70 including two failed runs.
- **Correction (2026-10-11, during S6):** the hat WAS seen. Before frames ~600-630 show the hats on the wardrobe, but the finger covers the bottom of the frame from ~590 to ~970 (13 s), and uniform sampling picked exactly those frames. So the hat miss is sampling plus occlusion, not "never observed".

### S7: LiDAR 3-state change map on pair01 (2026-10-11)
No pass/fail rule was written before running, so this is a result without a verdict.
- Method (`src/roomdiff/visibility.py`, `align.py`, `tools/visibility_map.py`): register the two LiDAR clouds (gravity-aware yaw sweep with FFT shift + 4-DoF ICP), carve 5 cm voxels with every 5th depth frame (hit / free / unknown), compare: removed, added, unchanged, unobserved. Laptop CPU, 44 s.
- Bugs found on the way, each fixed with a test:
  1. Generic 6-DoF FPFH+RANSAC flipped the room (168 deg tilt). Gravity-aware search fixes it.
  2. Unconstrained ICP tilted 6 deg (floor ~9% of points). 4-DoF ICP fixes it.
  3. **Stray poses are already OpenCV camera axes.** The ARKit flip in our loader smeared every map (frames 1 s apart disagreed by 18 cm; 2 cm without the flip). Regression test on the real capture added.
- Result after fixes: registration fitness 0.88 at 5 cm, rmse 2.4 cm. Voxels: 22,706 unchanged, 183 removed, 292 added, 13,141 unobserved.

| Change | LiDAR map |
|---|---|
| little suitcase, moved | **found**: removed blob 17 L + added blob 23 L, ~70 cm apart |
| blanket, removed | partial: 1.0 L removed blob on top of the suitcase (only at min blob 4 voxels) |
| hat, removed | wardrobe top falls in the **unobserved** band (LiDAR saw little of it in before) |
| black pouch, added | missed (below blob size / not resolved) |
| pen, blue tool | missed, far below 5 cm voxels |

- False alarms: 2 at min blob 15 voxels (cornice, wardrobe edge), 9 at min blob 4. All on thin edges.
- Takeaway: geometry and appearance fail differently. LiDAR catches big objects regardless of lighting and gives honest "unobserved"; SceneDiff (appearance) catches small objects but is fooled by light. Fusing the two is the case for head A.
- Artifacts: outputs/s7/ (bev.png, blobs.jpg, labels.npz, map.ply, result.json), outputs/s7_min4/.
