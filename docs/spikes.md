# Spikes: cheap tests before committing (2026-10-11)

Each spike has a pass/fail rule written BEFORE running it, so the result decides, not our hopes.

| ID | Question | Test | Decision rule | Cost |
|---|---|---|---|---|
| S1 | Is the SceneDiff baseline real and runnable? | Run SceneDiff on Modal on 5 SD-V val pairs | Runs end to end and per-pair scores are in the paper's ballpark -> fork it. Fails after 3 h of effort -> build our own lean pipeline (report option 2) | ~$2, 2-3 h |
| S2 | Does it work on OUR phone data? | Record 1 pair with Stray Scanner (3 changes), run SceneDiff | Finds >= 2 of 3 changes -> phone pipeline viable as is. Finds < 2 -> diagnose reconstruction vs segmentation first | 20 min capture + 30 min |
| S3 | What is Isaac Sim's role? | Render one furnished room on the laptop, headless: fps, VRAM, segmentation quality | >= 5 fps clean instance masks at 640x480 -> synthetic pairs feasible overnight. Else Isaac = robot demo only | 1-2 h |
| S4 | Keyless cost | Same 5 val pairs with SAM 2 + DINOv2 vs SAM 3 + DINOv3 | Gap < 10% relative -> keyless default. Larger -> gated models default, keyless as fallback | ~$2 |
| S5 | Which reconstruction model? | pi3X vs DA3 poses on S2 pair, scored against ARKit (ATE after Sim(3)) | Lowest ATE wins default | ~$1, day 2 |
| S6 | Does frame choice fix S2's misses? | Rerun SceneDiff on pair01 with 14 keyframes per video chosen by LiDAR coverage x sharpness x no-finger (same count as S2, only the choice changes). Drop predictions >= 50% inside the finger mask | Finds >= 4 of 6 with <= 4 false alarms -> keyframe selection becomes default. Finds <= 3 -> frame choice is not the bottleneck; keep uniform, look at resolution and lighting | ~$0.25 |
| S7 | Can LiDAR geometry give a 3-state map? | Voxel carving of both captures after registration | (no rule written in advance; result only) | laptop |
| S9 | Is 3D-first viable on plain video (D10)? | pi3 jointly on both videos -> S7 carving with model depth. (a) pair01 vs the LiDAR map. (b) the 5 S1 val pairs, 3D blobs projected to 2D masks, official SceneDiff evaluator | (a) suitcase move found (both blobs) with <= 4 false-alarm blobs. (b) obj/sc AP (no type) >= 26 (0.8 x SceneDiff's 32.0) -> 3D-first core. 13-26 -> 3D core plus SceneDiff appearance as a second channel. < 13 -> keep SceneDiff primary, LiDAR-only 3D layer | ~$1 |
| S10 | Was Pi3X treated fairly? (its confidence is calibrated differently: a 0.1 cut keeps ~100% of its pixels vs 82-95% for pi3) | Tuning pairs, full pipeline (ICP, SAM, DINOv2 matching). First geometry + ICP sweep: drop the least confident 10 / 20 / 30% of pixels per scene, voxel median/60 and /80; best one gets the full pipeline | Full-pipeline Pi3X >= 34.3 (pi3) -> Pi3X becomes the one model (metric scale for free). 29.3-34.3 -> tie, pi3 stays (held-out data already exists for it), Pi3X used for scale. < 29.3 -> pi3 confirmed, gap is real | pennies (evaluator) |
| S12 | Does VGGT-Omega beat pi3? (newer model, reported robust to wide viewpoint spans) | Same frames as S5/S9 (stride 10, both videos jointly). (a) pair01 geometry vs ARKit + LiDAR. (b) 15 tuning pairs, geometry + ICP at median/80, then full pipeline if (b) geometry >= 17.9 | Full pipeline >= 34.3 (pi3) -> run the 5 held-out pairs; it replaces pi3 only if it also beats pi3 there (18.0). Otherwise pi3 stays. Geometry < 17.9 -> stop, pi3 stays | ~$2 |
| S13 | Does ignoring free-space evidence at grazing angles cut false alarms? (sim: 8 false "removed" on a table seen edge-on; real rooms: wall patches) | Free evidence only from pixels whose surface is seen at incidence <= 70 / 75 / 80 deg (hits kept). Tuning pairs, full pipeline; sim room; pair01 | Adopt the best angle only if tuning AP >= 34.3 (no loss) AND sim false alarms drop; then held-out 5 pairs must stay >= 18.0. Otherwise keep the pipeline as is | pennies + laptop SAM |
| S14 | Does conservative carving cut false alarms? A voxel counts as free only if the camera saw past the whole voxel (nearest depth over its projected footprint), optionally with the free margin along the surface normal (slant). Replaces S13's angle cut, which broke the moved pair in sim (kind 2/3) | Tuning pairs, full pipeline: fp, slant, fp+slant | Best variant adopted only if tuning obj/sc AP >= 34.3 (no loss); ties broken by fewer predicted objects. Then held-out 5 must stay >= 18.0. Otherwise pipeline unchanged | pennies + laptop SAM |
| S15 | Does a minimum object size cut false alarms? (pair01: most of 46 false alarms are 5-10 cm fragments) | Drop blobs whose largest side is under 4 / 6 / 8 / 12 voxels (scale-free; 7 voxels ~ 12 cm on pair01), after SAM (cached masks). Current and fp+slant carving. Tuning pairs + own pairs (tools/score_own.py) | Adopt the largest threshold with tuning AP >= 33.3 (at most 1.0 below 34.3) AND no own-pair change lost; report false alarms cut. Held-out 5 must stay >= 17.0 | pennies |
| S16 | Does a VLM before/after check cut false alarms? Each change's best crop next to the same 3D spot from the closest camera in the other video; Qwen2.5-VL-7B answers 'changed' or 'same' | Drop a change when every one of its blobs is judged 'same'. Own pairs (current and fp+slant carving), then tuning pairs | Adopt if own pairs lose no true change AND false alarms drop >= 30%; then tuning AP >= 33.3 and held-out >= 17.0. Otherwise keep the verdict as information only | ~$0.5 Modal |

## Summary of all tests (2026-10-12)

| Test | Question | Verdict | What we concluded |
|---|---|---|---|
| S1 | Is SceneDiff real and runnable? | PASS | Runs, numbers near the paper's. It is the baseline: 32.0 AP on 5 val pairs, but 30 false alarms vs 7 hits |
| S2 | Does SceneDiff work on our phone video? | borderline FAIL | 3 of 6 changes on pair01. Misses blanket, hat, pen; light change causes wall false alarms |
| S3 | Can the laptop run Isaac Sim? | PASS | 9.4 fps, 4.4 GB VRAM. Fine for the robot demo |
| S4 | Keyless vs gated models | not run | Superseded: we got access to the gated models (DINOv3, SAM 3) |
| S5 | Which reconstruction model? | done | Best geometry: Pi3X (3.3 cm, near metric scale). Best change detection: pi3, at every setting. VGGT, DA3, MapAnything behind on both |
| S6 | Do better keyframes fix SceneDiff's misses? | FAIL | No. Same 3 of 6, more false alarms. Its misses are the detector, not the frames |
| S7 | Can LiDAR give a changed / unchanged / unobserved map? | result only | Yes: suitcase found, honest unobserved band, 2 false alarms. Needed gravity alignment and the Stray pose fix |
| S8 | Geometric veto of SceneDiff false alarms | folded into S9 | Became the fusion experiment |
| S9a | Does the 3D map work from plain video? | PASS | pi3 metric map finds the suitcase with 0 false alarms |
| S9b | 3D-first vs SceneDiff on the benchmark | middle band | Held-out 18.0 vs SceneDiff 32.0; fusion 33.0. Rule says: 3D core plus SceneDiff as a second channel. 5 pairs too few to rank |
| S10 | Was Pi3X compared fairly? | pi3 confirmed | Fair confidence cut makes Pi3X worse; full pipeline 28.1 vs pi3 34.3 |
| S11 | pair02 (corridor, mirror) | strong result | 3D pipeline 4 of 4 + boot correctly unobserved; SceneDiff 3 of 4; LiDAR fooled by the mirror |
| S12 | Does VGGT-Omega beat pi3? | pi3 stays | Best depth on pair01 (2.0%), but benchmark 13.0 vs pi3 17.9 at the geometry stage |
| S9e | Full 3D pipeline on our room, plain video | strong result | 5 of 6 changes vs SceneDiff's 3. False alarms on walls above the wardrobe |
| S14 | Conservative carving (whole voxel seen past, normal margin) | not adopted | Tuning 35.4 with 24% fewer objects, but held-out 17.2 < 18.0. 5 pairs too few to tell; off by default |
| S15 | Minimum object size | not adopted | Only 4 voxels keeps tuning AP (33.9); 6+ voxels cost 3+ AP. The benchmark is full of small objects; size cannot be the false-alarm filter |

Code tidy (2026-10-12): rejected options removed from the code (git history keeps them); one pipeline for video and LiDAR (`roomdiff.pipeline`). Re-running the final settings reproduced the tuning result exactly (34.33 / 23.56) and S5 pi3 (ATE 4.2 cm, AbsRel 0.032). Tool names in older entries below may no longer exist.

Component choices inside the 3D pipeline (tuning pairs, each kept only if it raised AP): finer voxels (median depth/80: 3.5 -> 15.3), robust ICP between videos (+2.6), SAM object outlines (+7.6 with ICP), DINOv2 moved matching (25.5 -> 34.3). Rejected: colour-histogram matching, depth-scaled tolerance, smart keyframes.

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

### S6: FAIL, frame choice is not the bottleneck (2026-10-11)
- pair01, 14 keyframes per video by LiDAR coverage x sharpness x no-finger (`tools/make_keyframe_pair.py`), SceneDiff default config, resample 1. Finger filter dropped 1 prediction (P5, 90% inside the finger). Modal ~6 min A100, ~$0.25.
- Keyframes were sharper (mean quality 0.75 vs 0.33 before, 0.74 vs 0.48 after) and showed every changed object at least once, including the hats from both sides.

| Change | S2 (uniform) | S6 (keyframes) |
|---|---|---|
| little suitcase, moved | found, matched | found, matched (P2) |
| black pouch, added | found | found (P10) |
| blue tool, moved | added only | removed only (P4): still wrong type |
| hat, removed | missed | **missed, although clearly in view in both videos** |
| blanket, removed | missed | missed |
| pen, moved | missed | missed |
| false alarms | 4 | **7** (P0 bed corner, P1 floor corner, P3 frame edge with an undetected finger, P6 duvet, P7 curtain corner, P8 floor, P9 suitcase fragment) |

- Rule: >= 4 of 6 with <= 4 false alarms. Got 3 of 6 (2 right) with 7. **FAIL: keep uniform sampling as SceneDiff default**; frame choice is not the bottleneck.
- Second correction on the hat: with clean views from both sides it is still missed, so it is a detector miss (a flat hat on a cluttered top, under a light change), not an observation problem.
- What did work: the finger filter removed the finger artifact it was built for.
- Observation for head A: P0, P6 and P8 lie on surfaces the S7 LiDAR map labels unchanged. A geometric veto (reject a 2D change whose 3D voxels are confidently unchanged) would remove them, and LiDAR independently found the blanket. Candidate spike S8.
- Artifacts: outputs/s6/objects.jpg (one tile per prediction), outputs/s6/pred/, data/scenediff_own/pair01_kf/keyframes.jpg.

### S5 + S9: reconstruction model comparison and 3D-first pipeline (2026-10-12, overnight, in progress)
Same frames for every model (every 10th frame, both videos in one call), same loader, same change-map code; only the model differs. Model runner: `infra/modal_models.py`; scorer: `tools/score_recon.py`; benchmark export: `tools/bench_predict.py` + official SceneDiff evaluator.

**(a) Geometry on pair01 vs ARKit poses + LiDAR depth (243 frames):**

| Model | ATE (cm, Sim3) | depth AbsRel, one Sim3 scale | Sim3 scale (1 = metric) | change map, metric 5 cm: suitcase / FP | scale-free (median/40): suitcase / FP |
|---|---|---|---|---|---|
| pi3 | 4.2 | 0.032 | 1.61 | both / 0 | both / 7 |
| **Pi3X** | **3.3** | **0.022** | **1.016** | both / 1 | both / 3 |
| VGGT | 4.1 | 0.025 | 2.31 | both / 1 | both / 12 |
| DA3 (GIANT-1.1) | 5.7 | 0.036 | 1.97 | both / 0 | both / 5 |
| MapAnything (apache) | 19.4 | 0.181 | 1.01 | both / 3 | both / 28 |

- MapAnything: focal 283 px vs ~362 true (22% off) explains its errors; may be our image loading, unresolved, not pursued (far behind).
- Pi3X is near-metric from plain video (scale 1.016): solves the scale problem. Peak memory 81 GB at 243 frames (pi3: 16 GB).
- S9a rule (suitcase found with <= 4 FP): **PASS** for pi3 metric (0 FP) and Pi3X both variants; scale-free pi3 at median/40 has 7 FP (fails), 0-1 FP at coarser ratios (post-hoc sensitivity, see below).

**(b) Change detection on 15 SD-V val tuning pairs (val[5:20], disjoint from the 5 S1 pairs), official evaluator, obj/sc AP without type (with type):**

| Config | AP | AP typed |
|---|---|---|
| pi3, voxel = median depth/25 | 3.5 | 1.8 |
| pi3, /40 | 8.9 | 4.4 |
| pi3, /60 | 12.2 | 6.9 |
| pi3, /60, SAM object layer | 15.2 | 7.5 |
| pi3, /80 | 15.3 | 7.7 |
| **pi3, /80, SAM object layer** | **22.0** | **10.7** |
| pi3, /110 | 8.9 | 5.1 |
| Pi3X /40, /60, /80 | 5.3, 9.2, 11.4 | |
| VGGT /60 (conf 10% / 40%) | 4.6 / 7.0 | |
| DA3 /60 (conf 10% / 40%) | 2.8 / 2.3 | |

- Geometry winner (Pi3X) and change-detection winner (pi3) differ. Open for the user (D11).
- Failure modes seen (bedroom_2_bedroom_3): carpet-texture noise blobs, wall corners, fragmented masks; SAM fixes mask quality.
- SceneDiff baseline on the same 15 pairs: running (needed for a like-for-like comparison).

**(c) Pipeline improvements on the tuning pairs (obj/sc AP, typed, obj/im AP, px IoU):**

| Config (all pi3, voxel = median/80) | AP | typed | obj/im | px IoU |
|---|---|---|---|---|
| geometry only | 15.3 | 7.7 | 31.2 | 0.45 |
| + moved matching by colour histogram (thr 0.5) | lower: typed 8.7 vs 10.7, dropped | | | |
| + robust ICP of video 2 onto video 1 | 17.9 | 9.3 | 34.1 | 0.50 |
| + SAM object layer | 22.0 | 10.7 | 38.7 | 0.50 |
| + ICP + SAM | 25.5 | 12.6 | 44.0 | 0.59 |
| SceneDiff (same 15 pairs) | 44.4 | 32.1 | 51.4 | 0.38 |
| Fusion: SceneDiff rescored by 3D support (x (0.5 + support)) | 47.2 | 33.1 | 52.5 | 0.38 |
| Fusion: gate (drop support < 0.1) | 43.0 | 30.0 | 49.7 | 0.45 |
| **Fusion: rescore + add 3D-only objects** | **50.8** | **35.2** | **60.5** | **0.51** |

**(d) Held-out test, the 5 S1 pairs, configs frozen before running:**

| Method | AP | typed | obj/im | px IoU |
|---|---|---|---|---|
| SceneDiff | 32.0 | 15.8 | 44.6 | 0.19 |
| 3D-first (pi3, /80, ICP, SAM) | 14.9 | 4.0 | 21.2 | **0.31** |
| Fusion (rescore + add) | 33.0 | 13.0 | 45.9 | 0.20 |

- **S9b verdict by the written rule: 14.9 is in 13-26 -> "3D core plus SceneDiff appearance as a second channel".** S9a passed.
- The tuning-set fusion gain (+6.4 AP) mostly does not transfer: on held-out it is +1.0 AP, -2.8 typed. 5 pairs hold only 13 GT objects, so the test cannot separate these. A larger held-out set is needed before claiming a gain (val[20:50], 30 pairs: ~$5.4 SceneDiff + ~$1 pi3).
- 3D-first has the best pixel IoU on both sets: it finds changed regions; it is weak at instances (fragments, noise blobs, failed reconstructions such as gas_station, where pi3 did not co-register the two videos: fitness 0.07).
- Typed AP is our weakest number: moved objects need a real matcher (appearance features, not colour histograms).

**(e) Later in the night (tuning pairs unless stated):**
- Depth-proportional surface tolerance (tau + k * depth), aimed at wall false alarms on pair01: k = 0.01 / 0.02 / 0.04 -> AP 16.4 / 15.6 / 8.5 vs 17.9 without. **Rejected**: it swallows the benchmark's small close-range objects.
- Moved matching with DINOv2 ViT-S/14 mean patch features under the SAM mask (cosine), greedy one-to-one with a voxel-size gate (`objects.DinoDescriptor`, `match_moved(kind="cos")`). Threshold 0.4 / 0.5 / 0.6 / 0.7 / 0.8 -> AP 30.8 / **34.3** / 32.6 / 32.2 / 26.6, typed 22.5 / **23.6** / 21.4 / 21.2 / 15.3 (from 25.5 / 12.6 without matching). **Kept at 0.5.**
- Held-out (5 S1 pairs, frozen): 3D-first with DINO matching 18.0 AP, 12.6 typed (SceneDiff 32.0 / 15.8). Fusion (rescore + add) 33.0 / 13.0: level with SceneDiff. Chart: outputs/s9/s9_results.png.
- **pair01 (own room, plain video, pi3, ICP, SAM; voxel median/80 as tuned on the benchmark):** top 12 detections hold suitcase removed + added, blanket removed, pouch added, hat removed (on the wardrobe), blue tool's old spot (removed). That is 5 of 6 changes found (pen missed; tool only as removed) vs SceneDiff's 3 of 6. Remaining false alarms are wall / cornice patches above the wardrobe and small fragments. outputs/s9/pair01_d80/objects.jpg (also /40: same changes minus the tool).
- Cost of the night on Modal: roughly $9 (pi3 recon 21 pairs, 4 models x 16 runs, SceneDiff on 15 pairs, evaluations).

### S10: pi3 confirmed, Pi3X gap is real (2026-10-12)
- Pi3X confidence is calibrated differently (an absolute 0.1 cut keeps ~100% of its pixels vs 82-95% for pi3), so we tested a share-of-pixels cut (`frames.from_recon_npz(conf_drop=...)`).
- Tuning pairs, voxel median/60 and /80, geometry + ICP (obj/sc AP): drop 10 / 20 / 30% -> /60: 5.5 / 4.2 / 7.0, /80: 7.3 / 7.1 / 4.2. Absolute 0.1 cut with ICP: **12.6** (11.4 without ICP). The share cut hurts Pi3X: its low-confidence pixels are still useful. The original comparison was not unfair against it.
- Full pipeline on the best Pi3X setting (/80, cut 0.1, ICP, SAM, DINOv2 matching 0.5): **28.1 AP, 20.5 typed**, obj/im 37.7, px IoU 0.58. pi3, same pipeline: 34.3 / 23.6 / 44.0 / 0.59.
- Rule: < 29.3 -> **pi3 confirmed, gap is real.** Pi3X stays a scale source only (near-metric on pair01).
- Note: SAM and matching narrow the gap (geometry 12.6 vs 17.9, full 28.1 vs 34.3), so object-level steps matter more than the model.

### S11: pair02, corridor with a large mirror (2026-10-12, result only, no rule)
- Ground truth (user): cap added, printer moved (removed + added), white shoe removed, black boot unobserved (in the before video only). Photos: data/raw/pair02/photos (`_a` = after, `_b` = before).
- Plain video: pi3 jointly (376 frames, stride 10, peak 22 GB), robust ICP (fitness 0.90), voxel median/80 (1.1 cm), SAM. LiDAR: S7 map, 5 cm, registration fitness 0.68. SceneDiff: default, resample 90.

| Change | 3D pipeline, plain video | LiDAR map | SceneDiff |
|---|---|---|---|
| cap, added | found (#4) | missed | found (P9) |
| printer old spot | found (#3) | found (B0, B2) | found, typed moved (P2) |
| printer new spot | found (#1) | found (B4) | found, typed moved (P2) |
| white shoe, removed | found, small (#5) | missed | missed |
| black boot, unobserved | **unobserved**: occupied in before, in view in 0 of 189 after frames | not checked | no notion of it |
| false alarms (to confirm with user) | box (#0, #2), 6 small wall / door edges, bench (#11) | 2 from the mirror, floor patch, a door (B3, 150 L) | box (P1), bench (P8), ~8 walls / doors |

- The mirror produced phantom changes in the LiDAR map (depth into the reflection), only edge noise in the plain-video map.
- Plain video beat LiDAR on small objects here (1.1 cm vs 5 cm voxels).
- Artifacts: outputs/s9/pair02_d80/ (objects.jpg, boot_before.jpg), outputs/s7_pair02/, outputs/sd/own/pair02/objects.jpg.
- User check of the extra detections: box moved a little while taking the printer out (real, acceptable); #2 likely the user's shadow; bench clothes and the LiDAR door (B3) unchanged; boot was seen in the before video only (confirms unobserved).
- Final count, pair02. 3D pipeline: 4 of 4 changes + boot unobserved, ~8 false alarms (shadow, bench, 6 small edges). LiDAR: printer only, 4 false alarms (2 mirror, floor, door). SceneDiff: 3 of 4, ~9 false alarms (bench, walls, doors).
- New failure source: a shadow made a pi3 "added" blob, so the plain-video map is not fully lighting-immune (pi3 depth comes from pixels). The LiDAR map is.

### S12: VGGT-Omega, pi3 stays (2026-10-12)
- Runner `infra/modal_models.py::vggt_omega` (checkpoint vggt_omega_1b_512.pt, gated, HF token in Modal secret `huggingface`). pair01: 243 frames at 592x448, peak 24.4 GB on A100-80GB.
- (a) pair01 geometry, same scorer as S5: ATE **4.5 cm** (pi3 4.2, Pi3X 3.3), depth AbsRel at one Sim3 scale **0.020** (best of six; Pi3X 0.022, pi3 0.032), Sim3 scale 2.20 (not metric). Change map: suitcase both blobs; 0 false-alarm blobs metric, 5 scale-free (pi3 7, Pi3X 3).
- (b) 15 tuning pairs, geometry + ICP (obj/sc AP): median/60 6.3, /80 8.2, /110 3.5, /80 with the stricter confidence cut (drop 40%) **13.0**. pi3 at the same stage: 17.9.
- Rule: geometry < 17.9 -> **stop, pi3 stays.** Best depth on pair01 does not turn into better change detection on the benchmark (same pattern as Pi3X in S10). Cost ~$1.50.

### S14: conservative carving, not adopted (2026-10-12)

| Variant | Tuning AP (typed) | Objects predicted | Held-out AP (typed) | Objects | Sim false alarms |
|---|---|---|---|---|---|
| Current | 34.3 (23.6) | 283 | 18.0 (12.6) | 218 | 8 |
| footprint | 34.0 (23.8) | 248 | | | 6 |
| slant | 35.0 (24.9) | 291 | | | 6 |
| footprint + slant | **35.4** (24.6) | **216** | 17.2 (8.0) | 148 | 5 |

- Rule: fp+slant passed tuning (>= 34.3, fewest objects), failed held-out (17.2 < 18.0). **Pipeline unchanged**; options stay in code, off.
- It cuts predictions by a quarter to a third on both sets at nearly equal AP: the removed blobs were mostly not matched to ground truth anyway. Typed AP on held-out fell 12.6 -> 8.0: fewer blobs leaves fewer moved pairs to match. 5 held-out pairs (13 objects) cannot separate 17.2 from 18.0; the 30-pair test could.
- Sim diagnosis: 2 of the 8 false alarms were grazing rays carving thin table edges (fixed by fp+slant). 3 come from the Simple_Room table asset rendering view-dependently: two visit 2 frames 27 cm apart, the arched leg is present in one and absent in the other (floor visible). Not our error; the bench needs a solid table.
- S13's angle cut (70/75/80) broke the moved cabinet in sim (kind 2 of 3) and was not run on the benchmark.

### S15: minimum object size, not adopted (2026-10-13)

| Min side (voxels) | 0 | 4 | 6 | 8 | 12 | fp+slant 8 | fp+slant 12 |
|---|---|---|---|---|---|---|---|
| Tuning AP | 34.3 | 33.9 | 31.3 | 31.4 | 27.5 | 32.4 | 28.1 |

- Only 4 voxels passes (>= 33.3); on pair01 that is ~7 cm and removes almost nothing (false alarms are mostly 6-10 cm). Size is not the lever.
- Own pairs scored per detection for the first time (tools/score_own.py, bench/own_gt/): pair01 current 5/6 found, 46 false alarms; fp+slant 5/6, 30. pair02 fp+slant 4/4, 27.

### S16 notes so far (2026-10-13, running)
- The written prompt ("changed" or "same") failed on own pairs: it said "same" for the blanket, pouch, cap and printer, although the crops show them plainly present vs absent. A 7B VLM defaults to "same".
- Prompt and crop iterated on own pairs, which are therefore development data now, not the test: a yes/no question ("is there an object inside the box in one image that is missing from the other?") said yes to every main real change with a crop, and to 36% of false alarms (26 of 72). Contour markers drowned small regions ("red pixelated shape"), so the marker is now a thin box around the region; the other-video search was widened (half the detections had no comparison view).
- The rule's decision moves to the benchmark: tuning AP >= 33.3 and held-out >= 17.0 with the filter on.
