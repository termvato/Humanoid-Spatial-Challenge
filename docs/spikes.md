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
