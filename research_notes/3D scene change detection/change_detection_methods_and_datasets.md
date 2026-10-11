# Scene change detection methods and datasets (2D, multiview, 3D, Gaussian splatting, scene-graph), 2023 to Oct 2026

Scope note: target task is two phone videos of the same indoor room, different times and trajectories, detect and localise objects that were moved, added or removed, without confusing viewpoint change, occlusion or unobserved regions with change. Pre-2023 items are flagged as BACKGROUND. All numbers are self-reported unless stated; most comparisons are on the authors' own benchmark. Research done 2026-10-11.

## Q1. Strongest 2024-2026 methods for multiview / video object-level change detection

### Takeaway
SceneDiff (arXiv 2512.16908, Dec 2025, rev. Mar 2026) is the closest match to the target task: training-free, takes two unposed videos, outputs per-frame object masks labelled Added/Removed/Moved, MIT code and MIT data openly downloadable, and it beats every prior method on its own benchmark by a wide margin. Absolute numbers are still low (SD-V obj/scene AP 22.8), so the task is far from solved. 3DGS-based methods (3DGS-CD, MV3DCD, GS-DIFF, O-SCD) need per-scene optimisation and dense pre-change capture, and score poorly on SceneDiff's free-trajectory benchmark.

### Cited Findings

**SceneDiff (Wu, Lin, Che, Tiwari, Zou, Wang, Hoiem; UIUC)**
- arXiv 2512.16908 submitted 18 Dec 2025, v2 31 Mar 2026; the arXiv page lists no peer-reviewed venue. Source: [arXiv abs](https://arxiv.org/abs/2512.16908)
- The GitHub README labels it "ECCV 2026"; I could not verify this independently against ECCV proceedings. Source: [GitHub yuqunw/scene_diff](https://github.com/yuqunw/scene_diff)
- Input: two captures (images or videos) of the same scene. Training-free pipeline: estimate poses, segment images into objects, compare objects using semantic and geometric features. Source: [arXiv abs](https://arxiv.org/abs/2512.16908)
- Pipeline detail: (1) pi^3 (Pi3) jointly estimates depth, pose, intrinsics for both sequences, point clouds normalised to unit cube; (2) frame pairing by bidirectional co-visibility (>50% or best match); (3) region scoring from depth reprojection (geometry), DINOv3 feature reprojection (appearance) and region-level DINOv3 matching, with regions from SAM masks; (4) maximum-entropy (Kapur) thresholding; (5) instance association adapted from ConceptGraphs (merge regions by feature and 3D point similarity); (6) greedy one-to-one matching across sequences to classify change type. Source: [arXiv HTML v2](https://arxiv.org/html/2512.16908v2)
- Ablations use VGGT, FASt3R, DINOv2, DINOv1. VGGT can "invert the scene dynamics" when most movable objects change together (marks moved objects static and walls as moved) because it regresses in the first frame's coordinate system; pi^3 is more robust. FASt3R gives noisier point clouds. Source: [arXiv HTML v2](https://arxiv.org/html/2512.16908v2)
- Results on SD-V (px/im IoU, obj/im AP, obj/sc AP): SceneDiff 39.1 / 43.8 / 22.8; MV3DCD IoU 22.1; CYWS-3D obj/im AP 13.9; CYWS-2D 19.4; CYWS-2D finetuned 24.5; VLM baseline (Qwen2.5-VL + SAM3) 22.9 / 7.1 / 5.3; 3DGS-CD 14.7 / 1.0 / 0.5. On SD-K: SceneDiff 20.8 / 20.9 / 10.6; MV3DCD IoU 9.1; CYWS-2D finetuned 16.8 AP; 3DGS-CD 4.8 / 0.1 / 0.1. Source: [arXiv HTML v2](https://arxiv.org/html/2512.16908v2)
- External benchmarks: PASLCD mIoU SceneDiff 47.3 vs MV3DCD 46.1 (essentially a tie), CYWS-2D 27.3; RC-3D AP50 SceneDiff 65.3 vs CYWS-3D 41.0 (50.0 with sensor depth), CYWS-2D 14.0; ChangeSim F1 SceneDiff 57.0 vs GeSCF 54.8, CSCDNet 43.1 (other methods' scores copied from their papers). Source: [arXiv HTML v2](https://arxiv.org/html/2512.16908v2)
- Headline claims: 53.0% relative AP gain on multiview benchmarks, 30.6% on two-view. Source: [arXiv abs](https://arxiv.org/abs/2512.16908)
- Runtime: about 159.4 s per sequence pair on one NVIDIA A40; dominant cost is SAM mask generation (~4 s per image). CYWS-2D 18.4 s, CYWS-3D 23.9 s, VLM 30.5 s, 3DGS-CD and MV3DCD each >30 min including 3DGS training. Source: [arXiv HTML v2](https://arxiv.org/html/2512.16908v2)
- Code: MIT licence, 31 commits, 27 stars, last push 2026-08-29 (GitHub API). Install: Python 3.10, PyTorch 2.5.1 CUDA 12.1, torch-scatter, bundled segment-anything; DINOv3 vith16plus checkpoint is gated (requires Meta access application, URL pasted into config). Scripts: demo.py (two-video demo), predict_multiview.py, evaluate_multiview.py. README unzip command has a typo (scenediff_bechmark.zip vs scenediff_benchmark.zip). Sources: [GitHub](https://github.com/yuqunw/scene_diff); [GitHub API](https://api.github.com/repos/yuqunw/scene_diff)

**3DGS-CD (Lu, Ye, Leonard; MIT)**
- Compares two unaligned image sets; renders novel views from a pre-change 3DGS, uses EfficientSAM for zero-shot 2D change segmentation, associates and fuses across views into 3D change masks and per-object rigid transformations. Works with as few as one post-change image, as fast as 18 s; no depth, instructions, classes or object models needed. Claims up to 14% higher accuracy than the prior radiance-field change method and about three orders of magnitude faster. Source: [arXiv 2411.03706](https://arxiv.org/abs/2411.03706)
- Published in IEEE RA-L 2025 (per repo citation). Apache-2.0. Data on Google Drive (images, GT masks, pre-change 3DGS weights). Custom captures tested with iPhone 13 mini; needs >150 pre-change images that include the moved objects. Old pinned stack (Python 3.8, PyTorch 2.1.2/CUDA 11.8, nerfstudio, tiny-cuda-nn, HLoc with pycolmap 0.4.0, LightGlue). Last push 2025-02-22. Sources: [GitHub 3DGS-CD](https://github.com/520xyxyzq/3DGS-CD); [GitHub API](https://api.github.com/repos/520xyxyzq/3DGS-CD)
- Its own dataset is tiny (5 videos, 20 frames per SceneDiff's Table 1). Source: [SceneDiff HTML](https://arxiv.org/html/2512.16908v2)

**MV3DCD, "Multi-View Pose-Agnostic Change Localization with Zero Labels" (Galappaththige et al., QUT)**
- CVPR 2025 per search snippet; the arXiv v2 page itself lists no venue. Sources: [search summary of arXiv 2412.03911](https://www.arxiv.org/pdf/2412.03911); [arXiv HTML v2](https://arxiv.org/html/2412.03911v2)
- Label-free but not training-free: per-scene 3DGS optimisation (7k iterations reference, 3k inference scene, 3k change-channel fine-tuning). COLMAP estimates reference poses and registers post-change images (works with as few as 5). DINOv2 features give feature-aware change masks plus structure-aware masks. Output: 2D change masks rendered from a change-aware 3DGS, renderable from unseen views. Source: [arXiv HTML v2](https://arxiv.org/html/2412.03911v2)
- PASLCD averaged mIoU / F1: MV3DCD 0.461 / 0.612; CYWS-2D 0.273 / 0.398; Feature diff 0.264 / 0.386; SplatPose 0.173 / 0.281; CSCDNet 0.125 / 0.204. ChangeSim change-class mIoU 0.407 vs CSCDNet 0.229. (PASLCD is the authors' own dataset.) Source: [arXiv HTML v2](https://arxiv.org/html/2412.03911v2)
- Repo Chumsy0725/MV3DCD: licence NOASSERTION (no recognised licence), 17 stars, last push 2026-02-02. Source: [GitHub API](https://api.github.com/repos/Chumsy0725/MV3DCD)

**O-SCD, "Changes in Real Time: Online Scene Change Detection with Multi-View Fusion" (Galappaththige et al.)**
- Accepted at CVPR 2026 (arXiv comments). Online, pose-free, label-free, >10 FPS; self-supervised fusion loss over multiple cues, PnP-based fast pose estimation against a reference scene, change-guided 3DGS update. Claims to beat online and offline baselines; abstract gives no numbers. Project page chumsy0725.github.io/O-SCD; no code link on arXiv page. Source: [arXiv 2511.12370](https://arxiv.org/abs/2511.12370)

**GS-DIFF / "From Pixels to Primitives: Scene Change Detection in 3D Gaussian Splatting" (same QUT group)**
- arXiv 2605.07203 (May 2026), no venue listed. Compares Gaussian primitives directly (position, anisotropic covariance, colour) instead of render-and-compare, with per-primitive observability term; multi-view consistent change maps; separates geometric (structural) vs appearance (surface) change; claims no external model dependencies and ~17% mIoU over prior SOTA on real benchmarks (no absolute numbers on abstract). Abstract inconsistently names it GD-DIFF and GS-DIFF. Project page chumsy0725.github.io/GS-DIFF. Source: [arXiv 2605.07203](https://arxiv.org/abs/2605.07203)
- A search snippet indicates it re-annotated PASLCD into structural (84% of changed pixels) vs surface (16%) labels, "released upon acceptance". Source: [search result for arXiv 2605.07203](https://arxiv.org/pdf/2605.07203)

**SCaR-3D, "3D Scene Change Modeling With Consistent Multi-View Aggregation" (Zhou, Ni, Zhang, Chen, Huang; BIGAI)**
- arXiv 2512.22830 (Dec 2025), no venue. Input: dense pre-change image sequence plus sparse post-change images. Signed-distance-based 2D differencing, then multi-view aggregation with voting and pruning on 3DGS; continual reconstruction updates only changed regions. Introduces CCS3D, a synthetic dataset. No numbers or code on abstract page. Source: [arXiv 2512.22830](https://arxiv.org/abs/2512.22830)

**Cross-Temporal 3DGS (An, Xiao, Leng, Li, Liang)**
- arXiv 2512.00534, accepted to AAAI 2026. This is a scene-update/reconstruction method rather than a change detector: cross-temporal camera alignment, interference-based confidence initialisation to find unchanged regions, progressive optimisation. Targets urban planning, disaster assessment, heritage. No datasets, numbers or code on abstract page. Low relevance for object-level change. Source: [arXiv 2512.00534](https://arxiv.org/abs/2512.00534)

**Gaussian Difference (Jiang, Huang, Zhao, Zhang)**
- ICASSP 2025. 4D Gaussians represent both captures; images segmented and instances given IDs, changes found by ID comparison; claims gains over C-NeRF and CYWS-3D under lighting variation. No numbers or code on abstract page. Source: [arXiv 2502.16941](https://arxiv.org/abs/2502.16941)

**VSCD, "Video-based Scene Change Detection in Unaligned Scenes" (Yoon, Kim)**
- arXiv 2605.20821, accepted to ICML 2026 per comments. Directly framed for the target setting: reference RGB video and query RGB video of the same indoor space, free camera motion, not synchronised, many instances appearing/disappearing. Trained (mask supervision); query-centric multi-reference model with local patch correspondence and confidence-weighted fusion. Outputs per-query-frame pixel change mask (no change type, no 3D). Benchmark of >1.1M annotated frames, apparently simulated, plus a real-world test set. No numbers or code link on abstract page. Source: [arXiv 2605.20821](https://arxiv.org/abs/2605.20821)

**CYWS family (Sachdeva, Zisserman; Oxford VGG)**
- CYWS-3D, "The Change You Want to See (Now in 3D)", ICCVW 2023, arXiv 2308.10417: two RGB images, register-and-difference using frozen self-supervised embeddings, trained only on synthetic data, outputs bounding boxes of changes in both images; releases RC-3D real evaluation set. Source: [search summary / ICCVW listing](https://mlanthology.org/iccvw/2023/sachdeva2023iccvw-change)
- Repo: no LICENSE file (legally all rights reserved by default), last push 2024-07-05, old stack (PyTorch 1.10.1, CUDA 11.3, pytorch3d 0.7.1, mmdet 2.25.3, SuperGlue submodule). Checkpoint cyws-3d.ckpt downloadable; RC-3D at thor.robots.ox.ac.uk/cyws-3d/rc3d.tar. Sources: [GitHub CYWS-3D](https://github.com/ragavsachdeva/CYWS-3D); [GitHub API](https://api.github.com/repos/ragavsachdeva/CYWS-3D)
- CYWS-2D (WACV 2023, arXiv 2209.14341) is BACKGROUND (2022). Repo ragavsachdeva/The-Change-You-Want-to-See: no licence reported by GitHub API, last push 2024-03-27. Source: [GitHub API](https://api.github.com/repos/ragavsachdeva/The-Change-You-Want-to-See)

**Scan-based (3RScan) methods**
- "Has Anything Changed? 3D Change Detection by 2D Segmentation Masks" (Adam, Karantzalos, Grammatikopoulos, Sattler), arXiv 2312.01148, Dec 2023: input is 3D scans at different times plus generic 2D segmentation masks; render-and-compare gives initial 3D changes, refined by graph optimisation over 2D masks; finds added, moved, removed objects without a vocabulary; SOTA on 3RScan (no numbers on abstract page, no code link found). Source: [arXiv 2312.01148](https://arxiv.org/abs/2312.01148)
- BACKGROUND: "Objects Can Move" (Adam, Sattler, Karantzalos, Pajdla), ECCV 2022: depth-map differences, keep regions that move rigidly, graph cut to extend; SOTA on 3RScan. Code github.com/katadam/ObjectsCanMove (no licence per GitHub API, last push 2022-07-18, 5 stars). Sources: [ECVA page](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6194_ECCV_2022_paper.php); [GitHub API](https://api.github.com/repos/katadam/ObjectsCanMove)
- "Living Scenes: Multi-object Relocalization and Reconstruction in Changing 3D Environments" (arXiv 2312.09138) surfaced in search; I did not open it, so details unverified. Source: [arXiv 2312.09138](https://arxiv.org/pdf/2312.09138)

**Scene-graph based**
- DSG (Liao, Ye, Fei, Lin), arXiv 2609.00619, Sep 2026, no venue: semantic 3D Gaussian map, "dual-view rendering-based" object change detection to update scene-graph nodes, LLM for spatial relations; new DynTHOR benchmark (AI2-THOR, synthetic); evaluated on DynTHOR, 3RScan, real scenes. No numbers or code on abstract page. Search snippet says input is RGB-D sequences at different times. Sources: [arXiv 2609.00619](https://arxiv.org/abs/2609.00619); [arXiv HTML](https://arxiv.org/html/2609.00619)
- SceneDiff itself borrows ConceptGraphs-style instance association, i.e. the strongest practical "scene-graph-like" approach is object-node building then cross-time matching. Source: [SceneDiff HTML](https://arxiv.org/html/2512.16908v2)
- Situat3DChange (arXiv 2510.11509, Oct 2025) is a situated 3D change understanding dataset for MLLMs; I only saw its title. Source: [arXiv 2510.11509](https://arxiv.org/pdf/2510.11509)
- BACKGROUND: 3D VSG (arXiv 2209.07896), POCD (arXiv 2205.01202). Sources: [3D VSG](https://arxiv.org/pdf/2209.07896); [POCD](https://arxiv.org/pdf/2205.01202)

### Inferences
- For a 3-4 day build from two phone videos, SceneDiff is the obvious starting point: it matches the input and output, is MIT, and runs in a few minutes per pair on one A40. Main engineering risks: gated DINOv3 weights (fall back to DINOv2 if access is slow, at some unknown accuracy cost) and SAM being the bottleneck (SAM2/SAM-HQ or fewer keyframes could help).
- 3DGS-based methods (3DGS-CD, MV3DCD) assume a dense pre-change capture and >30 min per scene; SceneDiff reports 3DGS-CD at near zero AP on free-trajectory pairs, though that is a competitor's evaluation on its own benchmark and may reflect untuned hyperparameters.
- On PASLCD (not SceneDiff's own data) SceneDiff and MV3DCD are within ~1 mIoU point, so SceneDiff's large margins mainly hold on its own benchmark.
- O-SCD, GS-DIFF and VSCD are newer and plausibly competitive but none report against SceneDiff in what I read and code availability is unconfirmed.

### Gaps
- No verified ECCV 2026 acceptance for SceneDiff beyond the README.
- Code/licence status for O-SCD, GS-DIFF, SCaR-3D, VSCD, DSG, Gaussian Difference, Has Anything Changed not confirmed (no repos on arXiv pages).
- No numbers obtained for SCaR-3D, O-SCD, GS-DIFF, VSCD, DSG, Has Anything Changed.
- DINOv2 vs DINOv3 ablation numbers in SceneDiff not extracted.

## Q2. Datasets with real change labels

### Takeaway
SceneDiff (350 real video pairs, per-frame masks with Added/Removed/Moved, MIT, ungated, 11.9 GB) is the only openly licensed real multiview video benchmark with change-type labels found. PASLCD (iPhone, 10 scenes, 500 masks, binary, MIT) and RC-3D (image pairs, boxes) are smaller. 3RScan is the main real RGB-D rescan dataset with object-level change/transform annotations but is form-gated and lacks a formal change benchmark. ChangeSim and VSCD's main set are synthetic.

### Cited Findings
- **SceneDiff benchmark**: 350 video pairs, 50 scenes, 20 scene categories, 1,009 annotated objects. SD-V: 200 pairs by 7 people, daily indoor AND outdoor scenes, larger objects, longer sequences. SD-K: 150 pairs drawn from the HD-Epic egocentric kitchen dataset (cooking changes, more deformables). Labels: full segmentation masks for each changed object in all visible frames (SAM2-assisted with sparse point prompts), per-object name, change type (Added/Removed/Moved), deformability, size. Splits: 50 val pairs per subset, remaining 250 test; same physical scene kept in one split. Capture device not stated in what I read. Source: [arXiv HTML v2](https://arxiv.org/html/2512.16908v2)
- HF card: 11.9 GB, MIT licence, no gating notice, per-pair original_video1/2.mp4, annotated video1/2.mp4, segments.pkl (RLE masks, pycocotools), metadata.json with in_video1/in_video2 flags and rigid/deformable tag; val_split.json/test_split.json; Flask visualiser. Source: [HF yuqun/SceneDiff](https://huggingface.co/datasets/yuqun/SceneDiff)
- Note: HD-Epic has its own licence terms; SD-K is derived from it, so MIT on the card may not override upstream terms (inference, not verified).
- **PASLCD** (from MV3DCD): 10 real scenes (5 indoor, 5 outdoor), iPhone 16:9 random independent trajectories per instance, each scene with same-lighting and varied-lighting instances, 50 human-annotated binary change masks per scene (500 total, Supervisely, two annotators), 91 changes (5 to 17 per scene); structural and surface changes plus distractors (shadows, reflections). Internal inconsistency: text says two scenes are 360 degrees, table lists four. Sources: [MV3DCD HTML](https://arxiv.org/html/2412.03911v2); [search summary](https://huggingface.co/datasets/ChamudithaJay/PASLCD)
- PASLCD HF: MIT tag, 31.3 GB, ~1,000 rows (images + gt masks), test split only, no gating notice. Source: [HF ChamudithaJay/PASLCD](https://huggingface.co/datasets/ChamudithaJay/PASLCD)
- **RC-3D** (CYWS-3D): 100 image pairs, one changed object per pair, two-view, box labels, AP50 metric (per SceneDiff Table). Downloadable tar from Oxford; repository has no licence file. Sources: [SceneDiff HTML](https://arxiv.org/html/2512.16908v2); [GitHub CYWS-3D](https://github.com/ragavsachdeva/CYWS-3D)
- **3RScan / RIO** (BACKGROUND, 2019): 1,482 RGB-D reconstructions of 478 indoor environments that naturally change; calibrated RGB-D with poses; dense instance segmentation with IDs consistent across rescans; ground-truth per-object transforms for changed objects with symmetry, global scan alignment, change info (rigid, non-rigid, removed, ambiguity mappings) in 3RScan.json; train/val/test split files. Toolkit MIT; data access via a "3RScan Terms of Use" form; no formal change-detection benchmark protocol. Capture sensor not named on pages I read (commonly reported as a Google Tango tablet; unverified here). Sources: [GitHub 3RScan](https://github.com/WaldJohannaU/3RScan); [RIO project page](https://waldjohannau.github.io/RIO/)
- **ChangeSim** (BACKGROUND, IROS 2021): synthetic photo-realistic industrial indoor environments, RGB, depth, semantic and change segmentation, poses, 3D reconstructions, raw unpaired sequences, nuisance variations (air turbidity, lighting). About 80 videos / ~130K frames per SceneDiff Table 1. Licence unconfirmed (aggregator lists "Other"). Sources: [arXiv 2103.05368](https://arxiv.org/pdf/2103.05368); [SceneDiff HTML](https://arxiv.org/html/2512.16908v2)
- **Others** listed in SceneDiff Table 1: VL-CMU-CD (152 videos, outdoor street, BACKGROUND), PSCD (770 images), 3DGS-CD dataset (5 videos, 20 frames). Source: [SceneDiff HTML](https://arxiv.org/html/2512.16908v2)
- **VSCD benchmark**: >1.1M annotated frames (apparently simulated) plus real test set; pixel masks; availability not stated. Source: [arXiv 2605.20821](https://arxiv.org/abs/2605.20821)
- **CCS3D** (SCaR-3D): synthetic, controlled combinations of change types. Source: [arXiv 2512.22830](https://arxiv.org/abs/2512.22830)
- **DynTHOR** (DSG): AI2-THOR-based synthetic dynamic scene graph benchmark. Source: [arXiv 2609.00619](https://arxiv.org/abs/2609.00619)

### Inferences
- For evaluating a phone-video system, use SceneDiff SD-V val (50 pairs) for tuning and its evaluate_multiview.py for metrics; SD-V includes outdoor scenes, so filter to indoor if you want the target domain. PASLCD indoor (5 scenes) is a secondary sanity check.
- 3RScan is the best source of 3D ground truth for "moved" (per-object transforms) but is RGB-D tablet scans, not phone videos, and is gated by a form, which may take time within 3-4 days.

### Gaps
- SD-V capture device(s) and resolution not found.
- RC-3D source and licence; ChangeSim licence; 3RScan data licence terms beyond the form.
- No real ScanNet-based rearrangement change benchmark found in this search.

## Q3. 3D outputs vs 2D masks, and explicit handling of "moved"

### Takeaway
Only a few methods output true 3D objects with explicit "moved" semantics: 3DGS-CD (3D masks plus rigid transforms), 3RScan-based Objects Can Move / Has Anything Changed (3D, added/moved/removed), and SceneDiff (3D instances internally, 2D masks out, Moved via one-to-one cross-sequence matching). MV3DCD, GS-DIFF, O-SCD, VSCD and CYWS output binary 2D change masks or boxes with no change type.

### Cited Findings
- SceneDiff: Moved = object present in both sequences at different positions; one-to-one greedy matching by feature cosine similarity >0.7 (descending); matched = Moved, unmatched = Added (post only) or Removed (pre only). Evaluation is on 2D masks per frame plus object-per-scene AP; instances are associated in 3D via point clouds. Source: [SceneDiff HTML](https://arxiv.org/html/2512.16908v2)
- 3DGS-CD outputs 3D object-level change masks and object transformations (i.e. moved as a single rigid event). Source: [arXiv 2411.03706](https://arxiv.org/abs/2411.03706)
- Has Anything Changed outputs added, moved and removed objects from 3D scans. Source: [arXiv 2312.01148](https://arxiv.org/abs/2312.01148)
- Objects Can Move keeps rigidly moving regions (geometric transformation consistency). Source: [ECVA](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6194_ECCV_2022_paper.php)
- MV3DCD outputs 2D change masks rendered from a 3DGS with change channels (no type). Source: [MV3DCD HTML](https://arxiv.org/html/2412.03911v2)
- GS-DIFF outputs multi-view consistent change maps split into geometric vs appearance change (not added/removed/moved). Source: [arXiv 2605.07203](https://arxiv.org/abs/2605.07203)
- SCaR-3D separates pre-change and post-change states and updates reconstruction (3D, object-level). Source: [arXiv 2512.22830](https://arxiv.org/abs/2512.22830)
- VSCD: pixel mask per query frame only. Source: [arXiv 2605.20821](https://arxiv.org/abs/2605.20821)
- CYWS-3D: boxes in both images. Source: [mlanthology](https://mlanthology.org/iccvw/2023/sachdeva2023iccvw-change)
- DSG: updates 3D scene-graph nodes; abstract does not specify per-type handling. Source: [arXiv 2609.00619](https://arxiv.org/abs/2609.00619)

### Inferences
- SceneDiff's "moved" is appearance-matching-based, so identical-looking objects will be mislabelled (it admits swaps are ambiguous). A geometric check (ICP / rigid transform as in 3DGS-CD or 3RScan methods) on matched pairs could add robustness.
- Getting 3D boxes out of SceneDiff should be straightforward since it already builds per-object point clouds in a shared pi^3 frame (inference from pipeline, not a documented feature).

### Gaps
- Whether O-SCD or SCaR-3D distinguish moved from added+removed not confirmed.

## Q4. Admitted failure modes

### Takeaway
Common failures: repetitive/identical objects, low overlap between captures (pose failure), strong lighting change, small objects in large scenes, colour-only changes, people/dynamic content, and dependence on dense pre-change captures for 3DGS methods.

### Cited Findings
- SceneDiff: identical objects swapping positions are ambiguous; pose estimation fails in extreme low light, textureless surfaces, sparse low-overlap views; semantic state changes and fine deformations out of scope; SD-K (cluttered egocentric) remains hard; dynamic content such as a person needs manual masking; SAM masks can misalign with RC-3D boxes; 3DGS rendering baselines struggle with large viewpoint change. Source: [SceneDiff HTML](https://arxiv.org/html/2512.16908v2)
- SceneDiff failure counts on SD-V: 11/200 pairs fail from large cluttered scenes with repetitive items (appearance scores unreliable, geometry alone insufficient); 2/200 from geometry reconstruction failure under limited overlap; strong lighting causes false positives (static pillows, quilts flagged). Source: [SceneDiff HTML](https://arxiv.org/html/2512.16908v2)
- MV3DCD: misses colour-only surface changes (no foundation-feature change); struggles with very small changes in large scenes; patch-to-pixel interpolation over-inflates masks; depends on COLMAP registration (fails in very dark scenes); false positives at fine edges and reflections; a miss in either mask removes a true change. Source: [MV3DCD HTML](https://arxiv.org/html/2412.03911v2)
- 3DGS-CD: sensitive to hyperparameters if data not captured carefully (mask_refine_sparse_view, pre_train_pred_bbox_expand, proj_check_cutoff); needs >150 pre-change images covering moved objects; sparse (4+4 views) operation listed as future work. Source: [GitHub 3DGS-CD](https://github.com/520xyxyzq/3DGS-CD)
- VLM baseline (Qwen2.5-VL + SAM3) has recognition and localisation bottlenecks (obj/im AP 7.1 on SD-V). Source: [SceneDiff HTML](https://arxiv.org/html/2512.16908v2)

### Inferences
- None of the papers I read report explicit handling of "unobserved region" as a separate output state; SceneDiff mitigates it implicitly with co-visibility-based frame pairing, and GS-DIFF with a per-primitive observability term. A build should explicitly mark regions not co-visible in both videos as "unknown" rather than changed.
- Capture protocol (overlapping coverage, consistent lighting, no people in frame) likely matters more than method choice given the admitted failure modes.

### Gaps
- No independent (third-party) re-evaluation of SceneDiff found; all its numbers are self-reported.
- Failure analyses for O-SCD, GS-DIFF, VSCD, SCaR-3D, DSG not read (full texts not fetched).
