# Synthetic data generation for indoor scene change detection, and sim-to-real evidence for perception (as of October 2026)

Scope note: research done 2026-10-11 under a tight tool budget (about 20 calls). Primary sources (arXiv, official docs, HF dataset cards) are preferred. Anything not backed by a fetched source is placed under Inferences or Gaps and flagged. Vendor and author numbers are self-reported unless stated otherwise.

## Q1. Tools for generating indoor room pairs with exact change ground truth (Isaac Sim Replicator, Infinigen Indoors, ProcTHOR/AI2-THOR, Habitat/HSSD/ReplicaCAD, BlenderProc, Kubric, 3D-FRONT)

### Takeaway
On an 8 GB RTX 4070 laptop, Isaac Sim 5.1 is officially below minimum spec (16 GB VRAM minimum), so it is a risky choice for a 1-day setup; a Blender-based pipeline (BlenderProc, or Infinigen Indoors on cloud CPUs/GPUs) or a game-engine simulator with built-in object manipulation (AI2-THOR/ProcTHOR, Habitat with HSSD) is the lower-risk path. For "move/add/remove object and re-render the same trajectory", every tool can do it because you control the scene graph; the differentiators are asset licences, VRAM, and setup time.

### Cited Findings
**Isaac Sim 5.1 / Replicator**
- Isaac Sim 5.1 official requirements: minimum GPU VRAM 16 GB (example GeForce RTX 4080), "Good" 16 GB, "Ideal" 48 GB; RAM minimum 32 GB. Page warns GPUs below 16 GB "may struggle with complex scenes" and that "due to VRAM constraints, some tutorials and benchmarks may not run on GPU below the minimum specifications." GPUs without RT cores (A100, H100) are not supported. The page also states 5.1.0 "is no longer supported", meaning a newer release exists. [Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- Older Isaac Sim 4.5 requirements listed 8 GB VRAM as minimum, so the bar rose for 5.x. [Isaac Sim 4.5 requirements](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html)
- A forum user reports Isaac Sim 5.x and 4.5 would not run on a 6 GB Quadro RTX 3000. [NVIDIA forum](https://forums.developer.nvidia.com/t/isaac-sim-5-x-and-4-5-wont-run-on-quadro-rtx-3000-6-gb-vram-recommended-path-for-internship/372141)
- No official Replicator headless throughput benchmark on 8 GB consumer GPUs was found. Forum anecdotes: a Replicator-API camera plus BasicWriter loop ran at about 8 FPS, versus about 60 FPS when the user created the camera manually and attached a segmentation annotator directly, and about 100 FPS with no Replicator (hardware per thread). [NVIDIA forum: FPS drop by camera creation method](https://forums.developer.nvidia.com/t/fps-drop-in-isaac-sim-depending-on-camera-creation-method/326207)
- Forum report of Replicator `rep.modify` pose/visibility calls slowing down each iteration when instance segmentation is retrieved each loop. [NVIDIA forum: Replicator gets slower](https://forums.developer.nvidia.com/t/replicator-gets-slower-after-each-iteration/274439)
- Forum reports of correctness issues: BasicWriter giving incorrect bboxes/instance segmentation for a class only inside Docker [forum](https://forums.developer.nvidia.com/t/incorrect-bboxes-in-basicwriter-output/312143); "Replicator's semantic segmentation has a lot of noise" thread, reported on RTX 50-series [forum](https://forums.developer.nvidia.com/t/replicators-semantic-segmentation-has-a-lot-of-noise/331525). Both anecdotal.
- Isaac Sim 4.5 user reporting about 9 FPS on a warehouse scene on recommended hardware [forum](https://forums.developer.nvidia.com/t/isaac-is-running-on-low-fps-9-fps-on-recommended-hardware-why/327202); 5.0 user on RTX 5090 reporting slowness, pointed to a "slightly outdated" Performance Optimization Handbook [forum](https://forums.developer.nvidia.com/t/isaac-sim-5-0-0-too-slow/352287).

**Infinigen Indoors**
- Procedural indoor generator, open-sourced under BSD; includes a constraint DSL plus solver for scene composition, and an export tool for real-time simulators "such as Omniverse and Unreal". [Infinigen Indoors abstract, arXiv 2406.11824](https://arxiv.org/abs/2406.11824)
- Export is "one-click" to USD and other formats; procedural Blender materials are baked to standard texture maps; single objects exportable as OBJ/FBX/STL plus URDF. Scenes imported into Isaac Sim ran at about 50 FPS with physics, and Unreal at about 60 FPS, both on RTX 4090s. [Infinigen Indoors paper HTML](https://arxiv.org/html/2406.11824)
- Paper-reported transfer evidence is thin: shadow-removal model trained on ISTD real plus 2k Infinigen pairs improved zero-shot SRD results (PSNR 22.83 to 24.56, RMSE 10.84 to 9.54); the occlusion-boundary experiment was tested on artist-made synthetic scenes, not real images. [Infinigen Indoors paper HTML](https://arxiv.org/html/2406.11824)
- No per-scene generation or render time was given in the parts of the paper read. An AWS Deadline Cloud guide estimates a single indoor dining-room job at about 15 minutes (GPU and scope unspecified); outputs include 1280x720 RGB, EXR, segmentation, camera intrinsics/extrinsics, and an optional second ground-truth pass for depth, normals and segmentation. [AWS Deadline Cloud Infinigen example](https://docs.aws.amazon.com/deadline-cloud/latest/developerguide/examples-jb-infinigen.html)

**AI2-THOR / ProcTHOR and the Rearrangement Challenge**
- The AI2-THOR Rearrangement challenge has a walkthrough phase (goal state) and an unshuffle phase where 1 to 5 objects are randomly changed in position, orientation or openness; the RoomR dataset uses 120 rooms and 70+ object categories. [Visual Room Rearrangement, CVPR 2021](https://arxiv.org/pdf/2103.16544); [ai2thor-rearrangement repo](https://github.com/allenai/ai2thor-rearrangement); [challenge page](https://ai2thor.allenai.org/rearrangement/)
- The 2022 1-phase winner was "ProcTHOR + Fine-Tuning" at 24.47% Fixed Strict (test), per the repo. [ai2thor-rearrangement repo](https://github.com/allenai/ai2thor-rearrangement)
- Multiple sources state AI2-THOR assets are Apache 2.0; ProcTHOR-specific licence not confirmed in this search. [ai2thor-rearrangement repo](https://github.com/allenai/ai2thor-rearrangement)

**Habitat / HSSD**
- HSSD-200: 211 scenes, 18,656 models of real-world objects; licensed CC BY-NC 4.0 and gated on Hugging Face (accept terms first). [HSSD-200 paper](https://arxiv.org/html/2306.11290v1); [hssd-models HF card](https://huggingface.co/datasets/hssd/hssd-models); [hssd-hab objects](https://huggingface.co/datasets/hssd/hssd-hab/tree/main/objects)

**BlenderProc / 3D-FRONT**
- BlenderProc is GPL v3. Original paper benchmark: about 3,000 images per hour on one GPU on complex SUNCG scenes (about 1.2 s per frame, old version); non-colour outputs (depth, segmentation) are fast because only colour needs multi-sampling. [BlenderProc paper, arXiv 1911.01911](https://arxiv.org/pdf/1911.01911); [licence listing](https://summary.ecosyste.ms/projects/357248)
- 3D-FRONT must be applied for and downloaded separately to use with BlenderProc. [ecosyste.ms BlenderProc 3D-FRONT project](https://science.ecosyste.ms/projects/11232)

**Kubric**
- Kubric couples PyBullet (physics) with Blender (rendering), designed to scale to thousands of machines; authors release code, assets and rendered datasets. Sources disagree on whether 11 or 13 datasets are presented. [Kubric CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Greff_Kubric_A_Scalable_Dataset_Generator_CVPR_2022_paper.html); [arXiv 2203.03570](https://arxiv.org/abs/2203.03570v1)

### Inferences
- Isaac Sim on an 8 GB laptop GPU is likely to work for small rooms at modest resolution but is officially unsupported; a 1-day setup risks being lost to VRAM, driver, and Replicator quirks. If Isaac Sim is wanted, renting a 24 GB or larger RT-core GPU (L4, L40S, A10G, RTX 4090/5090 class; NOT A100/H100) is the safer route. The camera-creation anecdote suggests attaching annotators directly to a manually created render product rather than the high-level BasicWriter path.
- For change pairs, the simplest correct design in any tool is: build room state A, render a trajectory; apply an explicit edit list (move: new pose for instance id k; add: spawn new instance; remove: hide/delete instance k); re-render a different (perturbed) trajectory for state B. Ground truth change masks come from instance-id images rendered in both states plus the known edit list, then projected across views via depth and poses. This is tool agnostic.
- Physical plausibility of edits (no floating or interpenetrating objects) matters for a verifier that must distinguish real changes from viewpoint artifacts. Tools with physics (Isaac Sim, Kubric/PyBullet, AI2-THOR, Habitat) can drop objects onto surfaces; with BlenderProc or Infinigen you can use Blender rigid-body settling or Infinigen's constraint solver.
- Rough 1-day suitability ranking for this user (my judgement, not sourced): (1) AI2-THOR/ProcTHOR: pip install, built-in object move/disable APIs, instance segmentation and depth out of the box, Apache-licensed assets, but stylised "game" look; (2) BlenderProc with 3D-FRONT or HSSD/Objaverse-style assets: better realism with path tracing, GPU friendly at 8 GB, but 3D-FRONT access requires an application and is likely non-commercial; (3) Habitat-sim with HSSD: fast rasterised rendering and rearrangement tooling, CC BY-NC assets, moderate learning curve; (4) Infinigen Indoors: most realistic and fully procedural (unlimited rooms, BSD), but generation is CPU heavy and slow, better on cloud; (5) Isaac Sim Replicator: strong writers and RTX rendering but steepest learning curve and below-spec hardware; (6) Kubric: good for object-level scenes but no indoor room layouts out of the box.

### Gaps
- No measured Replicator headless images/second on an 8 GB RTX 4070 laptop; must be benchmarked locally.
- Infinigen Indoors per-scene generation time (CPU hours) and render time not found in the parts of the paper read; widely reported anecdotally as slow (tens of minutes to hours per scene) but not verified here.
- Licences not verified in this session: ProcTHOR-10K houses, ReplicaCAD, 3D-FRONT/3D-FUTURE (believed non-commercial research licence, unverified), Kubric asset sets.
- Habitat rearrangement task tooling (Habitat 2.0/3.0 HAB, ReplicaCAD) not researched in depth.
- No head-to-head realism comparison of these generators for change detection transfer was found.

## Q2. Existing synthetic (and real) change-detection datasets that could be reused

### Takeaway
ChangeSim (industrial indoor, synthetic, IROS 2021) and the AI2-THOR Rearrangement (RoomR) data are directly reusable; newer 2025-2026 work (CCS3D synthetic, VSCD synthetic plus real test set, SceneDiff real benchmark) is highly relevant, with SceneDiff the best candidate for a held-out real evaluation if its data is released.

### Cited Findings
- ChangeSim (Park et al., IROS 2021): photorealistic simulated industrial indoor environments with non-target variations (air turbidity, lighting) plus object changes; provides RGB, depth, semantic segmentation, change segmentation, camera poses, 3D reconstructions, and raw unpaired sequences; authors report pair-based SCD models bottlenecked by the pairing step. Hosted at sammica.github.io/ChangeSim, code at SAMMiCA/ChangeSim. [ChangeSim arXiv 2103.05368](https://arxiv.org/pdf/2103.05368)
- AI2-THOR RoomR: 120 rooms, 70+ categories, 1 to 5 objects changed per episode (position, orientation, openness). [Visual Room Rearrangement](https://arxiv.org/pdf/2103.16544)
- CCS3D: synthetic 3D scene change dataset with four Blender scenes (Desk, Bookcase, Livingroom, Bedroom), human-like walking camera paths, controllable number of objects and change types. [3D Scene Change Modeling With Consistent Multi-View Aggregation, arXiv 2512.22830](https://arxiv.org/html/2512.22830)
- 3DGS-CD: 3D Gaussian Splatting based detection of object rearrangements from two unaligned image sets, with a real dataset of moves, swaps and insertions in cluttered scenes. [arXiv 2411.03706](https://arxiv.org/pdf/2411.03706)
- SceneDiff (Wu, Lin, Che, Tiwari, Zou, Wang, Hoiem; Dec 2025, v2 Mar 2026): 350 real video pairs, added/removed/moved objects, dense instance annotations; training-free method (pose solving, object segmentation, semantic plus geometric feature comparison) reporting 53.0% (multiview) and 30.6% (two-view) relative AP gains over prior methods (self-reported). Project page yuqunw.github.io/SceneDiff; data licence not confirmed. [SceneDiff arXiv 2512.16908](https://arxiv.org/abs/2512.16908)
- VSCD (Yoon and Kim, May 2026, listed as ICML 2026): video-based SCD in unaligned indoor scenes with free camera motion; over 1.1 million synthetic frames with pixel-accurate change masks plus an annotated real-world test set; abstract gives no quantitative sim-to-real numbers or data links. [VSCD arXiv 2605.20821](https://arxiv.org/abs/2605.20821)
- InteriorNet: large photorealistic multi-sensor indoor dataset (not change-specific). [InteriorNet arXiv 1809.00716](https://arxiv.org/pdf/1809.00716)

### Inferences
- SceneDiff is the closest match to the user's real-world target (phone-like video pairs, add/remove/move, differing trajectories) and its training-free pipeline is effectively a baseline the user's learned heads should beat. It is the obvious held-out real test set if downloadable.
- ChangeSim is industrial (warehouse/factory) not domestic, so its domain differs from typical phone-captured rooms; RoomR is domestic but stylised. VSCD's synthetic set, if released, could replace generation entirely.
- Reusing existing data for pretraining and generating only a small custom set targeted at the user's specific failure modes (viewpoint artifacts, reflections, occlusion) is likely the best use of a 3-4 day budget.

### Gaps
- Release status and licences of CCS3D, VSCD and SceneDiff data were not confirmed.
- VSCD's simulator, scene source and sim-to-real numbers were not available from the abstract.
- Habitat-based rearrangement datasets were not checked for per-frame change masks.

## Q3. Evidence on sim-to-real for perception: domain randomisation, co-training ratios, frozen foundation features, fraction of real data needed

### Takeaway
Evidence consistently shows a small amount of real data plus abundant synthetic beats either alone, but the best ratio is task specific and must be swept. Frozen self-supervised features (DINOv2, DINOv3) are the strongest documented lever for narrowing the synthetic-to-real gap in dense perception (Depth Anything V2 style), with larger backbones transferring better.

### Cited Findings
**Sim-and-real co-training (arXiv 2503.24361, RSS 2025, UT Austin/NVIDIA/Berkeley/NYU; manipulation policies, not pure perception)**
- Co-training ratio alpha is the probability a batch sample is from sim. Default 0.90 sim / 0.10 real on Panda Kitchen. On CupPnP (20 real, 1,000 sim demos) 50/50 was suboptimal, 99% sim was best (95% success), and 99.5% and 99.9% dropped to 60%, so the ratio needs tuning. [arXiv 2503.24361 HTML](https://arxiv.org/html/2503.24361v2)
- Average success: real only 45.3%, real plus task-aware "digital cousin" sim 81.1%, real plus task-agnostic prior sim 76.8%, all combined 83.2% (self-reported, real-robot evaluation). [arXiv 2503.24361 HTML](https://arxiv.org/html/2503.24361v2)
- Reducing sim data hurt (e.g., 67% at 10k sim to 53% at 500 sim; 95% at 1k to 75% at 100). Co-training still helped with up to 400 real demos. [arXiv 2503.24361 HTML](https://arxiv.org/html/2503.24361v2)
- Perception-relevant finding: misaligned sim camera viewpoints reduced performance (67% to 56%; 95% to 70%), yet perfect alignment was not required (sim did not model the real fisheye distortion). Same task definition and success criteria were essential; object and placement diversity helped generalisation. [arXiv 2503.24361 HTML](https://arxiv.org/html/2503.24361v2)
- Follow-ups: an ICML 2026 paper on a mechanistic understanding of sim-and-real co-training highlights balanced mixing ratio and structured representation alignment as key factors [ICML 2026 poster](https://icml.cc/virtual/2026/poster/64310); another proposes optimal transport to align sim and real feature spaces [arXiv 2509.18631](https://arxiv.org/pdf/2509.18631v3).

**Frozen foundation features and synthetic-to-real**
- DINOv3 paper: Depth Anything V2 trains on synthetic depth data and DINOv2 is the feature extractor that makes the transfer to real images work, which SAM lacks; DINOv3 authors freeze the backbone, train a DPT head on synthetic data, and argue DINOv3 inherits this sim-to-real ability. [DINOv3 arXiv 2508.10104](https://arxiv.org/pdf/2508.10104)
- BRIDGE (2025): depth trained on synthetic data with DINOv2 backbones; Giant and Large generalise well to real test sets, Small and Base do not (frozen vs fine-tuned not confirmed). [BRIDGE arXiv 2509.25077](https://arxiv.org/pdf/2509.25077)
- DINOv2: DPT head on frozen backbone competitive for depth; indoor-trained depth head transfers to other scenes (cross-domain, not sim-to-real). [DINOv2 arXiv 2304.07193](https://arxiv.org/html/2304.07193v2)
- Large study of pre-trained visual representations (Silwal et al., Meta, ICRA 2024): 5 PVRs, 3 robots, 348 experiments, 110+ hours of hardware; reports high sim-to-real predictivity of PVR-based policies and zero-shot real-world ImageNav transfer to a held-out scene; augmentation and fine-tuning helped in both sim and real. [arXiv 2310.02219](https://arxiv.org/html/2310.02219v2)
- Physically based structured-light simulation for segmentation: synthetic-to-real gap fell from 0.091 (RGB) to 0.052 (simulated depth) for one workpiece, i.e., geometric inputs transfer better than appearance. [arXiv 2407.12449](https://arxiv.org/pdf/2407.12449)

**How much real data (object detection literature, mostly older)**
- Synthetic pretraining plus fine-tuning on 10%, 5%, 2.5% of real data studied systematically. [How much real data do we actually need, arXiv 1907.07061](https://arxiv.org/pdf/1907.07061)
- Egocentric HOI detection: synthetic pretrain plus 10% real reached 72.69% mAP, above 70.84% for 100% real only; best 77.29% with 50% real. [arXiv 2204.07061](https://arxiv.org/pdf/2204.07061)
- NVIDIA TAO blog (vendor, 2021): synthetic plus 10% real nearly matched 100% real. Treat as best case. [NVIDIA blog](https://developer.nvidia.com/blog/preparing-models-for-object-detection-with-real-and-synthetic-data-and-tao-toolkit/)
- YOLOv3 study: no significant difference between pretrain-then-fine-tune and mixed training. [arXiv 2202.00632](https://arxiv.org/pdf/2202.00632)
- Maritime SAR (YOLOv9t): synthetic pretrain plus 10% real improved 13.7% over real only; best with 70% real. [SciTePress 2025](https://www.scitepress.org/Papers/2025/131127/131127.pdf)

### Inferences
- For the user's heads, the "real data needed" question will likely be dominated by how well frozen features already encode object identity and geometry. With DINOv2/v3 features, a few hundred real labelled pairs may suffice for calibration, but this is an extrapolation, not a cited result.
- The co-training paper's lesson that camera alignment matters translates to: render synthetic trajectories with phone-like intrinsics, motion blur, rolling exposure and handheld jitter, and use the same pair-sampling logic for synthetic and real.
- The optimal sim:real ratio varied wildly across tasks (50% bad, 99% best, 99.9% bad in one task), so the user should sweep at least 3 ratios (e.g., 50/50, 80/20, 95/5 by batch probability) and also compare pretrain-then-fine-tune.
- Geometric cues (depth consistency, reprojection error) likely transfer better than appearance cues, supporting a verifier that combines frozen semantic features with geometric residuals.

### Gaps
- No perception-specific (as opposed to policy) counterpart to 2503.24361 with a ratio sweep for change detection or feature matching was found.
- No direct evidence on sim-to-real for learned feature matchers or change verifiers on frozen DINO features; VSCD may contain such results but numbers were not in the abstract.
- Domain randomisation versus photorealism trade-off for indoor change detection specifically was not found.

## Q4. Training small heads on frozen features: data sizes, overfitting, curriculum learning

### Takeaway
Small heads on frozen DINOv2/v3 features are standard and data efficient, but I found no source giving specific data-size rules for change-verifier or object-matcher heads; curriculum learning evidence (Wu, Dyer, Neyshabur, ICLR 2021) shows only marginal benefit in standard settings, with gains mainly under noisy labels or limited training time.

### Cited Findings
- DINOv2 and DINOv3 are designed for frozen-backbone use with linear or light heads (linear probes, DPT decoders), with frozen DPT heads competitive on depth. [DINOv2](https://arxiv.org/html/2304.07193v2); [DINOv3](https://arxiv.org/pdf/2508.10104)
- "When Do Curricula Work?" (ICLR 2021 oral) ran thousands of orderings comparing curriculum, anti-curriculum and random orderings; found examples are learned in a consistent implicit order; third-party summary reports marginal benefits in standard settings but significant improvement with noisy data (not verified against full text). [Google Research page](https://research.google/pubs/when-do-curricula-work/); [arXiv 2012.03107](https://arxiv.org/abs/2012.03107v2); [Liner summary](https://liner.com/review/when-do-curricula-work)
- Silwal et al.: model size, augmentation and fine-tuning were the key design choices for pre-trained representations, and these effects carried from sim to real. [arXiv 2310.02219](https://arxiv.org/html/2310.02219v2)

### Inferences
- Overfitting risk for small heads on synthetic data is mostly overfitting to renderer artifacts (perfect edges, clean lighting, identical asset instances across train and test), not parameter count. Mitigations: split by room and by asset, add photometric and blur augmentations, keep heads small (MLP of 1 to 2 layers, or a linear probe plus a few geometric scalars).
- For the object matcher, the same asset appearing in train and test rooms will inflate synthetic metrics; split assets as well as rooms.
- Given the weak curriculum evidence, a 3-4 day project should not invest in curriculum design; an easy-to-hard ordering by change magnitude is cheap to try but should be treated as an ablation, not an assumption.

### Gaps
- No cited typical dataset sizes for change-verifier or cross-session object-matcher heads on frozen features.
- Full text of Wu et al. on limited-data regimes not checked.

## Q5. Fair evaluation design

### Takeaway
Split by scene/room (and by asset for matching), keep a held-out real test set that is never used for tuning, and report a 2x3 grid: sim-only, real-only, sim+real (with ratio sweep), each evaluated on held-out synthetic and held-out real.

### Cited Findings
- The co-training paper's main table compares real-only, real+task-aware sim, real+prior sim, and all, which is the template for reporting. [arXiv 2503.24361](https://arxiv.org/html/2503.24361v2)
- Silwal et al. evaluated zero-shot transfer to a held-out real scene and measured sim-to-real predictivity across many configurations, a model for checking whether synthetic validation ranks models the same way real validation does. [arXiv 2310.02219](https://arxiv.org/html/2310.02219v2)
- Object detection literature uses learning curves over real-data fractions (2.5%, 5%, 10%, ... 100%) to quantify the real-data requirement. [arXiv 1907.07061](https://arxiv.org/pdf/1907.07061); [arXiv 2204.07061](https://arxiv.org/pdf/2204.07061)
- SceneDiff provides 350 real pairs across 50 scenes with added/removed/moved labels and reports AP for multiview and two-view settings; a usable external real benchmark if released. [SceneDiff](https://arxiv.org/abs/2512.16908)

### Inferences
- Recommended protocol: (a) synthetic train/val/test split by procedural room seed, with disjoint asset sets for the matcher test; (b) real data split by physical room or capture session, never by frame; (c) real test set frozen on day 1; (d) report sim-only, real-only, mixed at 2 to 3 ratios, and pretrain-then-fine-tune, plus a real-data-fraction learning curve; (e) include a training-free baseline (e.g., SceneDiff-style feature comparison or plain cosine similarity of DINO features with a threshold) to show the heads add value; (f) report confidence intervals or multiple seeds because real test sets will be small.
- Measure sim-to-real predictivity: does model ranking on synthetic val match ranking on real val? If not, synthetic val should not be used for model selection.

### Gaps
- No change-detection-specific paper found that reports this full sim-only / real-only / mixed grid on a real held-out set; VSCD may, but numbers were not available.
