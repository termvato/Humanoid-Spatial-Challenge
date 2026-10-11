# Open-vocabulary 3D object maps, cross-session object re-identification, and VLMs for change description (as of October 2026)

Scope note: these notes cover building blocks for "two videos of a room -> object-level 3D maps with labels -> match objects across sessions (moved vs removed + added) -> natural-language change descriptions", on a 3-4 day budget with cloud GPUs. Licence facts and repo activity were checked through the GitHub API on 2026-10-11 unless stated otherwise. Works from before 2025 are marked [background]. Numbers are author-reported unless stated otherwise, and should be treated sceptically.

## Q1. Open-vocab 3D mapping and scene graphs (ConceptGraphs, HOV-SG, OpenMask3D, Open3DIS, successors incl. SAM 3 / DINOv3 / feed-forward reconstruction)

### Takeaway
The practical default is still a ConceptGraphs-style pipeline (posed RGB-D frames, 2D class-agnostic masks per frame, back-project, merge 3D segments when geometric overlap and feature similarity pass thresholds, attach CLIP/DINO features per object). ConceptGraphs and HOV-SG are MIT-licensed and maintained; 2025-2026 successors (TrackGraph, OASIS-Map front end, OVI-MAP, OVO-SLAM) mainly swap in DINOv3 for tracking/association and cut memory, but most lack released code. Feed-forward reconstruction (VGGT, MapAnything, Depth Anything 3) is now a credible way to get poses and depth from plain phone video, removing the need for an RGB-D sensor.

### Cited Findings
**Classic pipelines [background, but still the baseline everyone compares to]**
- ConceptGraphs repo is MIT-licensed, ~968 stars, last push 2025-10-16 (maintenance slowing) — [GitHub concept-graphs/concept-graphs](https://github.com/concept-graphs/concept-graphs)
- HOV-SG (RSS 2024) is MIT-licensed, ~550 stars, last push 2026-01-19; initial release of mapping and graph construction engine was July 2024 — [GitHub hovsg/HOV-SG](https://github.com/hovsg/HOV-SG)
- OpenMask3D (NeurIPS 2023) is MIT-licensed but the repo has not been pushed since 2023-12-15 — [GitHub OpenMask3D/openmask3d](https://github.com/OpenMask3D/openmask3d)
- Open3DIS: the VinAI-Research/Open3DIS GitHub path returned "Not Found" on the API; I could not confirm its current repo location or licence — (API check, see Gaps)
- ConceptGraphs-style merging is described in OASIS-Map as merging 3D masks "when geometric overlap and semantic similarity pass thresholds", supplemented by 2D tracking in low-FoV scenes — [OASIS-Map, arXiv 2607.14899](https://arxiv.org/html/2607.14899)
- Reported efficiency (from OVO-SLAM, on an RTX 3090, quoted by TrackGraph as "indicative only"): ConceptGraphs about 16 min and HOV-SG about 11 h per Replica scene — [TrackGraph, arXiv 2609.31005](https://arxiv.org/html/2609.31005)

**2025-2026 successors**
- TrackGraph (arXiv 2609.31005, Sept 2026): input RGB-D + odometry; FastSAM-x masks at keyframes, propagated between keyframes with DINOv3 ViT-S+ (29M) features; keyframe masks matched to tracks by IoU + DINOv3 similarity; fused into a TSDF with up to four weighted hypotheses per voxel; tracks merged into segments when vertex overlap and feature similarity pass thresholds; OpenCLIP ViT-H/14 features every 24 frames for open-vocab retrieval; built on Hydra — [TrackGraph](https://arxiv.org/html/2609.31005)
- TrackGraph reports 8.9 min vs 15.1 min and 8.3 GB vs 27.2 GB GPU memory against OVI-MAP (ViT-H) on Replica on an A100; no code release mentioned — [TrackGraph](https://arxiv.org/html/2609.31005)
- On the OpenLex3D benchmark, TrackGraph reports Replica object retrieval AP25 30.81 vs ConceptGraphs 22.39 vs HOV-SG 25.30, but on HM3D ConceptGraphs is better on AP (5.09 vs 2.16) and HOV-SG is better on ScanNet++ synonyms (0.40 vs 0.35). Absolute APs are low (single digits) for all methods — [TrackGraph](https://arxiv.org/html/2609.31005)
- Other 2025-2026 open-vocab mapping methods cited by TrackGraph: OVI-MAP (CVPR 2026), FindAnything (ICRA 2026), OpenVox (IROS 2025), ThinkGraphs (ECCV 2026), OVO-SLAM (RA-L 2025), OpenTrack3D (CVPR 2026 Findings), and the OpenLex3D benchmark (NeurIPS 2025) — [TrackGraph](https://arxiv.org/html/2609.31005)
- OpenTrack3D fuses DINO visual features with spatial cues to keep instance consistency across views — [OpenTrack3D, arXiv 2512.03532](https://arxiv.org/html/2512.03532)
- Other recent scene-graph papers surfaced but not read in detail: OVIP-SG (instance-preserving scene graphs for small objects, arXiv 2608.17633) and Prior-SG (arXiv 2608.06170, which mentions Grounding DINO and SAM3 as open-set primitives) — [OVIP-SG](https://arxiv.org/pdf/2608.17633); [Prior-SG](https://arxiv.org/pdf/2608.06170)
- "Bare Necessities" (arXiv 2412.01539) argues for simple, effective open-vocab scene graph designs [background] — [arXiv 2412.01539](https://arxiv.org/pdf/2412.01539)

**Feed-forward reconstruction (for plain video, no depth sensor)**
- MapAnything (Meta) is Apache-2.0, ~3.8k stars, last push 2026-08-07; it is a framework that lets VGGT, DUSt3R, MASt3R, MUSt3R, Pi3-X, DA3 and others be used through one interface and outputs metric reconstruction — [GitHub facebookresearch/map-anything](https://github.com/facebookresearch/map-anything); [MapAnything project](https://map-anything.github.io/)
- VGGT infers camera parameters, point maps, depth maps and 3D point tracks from one to hundreds of views — [VGGT, arXiv 2503.11651](https://arxiv.org/abs/2503.11651)
- The VGGT repo licence is reported by GitHub as NOASSERTION (custom licence), ~14.5k stars — [GitHub facebookresearch/vggt](https://github.com/facebookresearch/vggt)
- VGGT-SLAM 2.0 targets real-time dense feed-forward reconstruction (arXiv 2601.19887) — [VGGT-SLAM 2.0](https://arxiv.org/pdf/2601.19887)
- OVSeg3R takes a video, runs a 3D reconstructor (default MASt3R-SLAM) and a 2D instance segmenter (default DINO-X) and lifts 2D labels into 3D — [OVSeg3R, arXiv 2509.23541](https://arxiv.org/pdf/2509.23541)
- SegVGGT (ECCV 2026, IDEA-Research) does joint reconstruction and 3D instance segmentation from unposed multi-view images in one forward pass; open-vocab capability not confirmed — [GitHub IDEA-Research/SegVGGT](https://github.com/IDEA-Research/SegVGGT)
- FAST3DIS uses Depth Anything 3 as a frozen backbone with LoRA for instance semantics — [FAST3DIS, arXiv 2603.25993](https://arxiv.org/pdf/2603.25993)
- A VGGT-based semantic SLAM fuses 2D instance segmentation with the VGGT tracking head to get persistent 3D object IDs across occlusions and re-entries — [arXiv 2511.16282](https://arxiv.org/html/2511.16282v1)
- OASIS-Map shows Depth Anything 3 as a usable learned-depth substitute when no depth sensor exists, with wider keyframe thresholds (1.0 m / 30 deg) for learned depth — [OASIS-Map](https://arxiv.org/html/2607.14899)

### Inferences
- For a 3-4 day build, forking ConceptGraphs (MIT) or writing a lean equivalent (~500 lines: back-project masks, voxel downsample, greedy merge on 3D IoU + DINOv3/CLIP cosine) is lower risk than adopting any 2026 paper without code.
- Phone video path: MapAnything or VGGT for poses + metric depth per session, then the ConceptGraphs-style fuse. Long videos need keyframe subsampling (tens to low hundreds of frames) for feed-forward models to fit GPU memory.
- Low absolute OpenLex3D APs across all methods suggest open-vocab labels on 3D instances are noisy; a VLM relabel pass on best-view crops is likely worth it.

### Gaps
- Open3DIS repo location and licence not verified (API returned Not Found at the VinAI path).
- No paper found that combines SAM 3 + DINOv3 + feed-forward reconstruction end to end with released code.
- VGGT custom licence terms (commercial vs non-commercial checkpoint) not read directly.
- No independent speed benchmark of these pipelines on identical hardware.

## Q2. Segmentation and features: SAM 3 / SAM 3.1, DINOv3, CLIP / SigLIP 2, Grounding DINO / OWLv2

### Takeaway
SAM 3 (848M params) is the strongest single component: text-prompted exhaustive instance segmentation plus video tracking, under the custom "SAM License" that grants a royalty-free worldwide licence with no explicit commercial exclusion but with trade-control/military restrictions; checkpoints are gated on Hugging Face. DINOv3 is under a custom DINOv3 licence that secondary sources say permits commercial use with restrictions; access requires registration. Both are fine for a prototype; the gating is a practical setup step.

### Cited Findings
- SAM 3 (official repo facebookresearch/sam3): adds exhaustive segmentation of all instances of an open-vocabulary concept specified by a short text phrase or exemplars — [GitHub facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- SAM 3 has 848M parameters, with a detector and a tracker sharing a vision encoder; video predictor accepts text prompts and a JPEG folder or MP4 — [GitHub facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- SAM 3.1 (released 2026-03-27) adds "Object Multiplex", shared-memory joint multi-object tracking described as faster without accuracy loss; checkpoints at facebook/sam3.1 on Hugging Face — [GitHub facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- Requirements: Python 3.12+, PyTorch 2.7+, CUDA 12.6+; checkpoints are gated (request access on Hugging Face, then authenticate) — [GitHub facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- The README gives no GPU memory figure — [GitHub facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- SAM License text: grant is "non-exclusive, worldwide, non-transferable and royalty-free" with no field-of-use clause excluding commercial use; prohibits end uses barred by trade controls including military/warfare and ITAR; prohibits reverse engineering; redistribution must include a copy of the agreement; published research must acknowledge use; licence terminates if you sue Meta — [SAM LICENSE](https://github.com/facebookresearch/sam3/blob/main/LICENSE)
- DINOv3 weights are under a custom DINOv3 licence (dated 14 Aug 2025), not Apache 2.0 like DINOv2 — [Meta DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license); summarised as royalty-free for research and commercial use with trade-control, military, ITAR, nuclear, espionage exclusions by [Roboflow](https://inference-models.roboflow.com/models/dinov3-classification/)
- A DINOv3 GitHub issue notes the custom licence seems to allow commercial use but requires ad-hoc legal review — [GitHub dinov3 issue 31](https://github.com/facebookresearch/dinov3/issues/31)
- DINOv3 weight access requires sign-up/approval — [DrivenData forum](https://community.drivendata.org/t/is-dinov3-allowed-to-use/11488)
- Not all DINOv3-derived checkpoints share terms: Lightly's distilled ViT-T models are Apache 2.0 while EUPE models are FAIR Noncommercial — [Lightly docs](https://docs.lightly.ai/train/0.16.3/pretrain_distill/models/dinov3.html)
- DINOv3 as region/object descriptor in practice: OASIS-Map stores per-keyframe DINOv3 dense patch features and per-object feature tokens (not averaged) and reports DINOv3 at ~10 ms per keyframe on an RTX 4090; TrackGraph uses DINOv3 ViT-S+ for mask propagation and revisit matching — [OASIS-Map](https://arxiv.org/html/2607.14899); [TrackGraph](https://arxiv.org/html/2609.31005)
- OASIS-Map uses SAM2 (not SAM 3) at ~100 ms per keyframe for 10 objects, total GPU memory under 10 GB — [OASIS-Map](https://arxiv.org/html/2607.14899)
- Recent open-vocab mapping still commonly uses OpenCLIP ViT-H/14 for text-queryable features — [TrackGraph](https://arxiv.org/html/2609.31005)

### Inferences
- Recommended stack: SAM 3 (or 3.1) video predictor with a room-object vocabulary list or a generic "object" prompt for masks with consistent IDs across a session; DINOv3 (ViT-S+/B, cheap) for per-object appearance descriptors used in cross-session matching; CLIP or SigLIP 2 for text-queryable labels; VLM relabel of best crops for final names.
- Fallback if gating/approvals are slow: SAM 2 + Grounding DINO or OWLv2 for detection, DINOv2 (Apache 2.0) for descriptors.
- SAM 3 at 848M params plus video memory bank should fit on a 24-80 GB cloud GPU, but this is not documented; budget an A100/H100 or L40S.

### Gaps
- SAM 3 GPU memory/throughput on long videos not documented in the README; no independent figure found.
- SigLIP 2, Grounding DINO and OWLv2 licences and comparative accuracy were not re-verified in this pass (believed Apache 2.0 for SigLIP 2 and OWLv2 from prior knowledge, unconfirmed here).
- SAM 3 paper (arXiv 2511.16719) numbers not independently checked.

## Q3. Object re-identification across sessions / instance matching between two scans

### Takeaway
The strongest directly relevant 2026 work is OASIS-Map: dense DINOv3 patch correspondences aggregated per mask, Hungarian assignment on match proportion, joint thresholds, then centroid displacement to label static vs moved, and a visibility check (at least 50% of volume inside the other session's observed region) before calling something appeared/disappeared. Its own numbers show the problem is still hard (moved-object F1 0.353 on 3RScan change detection). SceneDiff (Dec 2025, revised Mar 2026) is the closest benchmark and training-free method for exactly the "two videos of the same scene" setting.

### Cited Findings
- OASIS-Map (arXiv 2607.14899, Oxford DRS): inputs are two sessions of RGB, depth (sensor, LiDAR, or Depth Anything 3) and poses in a shared metric frame; SAM2 masks; DINOv3 dense patch features — [OASIS-Map](https://arxiv.org/html/2607.14899)
- OASIS-Map matching: patch correlation between sessions with forward/backward best matches; aggregate matches per mask pair; compute mean confidence and match proportion; four cases (true match / partial visibility deferred / look-alike distinct object / no counterpart); Hungarian on proportion then joint threshold; geometry not used for matching, only afterwards (centroid displacement) for static vs moved — [OASIS-Map](https://arxiv.org/html/2607.14899)
- OASIS-Map change labels: unmatched objects are only appeared/disappeared if at least 50% of the object's 3D volume lies in the region the other session observed; otherwise "unknown" — [OASIS-Map](https://arxiv.org/html/2607.14899)
- OASIS-Map reported 3RScan results: change detection F1 static 0.663, appear/disappear 0.241, moved 0.353; object association F1 static 0.556, moved 0.667, removed 0.842. ConceptGraphs (run per session, matched offline) gets moved association 0.267 but removed 0.952 (better than OASIS-Map on removed) — [OASIS-Map](https://arxiv.org/html/2607.14899)
- OASIS-Map runtime on RTX 4090: front end ~160 ms/keyframe, GPU memory under 10 GB; code release not stated (project page dynamic.robots.ox.ac.uk/projects/oasis-map) — [OASIS-Map](https://arxiv.org/html/2607.14899)
- Related works cited by OASIS-Map: POCD, POV-SLAM, Khronos, Where's-my-glasses, Living Scenes, SG-Reg, ReScene4D, Lost & Found; datasets 3RScan, ChangeSim, "Nothing stands still" — [OASIS-Map](https://arxiv.org/html/2607.14899)
- Khronos (MIT-SPARK, spatio-temporal metric-semantic SLAM) is BSD-3-Clause, ~541 stars, pushed 2026-10-09 (active), ROS-based — [GitHub MIT-SPARK/Khronos](https://github.com/MIT-SPARK/Khronos)
- SceneDiff (arXiv 2512.16908, Hoiem / Wang group, UIUC): benchmark of 350 video pairs captured along different trajectories with dense instance annotations for added/removed/moved objects; training-free method estimates poses, segments objects, and compares objects with semantic and geometric features; reports 53.0% (multiview) and 30.6% (two-view) relative AP improvements over prior methods — [SceneDiff abstract](https://arxiv.org/abs/2512.16908); project page [yuqunw.github.io/SceneDiff](https://yuqunw.github.io/SceneDiff)
- 3DGS-CD (Lu, Ye, Leonard): change detection for physical object rearrangement from two unaligned image sets, outputs 3D segmentation of changed objects and their pose change, needs as few as one post-change image, no depth or class priors; Apache-2.0, repo last push Feb 2025 — [arXiv 2411.03706](https://arxiv.org/abs/2411.03706); [GitHub 520xyxyzq/3DGS-CD](https://github.com/520xyxyzq/3DGS-CD)
- 3D Semantic MapNet [background, 2024]: 3D detector on posed RGB-D video plus differentiable object matching between two sets of 3D boxes; zero-shot transfer to rearranged Replica, Active Vision and RIO scenes — [arXiv 2403.13190](https://arxiv.org/pdf/2403.13190)
- REMIND (arXiv 2607.09267): re-identification for indoor navigation with a relational disambiguator comparing pairwise spatial relations of detections against a historical distance graph (graph-style matching) — [REMIND](https://arxiv.org/pdf/2607.09267)
- ShelfChange3D (arXiv 2610.01283, Oct 2026): object-level change detection from two RGB-D observations with 3D boxes (retail shelves) — [ShelfChange3D](https://arxiv.org/html/2610.01283v1)
- [background] "Objects Can Move" (ECCV 2022) finds moved/added/removed objects between scan and rescan using RANSAC-estimated rigid transforms — [ECCV 2022 paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930104.pdf); "Has Anything Changed?" uses 2D mask constraints to propagate change to whole objects — [arXiv 2312.01148](https://arxiv.org/html/2312.01148)

### Inferences
- A buildable recipe in days: (1) register session B to session A (global alignment on static structure, e.g. ICP/RANSAC on the background point cloud or joint feed-forward reconstruction of frames from both videos); (2) per-object descriptors = DINOv3 (or patch-level matching as in OASIS-Map) + CLIP/label text + size/shape stats; (3) cost matrix = appearance distance with a label-compatibility gate, Hungarian assignment, reject above threshold; (4) matched + displacement > tau (e.g. 10-20 cm) = moved, matched + small displacement = static; (5) unmatched objects = removed/added only if their location was well observed in the other session, else "unknown". Geometry should not dominate the matching cost because moved objects by definition change position.
- Identical-instance ambiguity (two identical chairs) is the main failure mode; OASIS-Map's "low score, high proportion" case and REMIND's relational cues are the known mitigations. A VLM verification step on crop pairs is a cheap extra filter for ambiguous pairs.
- SceneDiff's benchmark is the best evaluation target for this exact task if its data is downloadable.

### Gaps
- SceneDiff code availability and absolute AP values not confirmed (PDF too large to fetch; abstract page lists no repo; GitHub yuqunw/SceneDiff not found via API).
- POCD and Khronos change-detection numbers on rearrangement benchmarks not read directly.
- No learned cross-session matcher (e.g. SuperGlue-style for objects) with released code found for 2025-2026.

## Q4. VLMs for spatial reasoning and change description (open models, 3D LLMs, benchmarks, frontier models)

### Takeaway
Frontier and open VLMs are reliable at "is object X present/absent" but weak at spatial/positional change: on VLM-SubtleBench (2026), the best model scores 59.9% on spatial-change items vs 95% for humans, and existence accuracy collapses beyond ~32 objects. This argues for computing the change set geometrically (Q3) and using the VLM only to name objects and phrase pre-computed changes, not to discover changes from raw frames. Qwen3-VL (Apache 2.0, 2B to 235B) is the natural keyless default; 3D-LLM-style change models (Situat3DChange/SCReasoner) exist but are research code.

### Cited Findings
- VLM-SubtleBench (arXiv 2603.07888) covers 10 difference types including Spatial (arrangement/relative position), Existence (appeared/disappeared) and Viewpoint — [VLM-SubtleBench](https://arxiv.org/html/2603.07888v1)
- Spatial-change accuracy: GPT-5-thinking 59.9, o3 55.1, GPT-5-main 50.1, Claude-sonnet-4 48.7, Qwen2.5-VL-72B 47.8, Gemini-2.5-pro 44.8, Qwen2.5-VL-7B 39.5, human 95.0 — [VLM-SubtleBench](https://arxiv.org/html/2603.07888v1)
- Existence accuracy: Claude-sonnet-4 87.5 (best), GPT-5-thinking 85.4, Qwen2.5-VL-32B 84.2, Gemini-2.5-pro 79.9; human 97.0 — [VLM-SubtleBench](https://arxiv.org/html/2603.07888v1)
- Existence accuracy stays near-perfect with four or fewer objects but drops below 60% beyond 32 objects (synthetic controls) — [VLM-SubtleBench](https://arxiv.org/html/2603.07888v1)
- Concatenating the two images into one composite input degraded accuracy in 9 of 10 categories; overlap/subtract images helped spatial/existence/viewpoint only for static-camera scenes; adding reasoning helped slightly (GPT-5-main 71.3 to 73.1 avg); grid/highlight prompts gave small or mixed changes; fine-tuning Qwen2.5-VL-7B raised avg 59.4 to 69.5 with modest spatial gains — [VLM-SubtleBench](https://arxiv.org/html/2603.07888v1)
- Note: VLM-SubtleBench evaluated Claude Sonnet 4, not current Claude models; results for newer frontier models were not found — [VLM-SubtleBench](https://arxiv.org/html/2603.07888v1)
- [background] MileBench included CLEVR-Change and Spot-the-Diff change captioning; GPT-4V, Gemini 1.0 and Claude 3 Opus scored low there — [MileBench, arXiv 2404.18532](https://arxiv.org/pdf/2404.18532)
- Situat3DChange (NeurIPS 2025 Datasets and Benchmarks): ~121K QA pairs, 36K change descriptions, 17K rearrangement instructions from ~11K human observations of changes, with egocentric/allocentric and coordinate spatial relations; SCReasoner is a 3D MLLM that compares point-cloud pairs with Mamba-based difference focusing and no extra decoder tokens; dataset and code public (CC-BY-4.0, small repo, ~10 stars) — [arXiv 2510.11509](https://arxiv.org/html/2510.11509v1); [GitHub RuipingL/Situat3DChange](https://github.com/RuipingL/Situat3DChange)
- [background] DyS2Change (WACV 2023): 3D change localisation and captioning from two registered point clouds; 120 synthetic AI2-THOR scenes, 37,715 change pairs, 661,345 captions; captions include object type, change type and spatial relations — [WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Qiu_3D_Change_Localization_and_Captioning_From_Dynamic_Scans_of_Indoor_WACV_2023_paper.pdf)
- C3-Bench (ECCV 2026): 4,996 human-labelled 2D image pairs over 51 change contexts — [C3-Bench](https://link.springer.com/chapter/10.1007/978-3-032-37435-6_14)
- OmniDiff (2025): fine-grained image difference captioning benchmark — [arXiv 2503.11093](https://arxiv.org/html/2503.11093v1)
- Holo-Captioning / HoloScan (July 2026): 15K+ indoor scenes, static scene description, not change — [arXiv 2607.02908](https://arxiv.org/abs/2607.02908)
- 3D VLM hallucination has been studied directly ("Understanding and Evaluating Hallucinations in 3D Visual Language Models") — [arXiv 2502.15888](https://arxiv.org/pdf/2502.15888)
- Qwen3-VL: Apache 2.0, dense 2B/4B/8B/32B plus 30B-A3B and 235B-A22B MoE, Instruct and Thinking variants, FP8 builds; README claims improved spatial perception (positions, viewpoints, occlusion) and 3D grounding. These details come from mirrors/secondary sites, not the official model card — [Kanaries overview](https://docs.kanaries.net/articles/qwen3-vl); [HF mirror README](https://huggingface.co/QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ/blob/main/README.md)

### Inferences
- Frontier models (GPT-5-thinking class, and by extension current Claude/Gemini) are better than open 7-72B models on subtle comparison, but no model is close to human on spatial change; do not trust a VLM to say "the lamp moved from the desk to the shelf" from two raw frames.
- Since existence detection degrades with object count, give the VLM one object (or one candidate pair) at a time rather than whole-room images.
- A keyless default of Qwen3-VL-8B or 30B-A3B on a single cloud GPU is adequate for crop labelling and templated-to-fluent rewriting; Claude via API is a quality upgrade for the final narrative and for ambiguous-pair verification.

### Gaps
- No public evaluation found of current Claude (4.x/5.x), GPT-5.x or Gemini 3 on 3D/multi-view room change description specifically.
- InternVL3/3.5, Molmo, SpatialVLM, SpatialRGPT, 3D-LLM, LEO, Chat-Scene and video-3D LLMs were not re-verified in this pass (search budget); their current status and licences remain unconfirmed here.
- Qwen3-VL official model card not fetched directly.

## Q5. Practical recipes: prompting VLMs with crops / rendered views; hallucination risks

### Takeaway
Evidence favours (a) separate images rather than side-by-side composites, (b) one object or pair per query, (c) letting geometry decide the change type and having the VLM only label, verify and phrase, and (d) structured outputs with an explicit "unsure" option. Hallucination risk is highest for spatial relations and for "missing" claims in cluttered views.

### Cited Findings
- Composite (concatenated) image pairs hurt accuracy in 9/10 categories vs separate images — [VLM-SubtleBench](https://arxiv.org/html/2603.07888v1)
- Highlighting and grid overlays gave only small or mixed improvements; brief reasoning helped slightly; two-step reasoning slightly hurt — [VLM-SubtleBench](https://arxiv.org/html/2603.07888v1)
- Existence judgements degrade sharply as object count rises (below 60% beyond 32 objects) — [VLM-SubtleBench](https://arxiv.org/html/2603.07888v1)
- Partial visibility is a key source of false "disappeared" claims; OASIS-Map defers decisions and requires the object's region to have been observed (at least 50% of volume) in the other session before declaring appeared/disappeared — [OASIS-Map](https://arxiv.org/html/2607.14899)
- SceneDiff frames occlusion/out-of-frame and large viewpoint changes as the core reasons objects falsely appear missing or different — [SceneDiff](https://arxiv.org/abs/2512.16908)
- 3D VLMs exhibit measurable hallucination of objects and relations — [arXiv 2502.15888](https://arxiv.org/pdf/2502.15888)

### Inferences
- Labelling: for each 3D object pick the top-k frames by mask area x sharpness x low truncation, crop with ~20-30% context padding, send 2-3 crops as separate images, ask for JSON {label, short description, confidence}, constrained to a vocabulary seeded by SAM 3/CLIP candidates plus "other".
- Pair verification (only for ambiguous matches): send crop A and crop B as separate images, ask "same physical object instance? yes/no/unsure" with a reason; treat "unsure" as unmatched-unknown.
- Change description: compute the structured diff first (object, label, old/new 3D position, nearest anchor objects and relation such as "on", "next to", computed from 3D boxes), then have the VLM/LLM turn the JSON into prose, instructing it not to add facts outside the JSON. Optionally show before/after crops for colour/naming polish. This keeps spatial claims grounded in geometry, where VLMs are weakest.
- Rendered views: top-down renders with numbered object markers can help an LLM produce coherent room-level summaries, but the evidence above suggests overlay tricks give marginal gains, so they should be secondary to the structured JSON.

### Gaps
- No controlled study found comparing crops vs rendered 3D views vs raw frames for VLM labelling of 3D instances.
- No evaluation found of "JSON-to-prose" grounding fidelity for change captions specifically.
