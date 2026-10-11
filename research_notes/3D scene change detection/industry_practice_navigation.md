# Industry Practice: Environment Change, Lifelong Mapping and Navigation (as of October 2026)

Scope note: research done 2026-10-11. Fifteen or so web searches and fetches. Facts are marked as company or vendor claims where that is the only source. No independent benchmarks of commercial systems were found.

## Q1. Lifelong / long-term SLAM and map maintenance in research (Khronos, Hydra, Panoptic Multi-TSDF, POCD, scene graphs, Nav2 costmaps)

### Takeaway
Research has moved from "filter out dynamics" to explicit object-level, time-aware maps that separate short-term motion (people) from long-term change (rearranged objects). Khronos (RSS 2024) is the reference formulation. Nav2 in practice handles change only at the 2D/2.5D costmap level (raytraced marking/clearing plus mask filters), with known stale-obstacle problems. Nothing standard sits between an object-level change model and the costmap.

### Cited Findings
- Khronos (Schmid, Abate, Chang, Carlone, RSS 2024) defines "Spatio-temporal Metric-semantic SLAM": building a dense metric-semantic model of the world at all times, incrementally — [arXiv 2402.13817](https://arxiv.org/pdf/2402.13817); [RSS 2024](https://roboticsconference.org/2024/program/papers/81/)
- Khronos argues that work on short-term dynamics has been disconnected from long-term change detection, where the scene changes a lot (e.g., furniture rearranged) while the robot is not watching — [arXiv 2402.13817](https://arxiv.org/html/2402.13817v1)
- Khronos architecture: a fast process tracks short-term dynamics in an active window. A slow process reasons about long-term changes using a factor graph. Its spatio-temporal local consistency assumption separates sensing noise, state-estimation error, dynamic objects and long-term change — [arXiv 2402.13817](https://arxiv.org/pdf/2402.13817)
- Khronos change detection keeps rays from background vertices to observing poses. These rays approximate free space and are used to label points as occluded, consistent, or evidence of absence. The first and last observations of presence or absence give appearance and disappearance times — [arXiv HTML](https://arxiv.org/html/2402.13817v1); [review summary](https://themoonlight.io/en/review/khronos-a-unified-approach-for-spatio-temporal-metric-semantic-slam-in-dynamic-environments)
- Panoptic Multi-TSDFs (Schmid et al., 2021/2022): a multi-resolution, per-object volumetric map for changing environments. It reasons at object level for semantic consistency over time and detects long-term object-level changes online. Open-source code is ETH ASL panoptic_mapping — [paper PDF](https://microsoft.com/en-us/research/uploads/prod/2022/10/2109.10165-1.pdf); [ar5iv](https://ar5iv.labs.arxiv.org/html/2109.10165)
- POCD (Qian, Chatrath et al., RSS 2022): each object state combines a stationarity score and a TSDF change measure. A Bayesian update fuses geometric and semantic evidence. It released TorWIC, a real warehouse dataset. The authors conclude that explicit object-level change estimation matters for long-term maintenance — [arXiv 2205.01202](https://arxiv.org/abs/2205.01202v2); [RSS 2022](https://roboticsconference.org/2022/program/papers/013/)
- 3D Variable Scene Graphs / DeltaVSG (ETH, 2023) add a "variability" attribute per scene-graph node to predict which objects are likely to change. 77.1% accuracy and 72.3% recall on 3RScan. Active change detection ran 66% faster than with a change-agnostic planner — [arXiv 2209.07896](https://arxiv.org/html/2209.07896v2)
- DovSG (2024/2025): dynamic open-vocabulary 3D scene graphs for long-term mobile manipulation. 35% long-term task success vs 5% for baselines, with lower memory and update time (per alphaXiv summary) — [alphaXiv 2410.11989](https://alphaxiv.org/overview/2410.11989v6)
- Lifelong LERF (2024): updates a language-embedded radiance field from SLAM poses and detects semantic changes. Up to 91% change-tracking accuracy, but only in small tabletop setups with 3 to 5 objects — [arXiv 2403.10494](https://arxiv.org/pdf/2403.10494)
- CubifyGS (arXiv, June 2026): object-centric 3D Gaussian Splatting for lifelong maintenance. Moved rigid objects are retrieve/transform/prune edits on reusable object assets, which avoids ghosting. Optimisation is event-triggered on affected regions — [arXiv 2606.28720](https://arxiv.org/abs/2606.28720)
- NVIDIA nvblox (Isaac ROS) assumes a static scene by default. It has a "human" mode where PeopleSemSegNet masks send people pixels into a separate decaying occupancy layer, and a general "dynamic" mode using a freespace-based method (Dynablox) — [nvblox paper arXiv 2311.00626](https://arxiv.org/pdf/2311.00626); [isaac_ros_nvblox mirror](https://github.com/GOFIRST-Robotics/isaac_ros_nvblox); [NVIDIA Isaac ROS DP3 blog](https://developer.nvidia.com/blog/build-high-performance-robotic-applications-with-nvidia-isaac-ros-developer-preview-3/)
- NVIDIA's stated design rule: people belong in the costmap for collision-free planning, but not in the 3D reconstruction. The reconstruction keeps static obstacles only, and dynamic parts decay over time — [NVIDIA blog](https://developer.nvidia.com/blog/build-high-performance-robotic-applications-with-nvidia-isaac-ros-developer-preview-3/); [arXiv 2311.00626](https://arxiv.org/pdf/2311.00626)
- One user report says nvblox costmap clearing lagged about 3 to 4 s behind reality (anecdotal, unresolved) — [NVIDIA forum](https://forums.developer.nvidia.com/t/nvblox-not-working-in-realtime/323644)
- Nav2 KeepoutFilter reads a keepout mask and marks restricted regions so the planner and controller avoid them. It is a mask-driven filter, not sensor-driven — [Nav2 API (Jazzy)](https://api.nav2.org/nav2-jazzy/html/classnav2__costmap__2d_1_1KeepoutFilter.html)
- Nav2 VoxelLayer exposes `clearStaticObservations(marking, clearing)` and area-clearing methods. Obstacle and voxel sources have per-source `marking`/`clearing` flags and `raytrace_max_range` — [Nav2 API VoxelLayer](https://api.nav2.org/nav2-jazzy/html/classnav2__costmap__2d_1_1VoxelLayer.html); [ROS Answers](https://answers.ros.org/question/403262/obstacle-and-voxel-layer-not-raytracing-free-space-properly/)
- Known Nav2 pain point: obstacles marked by obstacle and voxel layers often are not cleared after they move, leaving "trails" that block the robot. Users then switch to alternatives such as nonpersistent_voxel_layer — [ROS Answers](https://answers.ros.org/question/403262/obstacle-and-voxel-layer-not-raytracing-free-space-properly/); [nonpersistent_voxel_layer docs](https://docs.ros.org/en/kilted/p/nonpersistent_voxel_layer/)
- Nav2 has a speed filter alongside the keepout filter (Humble onward) — [Nav2 API (Humble)](https://api.nav2.org/nav2-humble/namespacenav2__costmap__2d.html)
- Cloud map sharing in warehouses (Magazino with Cartographer, ROSCon 2018): robots run finite-history SLAM and stream local map changes to a cloud service. Sharing changes kept localisation accurate in a dynamic customer warehouse — [Google Research](https://research.google/pubs/cloud-based-mapping-and-localization-in-dynamic-warehouse-environments/)

### Inferences
- Change in the stack has three timescales. (a) Seconds: people and forklifts, handled by decaying or nonpersistent costmap layers. (b) Hours to days: moved pallets, totes, carts and doors, which is semi-static change, the POCD/Khronos regime, and the gap a change detector fills. (c) Weeks: layout changes that need re-mapping. A submission that explicitly targets (b) and says how it hands off to (a) and (c) will read as credible.
- Nav2 has no native "object moved" concept. The cleanest integration points are: (1) publishing an updated static map or map-update patch to the static layer, (2) publishing a mask to a costmap filter (keepout or speed) for "changed, unverified" regions, or (3) a custom costmap layer plugin that ingests a change list. Option (3) is most flexible.
- The stale-trail problem in Nav2 is a concrete industry pain point. A change detector that confirms "this region is now free" (evidence of absence, as in Khronos) is as valuable as one that reports new obstacles.

### Gaps
- Could not fetch Nav2 static-layer docs (404). Static-layer parameters such as `subscribe_to_updates` and `map_subscribe_transient_local` are from prior knowledge, not verified in this session.
- Did not find Hydra follow-ups (e.g., Hydra-Multi, Clio) or how MIT SPARK connects scene graphs to Nav2-style planners. Searches were not run on this specifically.
- No published head-to-head comparison of Khronos vs POCD vs Panoptic Multi-TSDF on the same warehouse data was found.

## Q2. What deployed robots and companies do about rearranged environments (Boston Dynamics, ANYbotics, Agility, Figure, AMRs, Humanoid)

### Takeaway
Deployed systems mostly use a recorded map or route graph, local obstacle avoidance and rerouting, and human re-recording or editing when things change a lot. Commercial "change detection" is mostly an inspection product (did the gauge, spill or thermal reading change), not a map-maintenance feature. Humanoid (SKL Robotics) publicly says it wants "long-term persistent spatial memory" and environment-state tracking, but has published no technical detail on how it does it.

### Cited Findings
**Boston Dynamics Spot (GraphNav / Autowalk / Orbit / Site View)**
- GraphNav localises by comparing live sensor data with data stored during map recording. Docs list "the environment has changed dramatically since the map was recorded" as a cause of poor localisation, and say localisation quality correlates with the amount of change (example: an empty vs full parking lot) — [BD GraphNav Localization](https://dev.bostondynamics.com/docs/concepts/autonomy/localization); [Autonomy Technical Summary](https://dev.bostondynamics.com/docs/concepts/autonomy/graphnav_tech_summary)
- When lost, the GraphNav service refuses to navigate autonomously and does not self-recover. Re-initialisation (fiducials or local geometry) is required, and the recommended fix for site changes that cause STATUS_LOST is to record a new map — [BD GraphNav Service](https://dev.bostondynamics.com/docs/concepts/autonomy/graphnav_service); [BD Initialization](https://dev.bostondynamics.com/docs/concepts/autonomy/initialization)
- Directed Exploration lets Spot use its sensors to find a path to a nearby waypoint when blocked by unexpected obstacles or doors that changed state. It handles blockage, not lost localisation — [BD Directed Exploration](https://dev.bostondynamics.com/docs/concepts/autonomy/directed_exploration)
- Site View (Orbit) is described as "a clear visual history of your facility powered by 360° images captured by Spot" for remote review over time. The product page describes no automated cross-date image comparison — [BD blog, c. May 2025](https://bostondynamics.com/blog/see-your-facility/)
- Orbit visual inspections use "pre-programmed vision-language prompts" on photos of points of interest (spills, missing fire extinguishers, pallet status). Thermal anomaly detection uses statistical min/max thresholds across samples. These are vendor claims with no accuracy figures — [BD blog](https://bostondynamics.com/blog/see-your-facility/)
- Orbit 5.0 coverage: alerts on changes, search of past inspections, AI detection of visual anomalies such as debris, spills and corrosion (marketing, via trade press) — [The Robot Report](https://www.therobotreport.com/orbit-5-0-adds-features-boston-dynamics-spot-quadruped-robot/)
- Orbit 5.1: acoustic change detection flags anomalies relative to nominal-state captures (Fluke SV600 / Sorama L642) — [The Robot Report](https://www.therobotreport.com/boston-dynamics-releases-spot-and-orbit-5-1-with-new-spot-cam/)

**ANYbotics ANYmal**
- Missions are taught by guiding the robot and marking inspection points, then repeated autonomously. CAD models can be imported to set up points and paths — [ANYbotics ANYmal page](https://www.anybotics.com/anymal-autonomous-legged-robot/)
- Oct 2023 update: improved loop closure when recording large maps, and "self-recovery by setting anchor points for route map clarification." Claims faster detection of terrain, 3D layers and moving obstacles, and recognising inspection objects "even if it has moved" — [ANYbotics Oct 2023 update](https://www.anybotics.com/news/anymal-product-update-october-2023/)
- Release 23.02: operators can select and edit "Environment Objects" in Workforce to correct or update them in the field. Map maintenance here is a human-in-the-loop edit — [ANYbotics 23.02 release](https://www.anybotics.com/news/23-02-product-release-anybotics)

**Agility Robotics (Digit) / Figure**
- Agility Arc is a cloud platform covering "facility mapping and workflow definition to operational management and troubleshooting." No public detail on map-update mechanics — [BusinessWire, Mar 2024](https://businesswire.com/news/home/20240311483582/en/Agility-Robotics-Brings-Operational-Visibility-to-Deployment-of-Digit%e2%93%87-Fleets-with-the-Launch-of-Agility-Arc%e2%84%a2); [Control.com](https://control.com/news/agility-robotics-develops-cloud-automation-platform-for-robot-fleet-management/)
- Agility (2022) stresses a navigation stack "focused on real-world deployments" in dynamic warehouses with tight spaces — [Agility blog](https://agilityrobotics.com/2022/charting-a-path)
- Figure: no public material found on mapping or change handling.

**Humanoid (SKL Robotics Ltd, London)**
- Job posting "Senior/Staff Autonomy Engineer, Spatial Intelligence" (London) asks the hire to: lead state estimation, long-term mapping and localisation loops; integrate ML and "foundation geometry models" into the core navigation stack; build a "spatial intelligence layer" unifying 3D geometry with semantic scene understanding for safe navigation and locomotion; give robots "long-term persistent spatial memory"; implement 3D representations that "track and interpret environment state"; deploy low-latency models on edge hardware. It names PyTorch, VLMs and transformers, but not ROS 2, Nav2, nvblox or a SLAM framework — [vcbacked.co mirror of posting](https://www.vcbacked.co/jobs/role/humanoid-senior-autonomy-engineer-spatial-intelligence-67f1d4a4)
- Other listings (now closed or 404): "Senior Perception Engineer - Spatial Understanding & Navigation" and "Senior ML Engineer (VLA & Navigation)". The latter mentions open-ended navigation driven by VLA models (from search snippet) — [thehumanoid.ai vacancy (closed)](https://thehumanoid.ai/vacancy/perception-lead/); [itjobswatch (404 at fetch)](https://www.itjobswatch.co.uk/jv/Humanoid/Senior-Perception-Engineer-Spatial-Understanding-Navigation-Job-Slough-Berkshire-UK-4xnyy4)
- KinetIQ (Humanoid's AI stack, company claims): four layers. System 3 is agentic fleet orchestration. System 2 is an omni-modal model that calls tools including "navigation, manipulation policies, perception queries, recovery behaviors" and "re-localization", and monitors "status reports from navigation or perception systems". System 1 is a VLA at 10 Hz taking multi-camera input. System 0 is RL whole-body control. It acknowledges "objects move" and the environment "changes in ways that cannot be perfectly predicted". No mapping or spatial-memory details are published — [thehumanoid.ai/ai](https://thehumanoid.ai/ai/); [The Robot Report](https://www.therobotreport.com/kinetiq-framework-from-humanoid-orchestrates-robot-fleets/)
- Deployments: Siemens Erlangen POC with HMND 01 wheeled Alpha doing tote de-stacking to a conveyor. Reported 60 tote moves/hour, more than 30 min continuous autonomy, more than 8 h uptime, pick-and-place success above 90% (joint company claims) — [Robotics 24/7](https://www.robotics247.com/article/humanoid-siemens-complete-proof-of-concept-to-test-humanoid-robots-in-industrial-logistics); [The Robot Report](https://www.therobotreport.com/humanoid-siemens-proof-of-concept-may-lead-more-industrial-deployments/)
- Schaeffler: Oct 2025 POC picking bearing rings. May 2026 deal for "thousands" of robots, initial phase Dec 2026 to Jun 2027 at Herzogenaurach (box handling) and Schweinfurt (from press, no primary Schaeffler release found) — [AI Business](https://aibusiness.com/robotics/humanoid-schaeffler-bring-thousands-robots-to-factories); [humanoid.guide](https://humanoid.guide/?p=11552)
- HMND 01 Alpha Wheeled sensors (third-party product page): high-resolution cameras, depth sensing, proprioception — [Humanoid Robotics Technology](https://humanoidroboticstechnology.com/company/humanoid/hmnd-01-alpha-wheeled/). A French site claims 40 m lidar, RGB-D, 60 Hz fusion and an "83% obstacle bypass rate". Unverified and not in Humanoid's own material — [xpert.digital](https://xpert.digital/fr/robot-humanoide-modulaire)

### Inferences
- Humanoid's stated stack is ML-heavy (VLA, foundation geometry models, PyTorch, edge deployment) with a classical state-estimation and mapping core. A change detector that outputs structured, queryable results fits as either a "perception query" tool for System 2 or an input to the mapping layer that keeps "persistent spatial memory" current.
- Their deployments (tote stacks, conveyors, box handling, bearing rings) are factory and intralogistics cells where totes, pallets, carts and stacks move often. This is semi-static object change, not layout change.
- The industry default when the map goes stale is still "re-record the map" (Boston Dynamics) or "operator edits objects" (ANYbotics). Automated, trustworthy map patching is an open product gap.

### Gaps
- No public talks, papers or blog posts by Humanoid staff on perception or navigation found. KinetIQ pages name no staff.
- Humanoid's exact sensor suite, middleware (ROS 2 or not) and planner are not publicly confirmed.
- No details on how warehouse AMR vendors (Locus, OTTO, MiR) or Figure handle rearrangement. Not searched in depth.

## Q3. Known pain points (false positives, stale maps, relocalisation, compute, dynamic objects, sensor timing)

### Takeaway
The documented pain points are: viewpoint- and lighting-induced false changes, stale costmap trails, localisation loss after large changes, separating people from semi-static change, and latency on embedded GPUs. Sensor timing and synchronisation is implicitly critical (all 3D change methods depend on accurate poses) but I found no public industry write-up specific to it.

### Cited Findings
- SceneDiff (arXiv 2512.16908, Dec 2025) names viewpoint variation as the main obstacle: varying viewpoints can make objects falsely appear changed. Benchmark of 350 video pairs with object-level change annotations. Training-free 3D-aligned method with 94% and 37.4% relative AP gains (multi-view and two-view) — [papers.cool 2512.16908](https://papers.cool/arxiv/2512.16908)
- Multi-View Pose-Agnostic Change Localization (CVPR 2025): change-aware 3DGS from as few as 5 post-change images. Releases a real dataset with lighting variation. 1.7x mIoU and 1.5x F1 over baselines — [arXiv 2412.03911](https://arxiv.org/abs/2412.03911v2)
- Online follow-up (arXiv 2511.12370): claims the first online, pose-agnostic, label-free, multi-view-consistent scene change detection at more than 10 FPS. Stresses separating real object movement from shadows and reflections — [arXiv 2511.12370](https://arxiv.org/html/2511.12370v2)
- 2D foundation-model change detection evaluated on VL-CMU-CD and PSCD with viewpoint-varied versions, claiming robustness to photometric and geometric variation — [arXiv 2409.16850](https://arxiv.org/html/2409.16850v3)
- Stale maps causing localisation loss (Spot) and no autonomous recovery — [BD Localization](https://dev.bostondynamics.com/docs/concepts/autonomy/localization); [BD GraphNav Service](https://dev.bostondynamics.com/docs/concepts/autonomy/graphnav_service)
- Stale costmap obstacle trails in Nav2 — [ROS Answers](https://answers.ros.org/question/403262/obstacle-and-voxel-layer-not-raytracing-free-space-properly/)
- Dynamic vs static separation: nvblox routes people into a separate decaying layer. Users report costmap clearing latency of seconds — [arXiv 2311.00626](https://arxiv.org/pdf/2311.00626); [NVIDIA forum](https://forums.developer.nvidia.com/t/nvblox-not-working-in-realtime/323644)
- Khronos explicitly frames the need to separate state-estimation error, sensing noise, dynamic objects and true long-term change, which is the root of false positives in map-based change detection — [arXiv 2402.13817](https://arxiv.org/pdf/2402.13817)
- Humanoid's posting stresses low-latency edge deployment of large models, so compute is a stated constraint — [vcbacked.co posting](https://www.vcbacked.co/jobs/role/humanoid-senior-autonomy-engineer-spatial-intelligence-67f1d4a4)
- KinetIQ's own page notes current VLMs "perceive dynamics and physics poorly" because training data is largely static. This motivates explicit geometric change reasoning alongside VLAs (company opinion) — [thehumanoid.ai/ai](https://thehumanoid.ai/ai/)

### Inferences
- A credible submission should report false-positive behaviour under viewpoint and lighting change, not just recall. SceneDiff and the CVPR 2025 dataset are the current reference benchmarks.
- Pose error looks like change. Reporting sensitivity to pose or extrinsic noise and to camera-depth time offset is a cheap but differentiating validation step. It speaks to "validating perception quality", though no public industry source on timestamping was found.
- Gating changes by persistence (seen across N views or T seconds) and filtering a "dynamic" class (people, forklifts) before declaring semi-static change matches the nvblox and Khronos split.

### Gaps
- No public industry data on false-positive rates for commercial change detection (BD and ANYbotics publish none).
- No public sources found on camera timestamping or synchronisation validation practice at humanoid companies.
- On-robot compute budgets for HMND 01 are not public.

## Q4. What output format is most useful to a navigation stack

### Takeaway
No industry standard exists. The evidence points to a layered output: (1) an object-level change list (add, remove, move, with class, 3D box or mask, timestamps, confidence) as the primary semantic product, which is what POCD, Khronos, 3D VSG and DovSG reason over, and (2) a derived occupancy or costmap delta (free-to-occupied and occupied-to-free cells with confidence) that plugs into Nav2-style layers or filters. Scene-graph diffs are the natural form for a VLA/LLM planner such as KinetIQ System 2.

### Cited Findings
- Object-level probabilistic state (stationarity plus change measure per object) is POCD's representation, and the authors argue explicit object-level change estimation is key for long-term maintenance — [arXiv 2205.01202](https://arxiv.org/abs/2205.01202v2)
- Khronos outputs per-object appearance and disappearance time estimates from presence/absence evidence, so changes carry time intervals, not just labels — [arXiv 2402.13817](https://arxiv.org/html/2402.13817v1)
- 3D VSG attaches per-node variability (likelihood of change) to scene graphs and uses it to plan where to look, cutting active change-detection time by 66% — [arXiv 2209.07896](https://arxiv.org/html/2209.07896v2)
- CubifyGS represents moved objects as transform edits on object assets (retrieve, transform, prune) — [arXiv 2606.28720](https://arxiv.org/abs/2606.28720)
- Nav2 consumes change through layers and filters: obstacle/voxel marking and clearing, and mask-based filters (keepout, speed) — [Nav2 KeepoutFilter API](https://api.nav2.org/nav2-jazzy/html/classnav2__costmap__2d_1_1KeepoutFilter.html); [Nav2 VoxelLayer API](https://api.nav2.org/nav2-jazzy/html/classnav2__costmap__2d_1_1VoxelLayer.html)
- nvblox already produces a costmap slice from a 3D map with a separate dynamic occupancy layer, which is a ready template for "3D map to 2D costmap delta" — [arXiv 2311.00626](https://arxiv.org/pdf/2311.00626)
- KinetIQ System 2 consumes "status reports from navigation or perception systems" and calls "perception queries" and "re-localization" as tools, so structured, language-addressable change reports suit their top-level planner (company description) — [thehumanoid.ai/ai](https://thehumanoid.ai/ai/)
- Spot's response to blocked paths (Directed Exploration) and to lost localisation (re-record) shows the two downstream consumers: path re-planning and localisation map refresh — [BD Directed Exploration](https://dev.bostondynamics.com/docs/concepts/autonomy/directed_exploration); [BD Localization](https://dev.bostondynamics.com/docs/concepts/autonomy/localization)

### Inferences
- Suggested schema per change: id, type (added, removed, moved, unknown), semantic class and open-vocab label, 3D oriented box or voxel set in the map frame, old and new pose for moved objects, first and last seen timestamps, confidence or probability, evidence count or views, and "navigation relevance" (traversable-space impact, height band vs robot footprint).
- Derived products: (a) a 2D costmap delta or mask (OccupancyGrid-style) for a custom Nav2 layer or filter, with "changed but unverified" regions as elevated cost rather than lethal; (b) a flag for localisation-relevant change (large static structure changed) to trigger map refresh or re-localisation; (c) a scene-graph diff (node add, remove, move) for task planning and fleet-level spatial memory.
- Uncertainty matters because the costs are asymmetric. A false "removed" (clearing a real obstacle) is a safety risk. A false "added" mostly costs efficiency. So clearing should need higher confidence than marking. This follows from the Nav2 stale-trail and safety context and is not a published industry rule.
- For a Humanoid-facing submission: frame output as feeding (1) the mapping layer's "persistent spatial memory", (2) the costmap for safe locomotion, and (3) a perception-query API for System 2.

### Gaps
- No public document from any company specifies a change-detection output format consumed by its navigation stack.
- No standard ROS 2 message for object-level change sets was identified in this search. Whether one exists (e.g., in vision_msgs or scene-graph packages) is unverified.
