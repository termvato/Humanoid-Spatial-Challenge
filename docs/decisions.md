# Design decisions

Each decision with its date and reason, in the order I made them. Later decisions can override earlier ones; when they do, they say so. Tests named S1, S2, ... are in [spikes.md](spikes.md).

## D1. Input: plain video primary, iPhone LiDAR as reference (2026-10-11)
- I capture with an app that saves RGB video plus LiDAR depth and ARKit poses (Stray Scanner or Record3D).
- The main pipeline uses plain RGB video only (learned reconstruction, e.g. VGGT / MapAnything), so any phone works.
- LiDAR depth and poses are a measured reference: I report reconstruction error and its effect on change detection.
- LiDAR users can run on sensor data instead. Later: built as `roomdiff.pipeline.run_lidar` (`tools/analyse.py` on a Stray Scanner folder), not as a flag of the main command.
- Why: the brief allows any tools, but reviewers must be able to run it on a plain video, and video-only 3D is the harder, recognised problem.

## D2. Compute: Modal cloud GPU, ~$30 cap (2026-10-11)
- Heavy reconstruction models run on Modal; my laptop RTX 4070 (8 GB) handles small models and development.

## D3. Language: keyless default (2026-10-11)
- Default: local open-vocabulary models for labels, template sentences from 3D facts. A hosted language API would only ever be an optional extra.
- Why: a reviewer must be able to run the full pipeline with no API key or account.
- Later: labels come from Qwen2.5-VL-7B, an open-weight model, which keeps this rule (D15).

## D4. Real data (2026-10-11)
- I planned 6-10 labelled phone pairs, split into train and test by room, with the demo pairs in the test split only. In the end I recorded two pairs (pair01 bedroom, pair02 corridor), so they serve as development and demo data, and held-out SceneDiff pairs are the unseen test.

## D5. Learned components: A (change verifier) and B (object matcher); C (dense change map) only if time (2026-10-11)
- Both are small heads on frozen foundation-model features, trained on Isaac Sim and real pairs, with the sim/real ratio evaluated on held-out real rooms.
- C only after A and B are evaluated end to end.
- Later: not built. Zero-shot DINOv2 matching (B) and a VLM check (A, D15) were enough, and two rooms are too few to train on.

## D6. Compute: cloud for heavy models, no Jetson (2026-10-11)
- A Jetson is an on-robot inference device, slower than my RTX 4070 laptop, so it is no help for training or offline reconstruction. Spend goes to Modal.
- Big models (reconstruction, SAM, large VLMs) run on Modal A100/H100; the laptop runs Isaac Sim rendering and small jobs.

## D7. RTX 3080 Windows PC: deferred, add only if rendering is the bottleneck (2026-10-11)
- I measure Isaac render throughput on the laptop first. If the dataset needs more than ~6 h of rendering, I add the PC as a render worker over SSH.
- Later: never needed.

## D8. First plan: extend SceneDiff (2026-10-11, replaced by D10-D11)
- After a literature review, SceneDiff (ECCV 2026) was the strongest public method, so my first plan was to build on it: add 3D boxes, a 3-state (changed / unchanged / unobserved) map, a learned verifier that must beat SceneDiff's thresholds on its test split, and a costmap delta.
- Reconstruction candidates: pi3 / Pi3X or DA3 (not VGGT), compared in S5 and S9.
- The Isaac role (dataset vs robot demo) and co-training were left to spikes S1-S4.
- Principle I kept throughout: the best evidence decides, not my earlier choices.

## D9. Use case: a robot consumes the output (2026-10-12)
- Consumer: a mobile robot. Outputs serve navigation first: 3D change list, costmap delta, suggested viewpoints. Language is the simplest layer.
- Input: plain video primary (confirms D1); LiDAR + ARKit as an add-on, since S7 showed it is the easier problem.
- Uncertainty: report "unobserved" AND suggest a viewpoint that would resolve it.
- Scale: big (navigation-relevant, > shoebox) changes first; small objects are a bonus at lower confidence.
- Timing: offline, between two visits. Stated in the design note as a concession to the 3-4 day budget and compute; online is the product version.
- Memory: two-visit diff only.
- Robot demo: simple diff-drive robot in Isaac Sim that uses the change map.
- Priority: make the whole thing exist with a sound architecture before extras.

## D10. Architecture question: 3D-first vs building on SceneDiff (2026-10-12, decided by S9)
- The alternative: build a 3D model of each visit, then compare the 3D models.
- Evidence: S7 (3D-first with LiDAR) found the big change cleanly, was immune to the light change, and produced "unobserved" natively. SceneDiff (2D masks on a joint reconstruction) produced most false alarms from lighting (S2, S6), and its 2D masks would have to be lifted to 3D anyway.
- D9 needs per-voxel visibility (unobserved, viewpoints, costmap), which is a 3D-first output.
- Proposed: 3D-first core, with SceneDiff's appearance cues demoted to a small-object bonus that geometry verifies. Pending S9: does plain-video depth (feed-forward model) give a usable map?

## D11. Reconstruction model and pipeline shape (2026-10-12, decided after S10)
Evidence: S5 + S9.
- Model: pi3 gives the best change detection on the benchmark at every setting; Pi3X gives the best geometry on pair01 (ATE 3.3 cm, depth 2.2%) and near-metric scale (1.016) but worse benchmark AP (11.4 vs 15.3, geometry only). VGGT and DA3 are far behind on change detection; MapAnything far behind on geometry (focal 22% off, possibly my integration).
  Choice: pi3 for change detection; Pi3X only if metric scale is needed without the phone's sensors.
- Pipeline: 3D-first core (pi3 -> robust ICP -> voxel carving -> SAM object masks -> DINOv2 moved matching) as the primary system, because it is the only one that yields 3D boxes, unobserved space and a costmap (D9), and it found 5 of 6 changes on pair01 vs SceneDiff's 3. SceneDiff stays as a comparison.
- Not yet proven: that 3D-first beats SceneDiff on held-out benchmark pairs (5 pairs is too few).

## D12. Added / removed first, "moved" is a bonus (2026-10-12, partly superseded by D13)
- For a robot, what matters is space that became blocked (added) or free (removed). "Moved" is a nicer description of a removed + added pair, not a separate need.
- Headline metric: obj/sc AP without change type. Typed AP is reported but not optimised.
- Keep the DINOv2 matcher (it also raised untyped AP, 25.5 -> 34.3, by merging duplicate pairs), but no further work on moved matching.
- Occupancy / costmap grid: not built until D11 is settled.

## D13. "Moved" is required after all; add labels and a restore plan (2026-10-12, supersedes D12 in part)
- The brief's minimum: "Identify and visualize objects or regions that were moved, added, or removed in 3D", and "what changed and where it happened". D12 ranked moved as a bonus; that was a misreading.
- Output per change in 3D: added / removed with position and size; moved as one object with from -> to and distance (from the DINOv2 removed + added pairing).
- Semantic labels are also a minimum requirement and were missing: every change needs a name.
- Extra: a "put the room back" plan (move X from B back to A, return Y, take away Z), shown in the Isaac demo.
- D12 still holds for the robot maps: blocked / free cells do not depend on the change type.

## D14. Isaac demo frozen; focus on false alarms in real video (2026-10-12)
- The deliverable's demo is the two phone videos and what the pipeline finds in them. The Isaac robot demo (tools/demo.py, isaac/) stays as it is.
- Isaac is not the bench for false alarms either: its Simple_Room table renders view-dependently (S14), so its false alarms partly measure the asset.
- Next: cut false alarms on pair01 / pair02, scored per detection against my own ground truth, with no loss on the benchmark tuning AP.

## D15. VLM before/after check as a confidence score (2026-10-13, by the S17 rule)
- Each detection is shown to Qwen2.5-VL-7B next to the same 3D spot seen from the closest camera in the other video: "is an object in one image missing from the other?". "No" multiplies the score by 0.1.
- Evidence: S17 tuning 34.3 -> 38.1, held-out 18.0 -> 23.1, typed 12.6 -> 21.4. As a hard filter (S16) it lost real changes.
- The change list keeps every detection; "confident": false ones are listed last. The same VLM names each change (labels, D13).
- Open: which carving for own rooms. fp+slant keeps the blanket in pair01's confident list, current carving does not; benchmark slightly prefers current. Not decided on 2 rooms.

## D16. fp+slant carving for own videos, current carving on the benchmark (2026-10-13, from S14 + S17 numbers)
- Own pairs, VLM-rescored confident list (tools/score_own.py --confident): pair01 fp+slant 5/6 with 6 false alarms vs current 4/6 with 12 (current loses the blanket); pair02 4/4 with 6 vs 4/4 with 9. All detections: 30 vs 46 and 28 vs 33 false alarms. fp+slant wins every count.
- Benchmark: current carving slightly ahead (tuning 38.1 vs 37.6, held-out 23.1 vs 20.9, 13 objects).
- The deliverable is phone video of real rooms, so tools/analyse.py defaults to fp+slant (--footprint --slant on; --plain-carving to switch off). tools/bench_predict.py keeps the current carving. Both numbers are reported.

## D17. Stop VLM tuning; S20 majority verdict stays; finish the core (2026-10-14)
- S19-S21 showed the 7B VLM's yes/no flips one item in 9 true changes whatever the wording: tuning on 2 rooms is noise, so I stopped tuning the VLM.
- Kept: S18 merging, S19 views and names, S20 majority verdict (the white shoe in pair02 is listed as low confidence, not deleted).
- Remaining work, in order: 3D viewer per pair (the brief's "visualize in 3D"), a local web page to upload two videos and run, a check that refuses two videos of different rooms, README and design note.

## D18. Halves no-change false alarms documented, not tuned (2026-10-16)
- S25: the pipeline is clean when nothing changed and the two walks follow the same path (even/odd frames: 0 confident changes in 4 runs), but reports 2-6 confident false changes when the two "visits" are different stretches of one walk.
- Not fixed: 4 runs, no single cause, and any rule tuned on them would be fitted to them. Listed in the README limitations with the numbers.
- S26 kept as the evidence that carving, not later filtering, is what separates real changes from viewpoint and coverage differences.
