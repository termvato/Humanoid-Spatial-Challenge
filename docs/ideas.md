# Ideas under discussion (not decided)

## I1. Isaac Sim procedural room pairs (proposed by user, 2026-10-11)
- Auto-generate rooms, apply known move/add/remove edits, render two camera trajectories per pair. Exact ground truth.
- Uses: (a) benchmark with precision/recall at scale, (b) training data, (c) generalisation measurement, (d) robot nav demo in the same env.
- Curriculum learning suggested.
- Open risks: sim-to-real gap vs the judged real phone videos; asset variety; Isaac render throughput on 8 GB; 3-4 day budget.
- Open question to user: what exactly would the model learn (input, output)?

## I2. Sim+real co-training (user, 2026-10-11)
- Evidence: Maddukuri et al., RSS 2025, arXiv 2503.24361: sim data improves real manipulation by 38% on average; best sim share ~99% on CupPnP, extreme shares (99.9%) hurt. More sim data helps. RoboMIND 2.0 (arXiv 2512.24653): 1:5 real:sim beat 1:1 on some tasks.
- Caveat: those are manipulation policies, not change detection. Transfer of the finding is a hypothesis to test, not a given.
- Requirement: several labelled real pairs, split train/test by room. The demo videos must be in the real TEST split.
- Experiment: sim-only vs real-only vs sim+real (ratio sweep), scored on held-out real pairs.
- User agreed: use state-of-the-art pretrained models, no reinventing.
