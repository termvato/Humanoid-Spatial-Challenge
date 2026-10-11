# Ideas under discussion (not decided)

## I1. Isaac Sim procedural room pairs (proposed by user, 2026-10-11)
- Auto-generate rooms, apply known move/add/remove edits, render two camera trajectories per pair. Exact ground truth.
- Uses: (a) benchmark with precision/recall at scale, (b) training data, (c) generalisation measurement, (d) robot nav demo in the same env.
- Curriculum learning suggested.
- Open risks: sim-to-real gap vs the judged real phone videos; asset variety; Isaac render throughput on 8 GB; 3-4 day budget.
- Open question to user: what exactly would the model learn (input, output)?
