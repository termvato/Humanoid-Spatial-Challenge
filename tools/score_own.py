"""Score analysed own pairs against map-frame ground truth (bench/own_gt/<pair>.json).

Ground-truth positions are already map metres, so the world -> map transform is the identity on
the map plane. A change counts as found when a detection of a compatible kind lies within
`--radius` of it; every unmatched detection is a false alarm. A ground-truth entry without
positions (never located) counts as missed.

Usage: uv run python tools/score_own.py outputs/fa/pair01_base outputs/fa/pair01_fpslant --gt bench/own_gt/pair01.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from roomdiff.simeval import evaluate

MAP_AS_WORLD = np.array([[1.0, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])  # world (x, y) = map (x, y)


def score(changes: list[dict], gt: dict, radius: float) -> dict:
    located = {k: v for k, v in gt.items() if not k.startswith("_") and (v["visit1"] or v["visit2"])}
    unlocated = [k for k, v in gt.items() if not k.startswith("_") and not (v["visit1"] or v["visit2"])]
    e = evaluate(changes, located, MAP_AS_WORLD, 0.0, radius=radius)
    return {"found": e["found"], "kind_ok": e["kind_ok"], "changes": e["changed"] + len(unlocated),
            "detections": len(changes), "false_alarms": len(e["false_alarms"]),
            "missed": [k for k, r in e["props"].items() if r.get("found") is False] + unlocated,
            "false_alarm_ids": e["false_alarms"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, nargs="+")
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--radius", type=float, default=0.3, help="metres")
    ap.add_argument("--vlm-filter", action="store_true",
                    help="drop changes whose blobs the VLM all judged 'same' (needs tools/label.py; S16)")
    args = ap.parse_args()
    gt = json.loads(args.gt.read_text())
    for r in args.results:
        chs = json.loads((r / "changes.json").read_text())
        if args.vlm_filter:
            chs = [c for c in chs if not (c.get("changed") and all(v is False for v in c["changed"]))]
        s = score(chs, gt, args.radius)
        print(f"{r}: found {s['found']}/{s['changes']} (kind ok {s['kind_ok']}), "
              f"{s['detections']} detections, {s['false_alarms']} false alarms, missed {s['missed']}")


if __name__ == "__main__":
    main()
