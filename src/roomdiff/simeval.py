"""Score a change list against the simulated room's ground truth (isaac/record_visits.py gt.json).

Ground-truth positions are world (x, y) floor positions; detections are map-frame boxes.
`world_to_map` uses the run's world -> level transform (result.json), so both meet in
map metres. A prop counts as found when a detection of a compatible kind lies within
`radius` of where it should be. Stacked props at one spot count once.
"""

from __future__ import annotations

import numpy as np


def world_to_map(T_level: np.ndarray, xy: tuple[float, float], z: float = 0.0, scale: float = 1.0) -> tuple[float, float]:
    p = (np.asarray(T_level) @ [xy[0], xy[1], z, 1.0])[:3] * scale
    return float(p[0]), float(-p[2])


def map_to_world(T_level: np.ndarray, xy: tuple[float, float], height: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """World point of a map position at `height` above the floor."""
    p = np.array([xy[0], height, -xy[1]]) / scale
    return (np.linalg.inv(np.asarray(T_level)) @ [*p, 1.0])[:3]


def _near(box: dict | None, xy: tuple[float, float], radius: float) -> bool:
    return box is not None and float(np.hypot(box["centre"][0] - xy[0], box["centre"][1] - xy[1])) <= radius


def evaluate(changes: list[dict], gt: dict, T_level: np.ndarray, floor_z: float, radius: float = 0.75,
             look_targets: list[tuple[float, float]] = ()) -> dict:
    """Per ground-truth prop: found / kind_ok / matched change ids; plus unmatched detections (false alarms)."""
    used: set[int] = set()
    seen_spots: set = set()
    props = {}
    for name, g in gt.items():
        spot = (g["change"], tuple((g["visit1"] or {}).get("pos", ())), tuple((g["visit2"] or {}).get("pos", ())))
        if spot in seen_spots:
            continue
        seen_spots.add(spot)
        b = world_to_map(T_level, g["visit1"]["pos"], floor_z) if g["visit1"] else None
        a = world_to_map(T_level, g["visit2"]["pos"], floor_z) if g["visit2"] else None
        hits, kind_ok = [], False
        for c in changes:
            was = _near(c["before"], b, radius) if b else False
            now = _near(c["after"], a, radius) if a else False
            if g["change"] == "moved" and (was or now):
                hits.append(c["id"])
                kind_ok |= c["kind"] == "moved" and was and now
            elif g["change"] == "removed" and was:
                hits.append(c["id"])
                kind_ok |= c["kind"] == "removed"
            elif g["change"] == "added" and now:
                hits.append(c["id"])
                kind_ok |= c["kind"] == "added"
            elif g["change"] in ("unchanged", "unobserved") and (_near(c["before"], b, radius) or _near(c["after"], a, radius)):
                hits.append(c["id"])  # a detection on something that did not change
        r = {"change": g["change"], "detections": hits}
        if g["change"] in ("moved", "removed", "added"):
            used.update(hits)  # detections on unchanged or unobserved props stay false alarms
            r.update(found=bool(hits), kind_ok=kind_ok)
        if g["change"] == "unobserved":
            r["look_target"] = any(np.hypot(t[0] - b[0], t[1] - b[1]) <= radius for t in look_targets)
        props[name] = r
    changed = [r for r in props.values() if "found" in r]
    return {"props": props, "found": sum(r["found"] for r in changed), "kind_ok": sum(r["kind_ok"] for r in changed),
            "changed": len(changed), "false_alarms": [c["id"] for c in changes if c["id"] not in used]}
