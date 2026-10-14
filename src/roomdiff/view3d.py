"""One self-contained HTML page per result: both visits' point clouds, change boxes, moved arcs, names,
crop thumbnails and the robot map (template: viewer.html, three.js from a CDN).

Reads a result folder written by tools/analyse.py (changes.json, points.npz, crops/, map.png) after
tools/label.py has named the changes.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

TEMPLATE = Path(__file__).with_name("viewer.html")


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _img(p: Path) -> str | None:
    if not p.exists():
        return None
    kind = "png" if p.suffix == ".png" else "jpeg"
    return f"data:image/{kind};base64," + base64.b64encode(p.read_bytes()).decode()


def page_data(result: Path, title: str) -> dict:
    chs = json.loads((result / "changes.json").read_text())
    pts = np.load(result / "points.npz")
    out = []
    for c in sorted(chs, key=lambda c: not c.get("confident", True)):  # stable: confident first, by size
        was = c.get("was", [])
        heads = ([("was", was[0])] if was else []) + [("now", b) for b in c["blobs"] if b not in was][:1]
        thumbs = [{"side": s, "src": src} for s, b in heads
                  if (src := _img(result / "crops" / f"{c['id']}_{b}_0.jpg"))]
        out.append({k: c.get(k) for k in ("id", "kind", "label", "before", "after", "distance")}
                   | {"confident": c.get("confident", True), "thumbs": thumbs})
    n = {k: sum(c["kind"] == k for c in out if c["confident"]) for k in ("moved", "added", "removed")}
    return {
        "title": title,
        "subtitle": f"{n['moved']} moved, {n['added']} added, {n['removed']} removed; positions in metres, floor at 0",
        "changes": out,
        "points": {v: {"xyz": _b64(np.round(pts[f"{v}_xyz"] * 1000).astype("<i2")), "rgb": _b64(pts[f"{v}_rgb"])}
                   for v in ("before", "after")},
        "map": _img(result / "map.png"),
    }


def write_page(result: Path, title: str, out: Path | None = None) -> Path:
    """Write <result>/viewer.html (or `out`) and return its path."""
    data = json.dumps(page_data(result, title)).replace("</", "<\\/")
    out = out or result / "viewer.html"
    out.write_text(TEMPLATE.read_text().replace("/*DATA*/", data))
    return out
