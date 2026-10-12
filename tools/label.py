"""Name every change of an analysed pair with a VLM on Modal (infra/modal_vlm.py).

Reads <result>/crops/<change id>_<blob id>_<k>.jpg (written by tools/analyse.py), asks once per
blob, and writes the labels into changes.json ("label"; a moved change gets the label of its
"was" view, and both answers are kept) plus labels.json with the raw answers. A blob the VLM
calls "none" gets label null and "object": false. Each blob is also shown next to the same 3D spot seen
from the closest camera in the other video and judged changed (yes / no, "changed" per blob). A change
the VLM judges unchanged on every blob is marked "confident": false: listed last, not deleted
(S17: as a score it raised benchmark AP; as a filter it lost real changes, S16).

Usage: uv run python tools/label.py outputs/fa/pair02_base
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import modal

from roomdiff.label import COMPARE, PROMPT, changed, parse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result", type=Path)
    args = ap.parse_args()
    groups: dict[tuple[int, int], list[Path]] = defaultdict(list)
    other: dict[tuple[int, int], Path] = {}
    for p in sorted((args.result / "crops").glob("*.jpg")):
        cid, bid, k = p.stem.split("_")
        key = (int(cid), int(bid))
        if k.endswith("x"):
            other[key] = p
        else:
            groups[key].append(p)
    if not groups:
        raise SystemExit(f"no crops in {args.result / 'crops'}: rerun tools/analyse.py")
    keys = sorted(groups)
    VLM = modal.Cls.from_name("roomdiff-vlm", "VLM")
    vlm = VLM()
    answers = vlm.ask.remote([[p.read_bytes() for p in groups[k]] for k in keys], PROMPT)
    per_blob = {k: {"answer": a, "label": parse(a)} for k, a in zip(keys, answers, strict=True)}
    chs = json.loads((args.result / "changes.json").read_text())
    kinds = {}  # blob -> removed / added, from the change it belongs to
    for c in chs:
        for j, b in enumerate(c["blobs"]):
            kinds[(c["id"], b)] = c["kind"] if c["kind"] != "moved" else ("removed", "added")[j]
    pairs = [k for k in keys if k in other]
    # always first visit first: a removed blob's own crop is from visit 1, an added blob's from visit 2
    imgs = [[groups[k][0], other[k]] if kinds[k] == "removed" else [other[k], groups[k][0]] for k in pairs]
    verdicts = vlm.ask.remote([[p.read_bytes() for p in im] for im in imgs], COMPARE)
    for k, a in zip(pairs, verdicts, strict=True):
        per_blob[k].update(compare=a, changed=changed(a))
    for c in chs:
        got = [per_blob[(c["id"], b)] for b in c["blobs"] if (c["id"], b) in per_blob]
        c["label"] = next((g["label"] for g in got if g["label"]), None)
        c["object"] = any(g["label"] for g in got)
        c["answers"] = [g["answer"] for g in got]
        c["changed"] = [g.get("changed") for g in got]
        c["confident"] = not (c["changed"] and all(v is False for v in c["changed"]))  # S17: "no" ranks low
    (args.result / "changes.json").write_text(json.dumps(chs, indent=2))
    (args.result / "labels.json").write_text(json.dumps(
        [{"change": k[0], "blob": k[1], **v} for k, v in per_blob.items()], indent=2))
    for c in chs:
        print(c["id"], c["kind"], c["label"], c["answers"], c["changed"])


if __name__ == "__main__":
    main()
