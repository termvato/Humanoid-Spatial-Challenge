"""Name every change of an analysed result folder with a VLM (roomdiff.vlm).

Reads <result>/crops/<change id>_<blob id>_<k>.jpg (written by roomdiff.analyse) and asks once per
change, with the crops of its largest blob on each side (S19), and writes the name into changes.json
("label", raw "answer") plus labels.json with the raw answers. An answer of "none" or a structure
(wall, floor, door frame, ...) gets label null and "object": false, and the change is not confident.
Each blob is also shown next to the same 3D spot seen from the closest camera in the other video and
judged changed (yes / no), on up to 3 own views; the majority is the blob's "changed" (S20, a tie
counts as changed). A change the VLM judges unchanged on every blob is marked "confident": false:
listed last, not deleted (S17: as a score it raised benchmark AP; as a filter it lost real changes, S16).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from roomdiff.label import COMPARE, PROMPT, changed, majority, parse


def label_result(result: Path, ask: Callable[[list[list[bytes]], str], list[str]]) -> list[dict]:
    """Name and judge every change in `result` (crops from roomdiff.analyse); `ask(items, prompt)` is a VLM
    (LocalVLM.ask, or the Modal VLM's ask.remote). Rewrites changes.json, writes labels.json, returns the changes."""
    groups: dict[tuple[int, int], list[Path]] = defaultdict(list)
    other: dict[tuple[int, int], dict[int, Path]] = defaultdict(dict)  # own crop k -> other-video crop
    for p in sorted((result / "crops").glob("*.jpg")):
        cid, bid, k = p.stem.split("_")
        key = (int(cid), int(bid))
        if k.endswith("x"):
            other[key][int(k[:-1])] = p
        else:
            groups[key].append(p)
    chs = json.loads((result / "changes.json").read_text())
    if not chs:  # nothing changed: nothing to name
        (result / "labels.json").write_text(json.dumps({"names": {}, "compare": []}))
        return chs
    if not groups:
        raise RuntimeError(f"no crops in {result / 'crops'}: rerun tools/analyse.py")
    kinds = {}  # blob -> removed / added
    for c in chs:
        for b in c["blobs"]:
            kinds[(c["id"], b)] = "removed" if b in c.get("was", []) else "added"
    # S19: one name per change, from its largest blob on each side (2 + 2 crops when moved)
    asks = []
    for c in chs:
        heads = [k for k in ([(c["id"], c["was"][0])] if c.get("was") else []) +
                 [(c["id"], b) for b in c["blobs"] if kinds[(c["id"], b)] == "added"][:1] if k in groups]
        asks.append([p for k in heads for p in groups[k][:2 if len(heads) > 1 else 3]])
    named = ask([[p.read_bytes() for p in ps] for ps in asks if ps], PROMPT)
    it = iter(named)
    answers = {c["id"]: (next(it) if ps else "") for c, ps in zip(chs, asks, strict=True)}
    pairs = [(k, j) for k in sorted(groups) for j in sorted(other[k])]
    # always first visit first: a removed blob's own crop is from visit 1, an added blob's from visit 2
    imgs = [[groups[k][j], other[k][j]] if kinds[k] == "removed" else [other[k][j], groups[k][j]] for k, j in pairs]
    verdicts = ask([[p.read_bytes() for p in im] for im in imgs], COMPARE)
    votes: dict[tuple[int, int], list] = defaultdict(list)
    for (k, _), a in zip(pairs, verdicts, strict=True):
        votes[k].append(a)
    per_blob = {k: {"compare": v, "changed": majority([changed(a) for a in v])} for k, v in votes.items()}
    for c in chs:
        c["answer"] = answers[c["id"]]
        c["label"] = parse(c["answer"])
        c["object"] = c["label"] is not None
        c["changed"] = [per_blob.get((c["id"], b), {}).get("changed") for b in c["blobs"]]
        verdict = [v for v in c["changed"] if v is not None]
        # S17: all "unchanged" ranks low; S19: no object named (wall, floor, none) too
        c["confident"] = c["object"] and not (verdict and not any(verdict))
    (result / "changes.json").write_text(json.dumps(chs, indent=2))
    (result / "labels.json").write_text(json.dumps(
        {"names": answers, "compare": [{"change": k[0], "blob": k[1], **v} for k, v in per_blob.items()]}, indent=2))
    return chs

