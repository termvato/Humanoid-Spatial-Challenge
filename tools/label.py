"""Name every change of an analysed pair with a VLM on Modal (infra/modal_vlm.py).

Reads <result>/crops/<change id>_<blob id>_<k>.jpg (written by tools/analyse.py) and asks once per
change, with the crops of its largest blob on each side (S19), and writes the name into changes.json
("label", raw "answer") plus labels.json with the raw answers. An answer of "none" or a structure
(wall, floor, door frame, ...) gets label null and "object": false, and the change is not confident. Each blob of a named change is also shown next to the
same 3D spot seen from the closest camera in the other video, asking whether the named object is still
there (S21; "no" = changed), on up to 3 own views; the majority is the blob's "changed" (S20, a tie
counts as changed). A change
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

from roomdiff.label import PROMPT, STILL_THERE, gone, majority, parse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result", type=Path)
    args = ap.parse_args()
    groups: dict[tuple[int, int], list[Path]] = defaultdict(list)
    other: dict[tuple[int, int], dict[int, Path]] = defaultdict(dict)  # own crop k -> other-video crop
    for p in sorted((args.result / "crops").glob("*.jpg")):
        cid, bid, k = p.stem.split("_")
        key = (int(cid), int(bid))
        if k.endswith("x"):
            other[key][int(k[:-1])] = p
        else:
            groups[key].append(p)
    if not groups:
        raise SystemExit(f"no crops in {args.result / 'crops'}: rerun tools/analyse.py")
    chs = json.loads((args.result / "changes.json").read_text())
    kinds = {}  # blob -> removed / added
    for c in chs:
        for b in c["blobs"]:
            kinds[(c["id"], b)] = "removed" if b in c.get("was", []) else "added"
    VLM = modal.Cls.from_name("roomdiff-vlm", "VLM")
    vlm = VLM()
    # S19: one name per change, from its largest blob on each side (2 + 2 crops when moved)
    asks = []
    for c in chs:
        heads = [k for k in ([(c["id"], c["was"][0])] if c.get("was") else []) +
                 [(c["id"], b) for b in c["blobs"] if kinds[(c["id"], b)] == "added"][:1] if k in groups]
        asks.append([p for k in heads for p in groups[k][:2 if len(heads) > 1 else 3]])
    named = vlm.ask.remote([[p.read_bytes() for p in ps] for ps in asks if ps], PROMPT)
    it = iter(named)
    answers = {c["id"]: (next(it) if ps else "") for c, ps in zip(chs, asks, strict=True)}
    names = {c["id"]: parse(answers[c["id"]]) for c in chs}
    # S21: only named changes get the before/after question, own view first, naming the object
    pairs = [(k, j) for k in sorted(groups) if names[k[0]] for j in sorted(other[k])]
    imgs = [[groups[k][j], other[k][j]] for k, j in pairs]
    verdicts = vlm.ask.remote([[p.read_bytes() for p in im] for im in imgs],
                              [STILL_THERE.format(name=names[k[0]]) for k, _ in pairs])
    votes: dict[tuple[int, int], list] = defaultdict(list)
    for (k, _), a in zip(pairs, verdicts, strict=True):
        votes[k].append(a)
    per_blob = {k: {"compare": v, "changed": majority([gone(a) for a in v])} for k, v in votes.items()}
    for c in chs:
        c["answer"] = answers[c["id"]]
        c["label"] = parse(c["answer"])
        c["object"] = c["label"] is not None
        c["changed"] = [per_blob.get((c["id"], b), {}).get("changed") for b in c["blobs"]]
        verdict = [v for v in c["changed"] if v is not None]
        # S17: all "unchanged" ranks low; S19: no object named (wall, floor, none) too
        c["confident"] = c["object"] and not (verdict and not any(verdict))
    (args.result / "changes.json").write_text(json.dumps(chs, indent=2))
    (args.result / "labels.json").write_text(json.dumps(
        {"names": answers, "compare": [{"change": k[0], "blob": k[1], **v} for k, v in per_blob.items()]}, indent=2))
    for c in chs:
        print(c["id"], c["kind"], c["label"], repr(c["answer"]), c["changed"], "" if c["confident"] else "(low)")


if __name__ == "__main__":
    main()
