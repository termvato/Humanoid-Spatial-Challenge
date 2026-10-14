"""Name every change of an analysed pair (roomdiff.labelling) with the VLM on Modal (infra/modal_vlm.py)
or, with --local, on this machine's GPU (~16 GB).

Usage: uv run python tools/label.py outputs/fa/pair02_base [--local]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from roomdiff.labelling import label_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result", type=Path)
    ap.add_argument("--local", action="store_true", help="run the VLM here instead of on Modal")
    args = ap.parse_args()
    if args.local:
        from roomdiff.vlm import LocalVLM

        ask = LocalVLM().ask
    else:
        import modal

        ask = modal.Cls.from_name("roomdiff-vlm", "VLM")().ask.remote
    for c in label_result(args.result, ask):
        print(c["id"], c["kind"], c["label"], repr(c["answer"]), c["changed"], "" if c["confident"] else "(low)")


if __name__ == "__main__":
    main()
