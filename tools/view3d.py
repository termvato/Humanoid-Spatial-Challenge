"""3D viewer page for an analysed and labelled pair (roomdiff.view3d).

Usage: uv run python tools/view3d.py outputs/s20/pair01 --title "pair01: bedroom"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from roomdiff.view3d import write_page


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result", type=Path)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    p = write_page(args.result, args.title or args.result.name)
    print(f"{p} ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
