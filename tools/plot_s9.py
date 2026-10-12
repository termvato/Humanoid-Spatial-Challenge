"""Bar chart of S9 results (obj/sc AP and typed AP), tuning vs held-out. Numbers from outputs/s9/*.eval.log."""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROWS = [  # label, tuning tag, test tag (None = not run)
    ("SceneDiff", "sd_tune", "s1"),
    ("3D: geometry", "tune_pi3_d80_icp", None),
    ("3D: + SAM", "tune_pi3_d80_icp_sam", "test_pi3_d80_icp_sam"),
    ("3D: + SAM + DINO match", "tune_dino5", "test_dino5"),
    ("Fusion", "tune_fuse_add_dino5", "test_fuse_add_dino5"),
]


def metrics(tag):
    if tag is None:
        return None
    if tag == "s1":
        text = "".join(p.read_text() for p in Path("outputs/s1").glob("metrics*.txt"))
    elif tag == "sd_tune":
        text = Path("outputs/s9/sd_tune.log").read_text(errors="ignore")
    else:
        text = Path(f"outputs/s9/{tag}.eval.log").read_text(errors="ignore")
    v = [float(x) for x in re.findall(r"Metric 3[ab]: [^\n]*?: ([0-9.]+)", text)[:2]]
    return [100 * x for x in v]


fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, col, title in zip(axes, (1, 2), ("15 tuning pairs (configs chosen here)", "5 held-out pairs (frozen configs)")):
    labels, a, b = [], [], []
    for label, tune, test in ROWS:
        m = metrics(tune if col == 1 else test)
        if m is None:
            continue
        labels.append(label)
        a.append(m[0])
        b.append(m[1])
    x = np.arange(len(labels))
    ax.bar(x - 0.2, a, 0.4, label="obj/scene AP", color="#4c72b0")
    ax.bar(x + 0.2, b, 0.4, label="... with change type", color="#dd8452")
    for xi, (u, w) in enumerate(zip(a, b)):
        ax.text(xi - 0.2, u + 0.8, f"{u:.0f}", ha="center", fontsize=8)
        ax.text(xi + 0.2, w + 0.8, f"{w:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels, rotation=20, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
axes[0].set_ylabel("AP (%)")
axes[0].legend(fontsize=8)
fig.suptitle("SceneDiff benchmark (SD-V val): 3D-first pipeline vs SceneDiff", fontsize=11)
fig.tight_layout()
fig.savefig("outputs/s9/s9_results.png", dpi=120)
print("outputs/s9/s9_results.png")
