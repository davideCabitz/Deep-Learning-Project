"""
Tier comparison visualiser — macro overview of MEAN metrics across all tiers.

Grouped bar chart: x-axis = metric (R@1, R@5, R@10, P@1, P@5, P@10), one bar per tier.
Values are read directly from the MEAN row of each CSV (macro-average across queries,
each query weighted equally). No cross-metric averaging is performed.

Swap CSVs in TIERS to change what is compared.
Run: python src/visual/visualize.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display needed, safe in any terminal
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration — edit here to swap tiers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent

TIERS: list[tuple[str, Path]] = [
    ("Tier-0 baseline",         ROOT / "output" / "tier0_baseline.csv"),
    ("Tier-0 enhanced",         ROOT / "output" / "tier0_enhanced" / "tier0_enhanced_all_fixes.csv"),
    ("Tier-1 CLAY",             ROOT / "output" / "tier1_CLAY" / "tier1_CLAY_k50_rotH.csv"),
    ("Tier-1 GDE",              ROOT / "output" / "tier1_GDE" / "tier1_GDE.csv"),
    ("Tier-2a S k50",           ROOT / "output" / "tier2a_S" / "tier2a_S_percond_anchor_k50_50_lam0.1_rotH.csv"),
    ("Tier-2a Visual Ext α1.5", ROOT / "output" / "tier2a_visual_ext" / "tier2a_visual_ext_gde_alpha1.5.csv"),
]

ALL_METRICS = ["R@1", "R@5", "R@10", "P@1", "P@5", "P@10"]

OUT_DIR = ROOT / "output" / "visual"
OUT_DIR.mkdir(exist_ok=True)

_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3",
]
COLORS = {label: _PALETTE[i] for i, (label, _) in enumerate(TIERS)}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_mean(path: Path) -> dict[str, float]:
    # Reads only the MEAN row from a results CSV → {metric: value}.
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["query"] == "MEAN":
                return {m: float(row[m]) for m in ALL_METRICS if row.get(m, "") != ""}
    raise ValueError(f"No MEAN row found in {path}")


# ---------------------------------------------------------------------------
# Figure — macro overview: MEAN metrics, all tiers grouped
# ---------------------------------------------------------------------------
def fig_macro_overview():
    all_means = {label: _load_mean(path) for label, path in TIERS}

    n_tiers = len(TIERS)
    x = np.arange(len(ALL_METRICS))
    width = 0.8 / n_tiers

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (label, _) in enumerate(TIERS):
        vals = [all_means[label].get(m, 0.0) for m in ALL_METRICS]
        offset = (i - n_tiers / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=COLORS[label])
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=5.5, rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(ALL_METRICS)
    ax.set_ylabel("Score")
    ax.set_title("Macro-average per metric across all queries (MEAN row)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    fig.tight_layout()

    out = OUT_DIR / "macro_overview.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fig_macro_overview()
