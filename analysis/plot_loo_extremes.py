#!/usr/bin/env python3
"""Figure: SABER LOO extremes validation - quantile ratio (Qsim/Qobs) per percentile,
raw GloFAS vs SABER-LOO corrected, for the low-flow and high-flow tails."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = Path(__file__).resolve().parent / "saber_loo_out" / "loo_extremes_metrics.csv"
OUT = CSV.parent
df = pd.read_csv(CSV)
ok = df[df["status"]=="ok"].copy()
print(f"OK gauges: {len(ok)}")

LOW_PCTS = list(range(1, 21))
HIGH_PCTS = list(range(80, 101))

RAW_COLOR, LOO_COLOR = "#d73027", "#3182bd"
RAW_LABEL, LOO_LABEL = "Raw GloFAS", "SABER-corrected (LOO)"


def series_for(prefix, pct):
    col = f"{prefix}_ratio_q{pct:02d}"
    if col not in ok.columns:
        raise KeyError(
            f"Column '{col}' not found. Re-run saber_loo_extremes.py to regenerate "
            f"{CSV} with per-percentile ratio columns."
        )
    return ok[col].dropna().values


def draw_panel(ax, pcts, title):
    raw_data = [series_for("raw", p) for p in pcts]
    loo_data = [series_for("loo", p) for p in pcts]
    x = np.arange(1, len(pcts) + 1)
    width = 0.35

    ax.axhline(1.0, color="green", lw=1, ls="--", zorder=1)
    bp_raw = ax.boxplot(raw_data, positions=x - width/2, widths=width, patch_artist=True,
                         showfliers=False, zorder=2,
                         boxprops=dict(facecolor=RAW_COLOR, alpha=0.55, edgecolor=RAW_COLOR),
                         medianprops=dict(color="black"),
                         whiskerprops=dict(color=RAW_COLOR), capprops=dict(color=RAW_COLOR))
    bp_loo = ax.boxplot(loo_data, positions=x + width/2, widths=width, patch_artist=True,
                         showfliers=False, zorder=3,
                         boxprops=dict(facecolor=LOO_COLOR, alpha=0.55, edgecolor=LOO_COLOR),
                         medianprops=dict(color="black"),
                         whiskerprops=dict(color=LOO_COLOR), capprops=dict(color=LOO_COLOR))

    ax.set_xticks(x)
    ax.set_xticklabels(pcts, fontsize=8)
    ax.set_xlim(0.5, len(pcts) + 0.5)
    ax.set_xlabel("Percentile", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    return bp_raw, bp_loo


fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
draw_panel(axs[0], LOW_PCTS, "(a) Low flow")
bp_raw, bp_loo = draw_panel(axs[1], HIGH_PCTS, "(d) High flow")
axs[0].set_ylabel("Quantile Ratio (Qsim / Qobs)", fontsize=11)

fig.legend([bp_raw["boxes"][0], bp_loo["boxes"][0]], [RAW_LABEL, LOO_LABEL],
           loc="upper center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.03))
fig.suptitle(f"SABER LOO validation: quantile ratio at flow extremes ({len(ok)} gauges)",
             fontsize=13, y=1.08)
fig.tight_layout()
fig.savefig(OUT/"loo_extremes_quantile_ratio_tails.png", dpi=200, bbox_inches="tight")
print("Saved:", OUT/"loo_extremes_quantile_ratio_tails.png")
