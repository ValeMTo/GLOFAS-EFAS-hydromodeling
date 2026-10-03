#!/usr/bin/env python3
"""Figure S9: SABER LOO extremes validation - quantile ratio (Qsim/Qobs) per percentile,
raw GloFAS vs SABER-LOO corrected, for the low-flow (S9a) and high-flow (S9b) tails.
Saved as two separate images sharing the same y-axis, matching the paper's subcaption style."""
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


def collect_panel_data(pcts):
    raw_data = [series_for("raw", p) for p in pcts]
    loo_data = [series_for("loo", p) for p in pcts]
    return raw_data, loo_data


def whisker_extent(arr):
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - 1.5*iqr, q3 + 1.5*iqr
    inside = arr[(arr >= lo_fence) & (arr <= hi_fence)]
    return inside.min(), inside.max()


def shared_ylim(*panel_data_lists):
    los, his = [], []
    for panel_data in panel_data_lists:
        for arr in panel_data:
            if len(arr) == 0:
                continue
            lo, hi = whisker_extent(arr)
            los.append(lo); his.append(hi)
    lo, hi = min(los), max(his)
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def draw_panel(pcts, title, ylim, raw_data, loo_data, outfile):
    fig, ax = plt.subplots(figsize=(8, 5.5))
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
    ax.set_ylim(*ylim)
    ax.set_xlabel("Percentile", fontsize=11)
    ax.set_ylabel("Quantile Ratio (Qsim / Qobs)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    ax.legend([bp_raw["boxes"][0], bp_loo["boxes"][0]], [RAW_LABEL, LOO_LABEL],
              loc="upper right", frameon=True, framealpha=0.9, fontsize=10)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    print("Saved:", outfile)


low_raw, low_loo = collect_panel_data(LOW_PCTS)
high_raw, high_loo = collect_panel_data(HIGH_PCTS)
ylim = shared_ylim(low_raw, low_loo, high_raw, high_loo)

draw_panel(LOW_PCTS, "Low flow", ylim, low_raw, low_loo,
           OUT/"loo_extremes_quantile_ratio_S9a.png")
draw_panel(HIGH_PCTS, "High flow", ylim, high_raw, high_loo,
           OUT/"loo_extremes_quantile_ratio_S9b.png")
