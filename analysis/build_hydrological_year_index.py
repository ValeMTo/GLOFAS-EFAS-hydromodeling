#!/usr/bin/env python3
"""Hydrological year index (dry/normal/wet) from SABER-corrected inflows.

Indicator = capacity-weighted mean of annual inflow volume across plants:
    I(year) = sum_p [ V_p(year) * Cap_p ] / sum_p Cap_p
where V_p(year) = sum over days of (discharge[m3/s] * 86400) = annual volume [m3].
Then normalised to a dimensionless index (mean over the window = 1) so it reads
as "relative to an average hydrological year for the average installed plant".

Classification is restricted to the window [YEAR_MIN, YEAR_MAX], chosen so that
PyPSA-Eur has fully observed electricity demand (OPSD covers 2005 -> Sep 2020),
avoiding synthetic-load supplementation.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- config -----------------------------
CORR = "/work/cmcc/ds07226/GLOFAS-EFAS-hydromodeling/data/hydro_workflow/hydro_global/corrected_inflows_saber_Europe_glofas.pkl"
P    = "/work/cmcc/ds07226/GLOFAS-EFAS-hydromodeling/data/hydro_workflow/hydro_global/Eur_custom_ppls_GloHydroRes_filled.csv"
OUTDIR = Path(__file__).resolve().parent / "hydro_year_index_out"

EXCLUDE_TECH = {"Pumped Storage"}   # not natural inflow
SECONDS_PER_DAY = 86400.0

# --- classification window (PyPSA fully-observed demand: OPSD 2005 -> Sep 2020) ---
YEAR_MIN = 2005
YEAR_MAX = 2020

# --- how to pick dry / normal / wet ---
#   "extremes"   -> driest = min, wettest = max, normal = closest to median
#   "percentile" -> dry = closest to p{DRY_PCTL}, wet = closest to p{WET_PCTL}
METHOD = "extremes"
DRY_PCTL, WET_PCTL = 10, 90         # used only if METHOD == "percentile"

# macro-regions by country (edit freely)
REGIONS = {
    "Nordics":   {"NO", "SE", "FI"},
    "Alpine":    {"CH", "AT", "IT", "FR", "SI", "DE"},
    "Iberia":    {"ES", "PT"},
    "Balkans":   {"RO", "BG", "GR", "HR", "RS", "BA", "ME", "MK", "AL", "XK"},
    "British Isles": {"GB", "IE"},
}
# ------------------------------------------------------------------

OUTDIR.mkdir(parents=True, exist_ok=True)

# --- load plant metadata ---
meta = pd.read_csv(P)
meta["plant_index"] = meta["plant_index"].astype(str)
meta = meta.set_index("plant_index")

# --- load inflows ---
with open(CORR, "rb") as f:
    inflows = pickle.load(f)

# keep plants present in both, drop excluded tech
plants = [p for p in map(str, inflows.keys())
          if p in meta.index and meta.at[p, "Technology"] not in EXCLUDE_TECH]
print(f"Plants used: {len(plants)} (excluded tech: {EXCLUDE_TECH})")

cap = meta.loc[plants, "Capacity"].astype(float)
country = meta.loc[plants, "Country"]

# --- annual volume matrix: plants x years ---
vol_rows = {}
for p in plants:
    s = inflows[p].copy()
    s.index = pd.to_datetime(s.index)
    annual_vol = s.groupby(s.index.year).sum() * SECONDS_PER_DAY  # m3 per year
    vol_rows[p] = annual_vol

V = pd.DataFrame(vol_rows).T          # index=plant, columns=year
V = V.reindex(index=plants)

# restrict to the usable demand window BEFORE computing the index
all_years = sorted(V.columns)
V = V[[y for y in V.columns if YEAR_MIN <= y <= YEAR_MAX]]
years = sorted(V.columns)
print(f"Full inflow years available:    {all_years[0]}-{all_years[-1]} ({len(all_years)})")
print(f"Years used for classification:  {years[0]}-{years[-1]} ({len(years)})")

def weighted_index(vol_df, weights):
    w = weights.reindex(vol_df.index)
    num = (vol_df.mul(w, axis=0)).sum(axis=0, min_count=1)
    idx = num / w.sum()
    return idx / idx.mean()          # normalise: mean over window = 1

# --- Europe index ---
eu = weighted_index(V, cap).rename("Europe")

# --- regional indices ---
reg_series = {}
for name, ctry in REGIONS.items():
    mask = country.isin(ctry)
    if mask.sum() == 0:
        continue
    reg_series[name] = weighted_index(V.loc[mask.values], cap[mask.values]).rename(name)

out = pd.concat([eu] + list(reg_series.values()), axis=1)

# --- pick representative years ---
median = eu.median()
normal_year = (eu - median).abs().idxmin()

if METHOD == "extremes":
    dry_year = eu.idxmin()
    wet_year = eu.idxmax()
    dry_thr, wet_thr = eu.min(), eu.max()
elif METHOD == "percentile":
    dry_thr = np.percentile(eu, DRY_PCTL)
    wet_thr = np.percentile(eu, WET_PCTL)
    dry_year = (eu - dry_thr).abs().idxmin()
    wet_year = (eu - wet_thr).abs().idxmin()
else:
    raise ValueError(f"Unknown METHOD: {METHOD!r}")

def classify(v):
    if v <= dry_thr: return "dry"
    if v >= wet_thr: return "wet"
    return "normal"

out["EU_category"] = eu.apply(classify)
out["EU_pctl"] = eu.rank(pct=True).round(3)

print(f"\n=== Representative years (Europe, method={METHOD}) ===")
print(f"DRY    -> {dry_year}  (index {eu[dry_year]:.3f})")
print(f"NORMAL -> {normal_year}  (index {eu[normal_year]:.3f})")
print(f"WET    -> {wet_year}  (index {eu[wet_year]:.3f})")

print("\n=== Ranking within window (driest -> wettest) ===")
print(eu.sort_values().round(3).to_string())

# --- save ---
out.round(4).to_csv(OUTDIR / "hydrological_year_index.csv")
print(f"\nSaved: {OUTDIR/'hydrological_year_index.csv'}")

# --- diagnostic plot ---
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.plot(eu.index, eu.values, "-o", color="black", lw=1.6, ms=6,
        markerfacecolor="white", markeredgecolor="black", zorder=3, label="Europe")
ax.axhline(1.0, color="gray", ls="--", lw=0.8, zorder=1)

# highlighted years: label placed above wet, below dry/normal to avoid overlap
picks = [
    (dry_year,    eu[dry_year],    "#d73027", "dry",    (0, -28)),
    (normal_year, eu[normal_year], "#6e6e6e", "normal", (0, -28)),
    (wet_year,    eu[wet_year],    "#4575b4", "wet",    (0,  14)),
]
for yr, val, col, lab, off in picks:
    ax.scatter([yr], [val], s=170, color=col, zorder=5,
               edgecolor="black", linewidth=1.2)
    ax.annotate(f"{lab} · {yr}", (yr, val), textcoords="offset points",
                xytext=off, ha="center", fontsize=11, fontweight="bold", color=col)

# headroom so labels/title don't collide
ymin, ymax = eu.min(), eu.max()
pad = (ymax - ymin) * 0.28
ax.set_ylim(ymin - pad, ymax + pad)

ax.set_ylabel("Capacity-weighted inflow index\n(1 = average year in window)",
              fontsize=11)
ax.set_xlabel("Year", fontsize=11)
ax.set_title(f"European hydrological year index — SABER-corrected GloFAS  "
             f"({YEAR_MIN}–{YEAR_MAX})", fontsize=13, pad=12)
ax.set_xticks(list(eu.index))
ax.tick_params(axis="x", rotation=45, labelsize=9)
ax.tick_params(axis="y", labelsize=10)
ax.legend(loc="upper right", framealpha=0.95)
ax.grid(alpha=0.25, zorder=0)
fig.tight_layout()
fig.savefig(OUTDIR / "hydrological_year_index.png", dpi=200)
print(f"Saved: {OUTDIR/'hydrological_year_index.png'}")

# --- regional ranking table ---
print("\n=== Regional indices (per year) ===")
print(out[[c for c in out.columns if c in ['Europe'] + list(reg_series)]]
      .round(3).to_string())