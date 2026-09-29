#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ror_match_prototype.py

PURPOSE
-------
Standalone test of the new two-tier RoR matching rule discussed after
ror_plant_level_audit.py / ror_recoverability_test.py, WITHOUT touching the
real pipeline (extract_glofas_series_for_hydro_plants.py) and WITHOUT
re-extracting per-timestep series for every plant.

IDENTITY FIX (important)
-------------------------
An earlier version of this script identified plants via
GloHydroRes_vs1.csv's "ID" column (GHRxxxxx, digits stripped), assuming
that number equals the pipeline's "plant" coordinate in the inflow NetCDF.
It does not: the real pipeline (load_hydro_power_plants() in
extract_glofas_series_for_hydro_plants.py) reads plants from
Eur_custom_ppls_GloHydroRes_filled.csv, whose row index ("plant_index") is
the true identity used everywhere downstream (final_matches, rivid_map,
inflow NetCDF). The two numbering schemes disagree almost everywhere
(off-by-one and worse), which silently mixed in wrong plants (e.g. "GHR
ID 4086" = Mount Coffee, Liberia, while pipeline plant_index 4086 = Kaunas
HPP, Lithuania). This version loads plants directly from
Eur_custom_ppls_GloHydroRes_filled.csv and never touches GloHydroRes_vs1.csv
or norm_id(), so there is no re-identification step to get wrong.

HOMOGENEITY FIX (important)
----------------------------
The "old" (today's) assignment is read from the real final_matches parquet
(final_hydro_plant_glofas_points.parquet), which carries river_mean = the
RAW GloFAS mean discharge at the coordinates the pipeline actually chose --
computed by the same kind of long-term mean the new rule uses. Comparing
against the SABER-corrected inflow NetCDF mean instead (as the earlier
version did) mixes two different physical quantities and is not a fair
before/after comparison.

THE RULE
--------
For each plant, search drain-table nodes only (never raw GloFAS cells):

    1. within 10 km, keep nodes with 0.2 <= Q_node/Qmax <= 5.0;
       if any, take the nearest of these.
    2. else, within 25 km, same band; if any, take the nearest.
    3. else, within 25 km, relax to Q_node/Qmax >= 0.1 (no upper bound);
       if any, take the nearest.
    4. else: FAILED -- fall back to the nearest drain node within 25 km
       regardless of flow (same behaviour the pipeline has today), but the
       plant is explicitly flagged instead of silently absorbed.

Never searches beyond 25 km at any stage.

OUTPUTS (OUTDIR)
----------------
    ror_match_prototype.csv     one row per plant: old/new ratio, tier, dist
    transition_matrix.csv       old-in-band x new-in-band, capacity-weighted
    tier_summary.csv            capacity by tier (ok_in_band/relaxed/FAILED)
    failed_plants.csv           FAILED_* plants for manual inspection
    fig_transition.png, summary_match_prototype.txt
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG
# =============================================================================
ROOT = Path("/work/cmcc/ds07226/GLOFAS-EFAS-hydromodeling/data")

# The REAL pipeline's plant table (not GloHydroRes_vs1.csv): its row index
# ("plant_index") is the true plant identity used everywhere downstream.
EUR_PLANTS_FILE = ROOT / "hydro_workflow/hydro_global/Eur_custom_ppls_GloHydroRes_filled.csv"

# The REAL pipeline's own match output: river_mean here is the raw GloFAS
# mean at the coordinates actually assigned today -- the correct "before".
FINAL_MATCHES = ROOT / "hydro_workflow/saber_global/glofas/final_hydro_plant_glofas_points.parquet"

DRAIN_TABLE = ROOT / "hydro_workflow/saber_global/glofas/glofas_drain_lv3_europe_full.parquet"

# Shared with ror_recoverability_test.py: same field, computed once.
MEAN_FIELD_CACHE = Path("./ror_recover_out/glofas_mean_discharge.nc")

OUTDIR = Path("./ror_match_prototype_out")

# --- the rule ----------------------------------------------------------------
RADII_KM = [10.0, 25.0]
BAND_LOW, BAND_HIGH = 0.2, 5.0
RELAX_LOW = 0.1                     # stage 3, no upper bound, still <=25 km

EFFICIENCY = 0.85
ROR_TECH = "Run-Of-River"
RES_TECHS = ("Reservoir", "Pumped Storage")


# =============================================================================
# HELPERS
# =============================================================================
def add_qmax_turb(df, efficiency=EFFICIENCY):
    """Same formula as the pipeline's load_hydro_power_plants(): global
    per-technology median DamHeight_m imputation, computed on the FULL
    Europe table before any Technology filtering."""
    median_ror = df.loc[df.Technology == ROR_TECH, "DamHeight_m"].median(skipna=True)
    median_res = df.loc[df.Technology.isin(RES_TECHS), "DamHeight_m"].median(skipna=True)

    head = df["DamHeight_m"].copy()
    head = np.where(head.notna(), head,
                     np.where(df.Technology == ROR_TECH, median_ror, median_res))
    head = np.where(head <= 0, np.nan, head)

    df = df.copy()
    df["head_m"] = head
    df["qmax_turb"] = df["Capacity"] / (9.81 * df["head_m"] * 1e-3 * efficiency)
    return df


def load_ror_plants():
    eur = pd.read_csv(EUR_PLANTS_FILE, index_col=0)
    eur.index.name = "plant"
    eur = add_qmax_turb(eur)
    ror = eur[eur.Technology == ROR_TECH].copy()
    return ror.dropna(subset=["lat", "lon", "Capacity", "qmax_turb"])


def load_mean_field():
    if not MEAN_FIELD_CACHE.exists():
        sys.exit(f"[FATAL] {MEAN_FIELD_CACHE} not found -- run "
                  "ror_recoverability_test.py first to build the cache.")
    print(f"[info] using shared cached mean field {MEAN_FIELD_CACHE}")
    return xr.open_dataarray(MEAN_FIELD_CACHE)


def load_drain_table_with_flow(mean):
    """Real GloFAS drain nodes, each carrying the long-term mean discharge
    of the raw GloFAS cell nearest to it -- same sampling the pipeline's
    add_river_mean_to_candidates() does for drain candidates."""
    df = pd.read_parquet(DRAIN_TABLE)
    df["model_id"] = df["model_id"].astype(str)

    latname = "latitude" if "latitude" in mean.coords else "lat"
    lonname = "longitude" if "longitude" in mean.coords else "lon"
    lat_grid = mean[latname].values
    lon_grid = mean[lonname].values
    lat_idx = np.abs(lat_grid[:, None] - df["y"].to_numpy()[None, :]).argmin(axis=0)
    lon_idx = np.abs(lon_grid[:, None] - df["x"].to_numpy()[None, :]).argmin(axis=0)
    df["river_mean"] = mean.values[lat_idx, lon_idx]

    n_bad = int((~np.isfinite(df["river_mean"]) | (df["river_mean"] <= 0)).sum())
    print(f"[info] drain nodes: {len(df):,} ({n_bad:,} with no valid GloFAS "
          "flow at their nearest cell, excluded from candidacy)")
    df = df[np.isfinite(df["river_mean"]) & (df["river_mean"] > 0)].reset_index(drop=True)
    return df


def build_drain_tree(drain):
    lat0 = float(drain["y"].mean())
    x = drain["x"].to_numpy() * 111.0 * np.cos(np.deg2rad(lat0))
    y = drain["y"].to_numpy() * 111.0
    return cKDTree(np.column_stack([x, y])), lat0


def load_old_assignment(ror):
    """Today's pipeline assignment, straight from its own match output --
    river_mean there is already the raw GloFAS mean at the assigned cell."""
    fm = pd.read_parquet(FINAL_MATCHES)
    fm = fm.set_index("plant_index")
    fm = fm[~fm.index.duplicated()]
    common = ror.index.intersection(fm.index)
    print(f"[info] plants with a real final_matches row: {len(common)} of {len(ror)}")
    return fm.loc[common, "river_mean"]


def match_one(xq, yq, qmax, tree, drain):
    """Apply the tiered rule to a single plant. Returns a dict."""
    idx25 = np.asarray(tree.query_ball_point([xq, yq], r=25.0))
    if idx25.size == 0:
        return dict(tier="FAILED_no_drain_node_within_25km", n_cells=0)

    lat0 = drain["y"].mean()
    x_all = drain["x"].to_numpy()[idx25] * 111.0 * np.cos(np.deg2rad(lat0))
    y_all = drain["y"].to_numpy()[idx25] * 111.0
    d_km = np.hypot(x_all - xq, y_all - yq)
    ratio = drain["river_mean"].to_numpy()[idx25] / qmax

    for radius in RADII_KM:
        within = d_km <= radius
        in_band = within & (ratio >= BAND_LOW) & (ratio <= BAND_HIGH)
        if in_band.any():
            b = np.argmin(np.where(in_band, d_km, np.inf))
            return dict(tier=f"ok_in_band_{radius:g}km", n_cells=int(within.sum()),
                        matched_dist_km=float(d_km[b]), matched_ratio=float(ratio[b]),
                        matched_model_id=drain["model_id"].to_numpy()[idx25][b])

    relaxed = ratio >= RELAX_LOW
    if relaxed.any():
        b = np.argmin(np.where(relaxed, d_km, np.inf))
        return dict(tier="ok_relaxed_band", n_cells=int(idx25.size),
                    matched_dist_km=float(d_km[b]), matched_ratio=float(ratio[b]),
                    matched_model_id=drain["model_id"].to_numpy()[idx25][b])

    b = np.argmin(d_km)
    return dict(tier="FAILED_no_plausible_cell", n_cells=int(idx25.size),
                matched_dist_km=float(d_km[b]), matched_ratio=float(ratio[b]),
                matched_model_id=drain["model_id"].to_numpy()[idx25][b])


# =============================================================================
# MAIN
# =============================================================================
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ror = load_ror_plants()
    print(f"[info] RoR plants: {len(ror)} ({ror.Capacity.sum()/1e3:.2f} GW) | "
          f"rule: band [{BAND_LOW:g},{BAND_HIGH:g}] @10/25km, relax >={RELAX_LOW:g} @25km")

    mean = load_mean_field()
    drain = load_drain_table_with_flow(mean)
    tree, lat0 = build_drain_tree(drain)

    q_old = load_old_assignment(ror)
    ror["river_mean_old"] = q_old.reindex(ror.index)
    ror["ratio_old"] = ror.river_mean_old / ror.qmax_turb
    ror["in_band_old"] = (ror.ratio_old >= BAND_LOW) & (ror.ratio_old <= BAND_HIGH)

    rows = []
    for pid, p in ror.iterrows():
        xq = p.lon * 111.0 * np.cos(np.deg2rad(lat0))
        yq = p.lat * 111.0
        r = match_one(xq, yq, p.qmax_turb, tree, drain)
        r["plant"] = pid
        rows.append(r)

    match = pd.DataFrame(rows).set_index("plant")
    out = ror.join(match)
    out["ratio_new"] = out["matched_ratio"]
    out["in_band_new"] = (out["ratio_new"] >= BAND_LOW) & (out["ratio_new"] <= BAND_HIGH)
    out["failed"] = out["tier"].str.startswith("FAILED", na=False)

    out.to_csv(OUTDIR / "ror_match_prototype.csv")

    def bucket(in_band, failed):
        if failed:
            return "failed"
        if pd.isna(in_band):
            return "unknown"
        return "in_band" if in_band else "out_of_band"

    out["old_bucket"] = [bucket(b, False) for b in out.in_band_old]
    out["new_bucket"] = [bucket(b, f) for b, f in zip(out.in_band_new, out.failed)]

    trans = (out.pivot_table(index="old_bucket", columns="new_bucket",
                             values="Capacity", aggfunc="sum", fill_value=0.0) / 1e3)
    trans.to_csv(OUTDIR / "transition_matrix.csv")

    tiers = (out.groupby("tier").Capacity.sum() / 1e3).sort_values(ascending=False)
    tiers.to_csv(OUTDIR / "tier_summary.csv")

    show = ["Name", "Country", "Capacity", "qmax_turb", "ratio_old",
            "ratio_new", "matched_dist_km", "n_cells", "tier"]
    failed_plants = out[out.failed].nlargest(50, "Capacity")[show]
    failed_plants.to_csv(OUTDIR / "failed_plants.csv")

    tot = out.Capacity.sum() / 1e3
    became_ok = out[(out.old_bucket == "out_of_band") & (out.new_bucket == "in_band")]
    still_bad = out[(out.old_bucket == "out_of_band") & (out.new_bucket != "in_band")]
    broke = out[(out.old_bucket == "in_band") & (out.new_bucket != "in_band")]
    median_dist_fixed = became_ok.matched_dist_km.median() if len(became_ok) else np.nan

    with open(OUTDIR / "summary_match_prototype.txt", "w") as f:
        w = f.write
        w("ROR MATCH PROTOTYPE: two-tier drain-table rule vs today's assignment\n")
        w("=" * 72 + "\n\n")
        w("plant identity: Eur_custom_ppls_GloHydroRes_filled.csv row index "
          "(the real pipeline identity, not GloHydroRes_vs1.csv 'ID')\n")
        w("'old' source   : final_hydro_plant_glofas_points.parquet river_mean "
          "(raw GloFAS mean at today's assigned cell -- same physical "
          "quantity as 'new', not the SABER-corrected series)\n")
        w(f"rule: 10km band[{BAND_LOW:g},{BAND_HIGH:g}] -> 25km same band -> "
          f"25km relaxed>={RELAX_LOW:g} -> flagged fallback\n\n")
        w(f"RoR plants: {len(out)} ({tot:.2f} GW)\n\n")
        w("TIER SUMMARY (capacity by how the match was resolved)\n")
        w(tiers.to_string() + "\n\n")
        w("TRANSITION MATRIX old-bucket x new-bucket [GW]\n")
        w(trans.to_string() + "\n\n")
        w(f"Capacity moved from out-of-band to in-band : {became_ok.Capacity.sum()/1e3:.2f} GW "
          f"(median move: {median_dist_fixed:.1f} km)\n")
        w(f"Capacity still out-of-band or failed        : {still_bad.Capacity.sum()/1e3:.2f} GW\n")
        w(f"Capacity that WORSENED (was in-band, now not): {broke.Capacity.sum()/1e3:.2f} GW\n")
        w(f"Capacity FAILED (no plausible cell at all)  : "
          f"{out[out.failed].Capacity.sum()/1e3:.2f} GW\n")
    print("\n" + (OUTDIR / "summary_match_prototype.txt").read_text())

    _fig(trans)
    print(f"[done] outputs in {OUTDIR.resolve()}")


def _fig(trans):
    palette = {"in_band": "#4daf4a", "out_of_band": "#ff7f00", "failed": "#e41a1c"}
    colors = [palette.get(c, "grey") for c in trans.columns]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    trans.plot.bar(stacked=True, ax=ax, color=colors)
    ax.set_ylabel("capacity [GW]")
    ax.set_xlabel("bucket under today's assignment")
    ax.set_title("Where capacity lands under the new rule")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_transition.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
