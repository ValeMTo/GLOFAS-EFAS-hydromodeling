#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
reservoir_audit.py

PURPOSE
-------
Single-script audit for Reservoir plants (NOT Pumped Storage), mirroring
ror_recoverability_test.py but with a criterion appropriate for storage
plants: unlike run-of-river, a reservoir's Qmax can legitimately be much
larger than the mean inflow (it turbines from storage at need, not at
run-of-river pace), so there is no meaningful upper band on Q_cell/Qmax. The
only thing that can still be wrong is the assigned cell carrying too LITTLE
water to plausibly generate anything close to the plant's expected output --
that is a real matching error regardless of storage.

CRITERION
---------
    power_pot    = 9.81 * eta * H * 1e-3 * Q_cell_mean   [MW, if it turbined
                                                           the mean inflow]
    E_pot_annual = power_pot * 8760 h
    E_expected   = P_nom * CF_REF * 8760 h                [CF_REF: a rough,
                                                           not literature-
                                                           calibrated, default
                                                           expected capacity
                                                           factor]
    ratio_energy = E_pot_annual / E_expected = (Q_cell/Qmax) / CF_REF

    ratio_energy >= LOW_FRAC (0.2)  -> plausible
    ratio_energy <  LOW_FRAC        -> starved

No upper bound: a reservoir fed by a much richer cell than needed is normal
(it just doesn't use all of it), not a matching error.

IDENTITY (important, fixed 2026-09-18)
----------------------------------------
An earlier version identified plants via GloHydroRes_vs1.csv's "ID" column
(digits stripped) and assumed that number equals the pipeline's own plant
identity. It does not: the real pipeline reads plants from
Eur_custom_ppls_GloHydroRes_filled.csv, whose row index ("plant_index") is
the true identity used everywhere downstream (final_matches, inflow
NetCDF). This version loads plants directly from that file and reads the
"assigned cell" straight from the real final_hydro_plant_glofas_points.parquet
(river_mean = raw GloFAS mean at the assigned cell), instead of mixing in
the SABER-corrected inflow NetCDF or a mismatched plant id.

For every plant flagged "starved" it also searches EVERY GloFAS cell within
SEARCH_KM (not only the ones in the drainage table) for one meeting the
threshold, to tell apart "no adequate cell nearby" (data limit) from "one
existed and wasn't picked" (algorithmic).

OUTPUTS (OUTDIR)
----------------
    reservoir_audit.csv            one row per Reservoir plant
    audit_by_country.csv           capacity-weighted crosstab
    largest_starved_with_alternative.csv
    glofas_mean_discharge.nc        cached mean field (shared cache name)
    fig_verdict.png, fig_distance_to_best.png, summary_reservoir_audit.txt
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

# The REAL pipeline's plant table: its row index ("plant_index") is the true
# plant identity used everywhere downstream.
EUR_PLANTS_FILE = ROOT / "hydro_workflow/hydro_global/Eur_custom_ppls_GloHydroRes_filled.csv"

GLOFAS_DIR = ROOT / "hydro_workflow/glofas_europe"
GLOFAS_PATTERN = "glofas_eu_{year}.nc"
GLOFAS_VAR = "dis24"
MEAN_YEARS = [2015, 2016, 2017]

# The REAL pipeline's own match output. river_mean here is the raw GloFAS
# mean at the coordinates actually assigned today.
FINAL_MATCHES = ROOT / "hydro_workflow/saber_global/glofas/final_hydro_plant_glofas_points.parquet"

# Shared with ror_recoverability_test.py: same field, computed once.
MEAN_FIELD_CACHE = Path("./ror_recover_out/glofas_mean_discharge.nc")

OUTDIR = Path("./reservoir_audit_out")

# --- search geometry ---------------------------------------------------------
SEARCH_KM = 25.0
NEAR_KM = 10.0

# --- the one criterion: energy plausibility, not raw flow -------------------
CF_REF = 0.35                      # rough divisor, NOT a literature target
LOW_FRAC = 0.2                      # ratio_energy = (Q_cell/Qmax)/CF_REF >= this

EFFICIENCY = 0.85
RES_TECH = "Reservoir"             # Pumped Storage is deliberately excluded


# =============================================================================
# HELPERS
# =============================================================================
def load_plants():
    """DamHeight_m has no missing/non-positive values in the filled Europe
    table, so Qmax is a direct formula -- no median imputation needed."""
    eur = pd.read_csv(EUR_PLANTS_FILE, index_col=0)
    eur.index.name = "plant"

    res = eur[eur.Technology == RES_TECH].copy()
    print(f"[info] Europe plant table: {len(eur)} plants -> {len(res)} Reservoir")

    out = pd.DataFrame(index=res.index)
    out["name"] = res["Name"]
    out["country"] = res["Country"]
    out["p_nom"] = res["Capacity"]
    out["damheight_m"] = res["DamHeight_m"]
    out["qmax_turb"] = out["p_nom"] / (9.81 * out["damheight_m"] * 1e-3 * EFFICIENCY)
    return out.dropna(subset=["p_nom", "qmax_turb"])


def find_glofas_file(year):
    f = GLOFAS_DIR / GLOFAS_PATTERN.format(year=year)
    if f.exists():
        return f
    hits = sorted(GLOFAS_DIR.glob(f"*{year}*.nc"))
    if hits:
        print(f"[info] {year}: pattern missed, using {hits[0].name}")
        return hits[0]
    return None


def glofas_mean_field():
    """Long-term mean discharge, reusing ror_recoverability_test.py's cache
    when present so the field is computed only once across both scripts."""
    if MEAN_FIELD_CACHE.exists():
        print(f"[info] using shared cached mean field {MEAN_FIELD_CACHE}")
        return xr.open_dataarray(MEAN_FIELD_CACHE)

    cache = OUTDIR / "glofas_mean_discharge.nc"
    if cache.exists():
        print(f"[info] using cached mean field {cache}")
        return xr.open_dataarray(cache)

    fields = []
    for y in MEAN_YEARS:
        f = find_glofas_file(y)
        if f is None:
            print(f"[warn] no GloFAS file for {y} in {GLOFAS_DIR}")
            continue
        try:
            ds = xr.open_dataset(f, chunks={"time": 30})
        except Exception:
            ds = xr.open_dataset(f)
        var = GLOFAS_VAR
        da = ds[var]
        tdim = ("valid_time" if "valid_time" in da.dims
                else "time" if "time" in da.dims else da.dims[0])
        print(f"[info] {f.name}: var={var}, dims={dict(da.sizes)}")
        m = da.mean(dim=tdim)
        fields.append(m.compute() if hasattr(m.data, "compute") else m)
        ds.close()

    if not fields:
        sys.exit(f"[FATAL] no GloFAS files found in {GLOFAS_DIR}.")

    mean = xr.concat(fields, dim="__y").mean(dim="__y").rename("q_mean")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    mean.to_netcdf(cache)
    print(f"[info] cached mean field -> {cache}")
    return mean


def build_cell_tree(mean):
    latname = "latitude" if "latitude" in mean.coords else "lat"
    lonname = "longitude" if "longitude" in mean.coords else "lon"
    lon_g, lat_g = np.meshgrid(mean[lonname].values, mean[latname].values)
    q = mean.values
    ok = np.isfinite(q) & (q > 0)
    lat_f, lon_f, q_f = lat_g[ok], lon_g[ok], q[ok]
    print(f"[info] valid GloFAS cells: {len(q_f):,} of {q.size:,}")
    lat0 = float(np.mean(lat_f))
    x = lon_f * 111.0 * np.cos(np.deg2rad(lat0))
    y = lat_f * 111.0
    return cKDTree(np.column_stack([x, y])), lat_f, lon_f, q_f, lat0


def assigned_cell_discharge(res):
    """Mean discharge of the cell the pipeline actually assigned, straight
    from its own match output (already the raw GloFAS mean)."""
    fm = pd.read_parquet(FINAL_MATCHES).set_index("plant_index")
    fm = fm[~fm.index.duplicated()]
    common = res.index.intersection(fm.index)
    print(f"[info] assigned-cell discharge from final_matches "
          f"({len(common)} of {len(res)} plants, raw GloFAS mean)")
    return fm.loc[common, "river_mean"]


# =============================================================================
# MAIN
# =============================================================================
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    res = load_plants()
    print(f"[info] Reservoir plants: {len(res)} ({res.p_nom.sum()/1e3:.2f} GW) | "
          f"eta={EFFICIENCY}, CF_REF={CF_REF:g}, LOW_FRAC={LOW_FRAC:g}")

    mean = glofas_mean_field()
    tree, lat_f, lon_f, q_f, lat0 = build_cell_tree(mean)
    q_now = assigned_cell_discharge(res)

    res["q_cell_now"] = q_now.reindex(res.index)
    res["ratio_flow_now"] = res.q_cell_now / res.qmax_turb
    res["ratio_energy_now"] = res.ratio_flow_now / CF_REF
    res["plausible_now"] = res.ratio_energy_now >= LOW_FRAC

    # plant coordinates aren't in `res` (kept lean) -- pull them back in
    eur = pd.read_csv(EUR_PLANTS_FILE, index_col=0)
    eur.index.name = "plant"
    coords = eur.loc[res.index, ["lat", "lon"]]
    res = res.join(coords)

    rows = []
    for pid, p in res.iterrows():
        xq = p.lon * 111.0 * np.cos(np.deg2rad(lat0))
        yq = p.lat * 111.0
        idx = tree.query_ball_point([xq, yq], r=SEARCH_KM)
        rec = {"plant": pid}
        if not idx:
            rec.update(n_cells=0, n_ok=0)
            rows.append(rec)
            continue

        idx = np.asarray(idx)
        d_km = np.hypot(lon_f[idx] * 111.0 * np.cos(np.deg2rad(lat0)) - xq,
                        lat_f[idx] * 111.0 - yq)
        qc = q_f[idx]
        ratio_energy = (qc / p.qmax_turb) / CF_REF
        ok = ratio_energy >= LOW_FRAC

        rec.update(n_cells=len(idx), n_ok=int(ok.sum()))
        if ok.any():
            b = int(np.argmin(d_km[ok]))
            rec.update(best_dist_km=float(d_km[ok][b]),
                       best_q=float(qc[ok][b]),
                       best_ratio_energy=float(ratio_energy[ok][b]),
                       best_lat=float(lat_f[idx][ok][b]),
                       best_lon=float(lon_f[idx][ok][b]))
        rows.append(rec)

    out = res.join(pd.DataFrame(rows).set_index("plant"))
    out["availability"] = np.where(
        out.get("n_ok", 0).fillna(0) == 0, "no_good_cell",
        np.where(out.get("best_dist_km", np.nan) <= NEAR_KM,
                 "good_cell_near", "good_cell_far"))

    def verdict(r):
        if not np.isfinite(r.get("ratio_energy_now", np.nan)):
            return "unknown_assignment"
        if r.plausible_now:
            return "already_ok"
        if r.availability == "no_good_cell":
            return "not_recoverable"
        return ("recoverable_near" if r.availability == "good_cell_near"
                else "recoverable_far")
    out["verdict"] = out.apply(verdict, axis=1)

    VORDER = ["already_ok", "recoverable_near", "recoverable_far",
              "not_recoverable", "unknown_assignment"]
    present = [v for v in VORDER if v in set(out.verdict)]

    out.to_csv(OUTDIR / "reservoir_audit.csv")

    def table(by):
        gw = (out.pivot_table(index=by, columns="verdict", values="p_nom",
                              aggfunc="sum", fill_value=0.0) / 1e3)
        gw = gw.reindex(columns=present, fill_value=0.0)
        gw["total_GW"] = gw.sum(axis=1)
        pct = gw[present].div(gw.total_GW.replace(0, np.nan), axis=0) * 100
        pct.columns = [f"pct_{c}" for c in present]
        return pd.concat([gw.round(3), pct.round(1)], axis=1)

    t_ctry = table("country").sort_values("total_GW", ascending=False)
    t_ctry.to_csv(OUTDIR / "audit_by_country.csv")
    print("\n[by country]\n", t_ctry.to_string())

    show = ["name", "country", "p_nom", "damheight_m",
            "qmax_turb", "q_cell_now", "ratio_flow_now", "ratio_energy_now",
            "best_q", "best_ratio_energy", "best_dist_km", "verdict"]
    starved = out[~out.plausible_now.fillna(False)]
    big = starved.nlargest(30, "p_nom")[[c for c in show if c in out.columns]]
    big.to_csv(OUTDIR / "largest_starved_with_alternative.csv")
    print("\n[largest starved plants and the best nearby alternative]\n",
          big.round(3).to_string())

    tot = out.p_nom.sum() / 1e3
    with open(OUTDIR / "summary_reservoir_audit.txt", "w") as f:
        w = f.write
        w("RESERVOIR AUDIT: is the assigned cell's inflow energy-plausible?\n")
        w("=" * 64 + "\n\n")
        w("plant identity      : Eur_custom_ppls_GloHydroRes_filled.csv row "
          "index (the real pipeline identity)\n")
        w("assigned cell from  : final_hydro_plant_glofas_points.parquet "
          "(raw GloFAS mean)\n")
        w(f"criterion           : ratio_energy = (Q_cell/Qmax)/CF_REF >= "
          f"{LOW_FRAC:g}\n")
        w(f"CF_REF              : {CF_REF:g}  (rough divisor, not literature-"
          "calibrated)\n")
        w(f"                      no upper bound: reservoirs turbine from "
          "storage, not run-of-river\n")
        w(f"search radius       : {SEARCH_KM:g} km (near threshold {NEAR_KM:g} km)\n")
        w(f"GloFAS mean years   : {MEAN_YEARS}\n")
        w(f"Reservoir plants    : {len(out)}  ({tot:.2f} GW)\n\n")
        w("VERDICT\n")
        g = (out.groupby("verdict").p_nom.sum() / 1e3)
        for k in present:
            v = g.get(k, 0.0)
            w(f"  {k:18s} {v:7.2f} GW  ({100*v/tot:5.1f} %)\n")
        starved_gw = starved.p_nom.sum() / 1e3
        if starved_gw > 0:
            rec = starved[starved.availability != "no_good_cell"].p_nom.sum() / 1e3
            w(f"\nOf the {starved_gw:.2f} GW below ratio_energy {LOW_FRAC:g},\n"
              f"  {rec:.2f} GW ({100*rec/starved_gw:.1f} %) had a cell meeting "
              "the threshold within reach.\n")
            d = starved.dropna(subset=["best_dist_km"])
            if len(d):
                w(f"  median distance to that cell: "
                  f"{d.best_dist_km.median():.1f} km\n")
        w("BY COUNTRY\n" + t_ctry.to_string() + "\n")
    print("\n" + (OUTDIR / "summary_reservoir_audit.txt").read_text())

    _figs(out, present)
    print(f"[done] outputs in {OUTDIR.resolve()}")


# =============================================================================
# FIGURES
# =============================================================================
def _figs(out, present):
    g = (out.groupby("verdict").p_nom.sum() / 1e3).reindex(present, fill_value=0.0)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(g.index, g.values,
            color=["#4daf4a", "#377eb8", "#ff7f00", "#e41a1c", "grey"][:len(g)])
    ax.set_xlabel("capacity [GW]")
    ax.set_title(f"Reservoir plants: ratio_energy = (Q_cell/Qmax)/CF_REF "
                 f">= {LOW_FRAC:g} ?")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_verdict.png", dpi=180)
    plt.close(fig)

    d = out.dropna(subset=["best_dist_km"])
    d = d[~d.plausible_now.fillna(False)]
    if len(d):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.hist(d.best_dist_km, bins=np.linspace(0, SEARCH_KM, 40),
                weights=d.p_nom / 1e3, color="#377eb8",
                histtype="stepfilled", alpha=0.75)
        ax.axvline(NEAR_KM, color="k", ls="--", lw=1.2,
                   label=f"pipeline radius ~{NEAR_KM:g} km")
        ax.set_xlabel("distance to the nearest cell meeting the threshold [km]")
        ax.set_ylabel("capacity [GW]")
        ax.set_title("Starved reservoirs: how far away an adequate cell is")
        ax.legend(fontsize=8); ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(OUTDIR / "fig_distance_to_best.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
