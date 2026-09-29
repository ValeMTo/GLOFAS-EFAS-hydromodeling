#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ror_recoverability_test.py

THE QUESTION
------------
Is the RoR plant-to-cell mismatch avoidable? For every RoR plant, search
EVERY GloFAS cell within SEARCH_KM of the reported coordinates -- not only
the cells the pipeline's own drain-table search would consider -- and ask
whether any of them carries a long-term mean discharge compatible with the
plant's design discharge, within a factor of BAND.

BAND = 3 is deliberately generous: a well-sized RoR plant sits around
1.5-2, so a factor of 3 asks only whether something REASONABLE was
available, not whether the perfect cell was found.

IDENTITY (important, fixed 2026-09-18)
----------------------------------------
An earlier version of this script identified plants via GloHydroRes_vs1.csv's
"ID" column (digits stripped) and assumed that number equals the pipeline's
own plant identity in the inflow NetCDF / final_matches. It does not: the
real pipeline (load_hydro_power_plants() in
extract_glofas_series_for_hydro_plants.py) reads plants from
Eur_custom_ppls_GloHydroRes_filled.csv, whose row index ("plant_index") is
the true identity used everywhere downstream. The two numbering schemes
disagree for most rows (verified: "GHR ID 4086" = Mount Coffee, Liberia,
while pipeline plant_index 4086 = Kaunas HPP, Lithuania), which silently
compared unrelated plants. This version loads plants directly from
Eur_custom_ppls_GloHydroRes_filled.csv and never touches GloHydroRes_vs1.csv
or a digit-stripping id, so there is no re-identification step to get wrong.
A side effect: that file carries no "river" name, so the major_river/canal
river-type breakdown from earlier versions is dropped (not recoverable
reliably under the correct identity).

The "current assignment" is read directly from the real pipeline output,
final_hydro_plant_glofas_points.parquet, whose river_mean column is already
the raw GloFAS mean discharge at the assigned cell -- the same physical
quantity as the one this script searches for alternatives, so the
before/after comparison is homogeneous (no mixing with SABER-corrected
series).

WHAT IT DOES NOT PROVE
----------------------
Finding a correctly scaled cell is NECESSARY but NOT SUFFICIENT for a
correct assignment: a cell of the right size could belong to a different
river or a parallel valley. So the verdict is "a plausible alternative
existed and was not considered", not "these plants can be reassigned".

OUTPUTS (OUTDIR)
----------------
    recoverability.csv                   one row per plant
    recoverability_by_head.csv           capacity-weighted crosstab
    recoverability_by_country.csv        capacity-weighted crosstab
    largest_failed_with_alternative.csv  inspection list
    glofas_mean_discharge.nc             cached mean field (shared cache)
    fig_*.png, summary_recoverability.txt
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
PYPSA_DIR = ROOT / "pypsa"

# The REAL pipeline's plant table: its row index ("plant_index") is the true
# plant identity used everywhere downstream (final_matches, inflow NetCDF).
EUR_PLANTS_FILE = ROOT / "hydro_workflow/hydro_global/Eur_custom_ppls_GloHydroRes_filled.csv"

# Raw GloFAS downloads. Find the directory with:
#   python -c "from hydro_inflow.hydro_config import get_config as g; \
#              print(g()['hydro_data_dir'], g()['hydro_var_name'])"
GLOFAS_DIR = ROOT / "hydro_workflow/glofas_europe"
GLOFAS_PATTERN = "glofas_eu_{year}.nc"    # a *{year}*.nc glob is tried as well
GLOFAS_VAR = "dis24"
MEAN_YEARS = [2015, 2016, 2017]

# The REAL pipeline's own match output. river_mean here is the raw GloFAS
# mean at the coordinates actually assigned today -- the correct "before",
# homogeneous with the mean field this script searches for alternatives.
FINAL_MATCHES = ROOT / "hydro_workflow/saber_global/glofas/final_hydro_plant_glofas_points.parquet"

OUTDIR = Path("./ror_recover_out")

# --- search geometry --------------------------------------------------------
SEARCH_KM = 25.0                  # the pipeline effectively sees ~10-15 km
NEAR_KM = 10.0                    # the pipeline's MAX_DISTANCE_KM

# --- acceptability on Q_cell / Qmax -----------------------------------------
BAND = 3.0                        # acceptable: within a factor of 3
IDEAL = 2.0                       # ideal: within a factor of 2

# --- must match the pipeline -------------------------------------------------
EFFICIENCY = 0.85
ROR_TECH = "Run-Of-River"


# =============================================================================
# HELPERS
# =============================================================================
def load_plants():
    """DamHeight_m has no missing/non-positive values in the filled Europe
    table, so Qmax is a direct formula -- no median imputation needed."""
    eur = pd.read_csv(EUR_PLANTS_FILE, index_col=0)
    eur.index.name = "plant"

    df = pd.DataFrame(index=eur.index)
    df["name"] = eur["Name"]
    df["lat"] = eur["lat"]
    df["lon"] = eur["lon"]
    df["p_nom"] = eur["Capacity"]
    df["country"] = eur["Country"]
    df["technology"] = eur["Technology"]
    df["damheight_m"] = eur["DamHeight_m"]
    df["qmax_turb"] = df["p_nom"] / (9.81 * df["damheight_m"] * 1e-3 * EFFICIENCY)

    ror = df[df.technology == ROR_TECH].copy()
    print(f"[info] Europe plant table: {len(eur)} plants -> {len(ror)} RoR")
    return ror.dropna(subset=["lat", "lon", "p_nom", "qmax_turb"])


def find_glofas_file(year):
    """The pipeline's own naming, with a permissive glob as fallback."""
    f = GLOFAS_DIR / GLOFAS_PATTERN.format(year=year)
    if f.exists():
        return f
    hits = sorted(GLOFAS_DIR.glob(f"*{year}*.nc"))
    if hits:
        print(f"[info] {year}: pattern missed, using {hits[0].name}")
        return hits[0]
    return None


def glofas_mean_field():
    """Long-term mean discharge on the GloFAS grid, cached to NetCDF."""
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
        if var is None:
            cands = [v for v in ds.data_vars
                     if any(k in v.lower() for k in ("dis", "discharge"))]
            var = (cands or list(ds.data_vars))[0]
        da = ds[var]
        tdim = ("valid_time" if "valid_time" in da.dims
                else "time" if "time" in da.dims else da.dims[0])
        print(f"[info] {f.name}: var={var}, dims={dict(da.sizes)}")
        m = da.mean(dim=tdim)
        fields.append(m.compute() if hasattr(m.data, "compute") else m)
        ds.close()

    if not fields:
        sys.exit(f"[FATAL] no GloFAS files found in {GLOFAS_DIR}. "
                 "Adjust GLOFAS_DIR / GLOFAS_PATTERN.")

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


def assigned_cell_discharge(ror):
    """Mean discharge of the cell the pipeline actually assigned, straight
    from its own match output (already the raw GloFAS mean)."""
    fm = pd.read_parquet(FINAL_MATCHES).set_index("plant_index")
    fm = fm[~fm.index.duplicated()]
    common = ror.index.intersection(fm.index)
    print(f"[info] assigned-cell discharge from final_matches "
          f"({len(common)} of {len(ror)} plants, raw GloFAS mean)")
    return fm.loc[common, "river_mean"]


# =============================================================================
# MAIN
# =============================================================================
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ror = load_plants()
    print(f"[info] RoR plants: {len(ror)} ({ror.p_nom.sum()/1e3:.2f} GW) | eta={EFFICIENCY}")

    mean = glofas_mean_field()
    tree, lat_f, lon_f, q_f, lat0 = build_cell_tree(mean)
    q_now = assigned_cell_discharge(ror)
    q_now_src = "final_hydro_plant_glofas_points.parquet (raw GloFAS mean)"

    rows = []
    for pid, p in ror.iterrows():
        xq = p.lon * 111.0 * np.cos(np.deg2rad(lat0))
        yq = p.lat * 111.0
        idx = tree.query_ball_point([xq, yq], r=SEARCH_KM)
        rec = {"plant": pid}
        if not idx:
            rec.update(n_cells=0, n_in_band=0, n_in_ideal=0)
            rows.append(rec)
            continue

        idx = np.asarray(idx)
        d_km = np.hypot(lon_f[idx] * 111.0 * np.cos(np.deg2rad(lat0)) - xq,
                        lat_f[idx] * 111.0 - yq)
        qc = q_f[idx]
        ratio = qc / p.qmax_turb
        b = int(np.argmin(np.abs(np.log(np.maximum(ratio, 1e-12)))))

        in_band = (ratio >= 1.0 / BAND) & (ratio <= BAND)
        in_ideal = (ratio >= 1.0 / IDEAL) & (ratio <= IDEAL)

        rec.update(
            n_cells=len(idx), q_cells_min=float(qc.min()),
            q_cells_max=float(qc.max()),
            best_q=float(qc[b]), best_ratio=float(ratio[b]),
            best_dist_km=float(d_km[b]),
            best_lat=float(lat_f[idx][b]), best_lon=float(lon_f[idx][b]),
            n_in_band=int(in_band.sum()), n_in_ideal=int(in_ideal.sum()),
            band_min_dist_km=float(d_km[in_band].min()) if in_band.any() else np.nan,
            band_best_q=float(qc[in_band][np.argmin(d_km[in_band])])
            if in_band.any() else np.nan,
        )
        rows.append(rec)

    out = ror.join(pd.DataFrame(rows).set_index("plant"))

    # --- availability verdict (independent of what was chosen) --------------
    def avail(r):
        if r.get("n_in_band", 0) == 0:
            return "no_good_cell"
        return ("good_cell_near" if r.band_min_dist_km <= NEAR_KM
                else "good_cell_far")
    out["availability"] = out.apply(avail, axis=1)

    # --- comparison with what the pipeline actually assigned ----------------
    out["q_cell_now"] = q_now.reindex(out.index)
    out["ratio_now"] = out.q_cell_now / out.qmax_turb
    out["now_in_band"] = ((out.ratio_now >= 1.0 / BAND)
                          & (out.ratio_now <= BAND))
    out["log_err_now"] = np.abs(np.log(np.maximum(out.ratio_now, 1e-12)))
    out["log_err_best"] = np.abs(np.log(np.maximum(out.best_ratio, 1e-12)))
    out["improvement"] = out.log_err_now - out.log_err_best

    def verdict(r):
        if not np.isfinite(r.get("ratio_now", np.nan)):
            return "unknown_assignment"
        if r.now_in_band:
            return "already_ok"
        if r.availability == "no_good_cell":
            return "not_recoverable"
        return ("recoverable_near" if r.availability == "good_cell_near"
                else "recoverable_far")
    out["verdict"] = out.apply(verdict, axis=1)
    VCOL, VORDER = "verdict", ["already_ok", "recoverable_near",
                               "recoverable_far", "not_recoverable",
                               "unknown_assignment"]

    out.to_csv(OUTDIR / "recoverability.csv")
    present = [v for v in VORDER if v in set(out[VCOL])]

    def table(by):
        gw = (out.pivot_table(index=by, columns=VCOL, values="p_nom",
                              aggfunc="sum", fill_value=0.0) / 1e3)
        gw = gw.reindex(columns=present, fill_value=0.0)
        gw["total_GW"] = gw.sum(axis=1)
        pct = gw[present].div(gw.total_GW.replace(0, np.nan), axis=0) * 100
        pct.columns = [f"pct_{c}" for c in present]
        return pd.concat([gw.round(3), pct.round(1)], axis=1)

    t_ctry = table("country").sort_values("total_GW", ascending=False)
    t_ctry.to_csv(OUTDIR / "recoverability_by_country.csv")
    print("\n[by country]\n", t_ctry.to_string())

    show = ["name", "country", "p_nom", "damheight_m",
            "qmax_turb", "q_cell_now", "ratio_now",
            "best_q", "best_ratio", "best_dist_km", "band_min_dist_km",
            "n_in_band", VCOL]
    bad = out[~out.now_in_band.fillna(False)]
    big = bad.nlargest(30, "p_nom")[show]
    big.to_csv(OUTDIR / "largest_failed_with_alternative.csv")
    print("\n[largest mismatched plants and the best nearby cell]\n",
          big.round(3).to_string())

    with open(OUTDIR / "summary_recoverability.txt", "w") as f:
        w = f.write
        tot = out.p_nom.sum() / 1e3
        w("RECOVERABILITY TEST: was a correctly scaled cell available?\n")
        w("=" * 64 + "\n\n")
        w("plant identity     : Eur_custom_ppls_GloHydroRes_filled.csv row "
          "index (the real pipeline identity)\n")
        w("assigned cell from : final_hydro_plant_glofas_points.parquet "
          "(raw GloFAS mean, homogeneous with the search field)\n")
        w(f"search radius      : {SEARCH_KM:g} km (near threshold {NEAR_KM:g} km)\n")
        w(f"acceptable band    : Q_cell/Qmax within a factor of {BAND:g}\n")
        w(f"GloFAS mean years  : {MEAN_YEARS}\n")
        w(f"efficiency         : {EFFICIENCY}\n")
        w(f"RoR plants         : {len(out)}  ({tot:.2f} GW)\n\n")
        w("AVAILABILITY (is a correctly scaled cell reachable at all?)\n")
        g = (out.groupby("availability").p_nom.sum() / 1e3)
        for k, v in g.sort_values(ascending=False).items():
            w(f"  {k:18s} {v:7.2f} GW  ({100*v/tot:5.1f} %)\n")
        w("\nVERDICT vs THE CELL ACTUALLY ASSIGNED\n")
        g2 = (out.groupby("verdict").p_nom.sum() / 1e3)
        for k in present:
            v = g2.get(k, 0.0)
            w(f"  {k:18s} {v:7.2f} GW  ({100*v/tot:5.1f} %)\n")
        mis = out[~out.now_in_band.fillna(False)]
        rec = mis[mis.availability != "no_good_cell"].p_nom.sum() / 1e3
        w(f"\nOf the {mis.p_nom.sum()/1e3:.2f} GW whose assigned cell is "
          f"out of band,\n  {rec:.2f} GW "
          f"({100*rec/max(mis.p_nom.sum()/1e3,1e-9):.1f} %) had a "
          "correctly scaled cell within reach.\n")
        d = mis.dropna(subset=["band_min_dist_km"])
        if len(d):
            w(f"  median distance to that cell: "
              f"{d.band_min_dist_km.median():.1f} km\n")
            w(f"  share of that capacity with the cell inside "
              f"{NEAR_KM:g} km: "
              f"{100*d.loc[d.band_min_dist_km<=NEAR_KM,'p_nom'].sum()/d.p_nom.sum():.1f} %\n")
        w("BY COUNTRY\n" + t_ctry.to_string() + "\n")
    print("\n" + (OUTDIR / "summary_recoverability.txt").read_text())

    _figs(out, VCOL, present)
    print(f"[done] outputs in {OUTDIR.resolve()}")


# =============================================================================
# FIGURES
# =============================================================================
def _figs(out, VCOL, present):
    g = (out.groupby(VCOL).p_nom.sum() / 1e3).reindex(present, fill_value=0.0)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(g.index, g.values,
            color=["#4daf4a", "#377eb8", "#ff7f00", "#e41a1c", "grey"][:len(g)])
    ax.set_xlabel("capacity [GW]")
    ax.set_title("Was a correctly scaled GloFAS cell available nearby?")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_recoverability.png", dpi=180)
    plt.close(fig)

    d = out.dropna(subset=["band_min_dist_km"])
    if len(d):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.hist(d.band_min_dist_km, bins=np.linspace(0, SEARCH_KM, 40),
                weights=d.p_nom / 1e3, color="#377eb8",
                histtype="stepfilled", alpha=0.75)
        ax.axvline(NEAR_KM, color="k", ls="--", lw=1.2,
                   label=f"pipeline radius ~{NEAR_KM:g} km")
        ax.set_xlabel("distance to the nearest correctly scaled cell [km]")
        ax.set_ylabel("capacity [GW]")
        ax.set_title("How far away the right cell actually is")
        ax.legend(fontsize=8); ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(OUTDIR / "fig_distance_to_best.png", dpi=180)
        plt.close(fig)

    d = out.dropna(subset=["ratio_now", "best_ratio"])
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(np.log10(np.maximum(d.ratio_now, 1e-6)),
               np.log10(np.maximum(d.best_ratio, 1e-6)),
               s=3 + d.p_nom / 12, alpha=0.5, linewidths=0, c="#377eb8")
    lim = [-5, 3]
    ax.plot(lim, lim, "k-", lw=0.8)
    for v in (-np.log10(BAND), np.log10(BAND)):
        ax.axhline(v, color="green", ls=":", lw=1)
        ax.axvline(v, color="green", ls=":", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"$\log_{10}$(assigned $Q_{cell}/Q_{max}$)")
    ax.set_ylabel(r"$\log_{10}$(best available $Q_{cell}/Q_{max}$)")
    ax.set_title("Assigned vs best available cell (size = MW)")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_assigned_vs_best.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
