#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ror_plant_level_audit.py

PURPOSE
-------
Per-plant audit of the hydropower plant-to-cell assignment for run-of-river
(RoR) plants, over multiple inflow years, to establish how much of the fleet
produces a physically sensible amount of energy on the cell it was assigned to.

WHY CAPACITY FACTOR AND NOT THE FLOW-DURATION CRITERION
-------------------------------------------------------
An earlier version classified plants by p* = P(Q_cell > Qmax), on the grounds
that design practice sizes Qmax on the flow exceeded 20-40% of the time. That
criterion misclassifies large low-head plants on major rivers, where Qmax
legitimately sits near or above mean flow. So classification is by capacity
factor, the direct measure of whether the assignment yields sensible energy,
with p* and Qmax/Q_cell retained as continuous diagnostics.

    CF < 0.02        dead                  produces nothing at all
    0.02 - 0.15      severely_under        cell far too small for the plant
    0.15 - 0.65      plausible             physically sensible for a RoR plant
    0.65 - 0.85      suspect               high for run-of-river
    CF > 0.85        impossible            no RoR plant runs at 85%+ CF

IDENTITY (important, fixed 2026-09-18)
----------------------------------------
An earlier version identified plants via GloHydroRes_vs1.csv's "ID" column
(digits stripped) and assumed that number equals the pipeline's own plant
identity in the inflow NetCDF. It does not: the real pipeline
(load_hydro_power_plants() in extract_glofas_series_for_hydro_plants.py)
reads plants from Eur_custom_ppls_GloHydroRes_filled.csv, whose row index
("plant_index") is the true identity used everywhere downstream, including
the inflow NetCDF's "plant" coordinate. The two numbering schemes disagree
for most rows (verified: "GHR ID 4086" = Mount Coffee, Liberia, while
pipeline plant_index 4086 = Kaunas HPP, Lithuania), which silently compared
each plant's CF against a DIFFERENT, unrelated plant's inflow series. This
version loads plants directly from Eur_custom_ppls_GloHydroRes_filled.csv
and matches the inflow NetCDF by that same integer index -- no digit-
stripping id, so there is no re-identification step to get wrong.

A side effect: that file carries no "river" name or a separate "Canal" label
(canal schemes are already folded into "Run-Of-River" upstream), so the
major_river/canal_scheme river-type breakdown from earlier versions is
dropped -- not recoverable reliably under the correct identity.

MECHANISMS THIS SCRIPT SEPARATES
--------------------------------
1. FLAT SERIES. The pipeline fills all-NaN discharge series with a constant
   0.1 m3/s, so those plants produce exactly zero by construction. Detected
   as near-zero coefficient of variation.
2. SCORE DEGENERACY. In score_candidates(), diff_norm = |Qcell - Qmax| divided
   by the maximum of the same quantity. When Qmax greatly exceeds every
   candidate, diff_norm collapses near 1 and the score degenerates to
   distance alone.
3. WRONG HEAD. Head errors propagate into Qmax and, through the matching
   score, into the cell choice itself. Split out by head_class.
4. IRRECOVERABLE COORDINATES. No adequate cell nearby. Data limit.

CONSISTENCY WITH THE PIPELINE
-----------------------------
EFFICIENCY=0.85, matching the pipeline's own load_hydro_power_plants()
exactly. DamHeight_m has no missing/non-positive values in the Europe plant
table used here, so no head imputation is needed.

MEMORY
------
Years are streamed one at a time and only per-plant accumulators are kept,
so peak memory is one year of data regardless of how many years are
requested.

OUTPUTS (OUTDIR)
----------------
    plant_audit.csv            one row per RoR plant, all diagnostics
    plant_cf_by_year.csv       per-plant capacity factor per year
    audit_by_class.csv         overall buckets
    audit_by_headclass.csv     buckets x head class
    audit_by_country.csv       buckets x country
    worst_offenders.csv        largest dead and severely-underproducing plants
    overproducers.csv          largest impossible-CF plants
    fig_cf_hist.png, fig_cf_vs_qratio.png, fig_buckets.png, summary_audit.txt
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG
# =============================================================================
ROOT = Path("/work/cmcc/ds07226/GLOFAS-EFAS-hydromodeling/data")
PYPSA_DIR = ROOT / "pypsa"

# The REAL pipeline's plant table: its row index ("plant_index") is the true
# plant identity used everywhere downstream, including the inflow NetCDF.
EUR_PLANTS_FILE = ROOT / "hydro_workflow/hydro_global/Eur_custom_ppls_GloHydroRes_filled.csv"

INFLOW_PATTERN = "Europe_inflow_{year}_GloFAS_SABER.nc"
YEARS = list(range(2015, 2024))          # 2015-2023
OUTDIR = Path("./ror_audit_out")

# Optional: the pipeline's own match table, adds match_type / distance_km.
FINAL_MATCHES = ROOT / "hydro_workflow/saber_global/glofas/final_hydro_plant_glofas_points.parquet"

# --- MUST MATCH THE PIPELINE THAT BUILT THE INFLOW FILES --------------------
EFFICIENCY = 0.85
QMIN_FACTOR = 0.20

# --- capacity-factor classification -----------------------------------------
CF_DEAD = 0.02
CF_UNDER = 0.15
CF_PLAUSIBLE_HI = 0.65
CF_SUSPECT_HI = 0.85
CLASS_ORDER = ["dead", "severely_under", "plausible", "suspect", "impossible"]

FLAT_CV = 1e-3                           # detects the constant-0.1 m3/s fill

HEAD_BINS = [0, 5, 15, 50, np.inf]
HEAD_LABELS = ["H<5", "5-15", "15-50", ">50"]

ROR_TECH = "Run-Of-River"

ETA_CURVES = {
    "kaplan":  np.array([[.05, .55], [.10, .72], [.20, .86], [.30, .93],
                         [.40, .96], [.50, .98], [.70, 1.0], [1.0, 1.0]]),
    "francis": np.array([[.05, .10], [.10, .30], [.20, .58], [.30, .76],
                         [.40, .86], [.50, .92], [.70, .98], [.90, 1.0],
                         [1.0, .99]]),
    "pelton":  np.array([[.05, .70], [.10, .83], [.20, .92], [.30, .96],
                         [.40, .98], [.50, .99], [.70, 1.0], [1.0, 1.0]]),
}


# =============================================================================
# HELPERS
# =============================================================================
def eta_rel(qt, turbine):
    out = np.ones_like(qt)
    turbine = np.asarray(turbine)
    for name, curve in ETA_CURVES.items():
        cols = np.where(turbine == name)[0]
        if cols.size:
            sub = np.clip(qt[:, cols], 0.0, 1.0)
            out[:, cols] = np.interp(sub.ravel(), curve[:, 0],
                                     curve[:, 1]).reshape(sub.shape)
    return out


def classify_cf(cf):
    b = np.full(len(cf), "plausible", dtype=object)
    b[cf < CF_UNDER] = "severely_under"
    b[cf < CF_DEAD] = "dead"
    b[(cf > CF_PLAUSIBLE_HI) & (cf <= CF_SUSPECT_HI)] = "suspect"
    b[cf > CF_SUSPECT_HI] = "impossible"
    return b


# =============================================================================
# PLANT TABLE
# =============================================================================
def load_plants():
    """DamHeight_m has no missing/non-positive values in the filled Europe
    table, so Qmax is a direct formula -- no median imputation needed."""
    eur = pd.read_csv(EUR_PLANTS_FILE, index_col=0)
    eur.index.name = "plant"

    out = pd.DataFrame(index=eur.index)
    out["name"] = eur["Name"]
    out["country"] = eur["Country"]
    out["p_nom"] = eur["Capacity"]
    out["head_m"] = eur["DamHeight_m"]
    out["head_class"] = pd.cut(out["head_m"], HEAD_BINS, labels=HEAD_LABELS,
                               right=False).astype(str)
    out["turbine"] = np.where(out["head_m"] < 30, "kaplan",
                              np.where(out["head_m"] < 300, "francis", "pelton"))
    out["qmax"] = out["p_nom"] / (9.81 * out["head_m"] * 1e-3 * EFFICIENCY)
    out["tech"] = eur["Technology"]

    ror = out[out.tech == ROR_TECH].copy()
    print(f"[info] Europe plant table: {len(eur)} plants -> {len(ror)} RoR")
    return ror.dropna(subset=["p_nom", "qmax"])


def open_inflow(year):
    f = PYPSA_DIR / INFLOW_PATTERN.format(year=year)
    if not f.exists():
        return None, None
    ds = xr.open_dataset(f)
    if len(ds.data_vars) == 1:
        da = ds[list(ds.data_vars)[0]]
    else:
        cands = [v for v in ds.data_vars if "inflow" in v.lower()]
        if not cands:
            sys.exit(f"[FATAL] pick a variable from {list(ds.data_vars)}")
        da = ds[cands[0]]
    pdim = [d for d in da.dims if d != "time"][0]
    da = da.rename({pdim: "plant"}).transpose("time", "plant")
    # the inflow NetCDF's plant coordinate IS the pipeline's plant_index,
    # stored as plain digit strings -- cast back to int, no re-identification.
    da = da.assign_coords(plant=[int(p) for p in da.plant.values])
    t = pd.to_datetime(da.time.values)
    dt_h = float(np.median(np.diff(t) / np.timedelta64(1, "h"))) if len(t) > 1 else 1.0
    return da, dt_h


# =============================================================================
# MAIN
# =============================================================================
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    plants = load_plants()

    # identify the plant set from the first available year
    ror = ids = None
    for y in YEARS:
        da, _ = open_inflow(y)
        if da is None:
            continue
        common = plants.index.intersection(pd.Index(da.plant.values))
        if len(common) == 0:
            sys.exit("[FATAL] no plant ids in common -- identity mismatch?")
        ror = plants.loc[common].copy()
        ids = list(ror.index)
        da.close()
        break
    if ror is None:
        sys.exit("[FATAL] no inflow files found")

    print(f"[info] RoR: {len(ror)} plants, {ror.p_nom.sum()/1e3:.2f} GW | "
          f"EFFICIENCY={EFFICIENCY}")

    n = len(ids)
    qmax = ror.qmax.values.astype("float64")
    pnom = ror.p_nom.values.astype("float64")
    turb = ror.turbine.values

    # --- accumulators, streamed year by year --------------------------------
    acc = dict(steps=0, sum_p=np.zeros(n), sum_q=np.zeros(n),
               sum_q2=np.zeros(n), n_gt=np.zeros(n), n_lt=np.zeros(n),
               qmaxobs=np.zeros(n), energy=0.0)
    cf_year, used_years, dt_ref = {}, [], None

    for y in YEARS:
        da, dt_h = open_inflow(y)
        if da is None:
            print(f"[warn] missing inflow for {y}, skipped")
            continue
        Q = np.nan_to_num(da.sel(plant=ids).load().values.astype("float32"),
                          nan=0.0, posinf=0.0, neginf=0.0)
        da.close()
        q = Q / qmax[None, :].astype("float32")

        qt = np.clip(q, 0.0, 1.0)
        p = np.clip(eta_rel(qt, turb) * qt, 0.0, 1.0)
        p[q < QMIN_FACTOR] = 0.0

        steps = Q.shape[0]
        acc["steps"] += steps
        acc["sum_p"] += p.sum(axis=0, dtype="float64")
        acc["sum_q"] += Q.sum(axis=0, dtype="float64")
        acc["sum_q2"] += (Q.astype("float64") ** 2).sum(axis=0)
        acc["n_gt"] += (Q > qmax[None, :]).sum(axis=0)
        acc["n_lt"] += (q < QMIN_FACTOR).sum(axis=0)
        acc["qmaxobs"] = np.maximum(acc["qmaxobs"], Q.max(axis=0))
        cf_year[y] = p.mean(axis=0)
        used_years.append(y)
        dt_ref = dt_h if dt_ref is None else dt_ref
        print(f"[info] {y}: {steps} steps, dt={dt_h:g} h, "
              f"mean CF={p.mean():.3f}")
        del Q, q, qt, p

    if not used_years:
        sys.exit("[FATAL] no usable inflow years")

    steps = acc["steps"]
    cf = acc["sum_p"] / steps
    qbar = acc["sum_q"] / steps
    qstd = np.sqrt(np.maximum(acc["sum_q2"] / steps - qbar ** 2, 0.0))
    qcv = qstd / np.maximum(qbar, 1e-12)
    # annual-average energy, so it is comparable across different year counts
    E = acc["sum_p"] * pnom * dt_ref / 1e6 / len(used_years)

    audit = pd.DataFrame({
        "name": ror.name.values, "country": ror.country.values,
        "p_nom_MW": pnom, "head_m": ror.head_m.values,
        "head_class": ror.head_class.values, "turbine": turb,
        "qmax_m3s": qmax, "q_cell_mean_m3s": qbar,
        "q_cell_max_m3s": acc["qmaxobs"],
        "qmax_over_qcell": qmax / np.maximum(qbar, 1e-9),
        "log10_qmax_over_qcell": np.log10(qmax / np.maximum(qbar, 1e-9)),
        "q_cell_cv": qcv,
        "pstar": acc["n_gt"] / steps,
        "frac_below_qmin": acc["n_lt"] / steps,
        "cf": cf, "E_TWh_per_year": E,
        "flat_series": qcv < FLAT_CV,
    }, index=ror.index)
    audit.index.name = "plant"
    audit["cf_class"] = classify_cf(cf)

    cfy = pd.DataFrame(cf_year, index=ror.index)
    cfy.index.name = "plant"
    cfy["cf_all"] = cf
    cfy["cf_std_across_years"] = cfy[used_years].std(axis=1)
    cfy.to_csv(OUTDIR / "plant_cf_by_year.csv")

    if FINAL_MATCHES and Path(FINAL_MATCHES).exists():
        fm = pd.read_parquet(FINAL_MATCHES).set_index("plant_index")
        fm = fm[~fm.index.duplicated()]
        keep = [c for c in ["match_type", "correction_status", "distance_km",
                            "river_mean", "combined_score", "model_id"]
                if c in fm.columns]
        audit = audit.join(fm[keep], how="left")

    audit.to_csv(OUTDIR / "plant_audit.csv")

    # --- aggregations --------------------------------------------------------
    present = [c for c in CLASS_ORDER if c in set(audit.cf_class)]

    def table(by):
        gw = (audit.pivot_table(index=by, columns="cf_class", values="p_nom_MW",
                                aggfunc="sum", fill_value=0.0) / 1e3)
        gw = gw.reindex(columns=present, fill_value=0.0)
        gw["total_GW"] = gw.sum(axis=1)
        pct = gw[present].div(gw.total_GW.replace(0, np.nan), axis=0) * 100
        pct.columns = [f"pct_{c}" for c in present]
        twh = audit.pivot_table(index=by, columns="cf_class",
                                values="E_TWh_per_year", aggfunc="sum",
                                fill_value=0.0).reindex(columns=present,
                                                        fill_value=0.0)
        twh.columns = [f"TWh_{c}" for c in present]
        return pd.concat([gw.round(3), pct.round(1), twh.round(3)], axis=1)

    overall = audit.groupby("cf_class").agg(
        n=("p_nom_MW", "size"), GW=("p_nom_MW", lambda s: s.sum() / 1e3),
        TWh_per_year=("E_TWh_per_year", "sum"), mean_cf=("cf", "mean"),
        median_qmax_over_qcell=("qmax_over_qcell", "median"),
        mean_pstar=("pstar", "mean")).reindex(present)
    overall["flat_GW"] = (audit[audit.flat_series].groupby("cf_class").p_nom_MW.sum() / 1e3
                          ).reindex(present).fillna(0).round(3)
    tot_gw = audit.p_nom_MW.sum() / 1e3
    overall["pct_GW"] = (100 * overall.GW / tot_gw).round(1)
    overall.to_csv(OUTDIR / "audit_by_class.csv")

    t_hc = table("head_class"); t_hc.to_csv(OUTDIR / "audit_by_headclass.csv")
    t_ct = table("country").sort_values("total_GW", ascending=False)
    t_ct.to_csv(OUTDIR / "audit_by_country.csv")
    if "match_type" in audit.columns and audit.match_type.notna().any():
        table("match_type").to_csv(OUTDIR / "audit_by_matchtype.csv")

    pct_cols = [c for c in t_hc.columns if c.startswith("pct_")]
    print("\n[classes]\n", overall.to_string())
    print("\n[by head class]\n", t_hc[["total_GW"] + pct_cols].to_string())
    print("\n[by country]\n", t_ct[["total_GW"] + pct_cols].to_string())

    show = ["name", "country", "p_nom_MW", "head_m",
            "qmax_m3s", "q_cell_mean_m3s", "log10_qmax_over_qcell", "pstar", "cf"]
    worst = audit[audit.cf_class.isin(["dead", "severely_under"])
                  ].nlargest(30, "p_nom_MW")[show]
    worst.to_csv(OUTDIR / "worst_offenders.csv")
    over = audit[audit.cf_class == "impossible"].nlargest(30, "p_nom_MW")[show]
    over.to_csv(OUTDIR / "overproducers.csv")
    print("\n[worst: dead / severely under]\n", worst.round(3).to_string())
    print("\n[overproducers: CF > 0.85]\n", over.round(3).to_string())

    ok = audit.loc[audit.cf_class == "plausible", "p_nom_MW"].sum() / 1e3
    bad_lo = audit.loc[audit.cf_class.isin(["dead", "severely_under"]),
                       "p_nom_MW"].sum() / 1e3
    bad_hi = audit.loc[audit.cf_class.isin(["suspect", "impossible"]),
                       "p_nom_MW"].sum() / 1e3

    with open(OUTDIR / "summary_audit.txt", "w") as f:
        w = f.write
        w("PLANT-LEVEL ASSIGNMENT AUDIT (capacity-factor classification)\n")
        w("=" * 64 + "\n\n")
        w("plant identity   : Eur_custom_ppls_GloHydroRes_filled.csv row "
          "index (the real pipeline identity)\n")
        w(f"years used       : {used_years}\n")
        w(f"timesteps        : {steps}  (dt = {dt_ref:g} h)\n")
        w(f"EFFICIENCY       : {EFFICIENCY}\n")
        w(f"RoR plants       : {len(audit)}  ({tot_gw:.2f} GW, "
          f"{audit.E_TWh_per_year.sum():.1f} TWh/yr)\n")
        w(f"flat (0.1 fill)  : "
          f"{audit.loc[audit.flat_series,'p_nom_MW'].sum()/1e3:.2f} GW\n\n")
        w(f"PLAUSIBLE (0.15 <= CF <= 0.65) : {ok:.2f} GW "
          f"({100*ok/tot_gw:.1f} %)\n")
        w(f"TOO LOW   (CF < 0.15)          : {bad_lo:.2f} GW "
          f"({100*bad_lo/tot_gw:.1f} %)\n")
        w(f"TOO HIGH  (CF > 0.65)          : {bad_hi:.2f} GW "
          f"({100*bad_hi/tot_gw:.1f} %)\n\n")
        w(overall.to_string() + "\n\n")
        w("BY HEAD CLASS\n" + t_hc.to_string() + "\n\n")
        w("BY COUNTRY\n" + t_ct.to_string() + "\n\n")
        w("INTERANNUAL STABILITY\n")
        w(f"median CF std across years: {cfy.cf_std_across_years.median():.4f}\n")
        w(f"plants whose class would change between best and worst year: "
          f"{int((classify_cf(cfy[used_years].min(axis=1).values) != classify_cf(cfy[used_years].max(axis=1).values)).sum())}"
          f" of {len(cfy)}\n")
    print("\n" + (OUTDIR / "summary_audit.txt").read_text())

    _figs(audit, present)
    print(f"[done] outputs in {OUTDIR.resolve()}")


# =============================================================================
# FIGURES
# =============================================================================
def _figs(audit, present):
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.hist(audit.cf, bins=np.linspace(0, 1, 51), weights=audit.p_nom_MW / 1e3,
            histtype="stepfilled", alpha=0.65, color="#377eb8")
    ax.axvspan(CF_UNDER, CF_PLAUSIBLE_HI, color="green", alpha=0.10,
               label="plausible RoR range")
    ax.axvline(CF_DEAD, color="k", ls=":", lw=1)
    ax.axvline(CF_SUSPECT_HI, color="k", ls=":", lw=1)
    ax.set_xlabel("capacity factor (conversion V1)")
    ax.set_ylabel("capacity [GW]")
    ax.set_title("RoR capacity factor distribution")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUTDIR / "fig_cf_hist.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(HEAD_LABELS)))
    for hc, col in zip(HEAD_LABELS, colors):
        d = audit[audit.head_class == hc]
        if len(d):
            ax.scatter(d.log10_qmax_over_qcell, d.cf, s=3 + d.p_nom_MW / 12,
                       color=col, alpha=0.55, linewidths=0,
                       label=f"head {hc} ({d.p_nom_MW.sum()/1e3:.1f} GW)")
    ax.axhspan(CF_UNDER, CF_PLAUSIBLE_HI, color="green", alpha=0.08)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\log_{10}(Q_{max}/\bar{Q}_{cell})$")
    ax.set_ylabel("capacity factor")
    ax.set_title("Assignment scale error vs realised output (size = MW)")
    ax.legend(fontsize=7, markerscale=2)
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUTDIR / "fig_cf_vs_qratio.png", dpi=180)
    plt.close(fig)

    piv = (audit.pivot_table(index="head_class", columns="cf_class",
                             values="p_nom_MW", aggfunc="sum", fill_value=0)
           / 1e3).reindex(columns=present, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    piv.plot.barh(stacked=True, ax=ax, colormap="RdYlGn_r")
    ax.set_xlabel("capacity [GW]")
    ax.set_title("Capacity-factor class by head class")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout(); fig.savefig(OUTDIR / "fig_buckets.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
