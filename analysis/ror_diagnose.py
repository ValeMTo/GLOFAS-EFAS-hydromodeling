#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ror_diagnose.py

PURPOSE
-------
Single-purpose script answering Reviewer #4's request: quantify the error
introduced by representing run-of-river (RoR) output as a function of
discharge alone, ignoring the variation of net head with river flow.

Several conversions are applied to the SAME inflow series, so every
difference is attributable to head and nothing else -- in particular, NOT to
the turbine efficiency curve, which all formulas below omit on purpose so
that this report isolates the head effect alone. The turbine-curve question
(currently the head-based kaplan/francis/pelton classification and tabulated
efficiency-vs-flow curves used elsewhere in the analysis) is deliberately out
of scope here and will be addressed separately.

    q = Q / Qmax   relative discharge

V0  published formulation (affine ramp, constant head)
        p = 0                        for q < qmin
        p = (q - qmin)/(1 - qmin)    for qmin <= q <= 1
        p = 1                        for q > 1
    Slope 1.25, intercept -0.25. Not proportionality: at q = 0.5 it gives
    0.375 instead of 0.5.

V_head  same affine ramp, modulated by real head variation (no free params)
        p = min(1, p_V0(q) * h(q))
        h(q) = 1 - (y(q) - y(q=1)) / head_m
    y(Q) is the river's own water depth (tailwater), the normal depth from
    Manning's equation on the LISFLOOD compound cross-section (trapezoidal
    channel up to bankfull, floodplain above it), solved per plant from
    CEMS_SurfaceFields_2022 / GloFAS v4.0 static maps (chanbw, chans,
    changrad, chanman, chanbnkf, chanflpn), sampled at the SAME cell already
    matched to that plant for its discharge series (final_hydro_plant_
    glofas_points.parquet) -- not the plant's own reported lat/lon, which
    would pair a discharge series with a different river cell.
    y(q) rises with q: at a weir the headwater is held near crest level, but
    the tailwater rises as the river grows. Net head is the design head minus
    that rise, so h(q) > 1 at part load (head gain) and h(q) < 1 in flood
    (head loss) -- exactly the mechanism the reviewer's comment describes,
    with the tailwater excursion measured from real geometry instead of an
    assumed dz/b/lambda. Gain and loss have opposite signs and partly cancel
    on fleet aggregation.

V_head_adverse  same real geometry, part-load gain removed
        h_adverse(q) = min(1, h(q))
    Deliberately adverse, like the old literature-based V2b, but needs no
    extra assumption (no percentiles, no lambda_shut): capping the already
    real, measured h(q) at 1 keeps only the flood loss it already contains.

V_head_gain  mirror of V_head_adverse: flood loss removed
        h_gain(q) = h(q) for q < 1, frozen at 1 for q >= 1
    Keeps only the part-load gain.

V_head_cellcell  two-sided: drops the fixed-headwater assumption
    V_head/V_head_adverse/V_head_gain all assume the headwater (monte) level
    is fixed while only the tailwater (valle) moves with discharge. This
    variant lets BOTH sides move, each using its own real channel geometry:
    the matched cell (valle) and its immediate upstream neighbour in the
    drainage chain (monte, ~5 km apart, resolved from
    glofas_drain_lv3_europe_full.parquet's upstream_model_ids, largest
    drainage_area at confluences). Net head = design head minus the NET rise
    (downstream rise minus upstream rise) -- if both sides rise together in
    flood, the two terms partly cancel, so this is expected to show a
    SMALLER effect than the one-sided variants.

V_head_local100  same two-sided idea, at the plant's real local scale
    The monte/valle cells above are ~5 km apart -- far longer than a typical
    RoR intake-to-tailrace reach. This variant projects the plant's TRUE
    coordinates onto the monte-valle segment (t=0 at monte, t=1 at valle),
    uses that as "monte", and steps DOWNSTREAM_OFFSET_M (100 m by default,
    the assumed derivation length) further towards valle for "valle". Channel
    geometry at both points is a linear blend of the two endpoint cells (no
    new raster read). Because 100 m is a small fraction of the ~5 km cell
    spacing, geometry -- and therefore the computed depths -- barely differ
    between the two points, so this is expected to show NEAR-TOTAL
    cancellation: physically, an intake and tailrace only 100 m apart sit on
    the same short reach and their water levels rise together in flood.
    DOWNSTREAM_OFFSET_M is a config constant, meant to be varied.

V_head_cellcell_downstream  mirror of V_head_cellcell
    Same ~5 km cell-to-cell idea, but the matched cell now plays "monte" and
    its resolved immediate downstream neighbour (downstream_model_id, single-
    valued -- no confluence ambiguity) plays "valle". Coverage is expected to
    be much higher than V_head_cellcell/V_head_local100 (which need an
    upstream_model_ids match): every drainage node has a downstream neighbour
    except a true basin outlet.

V_head_cellcell_gain / V_head_local100_gain / V_head_cellcell_downstream_gain
    Gain-only mirrors of the three two-sided variants above (same construction
    as V_head_gain vs V_head): h(q) frozen at 1 for q >= 1, so only the
    part-load variation (between qmin and design flow) survives -- no flood
    derating from either side.

ROBUSTNESS AGAINST THE ASSIGNMENT AND MATCHING PROBLEMS
--------------------------------------------------------
Every result is reported against several denominators: the whole RoR fleet,
the well-assigned subset (Qmax/Qbar in [0.5, 4], independent of the geometry
match), the geometry-matched subset, and the low-/high-head split. Plants
without a valid channel geometry match contribute dE = 0 (p_head falls back
to p_V0 for them), which dilutes the whole-fleet percentage -- the
geometry-matched-only percentage is the one to quote for the reviewer. The
two-sided variants additionally require a resolved upstream cell
(geom_up_ok/valid_cellcell/valid_local), reported separately.

OUTPUTS (OUTDIR)
----------------
    sensitivity_summary.csv   V0 vs every V_head* variant: dE, % of
                              denominators, by subset
    baseline_energies.csv     annual energy of every scenario, per country
    plant_energies.csv        per-plant energy under every scenario, plus the
                              sampled channel geometry and projection details
    fig_curves.png            V0 vs V_head/V_head_adverse/V_head_gain for two
                              real matched plants (two-sided variants not
                              plotted here)
    fig_tornado.png           dE% under each denominator/subset (one-sided
                              variants only)
    summary_sensitivity.txt   the digest to read
"""

from pathlib import Path
import re
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
ENTSOE_DIR = ROOT / "hydro_workflow/hydro_global/ENTSOE/Production"

PLANTS_FILE = PYPSA_DIR / "GloHydroRes_vs1.csv"
INFLOW_PATTERN = "Europe_inflow_{year}_GloFAS_SABER.nc"
YEARS = list(range(2015, 2019))
OUTDIR = Path("./ror_headsens_out")

# must match the pipeline that produced the inflow files
EFFICIENCY = 0.85
QMIN_FACTOR = 0.20
IMPUTE_STRATEGY = "country"        # "country" (add_qmax_turb) | "global"

H_FLOOR = 0.30
Q_HEAD_CAP = 20.0

# --- data-driven head variation from real GloFAS v4.0 channel geometry ------
# CEMS_SurfaceFields_2022 / LISFLOOD static maps for GloFAS v1.1.1, sampled at
# the SAME cell already matched to each plant for its discharge series (no
# free dz/b/lambda: bankfull depth, width, side slope, gradient and Manning's
# n are read directly from data).
CHANNEL_GEOMETRY_DIR = ROOT / "hydro_workflow/hydro_global/channel_geometry"
CHANNEL_GEOMETRY_VARS = ["chanbw", "chanman", "changrad", "chans", "chanbnkf", "chanflpn"]
FINAL_MATCHES_PATH = (
    ROOT / "hydro_workflow/saber_global/glofas/final_hydro_plant_glofas_points.parquet"
)
MANNING_Q_GRID = np.unique(np.concatenate([np.geomspace(1e-3, Q_HEAD_CAP, 79), [1.0]]))
MANNING_BISECTION_ITERS = 40
MIN_CHANGRAD = 1e-4        # floor: flat/lake cells report changrad = 0
MIN_CHANBNKF = 0.10        # floor: degenerate/missing bankfull depth [m]

# --- two-sided head variation: monte (upstream) AND valle (downstream) can
# both move with discharge, instead of assuming the headwater is fixed -----
DRAIN_TABLE_PATH = (
    ROOT / "hydro_workflow/saber_global/glofas/glofas_drain_lv3_europe_full.parquet"
)
DOWNSTREAM_OFFSET_M = 100.0   # assumed intake-to-tailrace length, local variant

# --- well-assigned subset, defined independently ----------------------------
WELL_ASSIGNED_BAND = (0.5, 4.0)    # Qmax / Qbar

HEAD_SPLIT = 15.0                  # low-head vs the rest [m]

TECH_MAP = {"ROR": "Run-Of-River", "Canal": "Run-Of-River",
            "STO": "Reservoir", "PS": "Pumped Storage"}
ROR_TECH = "Run-Of-River"

MIN_WEEK_COVERAGE = 0.90
MIN_YEAR_COVERAGE = 0.50

NAME2CODE = {
    "Austria": "AT", "Belgium": "BE", "Bulgaria": "BG", "Switzerland": "CH",
    "Czechia": "CZ", "Czech Republic": "CZ", "Germany": "DE", "Denmark": "DK",
    "Estonia": "EE", "Spain": "ES", "Finland": "FI", "France": "FR",
    "United Kingdom": "GB", "Greece": "GR", "Croatia": "HR", "Hungary": "HU",
    "Ireland": "IE", "Italy": "IT", "Lithuania": "LT", "Luxembourg": "LU",
    "Latvia": "LV", "Montenegro": "ME", "North Macedonia": "MK",
    "Netherlands": "NL", "Norway": "NO", "Poland": "PL", "Portugal": "PT",
    "Romania": "RO", "Serbia": "RS", "Sweden": "SE", "Slovenia": "SI",
    "Slovakia": "SK", "Albania": "AL", "Bosnia and Herzegovina": "BA",
}


# =============================================================================
# HELPERS
# =============================================================================
def norm_id(x):
    s = str(x)
    d = re.sub(r"\D", "", s.split(".")[0])
    return str(int(d)) if d else s


def add_qmax_turb(df, efficiency=EFFICIENCY):
    """Head imputation as in the pipeline's add_qmax_turb(): country-and-
    technology median, global per-technology fallback. Note a quirk of the
    original: if a country has no reported head at all, the country median is
    NaN and the head stays NaN, so a final global fallback is applied here and
    the affected plants are counted."""
    df = df.copy()
    df["technology"] = df["technology"].fillna(ROR_TECH)
    ror_m = df["technology"] == ROR_TECH
    res_m = df["technology"].isin(["Reservoir", "Pumped Storage"])

    if IMPUTE_STRATEGY == "country":
        med_ror_c = df[ror_m].groupby("country")["head_m"].median()
        med_res_c = df[res_m].groupby("country")["head_m"].median()
    else:
        med_ror_c = med_res_c = pd.Series(dtype=float)
    med_ror_g = df.loc[ror_m, "head_m"].median(skipna=True)
    med_res_g = df.loc[res_m, "head_m"].median(skipna=True)

    nan_m = df["head_m"].isna()
    for c in df["country"].unique():
        m = nan_m & (df["country"] == c) & ror_m
        df.loc[m, "head_m"] = med_ror_c.get(c, med_ror_g)
        m = nan_m & (df["country"] == c) & res_m
        df.loc[m, "head_m"] = med_res_c.get(c, med_res_g)

    still = df["head_m"].isna()
    if still.any():
        print(f"[warn] {int(still.sum())} plants without head after country "
              "imputation; using the global per-technology median")
        df.loc[still & ror_m, "head_m"] = med_ror_g
        df.loc[still & res_m, "head_m"] = med_res_g
        df["head_m"] = df["head_m"].fillna(med_ror_g)
    df.loc[df["head_m"] <= 0, "head_m"] = 1.0

    df["qmax"] = df["p_nom"] / (9.81 * df["head_m"] * 1e-3 * efficiency)
    return df


def load_plants(domain_ids):
    raw = pd.read_csv(PLANTS_FILE)
    raw["__key"] = raw["ID"].map(norm_id)
    raw = raw[~raw["__key"].duplicated()].set_index("__key")

    df = pd.DataFrame(index=raw.index)
    df["name"] = raw.get("name", "")
    df["p_nom"] = pd.to_numeric(raw["capacity_mw"], errors="coerce")
    df["head_m"] = pd.to_numeric(raw["head_m"], errors="coerce")
    df["plant_lon"] = pd.to_numeric(raw.get("plant_lon"), errors="coerce")
    df["plant_lat"] = pd.to_numeric(raw.get("plant_lat"), errors="coerce")
    df["head_reported"] = df["head_m"].notna() & (df["head_m"] > 0)
    ctry = raw["country"].astype(str).str.strip()
    df["country"] = np.where(ctry.str.len() == 2, ctry.str.upper(),
                             ctry.map(NAME2CODE))
    df["technology"] = raw["plant_type"].map(TECH_MAP)
    df = df.dropna(subset=["p_nom"])

    # restrict to the modelled domain BEFORE imputing, so the medians are
    # computed on the same population the pipeline used
    keep = df.index.intersection(pd.Index(sorted(domain_ids)))
    print(f"[info] domain: {len(df)} global plants -> {len(keep)}")
    df = df.loc[keep]

    df = add_qmax_turb(df)
    df["head_class"] = np.where(df.head_m < HEAD_SPLIT,
                                f"H<{HEAD_SPLIT:g}", f"H>={HEAD_SPLIT:g}")
    return df.dropna(subset=["qmax"])


def open_inflow(year):
    f = PYPSA_DIR / INFLOW_PATTERN.format(year=year)
    if not f.exists():
        return None, None
    ds = xr.open_dataset(f)
    da = ds[list(ds.data_vars)[0]] if len(ds.data_vars) == 1 else \
        ds[[v for v in ds.data_vars if "inflow" in v.lower()][0]]
    pdim = [d for d in da.dims if d != "time"][0]
    da = da.rename({pdim: "plant"}).transpose("time", "plant")
    da = da.assign_coords(plant=[norm_id(p) for p in da.plant.values])
    t = pd.to_datetime(da.time.values)
    dt_h = float(np.median(np.diff(t) / np.timedelta64(1, "h"))) if len(t) > 1 else 1.0
    return da, dt_h


# =============================================================================
# CHANNEL GEOMETRY (data-driven head variation)
# =============================================================================
def load_matched_cells(ids):
    """Matched GloFAS cell (lon, lat, match_type) for each plant id, from the
    SAME plant-to-drainage-network matching already used upstream to extract
    each plant's discharge series (extract_glofas_series_for_hydro_plants.py).
    Sampling channel geometry anywhere else (e.g. the plant's own reported
    lat/lon) would pair a discharge series with a different river cell."""
    matches = pd.read_parquet(FINAL_MATCHES_PATH)
    matches["__key"] = matches["plant_index"].map(norm_id)
    matches = matches[~matches["__key"].duplicated()].set_index("__key")
    matches = matches.reindex(ids)
    return (
        matches["matched_lon"].to_numpy(dtype="float64"),
        matches["matched_lat"].to_numpy(dtype="float64"),
        matches["match_type"].to_numpy(),
    )


def sample_channel_geometry(lons, lats):
    """Nearest-cell sample of the GloFAS v4.0 channel geometry static maps
    (CEMS_SurfaceFields_2022) at the given (lon, lat) points. NaN coordinates
    (plant absent from the matching table) propagate to NaN geometry."""
    valid = np.isfinite(lons) & np.isfinite(lats)
    lon_da = xr.DataArray(np.where(valid, lons, 0.0), dims="plant")
    lat_da = xr.DataArray(np.where(valid, lats, 0.0), dims="plant")

    out = {}
    for name in CHANNEL_GEOMETRY_VARS:
        path = CHANNEL_GEOMETRY_DIR / f"{name}_glofas_v4_0.nc"
        with xr.open_dataset(path) as ds:
            values = ds["Band1"].sel(lon=lon_da, lat=lat_da, method="nearest").values
        out[name] = np.where(valid, values.astype("float64"), np.nan)
    return out


def _manning_discharge(y, bw, side_slope, grad, manning_n, bnkf, flpn):
    """Manning discharge for the LISFLOOD compound cross-section: trapezoidal
    channel (bottom width bw, side slope) up to bankfull depth, then a
    floodplain of width flpn added on top once y exceeds bnkf (floodplain
    side walls approximated as vertical -- the standard compound-channel
    simplification, adequate for a sensitivity study)."""
    y_channel = np.minimum(y, bnkf)
    area_channel = (bw + side_slope * y_channel) * y_channel
    wp_channel = bw + 2.0 * y_channel * np.sqrt(1.0 + side_slope ** 2)

    overbank = np.maximum(y - bnkf, 0.0)
    area = area_channel + overbank * flpn
    wp = wp_channel + np.where(
        overbank > 0.0, 2.0 * overbank + np.maximum(flpn - bw, 0.0), 0.0
    )

    area = np.maximum(area, 1e-6)
    wp = np.maximum(wp, 1e-6)
    return (1.0 / manning_n) * area ** (5.0 / 3.0) / wp ** (2.0 / 3.0) * np.sqrt(grad)


def solve_manning_depth(q_abs, bw, side_slope, grad, manning_n, bnkf, flpn,
                        n_iter=MANNING_BISECTION_ITERS):
    """Normal depth from Manning's equation via bisection: the compound
    section has no closed-form inverse, but discharge is monotonic in y."""
    lo = np.zeros_like(q_abs)
    hi = np.broadcast_to(np.maximum(bnkf * 20.0, 50.0), q_abs.shape).copy()

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        too_low = _manning_discharge(mid, bw, side_slope, grad, manning_n,
                                     bnkf, flpn) < q_abs
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)

    return 0.5 * (lo + hi)


def build_manning_head_table(qmax, head_m, bw, side_slope, grad, manning_n,
                             bnkf, flpn):
    """h(q) on MANNING_Q_GRID, one curve per plant (one-sided: only the
    downstream/tailwater level is allowed to move with discharge).

    y(q) is the river's own water depth (tailwater), which INCREASES with q
    -- that is not the turbine's head. Net head is the design head minus the
    rise of the tailwater relative to design flow: the excursion
    y(q) - y(q=1) is measured, in metres, from real channel geometry, and
    scaled by that plant's own reported design head."""
    delta_y = build_manning_delta_y_table(qmax, bw, side_slope, grad, manning_n, bnkf, flpn)
    h = 1.0 - delta_y / np.maximum(head_m[None, :], 0.1)
    return np.maximum(h, H_FLOOR)


def h_manning_series(q, head_table):
    """Interpolate each plant's own tabulated h(q) curve onto its actual q
    series (curves differ per plant, so this cannot be vectorised with a
    single shared table)."""
    out = np.empty_like(q, dtype="float64")
    for k in range(q.shape[1]):
        out[:, k] = np.interp(q[:, k], MANNING_Q_GRID, head_table[:, k],
                              left=head_table[0, k], right=head_table[-1, k])
    return out


def prepare_geometry(geom_raw):
    """Apply the same floors/defaults used for the matched (downstream) cell
    to any raw sample_channel_geometry() output, so upstream cells and
    interpolated points are treated identically."""
    bw = np.nan_to_num(geom_raw["chanbw"], nan=0.0)
    side = np.where(np.isfinite(geom_raw["chans"]), geom_raw["chans"], 1.0)
    grad = np.maximum(np.nan_to_num(geom_raw["changrad"], nan=MIN_CHANGRAD), MIN_CHANGRAD)
    manning_n = np.where(np.isfinite(geom_raw["chanman"]) & (geom_raw["chanman"] > 0),
                         geom_raw["chanman"], 0.035)
    bnkf = np.maximum(np.nan_to_num(geom_raw["chanbnkf"], nan=MIN_CHANBNKF), MIN_CHANBNKF)
    flpn = np.maximum(np.nan_to_num(geom_raw["chanflpn"], nan=3.0 * bw), bw)
    return {"bw": bw, "side_slope": side, "grad": grad, "manning_n": manning_n,
            "bnkf": bnkf, "flpn": flpn}


def geometry_is_valid(geom_raw):
    return (
        np.isfinite(geom_raw["chanbnkf"]) & (geom_raw["chanbnkf"] > 0)
        & np.isfinite(geom_raw["chanman"]) & (geom_raw["chanman"] > 0)
        & np.isfinite(geom_raw["chanbw"]) & (geom_raw["chanbw"] >= 0)
    )


def build_manning_delta_y_table(qmax, bw, side_slope, grad, manning_n, bnkf, flpn):
    """y(q) - y(q=1) on MANNING_Q_GRID, one curve per plant -- the raw
    tailwater excursion, before it is scaled by a design head."""
    q_abs = MANNING_Q_GRID[:, None] * qmax[None, :]
    y = solve_manning_depth(
        q_abs, bw[None, :], side_slope[None, :], grad[None, :],
        manning_n[None, :], bnkf[None, :], flpn[None, :],
    )
    design_row = int(np.argmin(np.abs(MANNING_Q_GRID - 1.0)))
    return y - y[design_row][None, :]


def freeze_above_design(head_table):
    """Mirror of the *_gain construction: keep head variation only at part
    load (q < 1), freeze h=1 for q >= 1 (no flood derating). Applies to any
    head table, one-sided or two-sided."""
    return np.where(MANNING_Q_GRID[:, None] >= 1.0, 1.0, head_table)


def build_manning_head_table_twoside(qmax, head_m, geom_down, geom_up):
    """Two-sided head table: drops the assumption that the headwater/intake
    level is fixed while only the tailwater varies. Both the downstream
    (valle) and upstream (monte) water levels are allowed to move with
    discharge, each from its own local channel geometry; net head is the
    design head minus the NET rise (downstream rise minus upstream rise).
    If both sides rise together, the two terms partly cancel."""
    delta_down = build_manning_delta_y_table(qmax, **geom_down)
    delta_up = build_manning_delta_y_table(qmax, **geom_up)
    h = 1.0 - (delta_down - delta_up) / np.maximum(head_m[None, :], 0.1)
    return np.maximum(h, H_FLOOR)


def load_upstream_cells(ids):
    """For each plant id, resolve the immediate upstream neighbour of its
    matched drainage node (the chain built by glofas_to_drainage_network.py),
    picking the branch with the largest drainage_area at confluences (main
    stem). Returns (lon_up, lat_up, has_upstream) aligned to `ids`; NaN/False
    where the plant has no model_id (fallback match) or sits at a headwater
    (empty upstream_model_ids)."""
    matches = pd.read_parquet(FINAL_MATCHES_PATH, columns=["plant_index", "model_id"])
    matches["__key"] = matches["plant_index"].map(norm_id)
    matches = matches[~matches["__key"].duplicated()].set_index("__key")
    matches = matches.reindex(ids)

    drain = pd.read_parquet(
        DRAIN_TABLE_PATH, columns=["model_id", "upstream_model_ids", "drainage_area", "x", "y"]
    )
    drain = drain[~drain["model_id"].duplicated()].set_index("model_id")

    n = len(ids)
    lon_up = np.full(n, np.nan)
    lat_up = np.full(n, np.nan)
    has_upstream = np.zeros(n, dtype=bool)

    for i, model_id in enumerate(matches["model_id"].to_numpy()):
        if not isinstance(model_id, str) or model_id not in drain.index:
            continue
        candidates = [c for c in str(drain.at[model_id, "upstream_model_ids"]).split(";") if c]
        candidates = [c for c in candidates if c in drain.index]
        if not candidates:
            continue
        best = max(candidates, key=lambda c: drain.at[c, "drainage_area"])
        lon_up[i] = drain.at[best, "x"]
        lat_up[i] = drain.at[best, "y"]
        has_upstream[i] = True

    return lon_up, lat_up, has_upstream


def load_downstream_cells(ids):
    """For each plant id, resolve the immediate downstream neighbour of its
    matched drainage node -- the mirror of load_upstream_cells(): here the
    matched cell itself plays the role of "monte", and we look one step
    further down the chain for "valle". `downstream_model_id` is single-
    valued (no confluence ambiguity), so coverage is expected to be much
    higher than for upstream_model_ids -- every node has a downstream
    neighbour except a true basin outlet (river mouth). Returns
    (lon_down, lat_down, has_downstream) aligned to `ids`; NaN/False where
    the plant has no model_id (fallback match) or sits at an outlet."""
    matches = pd.read_parquet(FINAL_MATCHES_PATH, columns=["plant_index", "model_id"])
    matches["__key"] = matches["plant_index"].map(norm_id)
    matches = matches[~matches["__key"].duplicated()].set_index("__key")
    matches = matches.reindex(ids)

    drain = pd.read_parquet(
        DRAIN_TABLE_PATH, columns=["model_id", "downstream_model_id", "x", "y"]
    )
    drain = drain[~drain["model_id"].duplicated()].set_index("model_id")

    n = len(ids)
    lon_down = np.full(n, np.nan)
    lat_down = np.full(n, np.nan)
    has_downstream = np.zeros(n, dtype=bool)

    for i, model_id in enumerate(matches["model_id"].to_numpy()):
        if not isinstance(model_id, str) or model_id not in drain.index:
            continue
        target = drain.at[model_id, "downstream_model_id"]
        if not isinstance(target, str) or target not in drain.index:
            continue
        lon_down[i] = drain.at[target, "x"]
        lat_down[i] = drain.at[target, "y"]
        has_downstream[i] = True

    return lon_down, lat_down, has_downstream


def project_onto_segment(plant_lon, plant_lat, up_lon, up_lat, down_lon, down_lat):
    """Project the plant's true coordinates onto the up->down segment (local
    equirectangular approximation, adequate at this scale: a few km at most).
    Returns t in [0,1] (0 = upstream cell, 1 = downstream/matched cell) and
    the segment length in metres."""
    kx = 111320.0 * np.cos(np.deg2rad(up_lat))   # metres per degree longitude
    ky = 111320.0                                # metres per degree latitude

    vx = (down_lon - up_lon) * kx
    vy = (down_lat - up_lat) * ky
    px = (plant_lon - up_lon) * kx
    py = (plant_lat - up_lat) * ky

    seg_len2 = vx ** 2 + vy ** 2
    t = np.where(seg_len2 > 0, (px * vx + py * vy) / np.maximum(seg_len2, 1e-12), 0.0)
    t = np.clip(t, 0.0, 1.0)
    return t, np.sqrt(seg_len2)


def blend_geometry(geom_a, geom_b, t):
    """Linear interpolation of prepared geometry dicts between two points
    (t=0 at geom_a, t=1 at geom_b) -- the two endpoint RASTER samples are
    already known, so this is a plain blend, no new raster read."""
    return {k: geom_a[k] * (1.0 - t) + geom_b[k] * t for k in geom_a}


# =============================================================================
# CONVERSIONS
# =============================================================================
def p_V0(q):
    p = np.clip((q - QMIN_FACTOR) / (1.0 - QMIN_FACTOR), 0.0, 1.0)
    p[q < QMIN_FACTOR] = 0.0
    return p


def p_head(q, h):
    """V0's affine ramp (no turbine efficiency curve) modulated by the real
    head ratio h(q) -- isolates the head effect alone against the published
    formula, with no efficiency curve mixed in."""
    return np.clip(p_V0(q.copy()) * h, 0.0, 1.0)


# =============================================================================
# ENTSO-E DENOMINATORS
# =============================================================================
def observed_denominators(years):
    frames = []
    for y in years:
        f = ENTSOE_DIR / f"Europe_Hydro_{y}.csv"
        if f.exists():
            frames.append(pd.read_csv(f, index_col=0, parse_dates=True))
    if not frames:
        print("[warn] no ENTSO-E files; observed denominators unavailable")
        return np.nan, np.nan
    d = pd.concat(frames).sort_index()
    d.index = pd.to_datetime(d.index, utc=True).tz_convert(None)

    def take(sfx):
        cols = [c for c in d.columns if c.endswith(sfx)]
        out = d[cols].copy()
        out.columns = [c[: -len(sfx)] for c in cols]
        return out

    ror, tot = take("_RoR"), take("_Total")
    cov = ror.notna().resample("W").mean()
    wk = ror.resample("W").mean().where(cov >= MIN_WEEK_COVERAGE)
    usable = [c for c in wk.columns
              if wk[c].notna().mean() >= MIN_YEAR_COVERAGE]
    print(f"[info] ENTSO-E usable countries: {len(usable)}")

    def annual(h):
        e = 0.0
        for c in [x for x in h.columns if x in usable]:
            s = h[c].dropna()
            if len(s):
                e += s.mean() * 8760.0 / 1e6      # TWh per year
        return float(e)

    return annual(ror), annual(tot)


# =============================================================================
# MAIN
# =============================================================================
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # plant set from the first available inflow year
    domain = None
    for y in YEARS:
        da, _ = open_inflow(y)
        if da is not None:
            domain = set(da.plant.values)
            da.close()
            break
    if domain is None:
        sys.exit("[FATAL] no inflow files found")

    plants = load_plants(domain)
    ror = plants[plants.technology == ROR_TECH].copy()
    ids = list(ror.index)
    print(f"[info] RoR: {len(ror)} plants, {ror.p_nom.sum()/1e3:.2f} GW | "
          f"eta={EFFICIENCY}, impute={IMPUTE_STRATEGY}")

    # --- load all years, then work in q space only -------------------------
    blocks, dt_ref, used = [], None, []
    for y in YEARS:
        da, dt_h = open_inflow(y)
        if da is None:
            print(f"[warn] missing inflow {y}")
            continue
        blocks.append(np.nan_to_num(da.sel(plant=ids).load().values
                                    .astype("float32"), nan=0.0,
                                    posinf=0.0, neginf=0.0))
        da.close()
        dt_ref = dt_h if dt_ref is None else dt_ref
        used.append(y)
    if not blocks:
        sys.exit("[FATAL] no usable inflow years")
    Q = np.concatenate(blocks, axis=0)
    del blocks
    T, N = Q.shape
    print(f"[info] inflow {T} steps x {N} plants, dt={dt_ref:g} h, years {used}")

    qmax = ror.qmax.values.astype("float64")
    pnom = ror.p_nom.values.astype("float64")
    head = ror.head_m.values

    qbar = Q.mean(axis=0, dtype="float64")
    well = ((qmax / np.maximum(qbar, 1e-9) >= WELL_ASSIGNED_BAND[0])
            & (qmax / np.maximum(qbar, 1e-9) <= WELL_ASSIGNED_BAND[1]))
    print(f"[info] well-assigned subset: {int(well.sum())} plants, "
          f"{pnom[well].sum()/1e3:.2f} GW of {pnom.sum()/1e3:.2f} GW")

    q = (Q / qmax[None, :].astype("float32"))
    del Q

    def energy(p):
        """annual-average energy per plant [TWh/yr]"""
        return p.sum(axis=0, dtype="float64") * pnom * dt_ref / 1e6 / len(used)

    E0 = energy(p_V0(q))
    print(f"[info] annual energy: V0={E0.sum():.1f} TWh")

    E_obs_ror, E_obs_tot = observed_denominators(used)
    print(f"[info] observed: RoR={E_obs_ror:.1f} TWh/yr, "
          f"total hydro={E_obs_tot:.1f} TWh/yr")

    print("\n[V_head] data-driven head variation (real GloFAS v4.0 channel geometry)")
    matched_lon, matched_lat, match_type = load_matched_cells(ror.index.values)
    geom = sample_channel_geometry(matched_lon, matched_lat)
    geom_bw = np.nan_to_num(geom["chanbw"], nan=0.0)
    geom_side = np.where(np.isfinite(geom["chans"]), geom["chans"], 1.0)
    geom_grad = np.maximum(np.nan_to_num(geom["changrad"], nan=MIN_CHANGRAD), MIN_CHANGRAD)
    geom_manning_n = np.where(np.isfinite(geom["chanman"]) & (geom["chanman"] > 0),
                              geom["chanman"], 0.035)
    geom_bnkf = np.maximum(np.nan_to_num(geom["chanbnkf"], nan=MIN_CHANBNKF), MIN_CHANBNKF)
    geom_flpn = np.maximum(np.nan_to_num(geom["chanflpn"], nan=3.0 * geom_bw), geom_bw)

    geom_ok = (
        np.isfinite(matched_lon) & np.isfinite(matched_lat)
        & np.isfinite(geom["chanbnkf"]) & (geom["chanbnkf"] > 0)
        & np.isfinite(geom["chanman"]) & (geom["chanman"] > 0)
        & np.isfinite(geom["chanbw"]) & (geom["chanbw"] >= 0)
    )
    print(f"  matched to channel geometry: {int(geom_ok.sum())}/{len(ror)} RoR plants "
          f"({pnom[geom_ok].sum()/1e3:.2f} of {pnom.sum()/1e3:.2f} GW); "
          f"fallback-matched cells: {int((match_type == 'fallback_nn_glofas').sum())}")

    manning_head_table = build_manning_head_table(
        qmax, head, geom_bw, geom_side, geom_grad, geom_manning_n, geom_bnkf, geom_flpn,
    )
    h_mat = h_manning_series(q, manning_head_table)
    E_head = energy(p_head(q, h_mat))
    E_head = np.where(geom_ok, E_head, E0)   # unmatched plants: no contribution to dE

    # adverse-only variant: same real geometry, part-load gain removed (h<=1
    # always) -- unlike the old V2b this needs no extra assumption (no
    # percentiles, no lambda_shut): it is simply min(1, h_manning(q)).
    manning_head_table_adverse = np.minimum(manning_head_table, 1.0)
    h_mat_adverse = h_manning_series(q, manning_head_table_adverse)
    E_head_adverse = energy(p_head(q, h_mat_adverse))
    E_head_adverse = np.where(geom_ok, E_head_adverse, E0)

    # gain-only variant: the mirror of V_head_adverse -- head varies only at
    # part load (q<1), frozen at h=1 (no loss) once the plant reaches design
    # flow. Same real geometry, no extra assumption: it is simply forcing
    # h(q)=1 for q>=1 instead of min(1, h(q)).
    manning_head_table_gain = freeze_above_design(manning_head_table)
    h_mat_gain = h_manning_series(q, manning_head_table_gain)
    E_head_gain = energy(p_head(q, h_mat_gain))
    E_head_gain = np.where(geom_ok, E_head_gain, E0)

    print("\n[V_head two-sided] monte AND valle both vary with discharge "
          "(drops the fixed-headwater assumption)")
    lon_up, lat_up, has_upstream = load_upstream_cells(ror.index.values)
    geom_up_raw = sample_channel_geometry(lon_up, lat_up)
    geom_up_ok = has_upstream & geometry_is_valid(geom_up_raw)
    print(f"  upstream cell resolved: {int(geom_up_ok.sum())}/{len(ror)} RoR plants "
          f"({int(has_upstream.sum())} with a chain neighbour, "
          f"{int((~has_upstream).sum())} headwater/unmatched)")

    geom_down = {"bw": geom_bw, "side_slope": geom_side, "grad": geom_grad,
                "manning_n": geom_manning_n, "bnkf": geom_bnkf, "flpn": geom_flpn}
    geom_up = prepare_geometry(geom_up_raw)

    # -- direct cell-to-cell: matched (valle) vs upstream drainage node (monte),
    # each with its own geometry, ~5 km apart (the raw grid-cell separation) --
    manning_head_table_cellcell = build_manning_head_table_twoside(qmax, head, geom_down, geom_up)
    h_mat_cellcell = h_manning_series(q, manning_head_table_cellcell)
    E_head_cellcell = energy(p_head(q, h_mat_cellcell))
    valid_cellcell = geom_ok & geom_up_ok
    E_head_cellcell = np.where(valid_cellcell, E_head_cellcell, E0)

    # -- local variant: project the plant's TRUE coordinates onto the
    # monte->valle segment (t=0 at monte, t=1 at valle), use that as the
    # "monte" reference, and move DOWNSTREAM_OFFSET_M further towards valle
    # for the "valle" reference -- both geometries are a linear blend of the
    # two already-sampled endpoint cells, no new raster read needed --
    t, seg_len_m = project_onto_segment(
        ror.plant_lon.values, ror.plant_lat.values, lon_up, lat_up, matched_lon, matched_lat,
    )
    has_coords = np.isfinite(ror.plant_lon.values) & np.isfinite(ror.plant_lat.values)
    t_valle = np.where(seg_len_m > 0, t + DOWNSTREAM_OFFSET_M / np.maximum(seg_len_m, 1e-6), t)
    offset_beyond_segment = t_valle > 1.0
    t_valle = np.clip(t_valle, 0.0, 1.0)
    print(f"  {DOWNSTREAM_OFFSET_M:g} m offset exceeds the monte-valle segment for "
          f"{int((offset_beyond_segment & valid_cellcell).sum())} plants (clipped to valle)")

    geom_monte_local = blend_geometry(geom_up, geom_down, t)
    geom_valle_local = blend_geometry(geom_up, geom_down, t_valle)
    manning_head_table_local = build_manning_head_table_twoside(
        qmax, head, geom_valle_local, geom_monte_local,
    )
    h_mat_local = h_manning_series(q, manning_head_table_local)
    E_head_local = energy(p_head(q, h_mat_local))
    valid_local = geom_ok & geom_up_ok & has_coords
    E_head_local = np.where(valid_local, E_head_local, E0)

    # -- mirror of the cell-to-cell variant: the matched cell now plays
    # "monte", and we resolve its immediate DOWNSTREAM neighbour as "valle".
    # downstream_model_id is single-valued (no confluence ambiguity), so
    # coverage is expected to be much higher than the upstream lookup --
    # every node has one except a true basin outlet (river mouth) --
    lon_down2, lat_down2, has_downstream = load_downstream_cells(ror.index.values)
    geom_down2_raw = sample_channel_geometry(lon_down2, lat_down2)
    geom_down2_ok = has_downstream & geometry_is_valid(geom_down2_raw)
    print(f"  downstream cell resolved: {int(geom_down2_ok.sum())}/{len(ror)} RoR plants "
          f"({int(has_downstream.sum())} with a chain neighbour, "
          f"{int((~has_downstream).sum())} outlet/unmatched)")

    geom_down2 = prepare_geometry(geom_down2_raw)
    manning_head_table_downcell = build_manning_head_table_twoside(qmax, head, geom_down2, geom_down)
    h_mat_downcell = h_manning_series(q, manning_head_table_downcell)
    E_head_downcell = energy(p_head(q, h_mat_downcell))
    valid_downcell = geom_ok & geom_down2_ok
    E_head_downcell = np.where(valid_downcell, E_head_downcell, E0)

    # gain-only mirrors of the three two-sided variants above: freeze h=1 for
    # q>=1 (no flood derating from either side), keep only the part-load
    # variation between qmin and design flow.
    h_mat_cellcell_gain = h_manning_series(q, freeze_above_design(manning_head_table_cellcell))
    E_head_cellcell_gain = energy(p_head(q, h_mat_cellcell_gain))
    E_head_cellcell_gain = np.where(valid_cellcell, E_head_cellcell_gain, E0)

    h_mat_local_gain = h_manning_series(q, freeze_above_design(manning_head_table_local))
    E_head_local_gain = energy(p_head(q, h_mat_local_gain))
    E_head_local_gain = np.where(valid_local, E_head_local_gain, E0)

    h_mat_downcell_gain = h_manning_series(q, freeze_above_design(manning_head_table_downcell))
    E_head_downcell_gain = energy(p_head(q, h_mat_downcell_gain))
    E_head_downcell_gain = np.where(valid_downcell, E_head_downcell_gain, E0)

    low = head < HEAD_SPLIT
    rows, keep_plant = [], {"E_V0": E0}

    def record(model, params, E2, baseline=E0):
        dE = E2 - baseline
        tot = float(dE.sum())
        r = {"model": model, **params,
             "dE_TWh_per_year": tot,
             "pct_of_base_RoR": 100 * tot / baseline.sum(),
             "pct_of_obs_RoR": 100 * tot / E_obs_ror
             if np.isfinite(E_obs_ror) else np.nan,
             "pct_of_obs_hydro": 100 * tot / E_obs_tot
             if np.isfinite(E_obs_tot) else np.nan,
             "pct_lowhead": 100 * dE[low].sum() / baseline[low].sum()
             if baseline[low].sum() > 0 else np.nan,
             "pct_highhead": 100 * dE[~low].sum() / baseline[~low].sum()
             if baseline[~low].sum() > 0 else np.nan,
             "dE_TWh_wellassigned": float(dE[well].sum()),
             "pct_of_base_wellassigned": 100 * dE[well].sum() / baseline[well].sum()
             if baseline[well].sum() > 0 else np.nan,
             "dE_TWh_geom_matched": float(dE[geom_ok].sum()),
             "pct_of_base_geom_matched": 100 * dE[geom_ok].sum() / baseline[geom_ok].sum()
             if baseline[geom_ok].sum() > 0 else np.nan}
        rows.append(r)
        print(f"  [{model}] {params} -> {tot:+7.2f} TWh/yr  "
              f"({r['pct_of_obs_RoR']:+6.2f} % obs RoR, "
              f"{r['pct_of_obs_hydro']:+6.2f} % obs hydro)")

    record("V_head", {}, E_head)
    keep_plant["E_head"] = E_head

    record("V_head_adverse", {}, E_head_adverse)
    keep_plant["E_head_adverse"] = E_head_adverse

    record("V_head_gain", {}, E_head_gain)
    keep_plant["E_head_gain"] = E_head_gain

    record("V_head_cellcell", {}, E_head_cellcell)
    keep_plant["E_head_cellcell"] = E_head_cellcell

    record("V_head_local100", {}, E_head_local)
    keep_plant["E_head_local100"] = E_head_local

    record("V_head_cellcell_downstream", {}, E_head_downcell)
    keep_plant["E_head_cellcell_downstream"] = E_head_downcell

    record("V_head_cellcell_gain", {}, E_head_cellcell_gain)
    keep_plant["E_head_cellcell_gain"] = E_head_cellcell_gain

    record("V_head_local100_gain", {}, E_head_local_gain)
    keep_plant["E_head_local100_gain"] = E_head_local_gain

    record("V_head_cellcell_downstream_gain", {}, E_head_downcell_gain)
    keep_plant["E_head_cellcell_downstream_gain"] = E_head_downcell_gain

    S = pd.DataFrame(rows)
    S.to_csv(OUTDIR / "sensitivity_summary.csv", index=False)

    pd.DataFrame({
        "name": ror.name.values, "country": ror.country.values,
        "p_nom_MW": pnom, "head_m": head,
        "head_reported": ror.head_reported.values,
        "qmax_m3s": qmax, "q_cell_mean": qbar,
        "well_assigned": well,
        "geom_matched": geom_ok, "geom_match_type": match_type,
        "chanbnkf_m": geom_bnkf, "chanbw_m": geom_bw,
        "changrad": geom_grad, "chanman": geom_manning_n,
        "upstream_resolved": geom_up_ok, "downstream_resolved": geom_down2_ok,
        "segment_length_m": seg_len_m, "projection_t": t,
        **keep_plant}, index=ror.index
    ).to_csv(OUTDIR / "plant_energies.csv")

    pd.DataFrame({"country": ror.country.values, **keep_plant}
                 ).groupby("country").sum().round(3).to_csv(
        OUTDIR / "baseline_energies.csv")

    vh = S[S.model == "V_head"].iloc[0]
    vh_adv = S[S.model == "V_head_adverse"].iloc[0]
    vh_gain = S[S.model == "V_head_gain"].iloc[0]
    vh_cellcell = S[S.model == "V_head_cellcell"].iloc[0]
    vh_local = S[S.model == "V_head_local100"].iloc[0]
    vh_downcell = S[S.model == "V_head_cellcell_downstream"].iloc[0]

    with open(OUTDIR / "summary_sensitivity.txt", "w") as f:
        w = f.write
        w("HEAD-VARIATION SENSITIVITY OF THE RoR CONVERSION\n" + "=" * 62 + "\n\n")
        w(f"years            : {used}  ({T} steps, dt = {dt_ref:g} h)\n")
        w(f"RoR plants       : {len(ror)}  ({pnom.sum()/1e3:.2f} GW)\n")
        w(f"well-assigned    : {int(well.sum())} plants, "
          f"{pnom[well].sum()/1e3:.2f} GW "
          f"(Qmax/Qbar in {WELL_ASSIGNED_BAND})\n")
        w(f"geom-matched     : {int(geom_ok.sum())} plants, "
          f"{pnom[geom_ok].sum()/1e3:.2f} GW\n")
        w(f"head < {HEAD_SPLIT:g} m    : {pnom[low].sum()/1e3:.2f} GW\n\n")
        w("BASELINE ANNUAL ENERGY [TWh/yr]\n")
        w(f"  V0 published : {E0.sum():8.2f}\n")
        w(f"  observed RoR : {E_obs_ror:8.2f}\n")
        w(f"  observed hydro: {E_obs_tot:7.2f}\n\n")
        w("V_head vs V0\n")
        w(f"  whole fleet (diluted by {len(ror) - int(geom_ok.sum())} unmatched plants):\n")
        w(f"    {vh.dE_TWh_per_year:+.2f} TWh/yr = {vh.pct_of_base_RoR:+.2f} % of V0 = "
          f"{vh.pct_of_obs_RoR:+.2f} % of observed RoR = "
          f"{vh.pct_of_obs_hydro:+.2f} % of observed total hydro\n")
        w(f"  geometry-matched plants only (the number to quote for the reviewer):\n")
        w(f"    {vh.dE_TWh_geom_matched:+.2f} TWh/yr = "
          f"{vh.pct_of_base_geom_matched:+.2f} % of their own V0 energy\n")
        w(f"  well-assigned plants only:\n")
        w(f"    {vh.dE_TWh_wellassigned:+.2f} TWh/yr = "
          f"{vh.pct_of_base_wellassigned:+.2f} % of their own V0 energy\n")
        w(f"  low-head plants (< {HEAD_SPLIT:g} m): {vh.pct_lowhead:+.2f} % of their V0 energy\n")
        w(f"  other plants (>= {HEAD_SPLIT:g} m):  {vh.pct_highhead:+.2f} % of their V0 energy\n\n")
        w("V_head_adverse vs V0 (part-load gain removed, flood loss only,\n"
          " same real geometry -- no extra assumption beyond min(1, h))\n")
        w(f"  whole fleet: {vh_adv.dE_TWh_per_year:+.2f} TWh/yr = "
          f"{vh_adv.pct_of_base_RoR:+.2f} % of V0 = "
          f"{vh_adv.pct_of_obs_RoR:+.2f} % of observed RoR = "
          f"{vh_adv.pct_of_obs_hydro:+.2f} % of observed total hydro\n")
        w(f"  geometry-matched plants only: {vh_adv.dE_TWh_geom_matched:+.2f} TWh/yr = "
          f"{vh_adv.pct_of_base_geom_matched:+.2f} % of their own V0 energy\n")
        w(f"  well-assigned plants only: {vh_adv.dE_TWh_wellassigned:+.2f} TWh/yr = "
          f"{vh_adv.pct_of_base_wellassigned:+.2f} % of their own V0 energy\n")
        w(f"  low-head plants (< {HEAD_SPLIT:g} m): {vh_adv.pct_lowhead:+.2f} % of their V0 energy\n")
        w(f"  other plants (>= {HEAD_SPLIT:g} m):  {vh_adv.pct_highhead:+.2f} % of their V0 energy\n\n")
        w("V_head_gain vs V0 (mirror of V_head_adverse: part-load gain kept,\n"
          " flood loss removed -- h(q) frozen at 1 for q>=1)\n")
        w(f"  whole fleet: {vh_gain.dE_TWh_per_year:+.2f} TWh/yr = "
          f"{vh_gain.pct_of_base_RoR:+.2f} % of V0 = "
          f"{vh_gain.pct_of_obs_RoR:+.2f} % of observed RoR = "
          f"{vh_gain.pct_of_obs_hydro:+.2f} % of observed total hydro\n")
        w(f"  geometry-matched plants only: {vh_gain.dE_TWh_geom_matched:+.2f} TWh/yr = "
          f"{vh_gain.pct_of_base_geom_matched:+.2f} % of their own V0 energy\n")
        w(f"  well-assigned plants only: {vh_gain.dE_TWh_wellassigned:+.2f} TWh/yr = "
          f"{vh_gain.pct_of_base_wellassigned:+.2f} % of their own V0 energy\n")
        w(f"  low-head plants (< {HEAD_SPLIT:g} m): {vh_gain.pct_lowhead:+.2f} % of their V0 energy\n")
        w(f"  other plants (>= {HEAD_SPLIT:g} m):  {vh_gain.pct_highhead:+.2f} % of their V0 energy\n\n")
        w("V_head_cellcell vs V0 (two-sided: monte AND valle both vary,\n"
          " using the upstream drainage node and the matched cell directly,\n"
          " ~5 km apart -- drops the fixed-headwater assumption)\n")
        w(f"  upstream resolved: {int(geom_up_ok.sum())}/{len(ror)} plants, "
          f"{pnom[geom_up_ok].sum()/1e3:.2f} GW\n")
        w(f"  whole fleet: {vh_cellcell.dE_TWh_per_year:+.2f} TWh/yr = "
          f"{vh_cellcell.pct_of_base_RoR:+.2f} % of V0 = "
          f"{vh_cellcell.pct_of_obs_RoR:+.2f} % of observed RoR = "
          f"{vh_cellcell.pct_of_obs_hydro:+.2f} % of observed total hydro\n")
        w(f"  geometry-matched plants only: {vh_cellcell.dE_TWh_geom_matched:+.2f} TWh/yr = "
          f"{vh_cellcell.pct_of_base_geom_matched:+.2f} % of their own V0 energy\n")
        w(f"  well-assigned plants only: {vh_cellcell.dE_TWh_wellassigned:+.2f} TWh/yr = "
          f"{vh_cellcell.pct_of_base_wellassigned:+.2f} % of their own V0 energy\n")
        w(f"  low-head plants (< {HEAD_SPLIT:g} m): {vh_cellcell.pct_lowhead:+.2f} % of their V0 energy\n")
        w(f"  other plants (>= {HEAD_SPLIT:g} m):  {vh_cellcell.pct_highhead:+.2f} % of their V0 energy\n\n")
        w(f"V_head_local{DOWNSTREAM_OFFSET_M:g} vs V0 (two-sided, but monte/valle are the\n"
          " plant's TRUE position projected onto the monte-valle segment and a\n"
          f" point {DOWNSTREAM_OFFSET_M:g} m further towards valle -- geometry blended between\n"
          " the two endpoint cells, no new raster read; expect near-total\n"
          " cancellation at short offsets, since monte and valle sit on the\n"
          " same short reach)\n")
        w(f"  {DOWNSTREAM_OFFSET_M:g} m offset exceeds the segment for "
          f"{int((offset_beyond_segment & valid_cellcell).sum())} plants (clipped to valle)\n")
        w(f"  whole fleet: {vh_local.dE_TWh_per_year:+.2f} TWh/yr = "
          f"{vh_local.pct_of_base_RoR:+.2f} % of V0 = "
          f"{vh_local.pct_of_obs_RoR:+.2f} % of observed RoR = "
          f"{vh_local.pct_of_obs_hydro:+.2f} % of observed total hydro\n")
        w(f"  geometry-matched plants only: {vh_local.dE_TWh_geom_matched:+.2f} TWh/yr = "
          f"{vh_local.pct_of_base_geom_matched:+.2f} % of their own V0 energy\n")
        w(f"  well-assigned plants only: {vh_local.dE_TWh_wellassigned:+.2f} TWh/yr = "
          f"{vh_local.pct_of_base_wellassigned:+.2f} % of their own V0 energy\n")
        w(f"  low-head plants (< {HEAD_SPLIT:g} m): {vh_local.pct_lowhead:+.2f} % of their V0 energy\n")
        w(f"  other plants (>= {HEAD_SPLIT:g} m):  {vh_local.pct_highhead:+.2f} % of their V0 energy\n\n")
        w("V_head_cellcell_downstream vs V0 (mirror of V_head_cellcell: the\n"
          " matched cell now plays monte, and its resolved downstream\n"
          " neighbour plays valle -- downstream_model_id is single-valued, so\n"
          " coverage should be much higher than the upstream lookup)\n")
        w(f"  downstream resolved: {int(geom_down2_ok.sum())}/{len(ror)} plants, "
          f"{pnom[geom_down2_ok].sum()/1e3:.2f} GW\n")
        w(f"  whole fleet: {vh_downcell.dE_TWh_per_year:+.2f} TWh/yr = "
          f"{vh_downcell.pct_of_base_RoR:+.2f} % of V0 = "
          f"{vh_downcell.pct_of_obs_RoR:+.2f} % of observed RoR = "
          f"{vh_downcell.pct_of_obs_hydro:+.2f} % of observed total hydro\n")
        w(f"  geometry-matched plants only: {vh_downcell.dE_TWh_geom_matched:+.2f} TWh/yr = "
          f"{vh_downcell.pct_of_base_geom_matched:+.2f} % of their own V0 energy\n")
        w(f"  well-assigned plants only: {vh_downcell.dE_TWh_wellassigned:+.2f} TWh/yr = "
          f"{vh_downcell.pct_of_base_wellassigned:+.2f} % of their own V0 energy\n")
        w(f"  low-head plants (< {HEAD_SPLIT:g} m): {vh_downcell.pct_lowhead:+.2f} % of their V0 energy\n")
        w(f"  other plants (>= {HEAD_SPLIT:g} m):  {vh_downcell.pct_highhead:+.2f} % of their V0 energy\n\n")
        w("GAIN-ONLY MIRRORS OF THE THREE TWO-SIDED VARIANTS\n" + "-" * 50 + "\n")
        w("(h(q) frozen at 1 for q>=1: no flood derating from either side,\n"
          " only the part-load variation between qmin and design flow)\n")
        for label, row in [("V_head_cellcell_gain", S[S.model == "V_head_cellcell_gain"].iloc[0]),
                           ("V_head_local100_gain", S[S.model == "V_head_local100_gain"].iloc[0]),
                           ("V_head_cellcell_downstream_gain",
                            S[S.model == "V_head_cellcell_downstream_gain"].iloc[0])]:
            w(f"  [{label}] whole fleet: {row.dE_TWh_per_year:+.2f} TWh/yr = "
              f"{row.pct_of_base_RoR:+.2f} % of V0 | "
              f"low-head: {row.pct_lowhead:+.2f} % | high-head: {row.pct_highhead:+.2f} %\n")
    print("\n" + (OUTDIR / "summary_sensitivity.txt").read_text())

    _figs(ror, head, geom_ok, manning_head_table, manning_head_table_adverse,
          manning_head_table_gain)
    print(f"[done] outputs in {OUTDIR.resolve()}")


# =============================================================================
# FIGURES
# =============================================================================
def _figs(ror, head, geom_ok, manning_head_table, manning_head_table_adverse,
         manning_head_table_gain):
    matched_idx = np.where(geom_ok)[0]
    if len(matched_idx) == 0:
        print("[warn] no geometry-matched plants; skipping fig_curves.png")
        example_idx = []
    else:
        example_idx = [matched_idx[np.argmin(head[matched_idx])],
                       matched_idx[np.argmax(head[matched_idx])]]

    qq = np.linspace(0, 4, 800)
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.plot(qq, p_V0(qq.copy()), "k-", lw=2.2, label="V0 published")
    for idx, col in zip(example_idx, ["#377eb8", "#e41a1c"]):
        h_curve = np.interp(qq, MANNING_Q_GRID, manning_head_table[:, idx])
        h_curve_adv = np.interp(qq, MANNING_Q_GRID, manning_head_table_adverse[:, idx])
        h_curve_gain = np.interp(qq, MANNING_Q_GRID, manning_head_table_gain[:, idx])
        label = f"{ror.name.values[idx]!s} (H={head[idx]:.1f} m)"
        ax.plot(qq, p_head(qq.copy(), h_curve), "-", color=col, lw=1.6,
                label=f"V_head, {label}")
        ax.plot(qq, p_head(qq.copy(), h_curve_adv), ":", color=col, lw=1.6,
                label=f"V_head_adverse, {label}")
        ax.plot(qq, p_head(qq.copy(), h_curve_gain), "--", color=col, lw=1.6,
                label=f"V_head_gain, {label}")
    ax.axvline(1, color="grey", lw=0.6)
    ax.grid(alpha=.3)
    ax.set_xlabel("$q = Q/Q_{max}$")
    ax.set_ylabel("$P/P_{max}$")
    ax.set_title("V0 vs V_head vs V_head_adverse vs V_head_gain (real matched-plant geometry)")
    ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_curves.png", dpi=180)
    plt.close(fig)

    S = pd.read_csv(OUTDIR / "sensitivity_summary.csv")
    denominators = ["pct_of_base_RoR", "pct_of_base_geom_matched",
                    "pct_of_base_wellassigned", "pct_lowhead", "pct_highhead"]
    labels = ["whole fleet", "geometry-matched only", "well-assigned only",
             f"head < {HEAD_SPLIT:g} m", f"head >= {HEAD_SPLIT:g} m"]

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    width = 0.25
    y = np.arange(len(labels))
    for offset, model, color in [(-width, "V_head", "#377eb8"),
                                  (0, "V_head_adverse", "#e41a1c"),
                                  (width, "V_head_gain", "#4daf4a")]:
        row = S[S.model == model].iloc[0]
        ax.barh(y + offset, [row[d] for d in denominators], height=width,
                color=color, label=model)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("$\\Delta E$ / own V0 energy of that subset  [%]")
    ax.set_title("V_head / V_head_adverse / V_head_gain - V0, by denominator")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_tornado.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
