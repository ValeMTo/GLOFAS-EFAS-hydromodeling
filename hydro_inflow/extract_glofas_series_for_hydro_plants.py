# This script loads hydropower plant locations and links each plant to the most suitable
# GloFAS drainage/grid point. It then extracts the corresponding GloFAS discharge time
# series, producing plant-level hydrological inflow.

from __future__ import annotations

from pathlib import Path
import logging
import os
import pickle
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from hydro_inflow.hydro_config import get_config, log_config_summary
from hydro_inflow.utils import (
    approx_dist_km,
    km_to_deg_lat,
    km_to_deg_lon,
    setup_logging,
)


logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

MAX_DISTANCE_KM = 10.0
WINDOW = 3
AREA_JUMP_THRESHOLD = 0.15
CORR_THRESHOLD = 0.95
MARGIN_DEG = 0.5
EFFICIENCY = 0.85
K_NEIGH_FALLBACK = 6
GRID_MATCH_TOLERANCE_DEG = 1e-6
MIN_RIVER_QMAX_RATIO = 0.2


# ============================================================
# LOADERS
# ============================================================

def load_hydro_power_plants(
    plants_path: Path,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    efficiency: float = 0.85,
) -> pd.DataFrame:
    """
    Load hydro power plants and compute qmax/qmin turbine discharge.
    """
    ppls = pd.read_csv(plants_path, index_col=0)

    hydro_ppls = ppls[ppls["Fueltype"] == "Hydro"].copy()
    hydro_ppls = hydro_ppls[hydro_ppls["Capacity"] != 0].copy()

    hydro_ppls["Technology"] = hydro_ppls["Technology"].fillna("Run-Of-River")

    logger.info("Hydro plants loaded: %s", len(hydro_ppls))

    hydro_ppls = hydro_ppls[
        (hydro_ppls["lon"] >= lon_min)
        & (hydro_ppls["lon"] <= lon_max)
        & (hydro_ppls["lat"] >= lat_min)
        & (hydro_ppls["lat"] <= lat_max)
    ].copy()

    logger.debug("Hydro plants filtered to model domain: %s", len(hydro_ppls))

    median_dam_height_ror = hydro_ppls.loc[
        hydro_ppls["Technology"] == "Run-Of-River",
        "DamHeight_m",
    ].median(skipna=True)

    median_dam_height_res = hydro_ppls.loc[
        hydro_ppls["Technology"].isin(["Reservoir", "Pumped Storage"]),
        "DamHeight_m",
    ].median(skipna=True)

    hydro_ppls["qmax_turb"] = np.where(
        hydro_ppls["DamHeight_m"].notna(),
        hydro_ppls["Capacity"]
        / (9.81 * hydro_ppls["DamHeight_m"] * 1e-3 * efficiency),
        np.where(
            hydro_ppls["Technology"] == "Run-Of-River",
            hydro_ppls["Capacity"]
            / (9.81 * median_dam_height_ror * 1e-3 * efficiency),
            hydro_ppls["Capacity"]
            / (9.81 * median_dam_height_res * 1e-3 * efficiency),
        ),
    )

    hydro_ppls["qmin_turb"] = hydro_ppls["qmax_turb"] * 0.2

    logger.debug("Qmax and Qmin extrapolated.")

    return hydro_ppls


def build_river_bbox_and_mean(
    hydro_ppls: pd.DataFrame,
    river_reference_path: Path,
    var_name: str,
    margin_deg: float = 2.0,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Build spatial GLoFAS subset around hydro plants and compute mean discharge.
    """
    lat_min = hydro_ppls["lat"].min() - margin_deg
    lat_max = hydro_ppls["lat"].max() + margin_deg
    lon_min = hydro_ppls["lon"].min() - margin_deg
    lon_max = hydro_ppls["lon"].max() + margin_deg

    logger.debug(
        "River bounding box: lat %.2f -> %.2f | lon %.2f -> %.2f",
        lat_min,
        lat_max,
        lon_min,
        lon_max,
    )

    ds = xr.open_dataset(river_reference_path)

    river = ds[var_name]

    lat_values = river["latitude"].values

    if lat_values[0] > lat_values[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    river_bbox = river.sel(
        latitude=lat_slice,
        longitude=slice(lon_min, lon_max),
    )

    logger.debug("River bbox dimensions: %s", dict(river_bbox.sizes))

    time_dim = "valid_time" if "valid_time" in river_bbox.dims else "time"

    river_mean = river_bbox.mean(dim=time_dim)
    river_mean = river_mean.where(river_mean > 0)

    logger.debug("River mean computed. Shape: %s", river_mean.shape)

    return river_bbox, river_mean


def build_storage_plants(hydro_ppls: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Select hydro plants with storage behavior.
    """
    plants_with_storage = hydro_ppls[
        (hydro_ppls["Technology"] != "Run-Of-River")
        & (hydro_ppls["Capacity"] > 0)
    ].copy()

    plants_with_storage = gpd.GeoDataFrame(
        plants_with_storage,
        geometry=gpd.points_from_xy(
            plants_with_storage["lon"],
            plants_with_storage["lat"],
        ),
        crs="EPSG:4326",
    )

    logger.debug("Storage plants: %s", len(plants_with_storage))

    return plants_with_storage

def load_drain_table(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    required_cols = {
        "model_id",
        "downstream_model_id",
        "upstream_model_ids",
        "drainage_area",
        "x",
        "y",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in drain table: {sorted(missing)}")

    df["model_id"] = df["model_id"].astype(str)
    df["downstream_model_id"] = df["downstream_model_id"].where(
        df["downstream_model_id"].notna(),
        None,
    )

    df["upstream_model_ids"] = df["upstream_model_ids"].fillna("").astype(str)

    if "HYBAS_ID" in df.columns:
        df["HYBAS_ID"] = df["HYBAS_ID"].astype(str)

    return df


def prepare_plants_gdf(all_hydro_plants: pd.DataFrame) -> gpd.GeoDataFrame:
    plants = all_hydro_plants.copy()

    if "lon" not in plants.columns or "lat" not in plants.columns:
        raise ValueError("all_hydro_plants must contain 'lon' and 'lat' columns.")

    plants = plants.dropna(subset=["lon", "lat"])

    return gpd.GeoDataFrame(
        plants,
        geometry=gpd.points_from_xy(plants["lon"], plants["lat"]),
        crs="EPSG:4326",
    )


def open_reference_river_mean(glofas_dir: Path, years: list[int], var_name: str) -> xr.DataArray:
    means = []

    for year in years:
        path = glofas_dir / f"glofas_eu_{year}.nc"

        with xr.open_dataset(path) as ds:
            da = ds[var_name]
            means.append(da.mean(dim=da.dims[0]).load())

    river_mean = xr.concat(means, dim="year").mean(dim="year")
    return river_mean


# ============================================================
# MATCH PLANTS TO DRAIN
# ============================================================

def build_glofas_grid_index(river_mean: xr.DataArray) -> dict:
    """
    Build a nearest-neighbour index over the full GloFAS grid.

    The KDTree is used only for fast preselection. Final distances are
    recomputed locally using approx_dist_km().
    """
    lats = river_mean.latitude.values
    lons = river_mean.longitude.values

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    lon_flat = lon_grid.ravel()
    lat_flat = lat_grid.ravel()
    river_values = river_mean.values.ravel()

    valid_mask = np.isfinite(lon_flat) & np.isfinite(lat_flat)

    lon_flat = lon_flat[valid_mask]
    lat_flat = lat_flat[valid_mask]
    river_values = river_values[valid_mask]

    mean_lat = float(np.mean(lat_flat))

    x_km = lon_flat * 111.0 * np.cos(np.deg2rad(mean_lat))
    y_km = lat_flat * 111.0

    coords_km = np.column_stack([x_km, y_km])
    tree = cKDTree(coords_km)

    return {
        "tree": tree,
        "lon_flat": lon_flat,
        "lat_flat": lat_flat,
        "river_values": river_values,
        "mean_lat": mean_lat,
    }


def build_drain_grid_lookup(
    df_drain: pd.DataFrame,
    tolerance_deg: float = 1e-6,
) -> dict[tuple[float, float], list[int]]:
    """
    Build a lookup from rounded GloFAS grid coordinates to drain-table row indices.
    """
    decimals = int(max(0, np.ceil(-np.log10(tolerance_deg))))

    lookup = {}

    for idx, row in df_drain.iterrows():
        key = (
            round(float(row["x"]), decimals),
            round(float(row["y"]), decimals),
        )
        lookup.setdefault(key, []).append(idx)

    return lookup


def get_k_nearest_glofas_cells(
    lon: float,
    lat: float,
    glofas_grid_index: dict,
    k_neigh: int,
) -> pd.DataFrame:
    """
    Return the k nearest GloFAS grid cells to a plant.

    A larger candidate set is queried first, then distances are recomputed
    locally at the plant latitude.
    """
    query_k = max(k_neigh * 5, 30)

    mean_lat = glofas_grid_index["mean_lat"]

    x_plant_global = lon * 111.0 * np.cos(np.deg2rad(mean_lat))
    y_plant_global = lat * 111.0

    _, idxs = glofas_grid_index["tree"].query(
        [x_plant_global, y_plant_global],
        k=query_k,
    )

    idxs = np.atleast_1d(idxs)

    lon_candidates = glofas_grid_index["lon_flat"][idxs]
    lat_candidates = glofas_grid_index["lat_flat"][idxs]

    dists_km = approx_dist_km(
        lon=lon,
        lat=lat,
        lon2=lon_candidates,
        lat2=lat_candidates,
    )

    order = np.argsort(dists_km)[:k_neigh]
    selected_idxs = idxs[order]

    return pd.DataFrame(
        {
            "x": glofas_grid_index["lon_flat"][selected_idxs],
            "y": glofas_grid_index["lat_flat"][selected_idxs],
            "distance_km": dists_km[order],
            "river_mean": glofas_grid_index["river_values"][selected_idxs],
            "glofas_flat_index": selected_idxs,
        }
    )


def attach_drain_rows_to_glofas_cells(
    glofas_cells: pd.DataFrame,
    df_drain: pd.DataFrame,
    drain_grid_lookup: dict[tuple[float, float], list[int]],
    tolerance_deg: float = 1e-6,
) -> pd.DataFrame:
    """
    Keep only GloFAS cells that are present in df_drain and attach drain metadata.
    """
    decimals = int(max(0, np.ceil(-np.log10(tolerance_deg))))

    rows = []

    for _, cell in glofas_cells.iterrows():
        key = (
            round(float(cell["x"]), decimals),
            round(float(cell["y"]), decimals),
        )

        drain_indices = drain_grid_lookup.get(key, [])

        for drain_idx in drain_indices:
            drain_row = df_drain.loc[drain_idx].copy()

            out = drain_row.to_dict()
            out["distance_km"] = float(cell["distance_km"])
            out["river_mean"] = float(cell["river_mean"])
            out["glofas_flat_index"] = int(cell["glofas_flat_index"])

            rows.append(out)

    return pd.DataFrame(rows)


def add_river_mean_to_candidates(
    candidates: pd.DataFrame,
    river_mean: xr.DataArray,
    cache: dict[tuple[float, float], float],
) -> pd.DataFrame:
    """
    Attach mean GloFAS discharge to drain candidates.
    """
    lats = river_mean.latitude.values
    lons = river_mean.longitude.values

    river_vals = np.empty(len(candidates), dtype=float)

    for j, (_, row) in enumerate(candidates.iterrows()):
        node_lat = float(row["y"])
        node_lon = float(row["x"])

        key = (round(node_lat, 5), round(node_lon, 5))

        if key not in cache:
            lat_idx = np.abs(lats - node_lat).argmin()
            lon_idx = np.abs(lons - node_lon).argmin()

            cache[key] = float(
                river_mean.isel(latitude=lat_idx, longitude=lon_idx).values
            )

        river_vals[j] = cache[key]

    candidates = candidates.copy()
    candidates["river_mean"] = river_vals

    return candidates


def get_drain_candidates_within_radius(
    lon: float,
    lat: float,
    df_drain: pd.DataFrame,
    river_mean: xr.DataArray,
    river_mean_cache: dict,
    max_distance_km: float,
) -> pd.DataFrame:
    """
    Search df_drain candidates within max_distance_km.
    """
    buf_lon = km_to_deg_lon(max_distance_km, lat)
    buf_lat = km_to_deg_lat(max_distance_km)

    bbox_candidates = df_drain[
        (df_drain["x"] >= lon - buf_lon)
        & (df_drain["x"] <= lon + buf_lon)
        & (df_drain["y"] >= lat - buf_lat)
        & (df_drain["y"] <= lat + buf_lat)
    ].copy()

    if bbox_candidates.empty:
        return pd.DataFrame()

    bbox_candidates["distance_km"] = approx_dist_km(
        lon=lon,
        lat=lat,
        lon2=bbox_candidates["x"].to_numpy(dtype=float),
        lat2=bbox_candidates["y"].to_numpy(dtype=float),
    )

    candidates = bbox_candidates[
        bbox_candidates["distance_km"] <= max_distance_km
    ].copy()

    if candidates.empty:
        return pd.DataFrame()

    candidates = add_river_mean_to_candidates(
        candidates=candidates,
        river_mean=river_mean,
        cache=river_mean_cache,
    )

    return candidates


def filter_candidates_by_minimum_flow(
    candidates: pd.DataFrame,
    qmax: float,
    min_fraction_of_qmax: float = 0.2,
) -> pd.DataFrame:
    """
    Remove candidates with river_mean < min_fraction_of_qmax * qmax,
    unless this would remove all candidates.
    """
    candidates = candidates.copy()

    if candidates.empty:
        return candidates

    if not np.isfinite(qmax) or qmax <= 0:
        return candidates

    if "river_mean" not in candidates.columns:
        return candidates

    threshold = min_fraction_of_qmax * qmax
    filtered = candidates[candidates["river_mean"] >= threshold].copy()

    if len(filtered) >= 1:
        return filtered

    return candidates


def score_candidates(
    candidates: pd.DataFrame,
    qmax: float,
    distance_norm_denominator_km: float,
    min_fraction_of_qmax: float = 0.2,
) -> pd.DataFrame:
    """
    Score candidates using:
        0.5 * normalized distance + 0.5 * normalized qmax mismatch.

    Before scoring, candidates with very low mean discharge are removed,
    unless this would remove all candidates.
    """
    candidates = candidates.copy()

    if candidates.empty:
        return candidates

    candidates = filter_candidates_by_minimum_flow(
        candidates=candidates,
        qmax=qmax,
        min_fraction_of_qmax=min_fraction_of_qmax,
    )

    if np.isfinite(qmax):
        candidates["diff_qmax"] = (candidates["river_mean"] - qmax).abs()
    else:
        candidates["diff_qmax"] = 0.0

    candidates["diff_qmax"] = candidates["diff_qmax"].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if distance_norm_denominator_km <= 0 or not np.isfinite(distance_norm_denominator_km):
        candidates["dist_norm"] = 0.0
    else:
        candidates["dist_norm"] = candidates["distance_km"] / distance_norm_denominator_km

    candidates["dist_norm"] = (
        candidates["dist_norm"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
    )

    finite_diff = candidates["diff_qmax"].dropna()

    if finite_diff.empty:
        candidates["diff_norm"] = 0.0
    else:
        max_diff = float(finite_diff.max())
        if max_diff == 0 or not np.isfinite(max_diff):
            candidates["diff_norm"] = 0.0
        else:
            candidates["diff_norm"] = candidates["diff_qmax"] / max_diff
            candidates["diff_norm"] = candidates["diff_norm"].fillna(1.0)

    candidates["combined_score"] = (
        0.5 * candidates["dist_norm"]
        + 0.5 * candidates["diff_norm"]
    )

    candidates["combined_score"] = (
        candidates["combined_score"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
    )

    return candidates


def make_match_row(
    plant_id,
    best: pd.Series,
    match_type: str,
) -> dict:
    """
    Build one standardized match row.
    """
    matched_river_mean = float(best["river_mean"])

    return {
        "plant_index": plant_id,
        "model_id": (
            str(best["model_id"])
            if "model_id" in best and pd.notna(best["model_id"])
            else None
        ),
        "HYBAS_ID": (
            str(best["HYBAS_ID"])
            if "HYBAS_ID" in best and pd.notna(best["HYBAS_ID"])
            else None
        ),
        "cluster": (
            int(best["cluster"])
            if "cluster" in best and pd.notna(best["cluster"])
            else None
        ),
        "node": (
            int(best["node"])
            if "node" in best and pd.notna(best["node"])
            else None
        ),
        "matched_lon": float(best["x"]),
        "matched_lat": float(best["y"]),
        "distance_km": float(best["distance_km"]),
        "diff_qmax": (
            float(best["diff_qmax"])
            if "diff_qmax" in best and pd.notna(best["diff_qmax"])
            else np.nan
        ),
        "combined_score": (
            float(best["combined_score"])
            if "combined_score" in best and pd.notna(best["combined_score"])
            else np.nan
        ),
        "river_mean": matched_river_mean,
        "match_type": match_type,
    }


def match_plants_to_drain(
    plants_gdf: gpd.GeoDataFrame,
    df_drain: pd.DataFrame,
    river_mean: xr.DataArray,
    max_distance_km: float,
    k_neigh: int,
) -> pd.DataFrame:
    """
    Match hydro plants with a safe priority order:

    1. Take the k nearest GloFAS grid cells.
       If any belongs to df_drain, select the best drain cell among them.

    2. If no drain cell is present among the k nearest GloFAS cells,
       search all drain nodes within max_distance_km and select the best.

    3. If no drain node exists within max_distance_km,
       fall back to the k nearest GloFAS cells, even if they are not in df_drain.

    All selections use:
        0.5 * normalized distance + 0.5 * normalized qmax mismatch.
    """
    logger.debug("Total plants to match: %s", len(plants_gdf))
    logger.debug("Drain nodes available: %s", len(df_drain))
    logger.debug("K nearest hydrological cells: %s", k_neigh)
    logger.debug("Drain radius fallback: %.1f km", max_distance_km)

    matched_info = []
    river_mean_cache = {}

    glofas_grid_index = build_glofas_grid_index(river_mean)

    drain_grid_lookup = build_drain_grid_lookup(
        df_drain=df_drain,
        tolerance_deg=GRID_MATCH_TOLERANCE_DEG,
    )

    nan_riverflow_count = 0
    low_riverflow_count = 0

    match_type_counts = {
        "drain_k_nearest": 0,
        "drain_radius": 0,
        "fallback_nn_glofas": 0,
        "unmatched": 0,
    }

    for plant_idx, plant in plants_gdf.iterrows():
        plant_id = plant_idx
        lon = float(plant.geometry.x)
        lat = float(plant.geometry.y)

        qmax = (
            float(plant["qmax_turb"])
            if "qmax_turb" in plant and pd.notna(plant["qmax_turb"])
            else np.nan
        )

        nearest_glofas = get_k_nearest_glofas_cells(
            lon=lon,
            lat=lat,
            glofas_grid_index=glofas_grid_index,
            k_neigh=k_neigh,
        )

        nearest_drain_candidates = attach_drain_rows_to_glofas_cells(
            glofas_cells=nearest_glofas,
            df_drain=df_drain,
            drain_grid_lookup=drain_grid_lookup,
            tolerance_deg=GRID_MATCH_TOLERANCE_DEG,
        )

        if not nearest_drain_candidates.empty:
            candidates = score_candidates(
                candidates=nearest_drain_candidates,
                qmax=qmax,
                distance_norm_denominator_km=max(
                    float(nearest_drain_candidates["distance_km"].max()),
                    1e-12,
                ),
                min_fraction_of_qmax=MIN_RIVER_QMAX_RATIO,
            )

            best = candidates.loc[candidates["combined_score"].idxmin()]
            match_type = "drain_k_nearest"

        else:
            radius_candidates = get_drain_candidates_within_radius(
                lon=lon,
                lat=lat,
                df_drain=df_drain,
                river_mean=river_mean,
                river_mean_cache=river_mean_cache,
                max_distance_km=max_distance_km,
            )

            if not radius_candidates.empty:
                candidates = score_candidates(
                    candidates=radius_candidates,
                    qmax=qmax,
                    distance_norm_denominator_km=max_distance_km,
                    min_fraction_of_qmax=MIN_RIVER_QMAX_RATIO,
                )

                best = candidates.loc[candidates["combined_score"].idxmin()]
                match_type = "drain_radius"

            else:
                fallback_candidates = nearest_glofas.copy()
                fallback_candidates["model_id"] = None
                fallback_candidates["HYBAS_ID"] = None
                fallback_candidates["cluster"] = None
                fallback_candidates["node"] = None

                candidates = score_candidates(
                    candidates=fallback_candidates,
                    qmax=qmax,
                    distance_norm_denominator_km=max(
                        float(fallback_candidates["distance_km"].max()),
                        1e-12,
                    ),
                    min_fraction_of_qmax=MIN_RIVER_QMAX_RATIO,
                )

                if candidates.empty:
                    match_type_counts["unmatched"] += 1
                    continue

                best = candidates.loc[candidates["combined_score"].idxmin()]
                match_type = "fallback_nn_glofas"

        row = make_match_row(
            plant_id=plant_id,
            best=best,
            match_type=match_type,
        )

        matched_river_mean = row["river_mean"]

        if pd.isna(matched_river_mean):
            nan_riverflow_count += 1
        elif matched_river_mean < 0.1:
            low_riverflow_count += 1

        matched_info.append(row)
        match_type_counts[match_type] += 1

    matched_df = pd.DataFrame(matched_info)

    logger.debug("Matching completed.")
    logger.debug("Matched plants: %s", len(matched_df))
    logger.debug("Matched points with river_mean NaN: %s", nan_riverflow_count)
    logger.debug("Matched points with river_mean < 0.1: %s", low_riverflow_count)
    logger.debug("Match type counts: %s", match_type_counts)

    return matched_df

# ============================================================
# STORAGE CORRECTION
# ============================================================

def compute_corr(s1: pd.Series, s2: pd.Series) -> float:
    df = pd.concat([s1, s2], axis=1).dropna()
    if len(df) < 10:
        return np.nan
    return float(np.corrcoef(df.iloc[:, 0], df.iloc[:, 1])[0, 1])


def parse_upstream_ids(value) -> list[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [x for x in str(value).split(";") if x]


def choose_prev_by_area_similarity(
    current_id: str,
    prev_candidates: list[str],
    drain_by_id: pd.DataFrame,
) -> str | None:
    curr_area = float(drain_by_id.loc[current_id, "drainage_area"])

    best_prev = None
    best_diff = np.inf

    for prev_id in prev_candidates:
        if prev_id not in drain_by_id.index:
            continue

        prev_area = float(drain_by_id.loc[prev_id, "drainage_area"])
        diff = abs(prev_area - curr_area) / curr_area

        if diff < best_diff:
            best_diff = diff
            best_prev = prev_id

    return best_prev


def plot_corr_break(
    s1: pd.Series,
    s2: pd.Series,
    n1: str,
    n2: str,
    corr: float,
    plant_name,
):
    df = pd.concat(
        [s1, s2],
        axis=1,
        keys=[f"node_{n1}", f"node_{n2}"],
    ).dropna()

    plt.figure(figsize=(6, 4))
    plt.plot(df.index, df.iloc[:, 0], label=f"node {n1}", linewidth=2)
    plt.plot(df.index, df.iloc[:, 1], label=f"node {n2}", linewidth=2, linestyle="--")
    plt.title(f"Worst hydrological break\n{plant_name}\ncorr = {corr:.3f}")
    plt.xlabel("Time")
    plt.ylabel("Discharge")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def choose_best_node_by_global_corr(
    node_chain: list[str],
    node_to_series: dict[str, pd.Series],
    corr_threshold: float,
    plant_index,
    do_plot: bool = True,
    verbose: bool = False,
) -> tuple[str, bool]:
    corr_list = []

    for i in range(len(node_chain) - 1):
        n1 = node_chain[i]
        n2 = node_chain[i + 1]

        s1 = node_to_series.get(n1)
        s2 = node_to_series.get(n2)

        if s1 is None or s2 is None:
            continue

        corr = compute_corr(s1, s2)

        if verbose:
            logger.debug("corr %s -> %s = %.3f", n1, n2, corr)

        if np.isfinite(corr):
            corr_list.append((corr, n1, n2))

    if len(corr_list) == 0:
        return node_chain[len(node_chain) // 2], False

    corr_min, n1_min, n2_min = min(corr_list, key=lambda x: x[0])

    if corr_min < corr_threshold:
        if do_plot:
            plot_corr_break(
                node_to_series[n1_min],
                node_to_series[n2_min],
                n1_min,
                n2_min,
                corr_min,
                plant_index,
            )

        return n1_min, True

    return node_chain[len(node_chain) // 2], False


def correct_storage_matches(
    matched_df: pd.DataFrame,
    df_drain: pd.DataFrame,
    river_bbox: xr.DataArray,
    storage_plant_indices: set,
    window: int,
    area_jump_threshold: float,
    corr_threshold: float,
    do_plot: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    if matched_df.empty:
        return matched_df

    corrected = matched_df.copy()
    corrected["correction_status"] = "original"
    drain_by_id = df_drain.set_index("model_id", drop=False)

    n_updated = 0

    for idx, match in corrected.iterrows():
        plant_index = match["plant_index"]

        if plant_index not in storage_plant_indices:
            continue

        model_id = match.get("model_id")

        if pd.isna(model_id) or model_id not in drain_by_id.index:
            continue

        center_id = str(model_id)
        node_chain = [center_id]

        current_id = center_id
        current_area = float(drain_by_id.loc[current_id, "drainage_area"])

        for _ in range(window):
            prev_ids = parse_upstream_ids(
                drain_by_id.loc[current_id, "upstream_model_ids"]
            )

            if not prev_ids:
                break

            prev_id = choose_prev_by_area_similarity(
                current_id=current_id,
                prev_candidates=prev_ids,
                drain_by_id=drain_by_id,
            )

            if prev_id is None:
                break

            prev_area = float(drain_by_id.loc[prev_id, "drainage_area"])

            if abs(prev_area - current_area) / current_area > area_jump_threshold:
                break

            node_chain.insert(0, prev_id)
            current_id = prev_id
            current_area = prev_area

        current_id = center_id
        current_area = float(drain_by_id.loc[current_id, "drainage_area"])

        for _ in range(window):
            succ_id = drain_by_id.loc[current_id, "downstream_model_id"]

            if pd.isna(succ_id) or succ_id is None:
                break

            succ_id = str(succ_id)

            if succ_id not in drain_by_id.index:
                break

            succ_area = float(drain_by_id.loc[succ_id, "drainage_area"])

            if abs(succ_area - current_area) / current_area > area_jump_threshold:
                break

            node_chain.append(succ_id)
            current_id = succ_id
            current_area = succ_area

        node_to_series = {}

        for node_id in node_chain:
            row = drain_by_id.loc[node_id]

            da = river_bbox.sel(
                longitude=float(row["x"]),
                latitude=float(row["y"]),
                method="nearest",
            )

            node_to_series[node_id] = da.to_series()

        final_node_id, break_detected = choose_best_node_by_global_corr(
            node_chain=node_chain,
            node_to_series=node_to_series,
            corr_threshold=corr_threshold,
            plant_index=plant_index,
            do_plot=do_plot,
            verbose=verbose,
        )

        if break_detected:
            best_row = drain_by_id.loc[final_node_id]

            corrected.loc[idx, "model_id"] = final_node_id
            corrected.loc[idx, "matched_lon"] = float(best_row["x"])
            corrected.loc[idx, "matched_lat"] = float(best_row["y"])
            corrected.loc[idx, "distance_km"] = np.nan
            corrected.loc[idx, "diff_qmax"] = np.nan
            corrected.loc[idx, "combined_score"] = np.nan
            corrected.loc[idx, "correction_status"] = "updated_storage"
            corrected.loc[idx, "match_type"] = "drain_storage_corrected"

            n_updated += 1

    logger.info("Storage plants hydrologically corrected: %s", n_updated)

    return corrected

def plot_hydro_matches_on_drainage_network_plotly(
    df_drain: pd.DataFrame,
    hydro_ppls: pd.DataFrame,
    final_matches: pd.DataFrame,
    output_html_path: Path,
    title: str = "Hydropower plants matched to GloFAS drainage network - Zambezi",
    max_hover_chars: int = 100,
) -> None:
    """
    Plot the drainage network, original hydropower plant locations,
    and matched GloFAS/drainage points.

    The plot includes:
    - drainage nodes colored by drainage area
    - downstream drainage links
    - original hydropower plant coordinates in green
    - matched GLoFAS/drainage points in red
    """

    required_drain_cols = {
        "model_id",
        "downstream_model_id",
        "drainage_area",
        "x",
        "y",
    }

    missing_drain_cols = required_drain_cols - set(df_drain.columns)
    if missing_drain_cols:
        raise ValueError(
            f"Missing required columns in df_drain: {sorted(missing_drain_cols)}"
        )

    required_plant_cols = {
        "lon",
        "lat",
    }

    missing_plant_cols = required_plant_cols - set(hydro_ppls.columns)
    if missing_plant_cols:
        raise ValueError(
            f"Missing required columns in hydro_ppls: {sorted(missing_plant_cols)}"
        )

    required_match_cols = {
        "plant_index",
        "matched_lon",
        "matched_lat",
        "model_id",
        "match_type",
    }

    missing_match_cols = required_match_cols - set(final_matches.columns)
    if missing_match_cols:
        raise ValueError(
            f"Missing required columns in final_matches: {sorted(missing_match_cols)}"
        )

    df_plot = df_drain.copy()
    df_plot["model_id"] = df_plot["model_id"].astype(str)

    coord_map = dict(
        zip(
            df_plot["model_id"],
            zip(df_plot["x"].astype(float), df_plot["y"].astype(float)),
        )
    )

    edge_x = []
    edge_y = []
    n_edges = 0
    n_missing_downstream = 0

    for _, row in df_plot.iterrows():
        model_id = row["model_id"]
        downstream_model_id = row["downstream_model_id"]

        if pd.isna(downstream_model_id) or downstream_model_id is None:
            continue

        downstream_model_id = str(downstream_model_id)

        if downstream_model_id not in coord_map:
            n_missing_downstream += 1
            continue

        x1, y1 = coord_map[model_id]
        x2, y2 = coord_map[downstream_model_id]

        edge_x.extend([x1, x2, None])
        edge_y.extend([y1, y2, None])
        n_edges += 1

    drain_hover_text = []

    for _, row in df_plot.iterrows():
        downstream_model_id = row["downstream_model_id"]

        if pd.isna(downstream_model_id) or downstream_model_id is None:
            downstream_text = "None"
        else:
            downstream_text = str(downstream_model_id)

        if len(downstream_text) > max_hover_chars:
            downstream_text = downstream_text[:max_hover_chars] + "..."

        hover_items = [
            f"model_id: {row['model_id']}",
            f"downstream_model_id: {downstream_text}",
            f"Drainage area: {float(row['drainage_area']):,.2f} km²",
            f"Longitude: {float(row['x']):.5f}",
            f"Latitude: {float(row['y']):.5f}",
        ]

        if "strahler_order" in row and pd.notna(row["strahler_order"]):
            hover_items.insert(2, f"Strahler order: {int(row['strahler_order'])}")

        drain_hover_text.append("<br>".join(hover_items))

    blues_no_white = sample_colorscale(
        "Blues",
        [0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0],
    )

    edge_trace = go.Scattergl(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(
            width=1,
            color="rgba(0, 0, 0, 0.25)",
        ),
        hoverinfo="skip",
        name="Downstream links",
    )

    drain_node_trace = go.Scattergl(
        x=df_plot["x"],
        y=df_plot["y"],
        mode="markers",
        marker=dict(
            size=4,
            color=df_plot["drainage_area"],
            colorscale=blues_no_white,
            showscale=True,
            colorbar=dict(
                title="Drainage area<br>(km²)",
            ),
            opacity=0.75,
        ),
        text=drain_hover_text,
        hoverinfo="text",
        name="Drainage nodes",
    )

    plants_plot = hydro_ppls.copy()
    plants_plot = plants_plot.dropna(subset=["lon", "lat"]).copy()

    plant_names = []

    for idx, row in plants_plot.iterrows():
        if "Name" in plants_plot.columns and pd.notna(row["Name"]):
            plant_name = str(row["Name"])
        elif "name" in plants_plot.columns and pd.notna(row["name"]):
            plant_name = str(row["name"])
        elif "PlantName" in plants_plot.columns and pd.notna(row["PlantName"]):
            plant_name = str(row["PlantName"])
        else:
            plant_name = str(idx)

        hover_items = [
            f"plant_index: {idx}",
            f"plant_name: {plant_name}",
            f"Original longitude: {float(row['lon']):.5f}",
            f"Original latitude: {float(row['lat']):.5f}",
        ]

        if "Technology" in plants_plot.columns and pd.notna(row["Technology"]):
            hover_items.append(f"Technology: {row['Technology']}")

        if "Capacity" in plants_plot.columns and pd.notna(row["Capacity"]):
            hover_items.append(f"Capacity: {float(row['Capacity']):,.2f} MW")

        if "qmax_turb" in plants_plot.columns and pd.notna(row["qmax_turb"]):
            hover_items.append(f"qmax_turb: {float(row['qmax_turb']):,.2f} m³/s")

        plant_names.append("<br>".join(hover_items))

    original_plants_trace = go.Scattergl(
        x=plants_plot["lon"],
        y=plants_plot["lat"],
        mode="markers",
        marker=dict(
            size=10,
            color="green",
            symbol="circle",
            line=dict(
                width=1,
                color="black",
            ),
            opacity=0.90,
        ),
        text=plant_names,
        hoverinfo="text",
        name="Original hydropower plants",
    )

    matches_plot = final_matches.copy()
    matches_plot = matches_plot.dropna(subset=["matched_lon", "matched_lat"]).copy()

    match_hover_text = []

    hydro_lookup = hydro_ppls.copy()

    for _, row in matches_plot.iterrows():
        plant_index = row["plant_index"]

        hover_items = [
            f"plant_index: {plant_index}",
            f"matched model_id: {row['model_id']}",
            f"Matched longitude: {float(row['matched_lon']):.5f}",
            f"Matched latitude: {float(row['matched_lat']):.5f}",
            f"match_type: {row['match_type']}",
        ]

        if "distance_km" in matches_plot.columns and pd.notna(row["distance_km"]):
            hover_items.append(f"distance_km: {float(row['distance_km']):.2f}")

        if "river_mean" in matches_plot.columns and pd.notna(row["river_mean"]):
            hover_items.append(f"river_mean: {float(row['river_mean']):,.2f} m³/s")

        if "correction_status" in matches_plot.columns and pd.notna(row["correction_status"]):
            hover_items.append(f"correction_status: {row['correction_status']}")

        if plant_index in hydro_lookup.index:
            plant_row = hydro_lookup.loc[plant_index]

            if "Name" in hydro_lookup.columns and pd.notna(plant_row.get("Name")):
                hover_items.insert(1, f"plant_name: {plant_row['Name']}")
            elif "name" in hydro_lookup.columns and pd.notna(plant_row.get("name")):
                hover_items.insert(1, f"plant_name: {plant_row['name']}")
            elif "PlantName" in hydro_lookup.columns and pd.notna(plant_row.get("PlantName")):
                hover_items.insert(1, f"plant_name: {plant_row['PlantName']}")

            if "Technology" in hydro_lookup.columns and pd.notna(plant_row.get("Technology")):
                hover_items.append(f"Technology: {plant_row['Technology']}")

            if "Capacity" in hydro_lookup.columns and pd.notna(plant_row.get("Capacity")):
                hover_items.append(f"Capacity: {float(plant_row['Capacity']):,.2f} MW")

        match_hover_text.append("<br>".join(hover_items))

    matched_points_trace = go.Scattergl(
        x=matches_plot["matched_lon"],
        y=matches_plot["matched_lat"],
        mode="markers",
        marker=dict(
            size=11,
            color="red",
            symbol="x",
            line=dict(
                width=2,
                color="red",
            ),
            opacity=0.95,
        ),
        text=match_hover_text,
        hoverinfo="text",
        name="Matched GLoFAS/drain points",
    )

    match_line_x = []
    match_line_y = []

    for _, row in matches_plot.iterrows():
        plant_index = row["plant_index"]

        if plant_index not in hydro_ppls.index:
            continue

        plant_row = hydro_ppls.loc[plant_index]

        if pd.isna(plant_row["lon"]) or pd.isna(plant_row["lat"]):
            continue

        x1 = float(plant_row["lon"])
        y1 = float(plant_row["lat"])
        x2 = float(row["matched_lon"])
        y2 = float(row["matched_lat"])

        match_line_x.extend([x1, x2, None])
        match_line_y.extend([y1, y2, None])

    plant_to_match_trace = go.Scattergl(
        x=match_line_x,
        y=match_line_y,
        mode="lines",
        line=dict(
            width=1,
            color="rgba(255, 0, 0, 0.35)",
            dash="dot",
        ),
        hoverinfo="skip",
        name="Plant-to-match links",
    )

    fig = go.Figure(
        data=[
            edge_trace,
            drain_node_trace,
            plant_to_match_trace,
            original_plants_trace,
            matched_points_trace,
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        template="plotly_white",
        width=1500,
        height=800,
        showlegend=True,
        hovermode="closest",
        margin=dict(l=40, r=40, t=70, b=40),
    )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    fig.write_html(
        str(output_html_path),
        include_plotlyjs="cdn",
        full_html=True,
        auto_open=False,
    )

    logger.info("Hydro match diagnostic Plotly HTML saved: %s", output_html_path)

    if n_missing_downstream > 0:
        logger.warning(
            "%s downstream links were skipped because downstream_model_id was not found in df_drain.",
            f"{n_missing_downstream:,}",
        )

# ============================================================
# SERIES EXTRACTION
# ============================================================

def build_final_matches(matched_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "plant_index",
        "model_id",
        "HYBAS_ID",
        "cluster",
        "node",
        "matched_lon",
        "matched_lat",
        "distance_km",
        "river_mean",
        "match_type",
        "correction_status",
    ]

    if matched_df.empty:
        return pd.DataFrame(columns=cols)

    return matched_df.reindex(columns=cols).copy()


def extract_glofas_series(
    final_matches: pd.DataFrame,
    glofas_dir: Path,
    years: list[int],
    var_name: str,
    file_template: str,
) -> dict[int, xr.DataArray]:
    plant_names = final_matches["plant_index"].astype(str).tolist()
    plant_lats = final_matches["matched_lat"].to_numpy(dtype=float)
    plant_lons = final_matches["matched_lon"].to_numpy(dtype=float)

    all_years_series_clean = {}

    ds0_path = glofas_dir / file_template.format(year=years[0])

    with xr.open_dataset(ds0_path) as ds0:
        river0 = ds0[var_name]

        lat_grid = river0["latitude"].values
        lon_grid = river0["longitude"].values

        lat_idx = np.abs(lat_grid[:, None] - plant_lats).argmin(axis=0)
        lon_idx = np.abs(lon_grid[:, None] - plant_lons).argmin(axis=0)

    bad_plants_mask = None
    all_nan_by_year = {}

    for i, year in enumerate(years):
        logger.info("Processing GLoFAS year %s: %s", year, glofas_dir / file_template.format(year=year))

        nc_path = glofas_dir / file_template.format(year=year)

        with xr.open_dataset(nc_path) as ds:
            river = ds[var_name]
            time_dim = "valid_time" if "valid_time" in river.dims else "time"

            data = river.values
            extracted = data[:, lat_idx, lon_idx].T

            time_vals = pd.to_datetime(river[time_dim].values) - pd.Timedelta(days=1)

            mask = ~((time_vals.month == 2) & (time_vals.day == 29))
            extracted = extracted[:, mask]
            time_vals = time_vals[mask]

            year_all_nan = np.all(np.isnan(extracted), axis=1)
            all_nan_by_year[year] = year_all_nan

            if i == 0:
                bad_plants_mask = year_all_nan.copy()

                logger.debug("Series fully NaN in first year: %s", int(bad_plants_mask.sum()))

                if bad_plants_mask.any():
                    bad_df = pd.DataFrame(
                        {
                            "plant": np.array(plant_names)[bad_plants_mask],
                            "matched_lon": plant_lons[bad_plants_mask],
                            "matched_lat": plant_lats[bad_plants_mask],
                        }
                    )

                    logger.debug("Plants with fully NaN series in first year:")
                    logger.debug(bad_df.to_string(index=False))

            if bad_plants_mask is not None and bad_plants_mask.any():
                extracted[bad_plants_mask, :] = 0.1

            da_year = xr.DataArray(
                extracted,
                dims=["plant", time_dim],
                coords={
                    "plant": plant_names,
                    time_dim: time_vals,
                },
                name="river_series",
            )

            all_years_series_clean[year] = da_year

    return all_years_series_clean


# ============================================================
# SABER INPUT
# ============================================================

def get_time_dim(da: xr.DataArray) -> str:
    """
    Return the temporal dimension name used by a GLoFAS DataArray.
    """
    if "valid_time" in da.dims:
        return "valid_time"
    if "time" in da.dims:
        return "time"

    non_plant_dims = [dim for dim in da.dims if dim != "plant"]

    if len(non_plant_dims) != 1:
        raise ValueError(f"Cannot infer time dimension from dims: {da.dims}")

    return non_plant_dims[0]


def prepare_rivid_map(
    final_matches: pd.DataFrame,
    all_years_series_clean: dict[int, xr.DataArray],
) -> pd.DataFrame:
    """
    Build the plant-to-rivid mapping directly from final_matches.

    No additional graph matching is needed because final_matches already
    contains the selected GLoFAS/SABER point for each hydro plant.
    """
    if final_matches.empty:
        raise ValueError("final_matches is empty. Cannot build rivid map.")

    year_ref = sorted(all_years_series_clean.keys())[0]
    da_ref = all_years_series_clean[year_ref]

    valid_plants = set(da_ref.coords["plant"].values.astype(str))

    rivid_map = final_matches.copy()
    rivid_map["plant"] = rivid_map["plant_index"].astype(str)

    rivid_map = rivid_map[rivid_map["plant"].isin(valid_plants)].copy()

    if rivid_map.empty:
        raise ValueError("No final_matches plants are present in all_years_series_clean.")

    rivid_map = rivid_map.sort_values("plant").reset_index(drop=True)
    rivid_map["rivid"] = np.arange(len(rivid_map), dtype=int)

    keep_cols = [
        "rivid",
        "plant",
        "plant_index",
        "model_id",
        "matched_lon",
        "matched_lat",
        "match_type",
    ]

    optional_cols = [
        "HYBAS_ID",
        "cluster",
        "node",
        "distance_km",
        "river_mean",
        "correction_status",
    ]

    keep_cols += [col for col in optional_cols if col in rivid_map.columns]

    return rivid_map[keep_cols].copy()


def build_saber_qsim_dataset(
    all_years_series_clean: dict[int, xr.DataArray],
    rivid_map: pd.DataFrame,
) -> xr.Dataset:
    """
    Build Qsim(time, rivid) dataset from extracted GLoFAS series.
    """
    plant_order = rivid_map.sort_values("rivid")["plant"].astype(str).tolist()
    rivid_order = rivid_map.sort_values("rivid")["rivid"].to_numpy(dtype=int)

    yearly_arrays = []

    for year in sorted(all_years_series_clean):
        da = all_years_series_clean[year]

        time_dim = get_time_dim(da)

        if time_dim != "time":
            da = da.rename({time_dim: "time"})

        available_plants = set(da.coords["plant"].values.astype(str))
        missing = [plant for plant in plant_order if plant not in available_plants]

        if missing:
            raise ValueError(
                f"Year {year}: {len(missing)} plants are missing from extracted series. "
                f"First missing plants: {missing[:10]}"
            )

        da = da.sel(plant=plant_order)
        yearly_arrays.append(da)

    da_all = xr.concat(yearly_arrays, dim="time")
    da_all = da_all.sortby("time")

    _, unique_index = np.unique(da_all["time"].values, return_index=True)
    da_all = da_all.isel(time=np.sort(unique_index))

    qsim = da_all.transpose("time", "plant").values.astype("float32")

    ds = xr.Dataset(
        data_vars={
            "Qsim": (("time", "rivid"), qsim),
        },
        coords={
            "time": pd.to_datetime(da_all["time"].values),
            "rivid": rivid_order,
        },
    )

    return ds


def save_outputs(
    final_matches: pd.DataFrame,
    all_years_series_clean: dict[int, xr.DataArray],
    final_matches_output_path: Path,
    series_output_path: Path,
    rivid_map_output_path: Path,
    zarr_output_path: Path,
) -> None:
    """
    Save intermediate outputs plus final SABER-ready files.
    """
    final_matches_output_path.parent.mkdir(parents=True, exist_ok=True)
    series_output_path.parent.mkdir(parents=True, exist_ok=True)
    rivid_map_output_path.parent.mkdir(parents=True, exist_ok=True)
    zarr_output_path.parent.mkdir(parents=True, exist_ok=True)

    final_matches.to_parquet(final_matches_output_path, index=False)

    with open(series_output_path, "wb") as f:
        pickle.dump(all_years_series_clean, f)

    rivid_map = prepare_rivid_map(
        final_matches=final_matches,
        all_years_series_clean=all_years_series_clean,
    )

    rivid_map.to_csv(rivid_map_output_path, index=False)

    ds_saber = build_saber_qsim_dataset(
        all_years_series_clean=all_years_series_clean,
        rivid_map=rivid_map,
    )

    ds_saber.to_zarr(zarr_output_path, mode="w")

    logger.info("Final matches output saved: %s", final_matches_output_path)
    logger.info("Series output saved: %s", series_output_path)
    logger.info("Rivid map output saved: %s", rivid_map_output_path)
    logger.info("SABER Zarr output saved: %s", zarr_output_path)
    logger.debug("SABER dataset:\n%s", ds_saber)

# ============================================================
# WORKFLOW
# ============================================================

def extract_glofas_series_for_hydro_plants(cfg: dict | None = None) -> None:
    cfg = cfg or get_config()

    log_config_summary(cfg)

    t0 = time.time()

    lon_min = cfg["lon_min"]
    lon_max = cfg["lon_max"]
    lat_min = cfg["lat_min"]
    lat_max = cfg["lat_max"]

    drain_full_path = cfg["drain_full_path"]
    plants_path = cfg["plants_path"]
    river_reference_path = cfg["river_reference_path"]

    glofas_data_dir = cfg["hydro_data_dir"]
    glofas_file_template = cfg["hydro_file_template"]
    glofas_var_name = cfg["hydro_var_name"]
    years = cfg["years"]

    final_matches_output_path = cfg["final_matches_output_path"]
    series_output_path = cfg["series_output_path"]
    rivid_map_output_path = cfg["rivid_map_output_path"]
    zarr_output_path = cfg["zarr_output_path"]

    diagnostic_plot_dir = cfg["base_output_dir"] / "diagnostic_plots"
    match_plot_html_output_path = diagnostic_plot_dir / (
        f"hydro_plants_{cfg['dataset']}_matches_drain_network.html"
    )

    make_match_plot = os.environ.get("HYDRO_MAKE_DIAGNOSTIC_PLOTS", "0") == "1"
    show_corr_break_plots = os.environ.get("HYDRO_SHOW_PLOTS", "0") == "1"
    debug = os.environ.get("HYDRO_LOG_LEVEL", "INFO").upper() == "DEBUG"

    hydro_ppls = load_hydro_power_plants(
        plants_path=plants_path,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        efficiency=EFFICIENCY,
    )

    river_bbox, river_mean = build_river_bbox_and_mean(
        hydro_ppls=hydro_ppls,
        river_reference_path=river_reference_path,
        var_name=glofas_var_name,
        margin_deg=MARGIN_DEG,
    )

    plants_with_storage = build_storage_plants(hydro_ppls)

    df_drain = load_drain_table(drain_full_path)

    plants_gdf = prepare_plants_gdf(hydro_ppls)

    matched_df = match_plants_to_drain(
        plants_gdf=plants_gdf,
        df_drain=df_drain,
        river_mean=river_mean,
        max_distance_km=MAX_DISTANCE_KM,
        k_neigh=K_NEIGH_FALLBACK,
    )

    storage_plant_indices = set(plants_with_storage.index)

    matched_df = correct_storage_matches(
        matched_df=matched_df,
        df_drain=df_drain,
        river_bbox=river_bbox,
        storage_plant_indices=storage_plant_indices,
        window=WINDOW,
        area_jump_threshold=AREA_JUMP_THRESHOLD,
        corr_threshold=CORR_THRESHOLD,
        do_plot=show_corr_break_plots,
        verbose=debug,
    )

    final_matches = build_final_matches(matched_df)

    logger.info("Total plants for extraction: %s", len(final_matches))

    all_years_series_clean = extract_glofas_series(
        final_matches=final_matches,
        glofas_dir=glofas_data_dir,
        years=years,
        var_name=glofas_var_name,
        file_template=glofas_file_template,
    )

    save_outputs(
        final_matches=final_matches,
        all_years_series_clean=all_years_series_clean,
        final_matches_output_path=final_matches_output_path,
        series_output_path=series_output_path,
        rivid_map_output_path=rivid_map_output_path,
        zarr_output_path=zarr_output_path,
    )

    if make_match_plot:
        plot_hydro_matches_on_drainage_network_plotly(
            df_drain=df_drain,
            hydro_ppls=hydro_ppls,
            final_matches=final_matches,
            output_html_path=match_plot_html_output_path,
            title=(
                f"Hydropower plants matched to {cfg['dataset'].upper()} "
                "drainage network"
            ),
        )
    else:
        logger.debug(
            "Hydro plant match diagnostic plot disabled. "
            "Set HYDRO_MAKE_DIAGNOSTIC_PLOTS=1 to enable it."
        )

    logger.info("Hydropower plant series extraction completed.")
    logger.debug("Total time: %.1f s", time.time() - t0)


def main() -> None:
    setup_logging()
    extract_glofas_series_for_hydro_plants()


if __name__ == "__main__":
    main()