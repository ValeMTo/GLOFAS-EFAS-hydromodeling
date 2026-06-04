# This script loads GRDC stations and links them to the most suitable
# EFAS drainage/grid point. It then extracts the corresponding EFAS discharge time
# series, producing model and observed time series for each station.

from __future__ import annotations

from pathlib import Path
import logging
import os
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

from scipy.spatial import cKDTree
from shapely.geometry import Point

from hydro_inflow.hydro_config import get_config, log_config_summary
from hydro_inflow.utils import (
    approx_dist_km,
    km_to_deg_lat,
    km_to_deg_lon,
    setup_logging,
)


logger = logging.getLogger(__name__)

# ======================================================
# CONSTANTS
# ======================================================

K_NEIGHBORS = 6

RATIO_MIN = 0.8
RATIO_MAX = 1.2

GRDC_VAR_NAME = "runoff_mean"

START_DATE = "1992-01-01"
END_DATE = "2025-12-31"

MIN_OBS_PER_MONTH = 100

MAX_DRAIN_DISTANCE_KM = 10.0
DRAIN_SCORE_DISTANCE_WEIGHT = 0.5
DRAIN_SCORE_AREA_WEIGHT = 0.5

# ======================================================
# LOAD GRDC METADATA
# ======================================================

def load_grdc_metadata(
    file_path: Path,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    threshold_uparea_km2: float,
) -> tuple[xr.Dataset, pd.DataFrame]:
    ds = xr.open_dataset(file_path)

    grdc_df = pd.DataFrame(
        {
            "id": ds["id"].values,
            "StationName": ds["station_name"].values,
            "RiverName": ds["river_name"].values,
            "Country": ds["country"].values,
            "Area_km2": ds["area"].values,
            "Longitude": ds["geo_x"].values,
            "Latitude": ds["geo_y"].values,
        }
    )

    logger.debug("GRDC stations loaded: %s", len(grdc_df))

    grdc_df["Area_km2"] = pd.to_numeric(grdc_df["Area_km2"], errors="coerce")
    grdc_df["Longitude"] = pd.to_numeric(grdc_df["Longitude"], errors="coerce")
    grdc_df["Latitude"] = pd.to_numeric(grdc_df["Latitude"], errors="coerce")

    grdc_df = grdc_df.dropna(
        subset=["Area_km2", "Longitude", "Latitude"]
    ).copy()

    grdc_df = grdc_df[
        (grdc_df["Longitude"] >= lon_min)
        & (grdc_df["Longitude"] <= lon_max)
        & (grdc_df["Latitude"] >= lat_min)
        & (grdc_df["Latitude"] <= lat_max)
    ].copy()

    logger.debug("GRDC stations filtered to model domain: %d", len(grdc_df))

    grdc_df = grdc_df[
        grdc_df["Area_km2"] > threshold_uparea_km2
    ].reset_index(drop=True)

    logger.debug("GRDC stations with area > %g km2: %d", threshold_uparea_km2, len(grdc_df))

    return ds, grdc_df


# ======================================================
# FILTER STATIONS WITH ENOUGH GRDC COVERAGE
# ======================================================

def filter_saber_ready_stations(
    ds: xr.Dataset,
    grdc_df: pd.DataFrame,
    var_name: str,
    start_date: str,
    end_date: str,
    min_obs_per_month_threshold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if var_name not in ds.data_vars:
        raise ValueError(f"Variable '{var_name}' not found in GRDC dataset.")

    logger.debug("Stations before Qobs coverage filter: %d", len(grdc_df))

    q_sub = ds[var_name].sel(time=slice(start_date, end_date))
    q_sub = q_sub.where(q_sub >= 0)

    valid_per_month = q_sub.notnull().groupby("time.month").sum("time")
    min_obs_per_month = valid_per_month.min("month")

    coverage_df = (
        min_obs_per_month.sel(id=grdc_df["id"].values)
        .to_dataframe(name="MinObsPerMonth")
        .reset_index()
    )

    df_coverage = grdc_df.merge(coverage_df, on="id", how="left")

    df_final = df_coverage[
        df_coverage["MinObsPerMonth"] >= min_obs_per_month_threshold
    ].reset_index(drop=True)

    logger.debug(
        "Stations with at least %d observations per month: %d",
        min_obs_per_month_threshold,
        len(df_final),
    )

    return df_coverage, df_final


# ======================================================
# LOAD EFAS DRAIN TABLE
# ======================================================

def load_drain_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    required_cols = {
        "model_id",
        "downstream_model_id",
        "drainage_area",
        "x",
        "y",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in drain table: {sorted(missing)}")

    df = df.copy()

    df["model_id"] = df["model_id"].astype(str)
    df["drainage_area"] = pd.to_numeric(df["drainage_area"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    if "HYBAS_ID" in df.columns:
        df["HYBAS_ID"] = df["HYBAS_ID"].astype(str)

    if "upstream_model_ids" in df.columns:
        df["upstream_model_ids"] = df["upstream_model_ids"].fillna("").astype(str)

    df = df.dropna(subset=["model_id", "drainage_area", "x", "y"]).copy()

    logger.debug("EFAS drain nodes loaded: %d", len(df))

    return df


# ======================================================
# MATCH GRDC STATIONS DIRECTLY TO EFAS DRAIN
# ======================================================

def score_drain_candidates(
    candidates: pd.DataFrame,
    station_lon: float,
    station_lat: float,
    target_area: float,
    max_distance_km: float,
    distance_weight: float,
    area_weight: float,
) -> pd.DataFrame:
    candidates = candidates.copy()

    candidates["distance_km"] = approx_dist_km(
        lon=station_lon,
        lat=station_lat,
        lon2=candidates["x"].to_numpy(dtype=float),
        lat2=candidates["y"].to_numpy(dtype=float),
    )

    candidates = candidates[
        candidates["distance_km"] <= max_distance_km
    ].copy()

    if candidates.empty:
        return candidates

    if target_area > 0 and np.isfinite(target_area):
        candidates["area_ratio"] = candidates["drainage_area"] / target_area
        candidates["area_rel_diff"] = (
            (candidates["drainage_area"] - target_area).abs() / target_area
        )

        candidates = candidates[
            (candidates["area_ratio"] >= RATIO_MIN)
            & (candidates["area_ratio"] <= RATIO_MAX)
        ].copy()

        if candidates.empty:
            return candidates
    else:
        candidates["area_ratio"] = np.nan
        candidates["area_rel_diff"] = 0.0

    candidates["dist_norm"] = candidates["distance_km"] / max_distance_km

    max_area_diff = float(candidates["area_rel_diff"].max())
    candidates["area_norm"] = (
        0.0 if max_area_diff == 0 else candidates["area_rel_diff"] / max_area_diff
    )

    candidates["combined_score"] = (
        distance_weight * candidates["dist_norm"]
        + area_weight * candidates["area_norm"]
    )

    return candidates


def build_match_row(
    station: pd.Series,
    best: pd.Series,
    candidate_search_mode: str,
) -> dict:
    gauge_id = str(station["gauge_uid"])
    lon = float(station["Longitude"])
    lat = float(station["Latitude"])
    target_area = float(station["Area_km2"])

    return {
        "gauge_id": gauge_id,
        "plant_lat": float(best["y"]),
        "plant_lon": float(best["x"]),
        "HYBAS_ID": str(best["HYBAS_ID"]) if "HYBAS_ID" in best and pd.notna(best["HYBAS_ID"]) else None,
        "cluster": (
            int(best["cluster"])
            if "cluster" in best and pd.notna(best["cluster"])
            else None
        ),
        "node_id": (
            int(best["node"])
            if "node" in best and pd.notna(best["node"])
            else (
                int(best["node_id"])
                if "node_id" in best and pd.notna(best["node_id"])
                else None
            )
        ),
        "model_id": str(best["model_id"]),
        "match_type": "efas_drain_nearest_then_radius_area_ratio",
        "candidate_search_mode": candidate_search_mode,
        "dist_m": float(best["distance_km"]) * 1000.0,
        "distance_km": float(best["distance_km"]),
        "station_area_km2": target_area,
        "drainage_area_km2": float(best["drainage_area"]),
        "area_ratio": float(best["area_ratio"]) if "area_ratio" in best else np.nan,
        "area_rel_diff": float(best["area_rel_diff"]),
        "combined_score": float(best["combined_score"]),
        "station_name": station["StationName"],
        "river": station["RiverName"],
        "country": station["Country"],
        "grdc_lon": lon,
        "grdc_lat": lat,
        "efas_lon": float(best["x"]),
        "efas_lat": float(best["y"]),
        "min_obs_per_month": int(station["MinObsPerMonth"]),
    }


def match_stations_to_drain(
    stations: pd.DataFrame,
    df_drain: pd.DataFrame,
    max_distance_km: float,
    distance_weight: float,
    area_weight: float,
    k_neighbors: int,
) -> pd.DataFrame:
    rows = []
    no_match_rows = []

    drain = df_drain.copy().reset_index(drop=True)

    if drain.empty:
        raise ValueError("EFAS drain table is empty after cleaning.")

    drain_coords = drain[["x", "y"]].to_numpy(dtype=float)
    drain_tree = cKDTree(drain_coords)

    k_query = min(k_neighbors, len(drain))

    for _, station in stations.iterrows():
        gauge_id = str(station["gauge_uid"])
        lon = float(station["Longitude"])
        lat = float(station["Latitude"])
        target_area = float(station["Area_km2"])

        _, nearest_idx = drain_tree.query([lon, lat], k=k_query)
        nearest_idx = np.atleast_1d(nearest_idx).astype(int)

        nearest_candidates = drain.iloc[nearest_idx].copy()

        nearest_candidates = score_drain_candidates(
            candidates=nearest_candidates,
            station_lon=lon,
            station_lat=lat,
            target_area=target_area,
            max_distance_km=max_distance_km,
            distance_weight=distance_weight,
            area_weight=area_weight,
        )

        if not nearest_candidates.empty:
            best = nearest_candidates.loc[nearest_candidates["combined_score"].idxmin()]
            rows.append(
                build_match_row(
                    station=station,
                    best=best,
                    candidate_search_mode=f"nearest_{k_query}_drain_nodes",
                )
            )
            continue

        buf_lon = km_to_deg_lon(max_distance_km, lat)
        buf_lat = km_to_deg_lat(max_distance_km)

        radius_candidates = drain[
            (drain["x"] >= lon - buf_lon)
            & (drain["x"] <= lon + buf_lon)
            & (drain["y"] >= lat - buf_lat)
            & (drain["y"] <= lat + buf_lat)
        ].copy()

        if radius_candidates.empty:
            no_match_rows.append(
                {
                    "gauge_id": gauge_id,
                    "station_lat": lat,
                    "station_lon": lon,
                    "reason": f"no EFAS drain node within {max_distance_km:.0f} km bbox",
                }
            )
            continue

        radius_candidates = score_drain_candidates(
            candidates=radius_candidates,
            station_lon=lon,
            station_lat=lat,
            target_area=target_area,
            max_distance_km=max_distance_km,
            distance_weight=distance_weight,
            area_weight=area_weight,
        )

        if radius_candidates.empty:
            no_match_rows.append(
                {
                    "gauge_id": gauge_id,
                    "station_lat": lat,
                    "station_lon": lon,
                    "reason": (
                        f"EFAS drain nodes found within {max_distance_km:.0f} km, "
                        f"but none with area_ratio in [{RATIO_MIN}, {RATIO_MAX}]"
                    ),
                }
            )
            continue

        best = radius_candidates.loc[radius_candidates["combined_score"].idxmin()]

        rows.append(
            build_match_row(
                station=station,
                best=best,
                candidate_search_mode=f"radius_{max_distance_km:.0f}_km",
            )
        )

    station_model_map = pd.DataFrame(rows)
    no_match_df = pd.DataFrame(no_match_rows)

    logger.info("Station to EFAS drain matching completed.")
    logger.info("Stations mapped: %s", len(station_model_map))
    logger.info("Stations without EFAS drain match: %s", len(no_match_df))

    if not no_match_df.empty:
        logger.debug(
            "Stations without EFAS drain match preview:\n%s",
            no_match_df.head(20).to_string(index=False),
        )

    if not station_model_map.empty:
        logger.debug(
            "Candidate search mode counts:\n%s",
            station_model_map["candidate_search_mode"].value_counts().to_string(),
        )

    return station_model_map


def log_drain_match_diagnostics(station_model_map: pd.DataFrame) -> None:
    if station_model_map.empty:
        logger.debug("Drain match diagnostics skipped: station_model_map is empty.")
        return

    logger.debug("EFAS drain match distance diagnostics")

    logger.debug(
        "Drain match distance summary [m]:\n%s",
        station_model_map["dist_m"].describe().to_string(),
    )

    logger.debug(
        "Stations with drain distance > 1 km: %s",
        int((station_model_map["dist_m"] > 1_000).sum()),
    )
    logger.debug(
        "Stations with drain distance > 5 km: %s",
        int((station_model_map["dist_m"] > 5_000).sum()),
    )
    logger.debug(
        "Stations with drain distance > 10 km: %s",
        int((station_model_map["dist_m"] > 10_000).sum()),
    )
    logger.debug(
        "Stations with drain distance > 20 km: %s",
        int((station_model_map["dist_m"] > 20_000).sum()),
    )

    logger.debug(
        "Drain match area relative difference summary:\n%s",
        station_model_map["area_rel_diff"].describe().to_string(),
    )

    logger.debug(
        "Stations with area_rel_diff > 0.2: %s",
        int((station_model_map["area_rel_diff"] > 0.2).sum()),
    )
    logger.debug(
        "Stations with area_rel_diff > 0.5: %s",
        int((station_model_map["area_rel_diff"] > 0.5).sum()),
    )
    logger.debug(
        "Stations with area_rel_diff > 1.0: %s",
        int((station_model_map["area_rel_diff"] > 1.0).sum()),
    )

    if "area_ratio" in station_model_map.columns:
        logger.debug(
            "Drain match area ratio summary:\n%s",
            station_model_map["area_ratio"].describe().to_string(),
        )
        logger.debug(
            "Stations with area_ratio < RATIO_MIN: %s",
            int((station_model_map["area_ratio"] < RATIO_MIN).sum()),
        )
        logger.debug(
            "Stations with area_ratio > RATIO_MAX: %s",
            int((station_model_map["area_ratio"] > RATIO_MAX).sum()),
        )

    cols = [
        "gauge_id",
        "station_name",
        "country",
        "grdc_lon",
        "grdc_lat",
        "efas_lon",
        "efas_lat",
        "model_id",
        "dist_m",
        "station_area_km2",
        "drainage_area_km2",
        "area_ratio",
        "area_rel_diff",
        "combined_score",
        "match_type",
        "candidate_search_mode",
    ]

    available_cols = [col for col in cols if col in station_model_map.columns]

    logger.debug(
        "Worst EFAS drain matches by distance:\n%s",
        station_model_map.sort_values("dist_m", ascending=False)
        [available_cols]
        .head(20)
        .to_string(index=False),
    )

    logger.debug(
        "Worst EFAS drain matches by area_rel_diff:\n%s",
        station_model_map.sort_values("area_rel_diff", ascending=False)
        [available_cols]
        .head(20)
        .to_string(index=False),
    )



# ======================================================
# EXTRACT EFAS TIMESERIES
# ======================================================

def extract_efas_timeseries_by_points(
    stations: pd.DataFrame,
    data_folder: Path,
    years: list[int],
    lat_col: str,
    lon_col: str,
    id_col: str,
    var_name: str,
    file_template: str,
    debug: bool,
) -> dict[str, pd.Series | None]:
    lats = stations[lat_col].to_numpy(dtype=float)
    lons = stations[lon_col].to_numpy(dtype=float)
    ids = stations[id_col].astype(str).to_numpy()

    logger.info("EFAS station series to extract: %s", len(stations))

    series_by_id = {sid: [] for sid in ids}

    first_existing_file = None

    for year in years:
        path = data_folder / file_template.format(year=year)

        if path.exists():
            first_existing_file = path
            break

    if first_existing_file is None:
        raise FileNotFoundError(
            f"No EFAS files found in {data_folder} using template '{file_template}'"
        )

    with xr.open_dataset(first_existing_file) as ds0:
        da0 = ds0[var_name]

        lat_grid = da0["latitude"].values
        lon_grid = da0["longitude"].values

        lat_idx = np.abs(lat_grid[:, None] - lats).argmin(axis=0)
        lon_idx = np.abs(lon_grid[:, None] - lons).argmin(axis=0)

        matched_lats = lat_grid[lat_idx]
        matched_lons = lon_grid[lon_idx]

    if debug:
        logger.debug("EFAS extraction grid indices computed.")
        logger.debug("Max lat diff: %s", float(np.abs(matched_lats - lats).max()))
        logger.debug("Max lon diff: %s", float(np.abs(matched_lons - lons).max()))

    for year in years:
        filepath = data_folder / file_template.format(year=year)

        if not filepath.exists():
            logger.warning(f"Missing EFAS file for year {year}: {filepath}")
            continue

        logger.info(f"Processing EFAS {year}")

        with xr.open_dataset(filepath) as ds:
            da = ds[var_name]
            time_dim = "valid_time" if "valid_time" in da.dims else "time"

            data = da.values
            extracted = data[:, lat_idx, lon_idx]

            time_vals = pd.to_datetime(da[time_dim].values)

            mask = ~((time_vals.month == 2) & (time_vals.day == 29))
            extracted = extracted[mask, :]
            time_vals = time_vals[mask]

            for i, sid in enumerate(ids):
                ts = pd.Series(extracted[:, i], index=time_vals, name="Qsim")
                series_by_id[sid].append(ts)

    for sid, chunks in series_by_id.items():
        if chunks:
            s = pd.concat(chunks).sort_index()
            s = s[~s.index.duplicated(keep="last")]
            series_by_id[sid] = s
        else:
            series_by_id[sid] = None

    logger.info("EFAS Qsim extracted: %s", sum(v is not None for v in series_by_id.values()))

    return series_by_id


# ======================================================
# EXTRACT GRDC OBSERVED TIMESERIES
# ======================================================

def extract_grdc_timeseries_by_id(
    ds_grdc: xr.Dataset,
    stations: pd.DataFrame,
    dataset_id_col: str,
    output_id_col: str,
    var_name: str,
    start_date: str,
    end_date: str,
) -> dict[str, pd.Series]:
    if var_name not in ds_grdc.data_vars:
        raise ValueError(f"Variable '{var_name}' not found in GRDC dataset.")

    qobs = ds_grdc[var_name]
    qobs = qobs.where(qobs >= 0)

    available_ids = set(qobs["id"].values.tolist())

    series_by_id = {}

    for _, row in stations.iterrows():
        output_id = str(row[output_id_col])
        station_id = row[dataset_id_col]

        if station_id not in available_ids:
            continue

        ts = (
            qobs.sel(id=station_id)
            .sel(time=slice(start_date, end_date))
            .to_pandas()
        )

        ts = ts.where(ts >= 0).dropna().sort_index()

        if ts.empty:
            continue

        ts.name = "Qobs"
        series_by_id[output_id] = ts

    logger.debug("GRDC Qobs extracted: %s", len(series_by_id))

    return series_by_id


# ======================================================
# WRITE SERIES
# ======================================================

def write_gauge_data_from_dict(
    grdc_series_by_station: dict[str, pd.Series],
    out_dir: Path,
) -> tuple[int, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = []

    for gauge_id, ts in grdc_series_by_station.items():
        if ts is None:
            skipped.append(gauge_id)
            continue

        df = ts.to_frame("Qobs")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().dropna()

        if df.empty:
            skipped.append(gauge_id)
            continue

        df.to_csv(out_dir / f"{gauge_id}.csv")
        written += 1

    return written, skipped


def write_station_qsim_from_dict(
    efas_series_by_station: dict[str, pd.Series | None],
    station_model_map: pd.DataFrame,
    out_dir: Path,
) -> tuple[pd.DataFrame, int, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = []
    valid_ids = set()

    for gauge_id, ts in efas_series_by_station.items():
        gauge_id = str(gauge_id)

        if ts is None:
            skipped.append(gauge_id)
            continue

        df = ts.to_frame("Qsim")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().dropna()

        if df.empty:
            skipped.append(gauge_id)
            continue

        df.to_csv(out_dir / f"{gauge_id}.csv")

        written += 1
        valid_ids.add(gauge_id)

    filtered_station_model_map = station_model_map[
        station_model_map["gauge_id"].astype(str).isin(valid_ids)
    ].copy()

    removed = len(station_model_map) - len(filtered_station_model_map)

    logger.debug("Qsim files written: %s", written)
    logger.debug("Qsim files skipped: %s", len(skipped))
    logger.debug("Stations removed from station_model_map due to missing/empty Qsim: %s", removed)
    
    if skipped:
        logger.debug(
            "Skipped Qsim gauge ids:\n%s",
            ", ".join(map(str, skipped[:30])),
        )

        if len(skipped) > 30:
            logger.debug("Additional skipped Qsim gauge ids: %s", len(skipped) - 30)

    return filtered_station_model_map.reset_index(drop=True), written, skipped


# ======================================================
# GAUGE TABLE AND GIS
# ======================================================

def build_gauge_table(station_model_map: pd.DataFrame) -> pd.DataFrame:
    stations_ok = station_model_map[
        station_model_map["model_id"].notna()
    ].copy()

    gauge_table = (
        stations_ok[["gauge_id", "model_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    logger.debug("Gauge table rows: %s", len(gauge_table))
    logger.info("Unique gauges: %s", gauge_table["gauge_id"].nunique())
    logger.debug("Unique model_ids: %s", gauge_table["model_id"].nunique())

    return gauge_table


def save_gauge_gis(
    station_model_map: pd.DataFrame,
    gauge_table: pd.DataFrame,
    out_path: Path,
) -> None:
    stations = station_model_map[
        station_model_map["gauge_id"].isin(gauge_table["gauge_id"])
    ].copy()

    gdf = gpd.GeoDataFrame(
        stations,
        geometry=[
            Point(xy)
            for xy in zip(
                stations["plant_lon"].astype(float),
                stations["plant_lat"].astype(float),
            )
        ],
        crs="EPSG:4326",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG")

    logger.debug(f"Gauge GIS saved: {out_path}")


# ======================================================
# OPTIONAL PLOT
# ======================================================

def save_overview_plot(
    station_model_map: pd.DataFrame,
    df_drain: pd.DataFrame,
    plants_path: Path,
    out_path: Path,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    threshold_uparea_km2: float,
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.colors import sample_colorscale
    except ImportError:
        logger.warning("Plotly not installed. Skipping overview plot.")
        return

    df_st = station_model_map.copy()

    df_drain_roi = df_drain[
        (df_drain["x"] >= lon_min)
        & (df_drain["x"] <= lon_max)
        & (df_drain["y"] >= lat_min)
        & (df_drain["y"] <= lat_max)
        & (df_drain["drainage_area"] > threshold_uparea_km2)
    ].copy()

    if df_drain_roi.empty:
        logger.warning("No EFAS drain nodes available for overview plot.")
        return

    df_drain_roi["log_drainage_area"] = np.log10(df_drain_roi["drainage_area"])

    if plants_path.exists():
        ppls = pd.read_csv(plants_path)

        hydro_ppls = ppls[
            (ppls["Fueltype"] == "Hydro")
            & (ppls["Capacity"] > 0)
            & (ppls["lon"] >= lon_min)
            & (ppls["lon"] <= lon_max)
            & (ppls["lat"] >= lat_min)
            & (ppls["lat"] <= lat_max)
        ][["Name", "Country", "Capacity", "lat", "lon"]].copy()
    else:
        hydro_ppls = pd.DataFrame(
            columns=["Name", "Country", "Capacity", "lat", "lon"]
        )

    blues_no_white = sample_colorscale(
        "Blues",
        [0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0],
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=df_drain_roi["x"],
            y=df_drain_roi["y"],
            mode="markers",
            hoverinfo="skip",
            marker=dict(
                size=4,
                color=df_drain_roi["log_drainage_area"],
                colorscale=blues_no_white,
                opacity=0.55,
                colorbar=dict(title="log10 drainage<br>area km²"),
            ),
            name="EFAS drain network",
        )
    )

    fig.add_trace(
        go.Scattergl(
            x=df_st["grdc_lon"],
            y=df_st["grdc_lat"],
            mode="markers",
            marker=dict(
                size=10,
                color="limegreen",
                symbol="circle",
                line=dict(color="black", width=0.8),
            ),
            name="Original GRDC stations",
            text=(
                "Gauge: "
                + df_st["gauge_id"].astype(str)
                + "<br>Station: "
                + df_st["station_name"].astype(str)
                + "<br>River: "
                + df_st["river"].astype(str)
                + "<br>Country: "
                + df_st["country"].astype(str)
                + "<br>Area km²: "
                + df_st["station_area_km2"].round(1).astype(str)
            ),
            hovertemplate="%{text}<br>Lon=%{x}<br>Lat=%{y}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scattergl(
            x=df_st["efas_lon"],
            y=df_st["efas_lat"],
            mode="markers",
            marker=dict(
                size=11,
                color="red",
                symbol="x",
                line=dict(color="red", width=2),
            ),
            name="Matched EFAS drain nodes",
            text=(
                "Gauge: "
                + df_st["gauge_id"].astype(str)
                + "<br>Model ID: "
                + df_st["model_id"].astype(str)
                + "<br>Match: "
                + df_st["match_type"].astype(str)
                + "<br>Search mode: "
                + df_st.get(
                    "candidate_search_mode",
                    pd.Series("", index=df_st.index),
                ).astype(str)
                + "<br>Distance m: "
                + df_st["dist_m"].round(1).astype(str)
                + "<br>Drainage area km²: "
                + df_st["drainage_area_km2"].round(1).astype(str)
                + "<br>Area ratio: "
                + df_st.get(
                    "area_ratio",
                    pd.Series(np.nan, index=df_st.index),
                ).round(3).astype(str)
                + "<br>Area rel diff: "
                + df_st["area_rel_diff"].round(3).astype(str)
            ),
            hovertemplate="%{text}<br>Lon=%{x}<br>Lat=%{y}<extra></extra>",
        )
    )

    link_x = []
    link_y = []

    for _, row in df_st.iterrows():
        link_x.extend([row["grdc_lon"], row["efas_lon"], None])
        link_y.extend([row["grdc_lat"], row["efas_lat"], None])

    fig.add_trace(
        go.Scattergl(
            x=link_x,
            y=link_y,
            mode="lines",
            line=dict(
                width=1,
                color="rgba(255, 0, 0, 0.35)",
                dash="dot",
            ),
            hoverinfo="skip",
            name="GRDC → EFAS drain links",
        )
    )

    if not hydro_ppls.empty:
        fig.add_trace(
            go.Scattergl(
                x=hydro_ppls["lon"],
                y=hydro_ppls["lat"],
                mode="markers",
                marker=dict(
                    size=10,
                    color="purple",
                    symbol="triangle-up",
                    line=dict(color="black", width=0.6),
                ),
                name="Hydropower plants",
                text=(
                    "Plant: "
                    + hydro_ppls["Name"].astype(str)
                    + "<br>Country: "
                    + hydro_ppls["Country"].astype(str)
                    + "<br>Capacity MW: "
                    + hydro_ppls["Capacity"].round(1).astype(str)
                ),
                hovertemplate="%{text}<br>Lon=%{x}<br>Lat=%{y}<extra></extra>",
            )
        )

    fig.update_layout(
        title="GRDC stations, matched EFAS drain nodes and hydropower plants",
        width=1300,
        height=850,
        template="plotly_white",
        xaxis=dict(
            title="Longitude",
            range=[lon_min, lon_max],
            showgrid=True,
        ),
        yaxis=dict(
            title="Latitude",
            range=[lat_min, lat_max],
            showgrid=True,
            scaleanchor="x",
            scaleratio=1,
        ),
        legend=dict(
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="black",
            borderwidth=1,
        ),
        margin=dict(l=40, r=40, t=70, b=40),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(
        str(out_path),
        include_plotlyjs="cdn",
        full_html=True,
        auto_open=False,
    )

    logger.info("Diagnostic overview plot saved: %s", out_path)


# ======================================================
# WORKFLOW
# ======================================================

def run_grdc_to_drain_efas(cfg: dict | None = None) -> None:
    cfg = cfg or get_config()

    log_config_summary(cfg)

    t0 = time.time()

    lon_min = cfg["lon_min"]
    lon_max = cfg["lon_max"]
    lat_min = cfg["lat_min"]
    lat_max = cfg["lat_max"]

    file_path_grdc = cfg["grdc_path"]
    efas_data_folder = cfg["hydro_data_dir"]
    efas_file_template = cfg["hydro_file_template"]
    efas_var_name = cfg["hydro_var_name"]

    drain_table_path = cfg["drain_table_path"]
    plants_path = cfg["plants_path"]

    base_output_dir = cfg["base_output_dir"]
    workdir_tables = base_output_dir / "workdir" / "tables"

    station_model_map_output_path = base_output_dir / "station_model_map.csv"
    gauge_table_output_path = workdir_tables / "gauge_table_all.csv"
    gauge_gis_output_path = base_output_dir / "gauge_gis_all.gpkg"

    gauge_data_dir = base_output_dir / "gauge_data_grdc"
    station_qsim_dir = base_output_dir / cfg["station_qsim_dir_name"]

    diagnostic_plot_dir = base_output_dir / "diagnostic_plots"
    plot_output_path = diagnostic_plot_dir / "stations_efas_overview.html"

    threshold_uparea_km2 = cfg["threshold_uparea_km2"]
    years = cfg["years"]

    debug = os.environ.get("HYDRO_LOG_LEVEL", "INFO").upper() == "DEBUG"
    make_overview_plot = os.environ.get("HYDRO_MAKE_DIAGNOSTIC_PLOTS", "0") == "1"

    workdir_tables.mkdir(parents=True, exist_ok=True)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    ds_grdc, grdc_df = load_grdc_metadata(
        file_path=file_path_grdc,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        threshold_uparea_km2=threshold_uparea_km2,
    )

    df_coverage, df_final = filter_saber_ready_stations(
        ds=ds_grdc,
        grdc_df=grdc_df,
        var_name=GRDC_VAR_NAME,
        start_date=START_DATE,
        end_date=END_DATE,
        min_obs_per_month_threshold=MIN_OBS_PER_MONTH,
    )

    df_final = df_final.copy()
    df_final["gauge_uid"] = df_final["id"].astype(str)

    logger.debug(
        "Final GRDC stations before EFAS drain mapping: %s",
        len(df_final),
    )

    df_drain = load_drain_table(drain_table_path)

    station_model_map = match_stations_to_drain(
        stations=df_final,
        df_drain=df_drain,
        max_distance_km=MAX_DRAIN_DISTANCE_KM,
        distance_weight=DRAIN_SCORE_DISTANCE_WEIGHT,
        area_weight=DRAIN_SCORE_AREA_WEIGHT,
        k_neighbors=K_NEIGHBORS,
    )

    log_drain_match_diagnostics(station_model_map)

    mapped_gauge_ids = set(station_model_map["gauge_id"].astype(str))

    df_final_mapped = df_final[
        df_final["gauge_uid"].astype(str).isin(mapped_gauge_ids)
    ].copy()

    efas_series_by_station = extract_efas_timeseries_by_points(
        stations=station_model_map,
        data_folder=efas_data_folder,
        years=years,
        lat_col="efas_lat",
        lon_col="efas_lon",
        id_col="gauge_id",
        var_name=efas_var_name,
        file_template=efas_file_template,
        debug=debug,
    )

    grdc_series_by_station = extract_grdc_timeseries_by_id(
        ds_grdc=ds_grdc,
        stations=df_final_mapped,
        dataset_id_col="id",
        output_id_col="gauge_uid",
        var_name=GRDC_VAR_NAME,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    written_obs, skipped_obs = write_gauge_data_from_dict(
        grdc_series_by_station=grdc_series_by_station,
        out_dir=gauge_data_dir,
    )

    logger.debug("Qobs files written: %s", written_obs)
    logger.debug("Qobs files skipped: %s", len(skipped_obs))

    if skipped_obs:
        logger.debug(
            "Skipped Qobs gauge ids: %s%s",
            ", ".join(map(str, skipped_obs[:30])),
            f" ... plus {len(skipped_obs) - 30} more"
            if len(skipped_obs) > 30
            else "",
        )

    station_model_map, written_sim, skipped_sim = write_station_qsim_from_dict(
        efas_series_by_station=efas_series_by_station,
        station_model_map=station_model_map,
        out_dir=station_qsim_dir,
    )

    station_model_map.to_csv(station_model_map_output_path, index=False)
    logger.info("Station model map saved: %s", station_model_map_output_path)

    gauge_table = build_gauge_table(station_model_map)

    gauge_table.to_csv(gauge_table_output_path, index=False)
    logger.info("Gauge table saved: %s", gauge_table_output_path)

    save_gauge_gis(
        station_model_map=station_model_map,
        gauge_table=gauge_table,
        out_path=gauge_gis_output_path,
    )

    if make_overview_plot:
        save_overview_plot(
            station_model_map=station_model_map,
            df_drain=df_drain,
            plants_path=plants_path,
            out_path=plot_output_path,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            threshold_uparea_km2=threshold_uparea_km2,
        )

    logger.debug("Final check:")
    logger.debug("  station_model_map: %s", len(station_model_map))
    logger.debug("  gauge_table: %s", len(gauge_table))
    logger.debug(
        "  Qsim extracted: %s",
        sum(v is not None for v in efas_series_by_station.values()),
    )
    logger.debug("  Qobs extracted: %s", len(grdc_series_by_station))

    ds_grdc.close()

    logger.info("GRDC-to-EFAS drain workflow completed.")
    logger.debug("Total time: %.1f s", time.time() - t0)


def main() -> None:
    setup_logging()
    run_grdc_to_drain_efas()


if __name__ == "__main__":
    main()