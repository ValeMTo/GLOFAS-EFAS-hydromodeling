# This script loads GRanD reservoir data, identifies major reservoirs in Europe,
# and links each reservoir to the most suitable hydrological drainage node.
# It then builds a regulate table mapping reservoirs for SABER.

from __future__ import annotations

from pathlib import Path
import logging
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from scipy.spatial import cKDTree

from hydro_inflow.hydro_config import get_config, log_config_summary
from hydro_inflow.utils import approx_dist_km, setup_logging


logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

CAPACITY_THRESHOLD_MCM = 250.0

WINDOW = 3
AREA_JUMP_THRESHOLD = 0.15
CORR_THRESHOLD = 0.95
MAX_DISTANCE_KM = 10.0

LOG_EVERY = 25
DRAIN_QUERY_K = 30

ALLOWED_ISO2 = {
    "AL",
    "AT",
    "BA",
    "BE",
    "BG",
    "CH",
    "CZ",
    "DE",
    "DK",
    "EE",
    "ES",
    "FI",
    "FR",
    "GB",
    "GR",
    "HR",
    "HU",
    "IE",
    "IT",
    "LT",
    "LU",
    "LV",
    "ME",
    "MK",
    "NL",
    "NO",
    "PL",
    "PT",
    "RO",
    "RS",
    "SE",
    "SI",
    "SK",
    "XK",
}


# ============================================================
# COUNTRY MAP
# ============================================================

EUROPE_COUNTRY_NAME_TO_ISO2 = {
    "albania": "AL",
    "andorra": "AD",
    "austria": "AT",
    "belarus": "BY",
    "belgium": "BE",
    "bosnia and herzegovina": "BA",
    "bosnia-herzegovina": "BA",
    "bulgaria": "BG",
    "croatia": "HR",
    "cyprus": "CY",
    "czech republic": "CZ",
    "czechia": "CZ",
    "denmark": "DK",
    "estonia": "EE",
    "finland": "FI",
    "france": "FR",
    "germany": "DE",
    "greece": "GR",
    "hungary": "HU",
    "iceland": "IS",
    "ireland": "IE",
    "italy": "IT",
    "kosovo": "XK",
    "latvia": "LV",
    "liechtenstein": "LI",
    "lithuania": "LT",
    "luxembourg": "LU",
    "malta": "MT",
    "moldova": "MD",
    "monaco": "MC",
    "montenegro": "ME",
    "netherlands": "NL",
    "norway": "NO",
    "north macedonia": "MK",
    "macedonia": "MK",
    "poland": "PL",
    "portugal": "PT",
    "romania": "RO",
    "russia": "RU",
    "russian federation": "RU",
    "san marino": "SM",
    "serbia": "RS",
    "slovakia": "SK",
    "slovenia": "SI",
    "spain": "ES",
    "sweden": "SE",
    "switzerland": "CH",
    "turkey": "TR",
    "turkiye": "TR",
    "ukraine": "UA",
    "united kingdom": "GB",
    "uk": "GB",
    "great britain": "GB",
}

# ============================================================
# LOADERS
# ============================================================

def load_grand_reservoirs(
    grand_path: Path,
    capacity_threshold_mcm: float,
    allowed_iso2: set[str],
) -> gpd.GeoDataFrame:
    reservoirs = gpd.read_file(grand_path)

    logger.info("GRanD reservoirs loaded: %s", len(reservoirs))

    reservoirs["COUNTRY_norm"] = (
        reservoirs["COUNTRY"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    reservoirs["ISO2"] = reservoirs["COUNTRY_norm"].map(
        EUROPE_COUNTRY_NAME_TO_ISO2
    )

    reservoirs_eu = reservoirs[reservoirs["ISO2"].notna()].copy()
    logger.debug("Reservoirs mapped to European ISO2: %s", len(reservoirs_eu))

    reservoirs_allowed = reservoirs_eu[
        reservoirs_eu["ISO2"].isin(allowed_iso2)
    ].copy()

    logger.debug("Reservoirs in selected countries: %s", len(reservoirs_allowed))

    reservoirs_allowed["CAP_MCM"] = pd.to_numeric(
        reservoirs_allowed["CAP_MCM"],
        errors="coerce",
    )

    major_reservoirs = reservoirs_allowed[
        reservoirs_allowed["CAP_MCM"] >= capacity_threshold_mcm
    ].copy()

    logger.debug(
        "Selected-country reservoirs with CAP_MCM >= %s: %s",
        f"{capacity_threshold_mcm:g}",
        len(major_reservoirs),
    )

    missing_name = (
        major_reservoirs["RES_NAME"].isna()
        | (major_reservoirs["RES_NAME"].astype(str).str.strip() == "")
    )

    major_reservoirs.loc[missing_name, "RES_NAME"] = (
        "RES_" + major_reservoirs.loc[missing_name, "GRAND_ID"].astype(str)
    )

    major_reservoirs["GRAND_ID"] = major_reservoirs["GRAND_ID"].astype(str)

    return major_reservoirs


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

    df["downstream_model_id"] = df["downstream_model_id"].where(
        df["downstream_model_id"].notna(),
        None,
    )

    if "upstream_model_ids" in df.columns:
        df["upstream_model_ids"] = df["upstream_model_ids"].fillna("").astype(str)

    df["drainage_area"] = pd.to_numeric(df["drainage_area"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    df = df.dropna(subset=["model_id", "drainage_area", "x", "y"]).copy()

    logger.debug("Drain nodes loaded: %s", len(df))

    return df


# ============================================================
# TOPOLOGY HELPERS
# ============================================================

def normalize_downstream_id(value) -> str | None:
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value in {"", "nan", "None", "NONE", "-1"}:
        return None

    return value


def parse_upstream_ids(value) -> list[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []

    return [
        item.strip()
        for item in str(value).split(";")
        if item.strip() not in {"", "nan", "None", "NONE", "-1"}
    ]


def build_topology_lookups(
    df_drain: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, str | None], dict[str, list[str]]]:
    area_map = dict(
        zip(
            df_drain["model_id"].astype(str),
            df_drain["drainage_area"].astype(float),
        )
    )

    down_map = {
        str(row["model_id"]): normalize_downstream_id(row["downstream_model_id"])
        for _, row in df_drain.iterrows()
    }

    up_map = {}

    if "upstream_model_ids" in df_drain.columns:
        for _, row in df_drain.iterrows():
            mid = str(row["model_id"])
            ups = parse_upstream_ids(row["upstream_model_ids"])
            if ups:
                up_map[mid] = ups
    else:
        for mid, downstream_id in down_map.items():
            if downstream_id is not None:
                up_map.setdefault(downstream_id, []).append(mid)

    return area_map, down_map, up_map


def choose_upstream_by_area_similarity(
    current_id: str,
    upstream_candidates: list[str],
    area_map: dict[str, float],
) -> str | None:
    current_area = area_map.get(current_id)

    if current_area is None or not np.isfinite(current_area) or current_area == 0:
        return None

    best_id = None
    best_diff = np.inf

    for upstream_id in upstream_candidates:
        upstream_area = area_map.get(upstream_id)

        if upstream_area is None or not np.isfinite(upstream_area):
            continue

        diff = abs(upstream_area - current_area) / current_area

        if diff < best_diff:
            best_diff = diff
            best_id = upstream_id

    return best_id


def build_local_window_chain(
    center_id: str,
    area_map: dict[str, float],
    down_map: dict[str, str | None],
    up_map: dict[str, list[str]],
    window: int,
    area_jump_threshold: float,
) -> tuple[list[str], int, int]:
    upstream_chain = []
    downstream_chain = []

    current_id = center_id
    current_area = area_map.get(current_id)

    if current_area is not None and np.isfinite(current_area) and current_area != 0:
        for _ in range(window):
            upstream_candidates = up_map.get(current_id, [])

            if not upstream_candidates:
                break

            upstream_id = choose_upstream_by_area_similarity(
                current_id=current_id,
                upstream_candidates=upstream_candidates,
                area_map=area_map,
            )

            if upstream_id is None:
                break

            upstream_area = area_map.get(upstream_id)

            if upstream_area is None or not np.isfinite(upstream_area):
                break

            if abs(upstream_area - current_area) / current_area > area_jump_threshold:
                break

            upstream_chain.insert(0, upstream_id)
            current_id = upstream_id
            current_area = upstream_area

    current_id = center_id
    current_area = area_map.get(current_id)

    if current_area is not None and np.isfinite(current_area) and current_area != 0:
        for _ in range(window):
            downstream_id = down_map.get(current_id)

            if downstream_id is None:
                break

            downstream_area = area_map.get(downstream_id)

            if downstream_area is None or not np.isfinite(downstream_area):
                break

            if abs(downstream_area - current_area) / current_area > area_jump_threshold:
                break

            downstream_chain.append(downstream_id)
            current_id = downstream_id
            current_area = downstream_area

    node_chain = upstream_chain + [center_id] + downstream_chain

    return node_chain, len(upstream_chain), len(downstream_chain)


# ============================================================
# HYDROLOGICAL BREAK
# ============================================================

def compute_corr(s1: pd.Series, s2: pd.Series) -> float:
    df = pd.concat([s1, s2], axis=1).dropna()

    if len(df) < 10:
        return np.nan

    return float(np.corrcoef(df.iloc[:, 0], df.iloc[:, 1])[0, 1])


def choose_dam_node_by_corr(
    center_id: str,
    node_chain: list[str],
    node_to_series: dict[str, pd.Series],
    corr_threshold: float,
) -> tuple[str, bool, float]:
    corr_list = []

    for i in range(len(node_chain) - 1):
        n_up = node_chain[i]
        n_dn = node_chain[i + 1]

        s_up = node_to_series.get(n_up)
        s_dn = node_to_series.get(n_dn)

        if s_up is None or s_dn is None:
            continue

        corr = compute_corr(s_up, s_dn)

        if np.isfinite(corr):
            corr_list.append((corr, n_up, n_dn))

    if not corr_list:
        return center_id, False, np.nan

    corr_min, n_up_min, n_dn_min = min(corr_list, key=lambda x: x[0])

    if corr_min < corr_threshold:
        return n_dn_min, True, float(corr_min)

    return center_id, False, float(corr_min)


# ============================================================
# GRID INDEXING / NEAREST DRAIN
# ============================================================

def build_model_to_grid_index(
    df_drain: pd.DataFrame,
    river_da: xr.DataArray,
) -> dict[str, tuple[int, int]]:
    lats = river_da["latitude"].values
    lons = river_da["longitude"].values

    model_to_ij = {}

    for _, row in df_drain.iterrows():
        mid = str(row["model_id"])

        iy = int(np.abs(lats - float(row["y"])).argmin())
        ix = int(np.abs(lons - float(row["x"])).argmin())

        model_to_ij[mid] = (iy, ix)

    return model_to_ij


def build_drain_tree(
    df_drain: pd.DataFrame,
) -> tuple[cKDTree, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Build a drain-node KDTree in approximate km coordinates.

    The tree is used only for fast preselection. The final nearest node is
    selected with approx_dist_km() at the reservoir latitude.
    """
    model_ids = df_drain["model_id"].astype(str).to_numpy()
    xs = df_drain["x"].to_numpy(dtype=float)
    ys = df_drain["y"].to_numpy(dtype=float)

    mean_lat = float(np.mean(ys))

    x_km = xs * 111.0 * np.cos(np.deg2rad(mean_lat))
    y_km = ys * 111.0

    coords_km = np.column_stack([x_km, y_km])
    tree = cKDTree(coords_km)

    return tree, model_ids, xs, ys, mean_lat


def find_nearest_model_id(
    lon: float,
    lat: float,
    tree: cKDTree,
    model_ids: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    mean_lat: float,
    max_distance_km: float,
    query_k: int = 30,
) -> tuple[str | None, float]:
    """
    Find the nearest drain node within max_distance_km.

    The KDTree preselects nearby nodes in approximate km coordinates.
    The final nearest is chosen using local distance at the reservoir latitude.
    """
    query_k = min(max(1, query_k), len(model_ids))

    x_res_global = lon * 111.0 * np.cos(np.deg2rad(mean_lat))
    y_res_global = lat * 111.0

    _, idxs = tree.query([x_res_global, y_res_global], k=query_k)
    idxs = np.atleast_1d(idxs)

    dists_km = approx_dist_km(
        lon=lon,
        lat=lat,
        lon2=xs[idxs],
        lat2=ys[idxs],
    )

    best_pos = int(np.argmin(dists_km))
    best_idx = int(idxs[best_pos])
    best_dist = float(dists_km[best_pos])

    if best_dist > max_distance_km:
        return None, best_dist

    return str(model_ids[best_idx]), best_dist


# ============================================================
# BUILD REGULATE TABLE
# ============================================================

def build_regulate_table(
    major_reservoirs: gpd.GeoDataFrame,
    df_drain: pd.DataFrame,
    river_da: xr.DataArray,
    window: int,
    area_jump_threshold: float,
    corr_threshold: float,
    max_distance_km: float,
    log_every: int,
) -> pd.DataFrame:
    t0 = time.time()

    tree_drain, model_ids, drain_xs, drain_ys, drain_mean_lat = build_drain_tree(
        df_drain
    )

    logger.debug("KDTree drain built in %.2f s", time.time() - t0)

    t0 = time.time()

    area_map, down_map, up_map = build_topology_lookups(df_drain)

    logger.debug("Topology lookups built in %.2f s", time.time() - t0)

    t0 = time.time()

    model_to_ij = build_model_to_grid_index(
        df_drain=df_drain,
        river_da=river_da,
    )

    logger.debug("model_id to river grid index built in %.2f s", time.time() - t0)

    series_cache = {}

    records = []
    skipped_records = []

    n_breaks = 0
    n_single_node = 0
    n_no_upstream = 0
    n_no_downstream = 0
    n_shifted_by_corr = 0

    t_loop = time.time()

    for i, (_, dam) in enumerate(major_reservoirs.iterrows(), start=1):
        reg_id = str(dam["GRAND_ID"])
        res_name = str(dam["RES_NAME"])
        lon = float(dam["LONG_DD"])
        lat = float(dam["LAT_DD"])

        center_id, center_distance_km = find_nearest_model_id(
            lon=lon,
            lat=lat,
            tree=tree_drain,
            model_ids=model_ids,
            xs=drain_xs,
            ys=drain_ys,
            mean_lat=drain_mean_lat,
            max_distance_km=max_distance_km,
            query_k=DRAIN_QUERY_K,
        )

        if center_id is None:
            skipped_records.append(
                {
                    "reg_id": reg_id,
                    "res_name": res_name,
                    "country": dam["COUNTRY"],
                    "iso2": dam["ISO2"],
                    "longitude": lon,
                    "latitude": lat,
                    "nearest_distance_km": center_distance_km,
                    "reason": f"no drain node within {max_distance_km:.1f} km",
                }
            )
            continue

        node_chain, n_upstream, n_downstream = build_local_window_chain(
            center_id=center_id,
            area_map=area_map,
            down_map=down_map,
            up_map=up_map,
            window=window,
            area_jump_threshold=area_jump_threshold,
        )

        if len(node_chain) == 1:
            n_single_node += 1

        if n_upstream == 0:
            n_no_upstream += 1

        if n_downstream == 0:
            n_no_downstream += 1

        node_to_series = {}

        for mid in node_chain:
            if mid not in series_cache:
                iy, ix = model_to_ij[mid]

                series_cache[mid] = river_da.isel(
                    latitude=iy,
                    longitude=ix,
                ).to_series()

            node_to_series[mid] = series_cache[mid]

        dam_id, break_detected, corr_min = choose_dam_node_by_corr(
            center_id=center_id,
            node_chain=node_chain,
            node_to_series=node_to_series,
            corr_threshold=corr_threshold,
        )

        if break_detected:
            n_breaks += 1

        if str(dam_id) != str(center_id):
            n_shifted_by_corr += 1

        records.append(
            {
                "reg_id": reg_id,
                "model_id": str(dam_id),
                "center_model_id": str(center_id),
                "center_distance_km": float(center_distance_km),
                "break_detected": bool(break_detected),
                "corr_min": float(corr_min) if np.isfinite(corr_min) else np.nan,
                "n_chain_nodes": int(len(node_chain)),
                "n_upstream": int(n_upstream),
                "n_downstream": int(n_downstream),
                "res_name": res_name,
                "country": dam["COUNTRY"],
                "iso2": dam["ISO2"],
                "longitude": lon,
                "latitude": lat,
            }
        )

        if i % log_every == 0:
            elapsed = time.time() - t_loop
            logger.debug(
                "Processed reservoirs: %s/%s | elapsed %.1f s",
                i,
                len(major_reservoirs),
                elapsed,
            )

    regulate_table = pd.DataFrame(records).drop_duplicates()
    skipped_table = pd.DataFrame(skipped_records)

    logger.debug("Regulate table built: %s", len(regulate_table))
    logger.debug("Reservoirs processed: %s", len(major_reservoirs))
    logger.debug("Reservoirs skipped: %s", len(skipped_table))
    logger.debug("Hydrological breaks detected: %s", n_breaks)
    logger.debug("Reservoirs shifted by correction: %s", n_shifted_by_corr)
    logger.debug("Single-node chains: %s", n_single_node)
    logger.debug("Chains with no upstream step: %s", n_no_upstream)
    logger.debug("Chains with no downstream step: %s", n_no_downstream)

    if not regulate_table.empty:
        logger.debug(
            "Correction diagnostics:\n%s",
            regulate_table[
                [
                    "reg_id",
                    "res_name",
                    "center_model_id",
                    "model_id",
                    "center_distance_km",
                    "break_detected",
                    "corr_min",
                    "n_chain_nodes",
                    "n_upstream",
                    "n_downstream",
                ]
            ]
            .sort_values("center_distance_km", ascending=False)
            .head(20)
            .to_string(index=False),
        )

    if not skipped_table.empty:
        logger.warning("Skipped reservoirs: %s", len(skipped_table))
        logger.debug(
            "Skipped reservoirs preview:\n%s",
            skipped_table[
                [
                    "reg_id",
                    "res_name",
                    "country",
                    "iso2",
                    "longitude",
                    "latitude",
                    "nearest_distance_km",
                    "reason",
                ]
            ].head(20).to_string(index=False),
        )

    return regulate_table


# ============================================================
# SAVE
# ============================================================

def save_outputs(
    regulate_table: pd.DataFrame,
    regulate_table_output_path: Path,
) -> None:
    regulate_table_output_path.parent.mkdir(parents=True, exist_ok=True)

    regulate_table.to_csv(regulate_table_output_path, index=False)

    logger.info("Regulate table saved: %s", regulate_table_output_path)

# ============================================================
# WORKFLOW
# ============================================================

def run_grand_to_drain(cfg: dict | None = None) -> None:
    cfg = cfg or get_config()

    log_config_summary(cfg)

    t0 = time.time()

    grand_path = cfg["grand_path"]
    drain_table_path = cfg["drain_table_path"]
    river_reference_path = cfg["river_reference_path"]
    river_var_name = cfg["hydro_var_name"]
    regulate_table_output_path = cfg["regulate_table_output_path"]

    major_reservoirs = load_grand_reservoirs(
        grand_path=grand_path,
        capacity_threshold_mcm=CAPACITY_THRESHOLD_MCM,
        allowed_iso2=ALLOWED_ISO2,
    )

    df_drain = load_drain_table(drain_table_path)

    logger.debug("Opening river reference: %s", river_reference_path)

    with xr.open_dataset(river_reference_path) as ds:
        river_da = ds[river_var_name]

        regulate_table = build_regulate_table(
            major_reservoirs=major_reservoirs,
            df_drain=df_drain,
            river_da=river_da,
            window=WINDOW,
            area_jump_threshold=AREA_JUMP_THRESHOLD,
            corr_threshold=CORR_THRESHOLD,
            max_distance_km=MAX_DISTANCE_KM,
            log_every=LOG_EVERY,
        )

    save_outputs(
        regulate_table=regulate_table,
        regulate_table_output_path=regulate_table_output_path,
    )

    logger.info("GRanD-to-drain workflow completed.")
    logger.debug("Total time: %.1f s", time.time() - t0)


def main() -> None:
    setup_logging()
    run_grand_to_drain()


if __name__ == "__main__":
    main()