# This workflow derives a simplified drainage network from EFAS upstream-area data.
# River candidate points are filtered, assigned to HydroBASINS, clustered basin by basin,
# and connected using upstream-area topology. The resulting base network is used as
# input for SABER and for later extraction/correction of hydropower plant locations.

# import
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import copy
import logging
import time

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt

from pyproj import Geod
from scipy.spatial import KDTree, cKDTree
from shapely.geometry import box
from sklearn.cluster import DBSCAN

from hydro_config import get_config, log_config_summary
from logging_utils import setup_logging


logger = logging.getLogger(__name__)

CFG = get_config()

# ============================================================
# CONFIGURATION
# ============================================================

# Europe bounding box
LON_MIN = CFG["lon_min"]
LON_MAX = CFG["lon_max"]
LAT_MIN = CFG["lat_min"]
LAT_MAX = CFG["lat_max"]

BUFFER_MARGIN_DEG = 0.5  # margin for hydrobasins shape selection
MIN_POSITIVE_VALUE = 1e-10

UPAREA_PATH = CFG["uparea_path"]
HYDROBASINS_PATH = CFG["hydrobasins_path"]

UPAREA_VARIABLE_NAME = CFG["uparea_variable_name"]
HYDROBASINS_CRS_EPSG = 4326
_GEOD = Geod(ellps="WGS84")

THRESHOLD_UPAREA_KM2 = CFG["threshold_uparea_km2"]

GRID_STEP = CFG["grid_step"]
TOLERANCE_FACTOR = CFG["tolerance_factor"]
MAX_DISTANCE_DEG = GRID_STEP * np.sqrt(2) * TOLERANCE_FACTOR
MIN_SAMPLES_DBSCAN = 1

DRAIN_CSV_OUTPUT_PATH = CFG["drain_csv_output_path"]
DRAIN_PARQUET_OUTPUT_PATH = CFG["drain_parquet_output_path"]
DRAIN_GIS_OUTPUT_PATH = CFG["drain_gis_output_path"]


# ============================================================
# 1) LOAD AND PREPARE UPSTREAM AREA
# ============================================================

def load_and_prepare_upstream_area(
    uparea_path: Path,
    variable_name: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    min_positive_value: float,
) -> xr.DataArray:
    """
    Load upstream area, clean invalid values, convert from m² to km²,
    normalize coordinate names, and subset to the target domain.
    """
    with xr.open_dataset(uparea_path) as ds:
        if variable_name not in ds:
            available_vars = list(ds.data_vars)
            raise KeyError(
                f"Variable '{variable_name}' not found in dataset. "
                f"Available variables: {available_vars}"
            )

        uparea = ds[variable_name].load()

    rename_map = {}

    if "Latitude" in uparea.coords:
        rename_map["Latitude"] = "latitude"
    if "Longitude" in uparea.coords:
        rename_map["Longitude"] = "longitude"
    if "lat" in uparea.coords:
        rename_map["lat"] = "latitude"
    if "lon" in uparea.coords:
        rename_map["lon"] = "longitude"

    if rename_map:
        uparea = uparea.rename(rename_map)

    if "latitude" not in uparea.coords or "longitude" not in uparea.coords:
        raise KeyError(
            "Could not find latitude/longitude coordinates. "
            f"Available coords: {list(uparea.coords)}. "
            f"Available dims: {list(uparea.dims)}"
        )

    uparea = uparea.fillna(0.0)
    uparea = uparea.where(uparea > 0, min_positive_value)

    uparea_km2 = uparea / 1e6

    lat_vals = uparea_km2["latitude"].values

    if lat_vals[0] < lat_vals[-1]:
        uparea_km2 = uparea_km2.sel(
            latitude=slice(lat_min, lat_max),
            longitude=slice(lon_min, lon_max),
        )
    else:
        uparea_km2 = uparea_km2.sel(
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min, lon_max),
        )

    logger.debug("Uparea loaded, cleaned, converted to km², and filtered to model domain.")
    logger.debug("Filtered uparea shape: %s", uparea_km2.shape)

    return uparea_km2


# ============================================================
# 2) BUILD GRID AND FILTER BY UPSTREAM AREA
# ============================================================

def build_grid_from_uparea(uparea_km2: xr.DataArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 2D longitude/latitude grids and flattened upstream area values.
    """
    longitude_coords, latitude_coords = np.meshgrid(
        uparea_km2["longitude"].values,
        uparea_km2["latitude"].values,
    )

    uparea_values = uparea_km2.values.flatten()
    uparea_values = np.where(uparea_values <= 0, MIN_POSITIVE_VALUE, uparea_values)

    return longitude_coords, latitude_coords, uparea_values


def filter_grid_by_uparea(
    longitude_coords: np.ndarray,
    latitude_coords: np.ndarray,
    uparea_values: np.ndarray,
    threshold_km2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Keep only grid points whose upstream area is above the threshold.
    """
    lon_flat = longitude_coords.ravel()
    lat_flat = latitude_coords.ravel()
    up_flat = np.asarray(uparea_values).ravel()

    mask = up_flat >= threshold_km2

    logger.debug("Total grid points: %s", f"{len(up_flat):,}")
    logger.debug("Points above upstream area threshold: %s", f"{mask.sum():,}")
    logger.info("Upstream area threshold: %.1f km²", threshold_km2)

    lon_sel = lon_flat[mask]
    lat_sel = lat_flat[mask]
    up_sel = up_flat[mask]

    return lon_sel, lat_sel, up_sel


# ============================================================
# 3) HYDROBASINS DOMAIN SELECTION
# ============================================================

def build_domain_bbox(
    longitude_coords: np.ndarray,
    latitude_coords: np.ndarray,
    buffer_margin_deg: float,
):
    """
    Create buffered model-domain bounding box from grid coordinates.
    """
    lon_min = float(longitude_coords.min())
    lon_max = float(longitude_coords.max())
    lat_min = float(latitude_coords.min())
    lat_max = float(latitude_coords.max())

    logger.debug(
        "Model domain bounding box: lon %.3f -> %.3f | lat %.3f -> %.3f",
        lon_min,
        lon_max,
        lat_min,
        lat_max,
    )

    domain_bbox = box(
        lon_min - buffer_margin_deg,
        lat_min - buffer_margin_deg,
        lon_max + buffer_margin_deg,
        lat_max + buffer_margin_deg,
    )

    return domain_bbox


def load_and_clip_hydrobasins(
    hydrobasins_path: Path,
    domain_bbox,
    crs_epsg: int = 4326,
) -> gpd.GeoDataFrame:
    """
    Load HydroBASINS shapefile, enforce CRS, normalize HYBAS_ID,
    and keep only basins intersecting the domain bbox.
    """
    hydrobasins = gpd.read_file(hydrobasins_path)

    if hydrobasins.crs is None:
        hydrobasins = hydrobasins.set_crs(epsg=crs_epsg)
    else:
        hydrobasins = hydrobasins.to_crs(epsg=crs_epsg)

    if "HYBAS_ID" not in hydrobasins.columns:
        raise KeyError("Column 'HYBAS_ID' not found in HydroBASINS shapefile.")

    hydrobasins["HYBAS_ID"] = hydrobasins["HYBAS_ID"].astype(np.int64).astype(str)

    hydrobasins_clipped = hydrobasins[hydrobasins.intersects(domain_bbox)].copy()

    logger.debug("HydroBASINS selected: %s", len(hydrobasins_clipped))

    return hydrobasins_clipped

# ============================================================
# 4) BUILD RIVER POINTS GDF
# ============================================================

def build_river_points_gdf(
    lon_sel: np.ndarray,
    lat_sel: np.ndarray,
    up_sel: np.ndarray,
) -> gpd.GeoDataFrame:
    """
    Build GeoDataFrame of selected river candidate points.
    """
    gdf_river = gpd.GeoDataFrame(
        {
            "Longitude": lon_sel,
            "Latitude": lat_sel,
            "UpstreamArea": up_sel,
        },
        geometry=gpd.points_from_xy(lon_sel, lat_sel),
        crs="EPSG:4326",
    )

    logger.debug("River points kept: %s", f"{len(gdf_river):,}")
    return gdf_river


# ============================================================
# 5) ASSIGN HYDROBASINS TO RIVER POINTS
# ============================================================

def assign_hydrobasins_to_river_points(
    gdf_river: gpd.GeoDataFrame,
    hydrobasins_clipped: gpd.GeoDataFrame,
    max_distance_deg: float,
) -> gpd.GeoDataFrame:
    """
    Assign HYBAS_ID to river points:
    1. spatial join with 'within'
    2. nearest-basin recovery for unassigned points within max_distance_deg
    """
    gdf_river = gpd.sjoin(
        gdf_river,
        hydrobasins_clipped[["HYBAS_ID", "geometry"]],
        how="left",
        predicate="within",
    )

    assigned = gdf_river["HYBAS_ID"].notna().sum()
    logger.debug("Assigned by within: %s", f"{assigned:,}")

    unassigned = gdf_river[gdf_river["HYBAS_ID"].isna()].copy()
    logger.debug("Unassigned after within: %s", f"{len(unassigned):,}")

    if not unassigned.empty:
        if "index_right" in unassigned.columns:
            unassigned = unassigned.drop(columns=["index_right"])

        nearest = gpd.sjoin_nearest(
            unassigned,
            hydrobasins_clipped[["HYBAS_ID", "geometry"]],
            how="left",
            max_distance=max_distance_deg,
            distance_col="dist_deg",
        )

        hybas_right_col = "HYBAS_ID_right" if "HYBAS_ID_right" in nearest.columns else "HYBAS_ID"
        recovered = nearest[hybas_right_col].notna().sum()
        logger.debug("Recovered by nearest: %s", f"{recovered:,}")

        gdf_river.loc[nearest.index, "HYBAS_ID"] = nearest[hybas_right_col]

    gdf_river = gdf_river.dropna(subset=["HYBAS_ID"]).copy()
    gdf_river["HYBAS_ID"] = (
        gdf_river["HYBAS_ID"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )

    gdf_river = gdf_river.drop(
        columns=[col for col in gdf_river.columns if col.startswith("index_")],
        errors="ignore",
    )

    logger.debug("Final river points with basin: %s", f"{len(gdf_river):,}")
    logger.debug("Basins retained: %s", f"{gdf_river['HYBAS_ID'].nunique():,}")

    return gdf_river


# ============================================================
# 6) CLUSTER RIVER POINTS BASIN BY BASIN
# ============================================================

def cluster_river_points_by_basin(
    gdf_river: gpd.GeoDataFrame,
    max_distance_deg: float,
    min_samples: int = 1,
) -> pd.DataFrame:
    """
    Cluster river points independently within each basin using DBSCAN
    and return the cleaned cluster table ready for downstream steps.
    """
    required_cols = {"HYBAS_ID", "Longitude", "Latitude"}
    missing_cols = required_cols - set(gdf_river.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in gdf_river: {sorted(missing_cols)}")

    logger.debug("Clustering river points basin by basin using DBSCAN.")
    logger.debug("DBSCAN max_distance_deg: %.4f deg", max_distance_deg)

    clustered_list = []

    for hybas_id, group in gdf_river.groupby("HYBAS_ID"):
        coords = group[["Longitude", "Latitude"]].to_numpy()
        if len(coords) == 0:
            continue

        db = DBSCAN(
            eps=max_distance_deg,
            min_samples=min_samples,
            metric="euclidean",
        ).fit(coords)

        group = group.copy()
        group["cluster"] = db.labels_

        n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        #print(f"  HYBAS_ID {hybas_id}: {n_clusters} cluster(s), {len(group)} point(s)")

        clustered_list.append(group)

    if not clustered_list:
        raise ValueError("No clusters were produced. Check gdf_river input.")

    clustered_points = pd.concat(clustered_list, ignore_index=True)
    logger.debug("Clustering completed for %s river points.", f"{len(clustered_points):,}")

    filtered_clusters = clustered_points[clustered_points["cluster"] != -1].copy()

    filtered_clusters["HYBAS_ID"] = (
        filtered_clusters["HYBAS_ID"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )

    filtered_clusters["cluster"] = filtered_clusters["cluster"].astype(int)

    n_clusters_retained = (
        filtered_clusters[["HYBAS_ID", "cluster"]]
        .drop_duplicates()
        .shape[0]
    )

    logger.debug("Clusters retained: %s", n_clusters_retained)

    return filtered_clusters


# ============================================================
# 7) DRAINAGE NETWORK
# ============================================================

def append_unique_prev(results, node_id, upstream_id):
    """
    Append upstream_id to results[node_id]["prev"] only if not already present.
    """
    if upstream_id not in results[node_id]["prev"]:
        results[node_id]["prev"].append(upstream_id)


def compute_min_delta_from_cluster(df_cluster, grid_step, factor=0.85):
    """
    Compute min_delta as a fraction of one grid-cell area (km²),
    using the most conservative location in the cluster
    (maximum latitude -> minimum cell area).
    """
    lat0 = float(df_cluster["Latitude"].max())
    lon0 = float(df_cluster.loc[df_cluster["Latitude"].idxmax(), "Longitude"])

    h = grid_step / 2.0

    lons = [lon0 - h, lon0 + h, lon0 + h, lon0 - h, lon0 - h]
    lats = [lat0 - h, lat0 - h, lat0 + h, lat0 + h, lat0 - h]

    area_m2, _ = _GEOD.polygon_area_perimeter(lons, lats)
    cell_km2 = abs(area_m2) / 1e6

    min_delta = factor * cell_km2
    return float(min_delta)


def build_downstream_graph_single_basin(df, max_distance_deg, min_delta, max_gap):
    """
    Build an initial downstream graph for a single basin/cluster using UpstreamArea.
    Internal orientation is reversed and will be flipped later.
    """
    df_sorted = df.sort_values("UpstreamArea", ascending=False).reset_index(drop=True)
    coords = df_sorted[["Longitude", "Latitude"]].to_numpy()
    upvals = df_sorted["UpstreamArea"].to_numpy()
    kdt = KDTree(coords)
    n = len(df_sorted)

    prev = [None] * n
    succ = [None] * n
    used = [False] * n

    for start in range(n):
        if prev[start] is not None or succ[start] is not None or used[start]:
            continue

        current = start

        while True:
            used[current] = True
            neigh = kdt.query_ball_point(coords[current], max_distance_deg)

            candidates = []
            for j in neigh:
                if j == current:
                    continue

                delta = upvals[current] - upvals[j]

                if delta < min_delta:
                    continue

                if used[j]:
                    continue

                candidates.append(j)

            if not candidates:
                break

            target = upvals[current] - min_delta

            filtered = []
            for j in candidates:
                if upvals[j] <= target:
                    diff = target - upvals[j]
                    filtered.append((j, diff))

            if not filtered:
                break

            best, best_diff = min(filtered, key=lambda x: x[1])

            if best_diff > max_gap:
                break

            succ[current] = best
            prev[best] = current
            current = best

    results = {}
    for i in range(n):
        results[i] = {
            "prev": prev[i],
            "succ": succ[i],
            "uparea": float(upvals[i]),
        }

    return results, df_sorted


def run_downstream_graph_all_basins(
    filtered_clusters,
    max_distance_deg,
    grid_step,
    max_gap,
    min_delta_factor=0.85,
):
    """
    Apply the initial graph construction to all basin/cluster subsets.
    """
    graphs = {}

    for hybas_id, df_basin in filtered_clusters.groupby("HYBAS_ID"):
        graphs[hybas_id] = {}

        for cluster_id, df_cluster in df_basin.groupby("cluster"):
            min_delta = compute_min_delta_from_cluster(
                df_cluster,
                grid_step=grid_step,
                factor=min_delta_factor,
            )

            results, df_sorted = build_downstream_graph_single_basin(
                df_cluster,
                max_distance_deg=max_distance_deg,
                min_delta=min_delta,
                max_gap=max_gap,
            )

            graphs[hybas_id][cluster_id] = {
                "results": results,
                "df_sorted": df_sorted,
                "min_delta": min_delta,
            }

    return graphs


def flip_graph_orientation(all_graphs):
    """
    Flip graph orientation so that:
    - prev = list of upstream nodes
    - succ = unique downstream node
    """
    flipped = copy.deepcopy(all_graphs)

    for hyb in flipped:
        for clu in flipped[hyb]:
            results = flipped[hyb][clu]["results"]

            for _, info in results.items():
                old_prev = info["prev"]
                old_succ = info["succ"]

                if old_succ is None:
                    new_prev = []
                else:
                    new_prev = [old_succ]

                if isinstance(old_prev, list):
                    new_succ = old_prev[0] if old_prev else None
                else:
                    new_succ = old_prev

                info["prev"] = new_prev
                info["succ"] = new_succ

    return flipped


def connect_lonely_extremes(results, df_sorted, max_distance_deg):
    """
    Connect mutual lonely extremes that are within max_distance_deg.
    """
    coords = df_sorted[["Longitude", "Latitude"]].to_numpy()
    upvals = df_sorted["UpstreamArea"].to_numpy()
    kdt = KDTree(coords)

    extremes = [
        i for i, info in results.items()
        if len(info["prev"]) == 0 or info["succ"] is None
    ]
    extremes_set = set(extremes)

    neighbor_map = {}
    for i in extremes:
        neigh = kdt.query_ball_point(coords[i], max_distance_deg)
        neighbor_map[i] = [j for j in neigh if j in extremes_set and j != i]

    pairs = []
    used = set()

    for i in extremes:
        if i in used:
            continue

        neighs = neighbor_map[i]
        if len(neighs) != 1:
            continue

        j = neighs[0]
        if j in used:
            continue

        if len(neighbor_map[j]) != 1:
            continue

        pairs.append((i, j))
        used.add(i)
        used.add(j)

    for a, b in pairs:
        if upvals[a] < upvals[b]:
            upstream_node, downstream_node = a, b
        else:
            upstream_node, downstream_node = b, a

        results[upstream_node]["succ"] = downstream_node
        append_unique_prev(results, downstream_node, upstream_node)

    return results


def run_connect_lonely_extremes_all(all_graphs, max_distance_deg):
    """
    Apply lonely-extreme connection to all basin/cluster graphs.
    """
    for hyb in all_graphs:
        for clu in all_graphs[hyb]:
            pack = all_graphs[hyb][clu]
            pack["results"] = connect_lonely_extremes(
                pack["results"],
                pack["df_sorted"],
                max_distance_deg=max_distance_deg,
            )

    return all_graphs


def get_existing_chains(results):
    """
    Return connected components of the graph, ignoring direction.
    """
    graph = defaultdict(list)

    for node_id, info in results.items():
        for p in info["prev"]:
            graph[node_id].append(p)
            graph[p].append(node_id)

        s = info["succ"]
        if s is not None:
            graph[node_id].append(s)
            graph[s].append(node_id)

    visited = set()
    chains = []

    for node in results:
        if node in visited:
            continue

        stack = [node]
        component = []
        visited.add(node)

        while stack:
            u = stack.pop()
            component.append(u)

            for v in graph[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)

        chains.append(set(component))

    return chains


def group_extreme_points_with_chains(results, df_sorted, max_distance_deg):
    """
    Identify groups of extreme nodes that are spatially close
    but belong to different existing chains.
    """
    coords = df_sorted[["Longitude", "Latitude"]].to_numpy()

    extremes = [
        i for i, info in results.items()
        if len(info["prev"]) == 0 or info["succ"] is None
    ]
    extremes_set = set(extremes)

    chains = get_existing_chains(results)

    chain_id = {}
    for idx, comp in enumerate(chains):
        for n in comp:
            chain_id[n] = idx

    kdt = KDTree(coords)
    neighbors = {}

    for i in extremes:
        neigh = kdt.query_ball_point(coords[i], max_distance_deg)

        valid_neighbors = [
            j for j in neigh
            if j in extremes_set
            and j != i
            and chain_id[j] != chain_id[i]
        ]

        if valid_neighbors:
            neighbors[i] = valid_neighbors

    visited = set()
    groups = []

    for i in neighbors:
        if i in visited:
            continue

        stack = [i]
        group = []
        visited.add(i)

        while stack:
            u = stack.pop()
            group.append(u)

            for nb in neighbors.get(u, []):
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)

        if len(group) >= 3:
            groups.append(group)

    return groups


def resolve_extreme_group(
    results,
    df_sorted,
    group,
    max_distance_deg,
    min_delta,
    max_gap,
):
    """
    Resolve one group of extreme nodes by selecting the dominant node
    and attaching a consistent subset of nearby children.
    """
    from itertools import combinations

    coords = df_sorted[["Longitude", "Latitude"]].to_numpy()
    upvals = df_sorted["UpstreamArea"].to_numpy()

    node = max(group, key=lambda i: upvals[i])
    node_up = upvals[node]
    node_xy = coords[node]

    others = [i for i in group if i != node]

    children_near = [
        i for i in others
        if np.linalg.norm(coords[i] - node_xy) <= max_distance_deg
    ]

    if len(children_near) == 0:
        return results

    if len(children_near) == 1:
        c = children_near[0]
        append_unique_prev(results, node, c)
        results[c]["succ"] = node
        return results

    if len(group) == 3 and len(children_near) == 2:
        c1, c2 = children_near
        s = upvals[c1] + upvals[c2]

        if not (s + min_delta < node_up and (node_up - s) < max_gap):
            return results

        selected = [c1, c2]

    else:
        best_subset = None
        best_gap = None

        for r in range(2, len(children_near) + 1):
            for subset in combinations(children_near, r):
                s = sum(upvals[i] for i in subset)

                if s + min_delta < node_up and (node_up - s) < max_gap:
                    gap = node_up - s
                    if best_gap is None or gap < best_gap:
                        best_gap = gap
                        best_subset = subset

        if best_subset is None:
            return results

        selected = list(best_subset)

    for c in selected:
        append_unique_prev(results, node, c)
        results[c]["succ"] = node

    main_child = max(selected, key=lambda i: upvals[i])
    results[main_child]["succ"] = node

    return results


def resolve_all_extreme_groups_single_basin(
    results,
    df_sorted,
    max_distance_deg,
    grid_step,
    threshold_uparea,
):
    """
    Resolve all extreme groups in one basin/cluster.
    """
    min_delta = compute_min_delta_from_cluster(
        df_sorted,
        grid_step=grid_step,
        factor=0.85,
    )

    max_gap = threshold_uparea

    groups = group_extreme_points_with_chains(
        results,
        df_sorted,
        max_distance_deg=max_distance_deg,
    )

    for g in groups:
        results = resolve_extreme_group(
            results,
            df_sorted,
            g,
            max_distance_deg=max_distance_deg,
            min_delta=min_delta,
            max_gap=max_gap,
        )

    return results


def resolve_fail_3point_case(results, df_sorted, group, min_delta):
    """
    Relaxed fix for unresolved 3-point groups.
    """
    upvals = df_sorted["UpstreamArea"].to_numpy()

    node = max(group, key=lambda i: upvals[i])
    node_up = upvals[node]
    children = [i for i in group if i != node]

    s = upvals[children[0]] + upvals[children[1]]

    if not (s + min_delta < node_up):
        return results

    for c in children:
        append_unique_prev(results, node, c)
        results[c]["succ"] = node

    main_child = max(children, key=lambda i: upvals[i])
    results[main_child]["succ"] = node

    return results


def resolve_no_subset_case(results, df_sorted, group, max_distance_deg):
    """
    Relaxed fix for unresolved groups where no valid subset was found.
    """
    from itertools import combinations

    coords = df_sorted[["Longitude", "Latitude"]].to_numpy()
    upvals = df_sorted["UpstreamArea"].to_numpy()

    node = max(group, key=lambda i: upvals[i])
    node_up = upvals[node]
    node_xy = coords[node]

    candidates = [
        i for i in results
        if i != node and np.linalg.norm(coords[i] - node_xy) <= max_distance_deg
    ]

    if len(candidates) < 2:
        return results

    best_subset = None
    best_gap = None

    for r in range(2, len(candidates) + 1):
        for subset in combinations(candidates, r):
            s = sum(upvals[i] for i in subset)
            if s < node_up:
                gap = node_up - s
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best_subset = subset

    if best_subset is None:
        return results

    selected = list(best_subset)
    local = set(selected + [node])

    for i in selected:
        results[i]["prev"] = [
            p for p in results[i]["prev"]
            if p not in local or p == node
        ]

        s = results[i]["succ"]
        if s is not None and s in local and s != node:
            results[i]["succ"] = None

    for c in selected:
        append_unique_prev(results, node, c)
        results[c]["succ"] = node

    main_child = max(selected, key=lambda i: upvals[i])
    results[main_child]["succ"] = node

    return results


def iterate_extreme_group_resolution(
    all_graphs,
    max_distance_deg,
    grid_step,
    threshold_uparea,
    max_iter=20,
):
    """
    Iteratively resolve extreme groups, then apply one relaxed correction phase.
    """
    graphs = copy.deepcopy(all_graphs)

    for _ in range(max_iter):
        merged_in_this_iter = 0

        for hyb, clusters in graphs.items():
            for clu, pack in clusters.items():
                before = copy.deepcopy(pack["results"])

                pack["results"] = resolve_all_extreme_groups_single_basin(
                    pack["results"],
                    pack["df_sorted"],
                    max_distance_deg=max_distance_deg,
                    grid_step=grid_step,
                    threshold_uparea=threshold_uparea,
                )

                if before != pack["results"]:
                    merged_in_this_iter += 1

        if merged_in_this_iter == 0:
            break

    for hyb, clusters in graphs.items():
        for clu, pack in clusters.items():
            results = pack["results"]
            df_sorted = pack["df_sorted"]

            min_delta = compute_min_delta_from_cluster(
                df_sorted,
                grid_step=grid_step,
                factor=0.85,
            )

            groups = group_extreme_points_with_chains(
                results,
                df_sorted,
                max_distance_deg=max_distance_deg,
            )

            for g in groups:
                if len(g) == 3:
                    results = resolve_fail_3point_case(
                        results,
                        df_sorted,
                        g,
                        min_delta=min_delta,
                    )
                else:
                    results = resolve_no_subset_case(
                        results,
                        df_sorted,
                        g,
                        max_distance_deg=max_distance_deg,
                    )

            pack["results"] = results

    return graphs


def attach_dangling_sinks_single(
    results,
    df_sorted,
    max_distance_deg,
    min_delta,
    max_gap,
):
    """
    Attach non-terminal local sinks to nearby downstream nodes from other chains.
    """
    coords = df_sorted[["Longitude", "Latitude"]].to_numpy()
    upvals = df_sorted["UpstreamArea"].to_numpy()

    real_outlet = int(np.argmax(upvals))

    sinks = [
        i for i, info in results.items()
        if info["succ"] is None and i != real_outlet
    ]

    if not sinks:
        return results

    chains = get_existing_chains(results)
    chain_id = {}
    for idx, comp in enumerate(chains):
        for node in comp:
            chain_id[node] = idx

    kdt = KDTree(coords)

    for i in sinks:
        up_i = upvals[i]
        neigh = kdt.query_ball_point(coords[i], max_distance_deg)

        candidates = [
            j for j in neigh
            if j != i and j in results and chain_id.get(j) != chain_id.get(i)
        ]

        if not candidates:
            continue

        valid_candidates = []

        for j in candidates:
            up_j = upvals[j]

            if up_j < up_i + min_delta:
                continue

            children_j = [k for k, info in results.items() if info["succ"] == j]

            if children_j:
                sum_children = sum(upvals[k] for k in children_j)
                sum_new = sum_children + up_i
            else:
                sum_new = up_i

            if sum_new + min_delta <= up_j and (up_j - sum_new) <= max_gap:
                gap = abs((up_j - sum_new) - min_delta)
                valid_candidates.append((j, gap))

        if valid_candidates:
            j_best = min(valid_candidates, key=lambda x: x[1])[0]
        else:
            best_eff_list = []
            for j in candidates:
                up_j = upvals[j]
                if up_j <= up_i:
                    continue
                diff = abs((up_j - up_i) - min_delta)
                best_eff_list.append((j, diff))

            if not best_eff_list:
                continue

            j_best = min(best_eff_list, key=lambda x: x[1])[0]

        old_succ = results[i]["succ"]
        if old_succ is not None and old_succ != j_best:
            results[old_succ]["prev"] = [
                p for p in results[old_succ]["prev"] if p != i
            ]

        results[i]["succ"] = j_best
        append_unique_prev(results, j_best, i)

    return results


def attach_dangling_sinks_all(
    graphs_after,
    max_distance_deg,
    grid_step,
    threshold_uparea,
    min_delta_factor=0.85,
):
    """
    Apply dangling-sink attachment to all basin/cluster graphs.
    """
    graphs = copy.deepcopy(graphs_after)

    for hyb in graphs:
        for clu in graphs[hyb]:
            pack = graphs[hyb][clu]

            min_delta = compute_min_delta_from_cluster(
                pack["df_sorted"],
                grid_step=grid_step,
                factor=min_delta_factor,
            )

            pack["min_delta"] = min_delta
            pack["results"] = attach_dangling_sinks_single(
                pack["results"],
                pack["df_sorted"],
                max_distance_deg=max_distance_deg,
                min_delta=min_delta,
                max_gap=threshold_uparea,
            )

    return graphs


def split_by_final_sink(results):
    """
    Group nodes by their terminal sink.
    """
    sink_of = {}

    for node in results:
        curr = node
        visited = set()

        while results[curr]["succ"] is not None:
            if curr in visited:
                raise RuntimeError("Cycle detected (should never happen)")
            visited.add(curr)
            curr = results[curr]["succ"]

        sink_of[node] = curr

    groups = {}
    for node, sink in sink_of.items():
        groups.setdefault(sink, []).append(node)

    return groups


def remap_cluster(results, df_sorted, nodes):
    """
    Build a new (results, df_sorted) pair with node ids remapped to 0..N-1.
    """
    nodes = sorted(nodes)
    id_map = {old: new for new, old in enumerate(nodes)}

    new_df = df_sorted.iloc[nodes].reset_index(drop=True)
    new_results = {}

    for old_i in nodes:
        info = results[old_i]
        new_i = id_map[old_i]

        new_prev = [id_map[p] for p in info["prev"] if p in id_map]
        new_succ = id_map[info["succ"]] if info["succ"] in id_map else None

        new_results[new_i] = {
            "prev": new_prev,
            "succ": new_succ,
            "uparea": info["uparea"],
        }

    return new_results, new_df


def split_clusters_with_multiple_chains(graphs):
    """
    Split each cluster with multiple final sinks into one cluster per chain.
    """
    new_graphs = {}

    for hyb in graphs:
        new_graphs[hyb] = {}

        existing_clusters = sorted(graphs[hyb].keys())
        next_cluster_id = max(existing_clusters) + 1 if existing_clusters else 0

        for clu in existing_clusters:
            pack = graphs[hyb][clu]
            results = pack["results"]
            df_sorted = pack["df_sorted"]

            sink_groups = split_by_final_sink(results)
            chains = list(sink_groups.values())

            if len(chains) == 1:
                new_graphs[hyb][clu] = pack
                continue

            for k, chain in enumerate(chains):
                new_results, new_df = remap_cluster(results, df_sorted, chain)

                if k == 0:
                    new_clu = clu
                else:
                    new_clu = next_cluster_id
                    next_cluster_id += 1

                new_graphs[hyb][new_clu] = {
                    "results": new_results,
                    "df_sorted": new_df,
                    "min_delta": pack.get("min_delta", 0.0),
                }

    return new_graphs


def _merge_two_clusters(graphs, upstream_cluster, downstream_cluster, sink_node, head_node):
    """
    Merge upstream_cluster into downstream_cluster and connect:
        sink_node (upstream cluster) -> head_node (downstream cluster)
    """
    hyb_u, clu_u = upstream_cluster
    hyb_d, clu_d = downstream_cluster

    pack_u = graphs[hyb_u][clu_u]
    pack_d = graphs[hyb_d][clu_d]

    res_u = pack_u["results"]
    res_d = pack_d["results"]

    df_u = pack_u["df_sorted"]
    df_d = pack_d["df_sorted"]

    offset = max(res_d.keys()) + 1 if res_d else 0
    id_map = {old: offset + i for i, old in enumerate(sorted(res_u.keys()))}

    df_u_remap = df_u.copy()
    df_u_remap.index = [id_map[i] for i in df_u_remap.index]
    pack_d["df_sorted"] = pd.concat([df_d, df_u_remap], axis=0)

    for old_id, info in res_u.items():
        new_id = id_map[old_id]
        pack_d["results"][new_id] = {
            "prev": [id_map[p] for p in info["prev"]],
            "succ": id_map[info["succ"]] if info["succ"] is not None else None,
            "uparea": info["uparea"],
        }

    sink_new = id_map[sink_node]
    head_existing = head_node

    pack_d["results"][sink_new]["succ"] = head_existing
    append_unique_prev(pack_d["results"], head_existing, sink_new)

    pack_d["min_delta"] = max(
        pack_d.get("min_delta", 0.0),
        pack_u.get("min_delta", 0.0),
    )

    del graphs[hyb_u][clu_u]


def _reorder_all_clusters(graphs):
    """
    Reorder node ids inside each cluster to a compact 0..N-1 indexing.
    """
    for hyb in graphs:
        for clu, pack in graphs[hyb].items():
            res = pack["results"]
            df = pack["df_sorted"]

            visited = set()
            order = []

            def dfs(n):
                if n in visited:
                    return
                visited.add(n)
                order.append(n)
                if res[n]["succ"] is not None:
                    dfs(res[n]["succ"])

            for node in res:
                if node not in visited:
                    dfs(node)

            df2 = df.loc[order].copy()
            df2.index = range(len(df2))

            id_map = dict(zip(order, df2.index))

            new_res = {}
            for old_id in order:
                info = res[old_id]
                new_res[id_map[old_id]] = {
                    "prev": [id_map[p] for p in info["prev"] if p in id_map],
                    "succ": id_map[info["succ"]] if info["succ"] is not None else None,
                    "uparea": info["uparea"],
                }

            pack["df_sorted"] = df2
            pack["results"] = new_res

    return graphs


def merge_clusters_by_sink_headwater(
    graphs_input,
    max_distance_deg,
    max_iter=10,
    verbose=True,
):
    """
    Merge clusters across basin-local components using strict sink->headwater matching.
    """
    graphs = copy.deepcopy(graphs_input)
    total_merges = 0

    for it in range(max_iter):
        if verbose:
            logger.debug("Merge iteration %s/%s", it + 1, max_iter)

        sinks = []
        heads = []
        head_index = []

        for hyb in graphs:
            for clu, pack in graphs[hyb].items():
                res = pack["results"]
                df = pack["df_sorted"]
                coords = df[["Longitude", "Latitude"]].to_numpy()

                for i, info in res.items():
                    if info["succ"] is None:
                        sinks.append((hyb, clu, i, coords[i, 0], coords[i, 1]))

                    if len(info["prev"]) == 0:
                        heads.append((coords[i, 0], coords[i, 1]))
                        head_index.append((hyb, clu, i))

        if not sinks or not heads:
            break

        tree = cKDTree(np.array(heads))
        sink_to_heads = defaultdict(list)
        head_to_sinks = defaultdict(list)

        for hyb_s, clu_s, sink_node, lon, lat in sinks:
            idxs = tree.query_ball_point([lon, lat], r=max_distance_deg)

            for idx in idxs:
                hyb_h, clu_h, head_node = head_index[idx]

                if hyb_s == hyb_h and clu_s == clu_h:
                    continue

                sink_to_heads[(hyb_s, clu_s, sink_node)].append((hyb_h, clu_h, head_node))
                head_to_sinks[(hyb_h, clu_h, head_node)].append((hyb_s, clu_s, sink_node))

        merges_this_iter = 0
        used_clusters = set()

        for sink_key, head_list in sink_to_heads.items():
            if len(head_list) != 1:
                continue

            head_key = head_list[0]

            if len(head_to_sinks[head_key]) != 1:
                continue

            hyb_u, clu_u, sink_node = sink_key
            hyb_d, clu_d, head_node = head_key

            if (hyb_u, clu_u) in used_clusters or (hyb_d, clu_d) in used_clusters:
                continue

            df_u = graphs[hyb_u][clu_u]["df_sorted"]
            df_d = graphs[hyb_d][clu_d]["df_sorted"]

            up_sink = float(df_u.loc[sink_node, "UpstreamArea"])
            up_head = float(df_d.loc[head_node, "UpstreamArea"])

            min_delta_u = graphs[hyb_u][clu_u].get("min_delta", 0.0)
            min_delta_d = graphs[hyb_d][clu_d].get("min_delta", 0.0)
            min_delta_eff = max(min_delta_u, min_delta_d)

            if verbose:
                logger.debug("[MERGE] upstream %s -> downstream %s", (hyb_u, clu_u), (hyb_d, clu_d))

            _merge_two_clusters(
                graphs,
                upstream_cluster=(hyb_u, clu_u),
                downstream_cluster=(hyb_d, clu_d),
                sink_node=sink_node,
                head_node=head_node,
            )

            used_clusters.add((hyb_u, clu_u))
            used_clusters.add((hyb_d, clu_d))

            merges_this_iter += 1
            total_merges += 1

        if verbose:
            logger.debug("Iteration %s: merges=%s", it + 1, merges_this_iter)

        if merges_this_iter == 0:
            break

    if verbose:
        logger.debug("Total merges performed: %s", total_merges)

    return _reorder_all_clusters(graphs)

def compute_strahler(results):
    """
    Compute Strahler order for a directed graph without recursion.
    Uses only prev links, so it is robust to minor prev/succ inconsistencies.
    """
    valid_prev = {
        node: [p for p in info["prev"] if p in results and p != node]
        for node, info in results.items()
    }

    downstream_from_prev = {node: [] for node in results}

    for node, prev_nodes in valid_prev.items():
        for p in prev_nodes:
            downstream_from_prev[p].append(node)

    pending = {
        node: len(prev_nodes)
        for node, prev_nodes in valid_prev.items()
    }

    upstream_orders = {node: [] for node in results}

    queue = [
        node
        for node, n_prev in pending.items()
        if n_prev == 0
    ]

    strahler = {}

    while queue:
        node = queue.pop()

        orders = upstream_orders[node]

        if not orders:
            order = 1
        else:
            max_order = max(orders)
            order = max_order + 1 if orders.count(max_order) >= 2 else max_order

        strahler[node] = order

        for downstream_node in downstream_from_prev[node]:
            upstream_orders[downstream_node].append(order)
            pending[downstream_node] -= 1

            if pending[downstream_node] == 0:
                queue.append(downstream_node)

    if len(strahler) != len(results):
        missing = set(results) - set(strahler)
        raise RuntimeError(
            f"Could not compute Strahler for all nodes. "
            f"Possible closed prev-component or duplicated/inconsistent upstream links. "
            f"Missing nodes: {list(missing)[:20]}"
        )

    return strahler

def build_saber_drain_table(graphs):
    """
    Build final drainage table from merged drainage graphs.

    The returned table contains both:
    - SABER-ready fields
    - extra topology fields useful for later EFAS extraction/correction
    """
    rows = []

    for hybas_id, basin_clusters in graphs.items():
        for cluster_id, cluster_data in basin_clusters.items():
            results = cluster_data["results"]
            df_coords = cluster_data["df_sorted"]

            coords = {
                int(idx): (row["Longitude"], row["Latitude"])
                for idx, row in df_coords.iterrows()
            }

            strahler = compute_strahler(results)

            for node, attrs in results.items():
                succ = attrs["succ"]
                prev_nodes = attrs["prev"]
                uparea = attrs["uparea"]

                model_id = f"{hybas_id}_{cluster_id}_{node}"

                downstream_model_id = (
                    None if succ is None else f"{hybas_id}_{cluster_id}_{succ}"
                )

                upstream_model_ids = ";".join(
                    f"{hybas_id}_{cluster_id}_{p}" for p in prev_nodes
                )

                lon, lat = coords[node]

                rows.append(
                    {
                        "HYBAS_ID": str(hybas_id),
                        "cluster": int(cluster_id),
                        "node": int(node),
                        "model_id": model_id,
                        "downstream_model_id": downstream_model_id,
                        "upstream_model_ids": upstream_model_ids,
                        "strahler_order": int(strahler[node]),
                        "drainage_area": float(uparea),
                        "x": float(lon),
                        "y": float(lat),
                    }
                )

    return pd.DataFrame(rows)


def save_drain_outputs(
    df_drain,
    saber_csv_path,
    full_parquet_path,
):
    """
    Save two versions of the drainage table:
    - compact CSV for SABER
    - full Parquet for later EFAS matching/correction
    """
    saber_cols = [
        "model_id",
        "downstream_model_id",
        "strahler_order",
        "drainage_area",
        "x",
        "y",
    ]

    df_drain[saber_cols].to_csv(saber_csv_path, index=False)
    df_drain.to_parquet(full_parquet_path, index=False)

    logger.info("SABER drain CSV saved: %s", saber_csv_path)
    logger.info("Full drain Parquet saved: %s", full_parquet_path)


def build_drain_gis(df_drain, output_path, crs="EPSG:4326"):
    """
    Create a GeoPackage with one point per drainage node.
    """
    gdf = gpd.GeoDataFrame(
        df_drain[["model_id"]].copy(),
        geometry=gpd.points_from_xy(df_drain["x"], df_drain["y"]),
        crs=crs,
    )

    gdf.to_file(output_path, driver="GPKG")
    return gdf

# ============================================================
# MAIN
# ============================================================

def main() -> None:
    setup_logging()
    log_config_summary(CFG)

    t0 = time.time()

    uparea_km2 = load_and_prepare_upstream_area(
        uparea_path=UPAREA_PATH,
        variable_name=UPAREA_VARIABLE_NAME,
        lon_min=LON_MIN,
        lon_max=LON_MAX,
        lat_min=LAT_MIN,
        lat_max=LAT_MAX,
        min_positive_value=MIN_POSITIVE_VALUE,
    )

    longitude_coords, latitude_coords, uparea_values = build_grid_from_uparea(uparea_km2)

    lon_sel, lat_sel, up_sel = filter_grid_by_uparea(
        longitude_coords=longitude_coords,
        latitude_coords=latitude_coords,
        uparea_values=uparea_values,
        threshold_km2=THRESHOLD_UPAREA_KM2,
    )

    domain_bbox = build_domain_bbox(
        longitude_coords=longitude_coords,
        latitude_coords=latitude_coords,
        buffer_margin_deg=BUFFER_MARGIN_DEG,
    )

    hydrobasins_clipped = load_and_clip_hydrobasins(
        hydrobasins_path=HYDROBASINS_PATH,
        domain_bbox=domain_bbox,
        crs_epsg=HYDROBASINS_CRS_EPSG,
    )

    gdf_river = build_river_points_gdf(
        lon_sel=lon_sel,
        lat_sel=lat_sel,
        up_sel=up_sel,
    )

    gdf_river = assign_hydrobasins_to_river_points(
        gdf_river=gdf_river,
        hydrobasins_clipped=hydrobasins_clipped,
        max_distance_deg=MAX_DISTANCE_DEG,
    )

    filtered_clusters = cluster_river_points_by_basin(
        gdf_river=gdf_river,
        max_distance_deg=MAX_DISTANCE_DEG,
        min_samples=MIN_SAMPLES_DBSCAN,
    )

    logger.debug("Workflow completed up to filtered clusters.")
    logger.debug("Final filtered cluster rows: %s", f"{len(filtered_clusters):,}")

    logger.info("Starting drainage network construction.")

    all_graphs = run_downstream_graph_all_basins(
        filtered_clusters,
        max_distance_deg=MAX_DISTANCE_DEG,
        grid_step=GRID_STEP,
        max_gap=THRESHOLD_UPAREA_KM2,
        min_delta_factor=0.85,
    )

    all_graphs = flip_graph_orientation(all_graphs)

    all_graphs = run_connect_lonely_extremes_all(
        all_graphs,
        max_distance_deg=MAX_DISTANCE_DEG,
    )

    graphs_after = iterate_extreme_group_resolution(
        all_graphs,
        max_distance_deg=MAX_DISTANCE_DEG,
        grid_step=GRID_STEP,
        threshold_uparea=THRESHOLD_UPAREA_KM2,
    )

    graphs_fixed = attach_dangling_sinks_all(
        graphs_after,
        max_distance_deg=MAX_DISTANCE_DEG,
        grid_step=GRID_STEP,
        threshold_uparea=THRESHOLD_UPAREA_KM2,
        min_delta_factor=0.85,
    )

    graphs_fixed = split_clusters_with_multiple_chains(graphs_fixed)

    graphs_merged = merge_clusters_by_sink_headwater(
        graphs_fixed,
        max_distance_deg=MAX_DISTANCE_DEG,
        max_iter=10,
        verbose=False,
    )

    df_drain = build_saber_drain_table(graphs_merged)

    save_drain_outputs(
        df_drain=df_drain,
        saber_csv_path=DRAIN_CSV_OUTPUT_PATH,
        full_parquet_path=DRAIN_PARQUET_OUTPUT_PATH,
    )

    gdf_drain = build_drain_gis(
        df_drain=df_drain,
        output_path=DRAIN_GIS_OUTPUT_PATH,
    )

    logger.debug("Drain GIS saved: %s", DRAIN_GIS_OUTPUT_PATH)
    logger.debug("Drain GIS created: %s points", f"{len(gdf_drain):,}")

    logger.info("Drainage network construction completed.")
    logger.debug("Total time: %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()