#!/usr/bin/env python3

from pathlib import Path
import argparse
import logging
import warnings

import matplotlib
matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from matplotlib.lines import Line2D
import os


# ============================================================
# LOGGING / WARNINGS
# ============================================================

logging.getLogger("pypsa").setLevel(logging.ERROR)
logging.getLogger("pypsa.network.io").setLevel(logging.ERROR)

warnings.filterwarnings(
    "ignore",
    message="Importing network from PyPSA version.*",
    category=Warning,
)

warnings.filterwarnings(
    "ignore",
    message="New version .* available.*",
    category=Warning,
)


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_BUS_NAME = "NO0 0"

ELECTRIC_CARRIERS = {
    "nuclear",
    "CCGT",
    "OCGT",
    "coal",
    "lignite",
    "oil",
    "biomass",
    "geothermal",
    "onwind",
    "offwind-ac",
    "offwind-dc",
    "offwind-float",
    "solar",
    "solar-hsat",
    "hydro",
    "ror",
    "PHS",
}

DRAW_TRANSMISSION = True
FLOW_CLIP_THRESHOLD = 1.0
MIN_LINEWIDTH = 2
MAX_LINEWIDTH = 10
TRANSMISSION_COLOR = "#d62728"

RESERVOIR_LINK_COLOR = "#9ecae1"
RESERVOIR_LINK_STYLE = "--"
RESERVOIR_LINK_WIDTH = 0.5

BUS_NODE_SIZE = 2000

CARRIER_RING_FACTOR = 1.18
ARROW_END_FACTOR = 1.5
PLOT_PADDING_FACTOR = 0.4

CARRIER_ANGLE_MARGIN_DEG = 16
ARROW_LABEL_DISTANCE_FACTOR = 1.72

SHOW_RESERVOIR_LABELS = False
MAX_RESERVOIR_LABELS = 40
RESERVOIR_LABEL_YOFFSET = 2500

CAPACITY_SIZE_MIN = 50
CAPACITY_SIZE_MAX = 3000

CAPACITY_LEGEND_VALUES = [100, 500, 2000]
FLOW_LEGEND_VALUES = [100, 500, 2000]

USE_FIXED_GLOBAL_SCALES = True

CAPACITY_REF_MIN = 0.0
CAPACITY_REF_MAX = 5000.0

FLOW_REF_MIN = 0.0
FLOW_REF_MAX = 3000.0

# ============================================================
# DEFAULT PATHS
# ============================================================

def get_repo_root():
    return Path(
        os.environ.get(
            "HYDRO_REPO_ROOT",
            Path(__file__).resolve().parents[1],
        )
    ).resolve()


def get_hydro_results_root():
    repo_root = get_repo_root()

    return Path(
        os.environ.get(
            "HYDRO_RESULTS_ROOT",
            repo_root / "hydro_results",
        )
    ).resolve()


DEFAULT_HYDRO_RESULTS_PATH = get_hydro_results_root()

# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate hydro representation comparison figure."
    )

    parser.add_argument(
        "--hydro-results",
        type=Path,
        default=DEFAULT_HYDRO_RESULTS_PATH,
        help="Path to hydro_results directory.",
    )

    parser.add_argument(
        "--bus-name",
        type=str,
        default=DEFAULT_BUS_NAME,
        help="Bus name to plot.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for generated figure.",
    )

    return parser.parse_args()


# ============================================================
# GENERAL HELPERS
# ============================================================

def save_figure(fig, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.close(fig)


def load_network(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing network file: {path}")

    return pypsa.Network(path, engine="netcdf4")


def find_bus_column(gdf, bus_name):
    candidates = ["name", "bus", "Bus", "id", "ID", "index", "region", "node"]

    for col in candidates:
        if col in gdf.columns:
            return col

    for col in gdf.columns:
        if gdf[col].dtype == "object":
            values = gdf[col].astype(str)
            if (values == bus_name).any():
                return col

    raise ValueError(
        "Could not identify the bus-name column in the geojson. "
        f"Available columns: {list(gdf.columns)}"
    )


def find_capacity_column(df):
    candidates = ["Capacity", "capacity", "p_nom", "p_nom_opt"]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "No capacity column found in powerplants CSV. "
        f"Available columns: {list(df.columns)}"
    )


def get_carrier_color(network, carrier, default="#bdbdbd"):
    if hasattr(network, "carriers") and carrier in network.carriers.index:
        color = network.carriers.loc[carrier].get("color", default)

        if pd.notna(color) and str(color).strip() != "":
            return str(color)

    return default


def get_carrier_nice_name(network, carrier):
    if hasattr(network, "carriers") and carrier in network.carriers.index:
        nice_name = network.carriers.loc[carrier].get("nice_name", carrier)

        if pd.notna(nice_name) and str(nice_name).strip() != "":
            return str(nice_name)

    return str(carrier)


def get_bus_capacities_from_network(network):
    parts = []

    if hasattr(network, "generators") and not network.generators.empty:
        generators = network.generators.copy()
        generators = generators[
            generators["carrier"].astype(str).str.lower() != "load"
        ]

        if "p_nom_opt" in generators.columns:
            parts.append(
                generators.groupby(["bus", "carrier"])["p_nom_opt"].sum()
            )
        elif "p_nom" in generators.columns:
            parts.append(
                generators.groupby(["bus", "carrier"])["p_nom"].sum()
            )

    if hasattr(network, "storage_units") and not network.storage_units.empty:
        storage_units = network.storage_units.copy()

        if "p_nom_opt" in storage_units.columns:
            parts.append(
                storage_units.groupby(["bus", "carrier"])["p_nom_opt"].sum()
            )
        elif "p_nom" in storage_units.columns:
            parts.append(
                storage_units.groupby(["bus", "carrier"])["p_nom"].sum()
            )

    if len(parts) == 0:
        return pd.DataFrame()

    return pd.concat(parts).groupby(level=[0, 1]).sum().unstack(fill_value=0)


def reservoir_mask(df):
    fuel = (
        df["Fueltype"].astype(str).str.lower()
        if "Fueltype" in df.columns
        else pd.Series("", index=df.index)
    )

    technology = (
        df["Technology"].astype(str).str.lower()
        if "Technology" in df.columns
        else pd.Series("", index=df.index)
    )

    fuel_ok = fuel.str.contains("hydro", na=False)
    technology_ok = (
        technology.str.contains("reservoir", na=False)
        | technology.str.contains("dam", na=False)
    )

    return fuel_ok & technology_ok


def get_other_aggregated_carriers_for_bus(network, bus_name, exclude_aggregated_carriers=None):
    if exclude_aggregated_carriers is None:
        exclude_aggregated_carriers = []

    capacities = get_bus_capacities_from_network(network)

    if bus_name not in capacities.index:
        return pd.Series(dtype=float)

    series = capacities.loc[bus_name].copy()
    series = series[series > 0]
    series = series[series.index.astype(str).isin(ELECTRIC_CARRIERS)]

    keep_mask = ~series.index.astype(str).str.lower().str.contains("reservoir|dam")
    series = series[keep_mask]

    for carrier in exclude_aggregated_carriers:
        if carrier in series.index:
            series = series.drop(carrier)

    return series.sort_values(ascending=False)


def make_size_mapper(vmin, vmax, size_min, size_max):
    def mapper(values):
        arr = np.asarray(values, dtype=float)

        if arr.size == 0:
            return np.array([])

        arr = np.nan_to_num(arr, nan=0.0)

        if np.isclose(vmax, vmin):
            return np.full(
                arr.shape,
                0.5 * (size_min + size_max),
                dtype=float,
            )

        clipped = np.clip(arr, vmin, vmax)

        return size_min + (clipped - vmin) * (size_max - size_min) / (vmax - vmin)

    return mapper


def make_linewidth_mapper(vmin, vmax, lw_min, lw_max):
    def mapper(values):
        arr = np.asarray(values, dtype=float)

        if arr.size == 0:
            return np.array([])

        arr = np.nan_to_num(arr, nan=0.0)

        if np.isclose(vmax, vmin):
            return np.full(
                arr.shape,
                0.5 * (lw_min + lw_max),
                dtype=float,
            )

        clipped = np.clip(arr, vmin, vmax)

        return lw_min + (clipped - vmin) * (lw_max - lw_min) / (vmax - vmin)

    return mapper


def get_connected_lines_for_bus(network, bus_name):
    if not hasattr(network, "lines") or network.lines.empty:
        return pd.DataFrame()

    lines = network.lines.copy()

    mask = (
        (lines["bus0"].astype(str) == bus_name)
        | (lines["bus1"].astype(str) == bus_name)
    )

    lines_bus = lines.loc[mask, ["bus0", "bus1"]].copy()

    if lines_bus.empty:
        return lines_bus

    if "s_nom_opt" in lines.columns:
        lines_bus["s_nom_opt"] = lines.loc[lines_bus.index, "s_nom_opt"]
    elif "s_nom" in lines.columns:
        lines_bus["s_nom_opt"] = lines.loc[lines_bus.index, "s_nom"]
    else:
        lines_bus["s_nom_opt"] = np.nan

    if (
        hasattr(network, "lines_t")
        and hasattr(network.lines_t, "p0")
        and not network.lines_t.p0.empty
    ):
        p0 = network.lines_t.p0.reindex(columns=lines_bus.index)
        lines_bus["power_mean"] = p0.mean(axis=0).reindex(lines_bus.index).fillna(0.0)
        lines_bus["power_mean_abs"] = p0.abs().mean(axis=0).reindex(lines_bus.index).fillna(0.0)
    else:
        lines_bus["power_mean"] = 0.0
        lines_bus["power_mean_abs"] = 0.0

    lines_bus.loc[
        np.abs(lines_bus["power_mean"]) < FLOW_CLIP_THRESHOLD,
        "power_mean",
    ] = 0.0

    lines_bus["connected_bus"] = np.where(
        lines_bus["bus0"].astype(str) == bus_name,
        lines_bus["bus1"].astype(str),
        lines_bus["bus0"].astype(str),
    )

    denom = 0.7 * lines_bus["s_nom_opt"].replace(0, np.nan)

    lines_bus["utilization"] = (
        lines_bus["power_mean_abs"] / denom
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return lines_bus


def build_bus_centroids_from_regions(regions_gdf, bus_col):
    regions_proj_all = regions_gdf.to_crs(epsg=3035).copy()
    regions_proj_all["centroid_geom"] = regions_proj_all.geometry.centroid

    centroid_df = pd.DataFrame(
        {
            "bus": regions_proj_all[bus_col].astype(str).values,
            "x": regions_proj_all["centroid_geom"].x.values,
            "y": regions_proj_all["centroid_geom"].y.values,
        }
    ).drop_duplicates(subset="bus")

    return regions_proj_all, centroid_df


def get_layout_radius(region_geom, centroid):
    geom = region_geom

    if geom.geom_type == "MultiPolygon":
        coords = np.vstack(
            [
                np.array(poly.exterior.coords)
                for poly in geom.geoms
            ]
        )
    else:
        coords = np.array(geom.exterior.coords)

    dx = coords[:, 0] - centroid.x
    dy = coords[:, 1] - centroid.y

    return np.sqrt(dx**2 + dy**2).max()


def wrap_to_pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def angle_distance(a, b):
    return np.abs(wrap_to_pi(a - b))


def choose_best_carrier_offset(n_carriers, arrow_angles, margin_deg=16):
    if n_carriers == 0:
        return 0.0

    candidate_offsets = np.linspace(0, 2 * np.pi, 72, endpoint=False)

    best_offset = 0.0
    best_score = -1e18
    margin = np.deg2rad(margin_deg)

    for offset in candidate_offsets:
        angles = np.linspace(0, 2 * np.pi, n_carriers, endpoint=False) + offset
        score = 0.0

        for angle in angles:
            if len(arrow_angles) == 0:
                score += 10.0
            else:
                dmin = np.min(
                    [
                        angle_distance(angle, arrow_angle)
                        for arrow_angle in arrow_angles
                    ]
                )

                score += dmin

                if dmin < margin:
                    score -= 100.0

        if score > best_score:
            best_score = score
            best_offset = offset

    return best_offset


def get_carrier_positions(cx, cy, radius, n_carriers, offset):
    if n_carriers == 0:
        return np.array([]), np.array([]), np.array([])

    angles = np.linspace(0, 2 * np.pi, n_carriers, endpoint=False) + offset
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)

    return xs, ys, angles


def adjust_arrow_angle(theta, forbidden_angles, margin_deg=16):
    if len(forbidden_angles) == 0:
        return theta

    margin = np.deg2rad(margin_deg)
    theta_new = theta

    for forbidden_angle in forbidden_angles:
        distance = wrap_to_pi(theta_new - forbidden_angle)

        if np.abs(distance) < margin:
            theta_new = forbidden_angle + np.sign(distance if distance != 0 else 1.0) * margin

    return theta_new


def draw_center_legend(
    legend_ax,
    capacity_values,
    capacity_size_mapper,
    flow_values,
    flow_lw_mapper,
    line_color,
    reservoir_color,
):
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")

    reservoir_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=8,
        markerfacecolor=reservoir_color,
        markeredgecolor="black",
        markeredgewidth=0.8,
        label="Individual reservoir plants",
    )

    reservoir_legend = legend_ax.legend(
        handles=[reservoir_handle],
        loc="center",
        bbox_to_anchor=(0.5, 0.82),
        frameon=False,
        fontsize=13,
        handletextpad=0.6,
        borderaxespad=0.0,
    )

    legend_ax.add_artist(reservoir_legend)

    legend_ax.text(
        0.5,
        0.66,
        "Capacity size [MW]",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )

    x_positions = [0.22, 0.50, 0.78]
    y_circle = 0.55

    for x, value in zip(x_positions, capacity_values):
        size = float(capacity_size_mapper([value])[0])

        legend_ax.scatter(
            [x],
            [y_circle],
            s=size,
            facecolor="none",
            edgecolor="0.65",
            linewidth=1.4,
            zorder=3,
        )

        legend_ax.text(
            x,
            y_circle - 0.11,
            f"{value}",
            ha="center",
            va="top",
            fontsize=12,
            color="0.45",
        )

    legend_ax.text(
        0.5,
        0.30,
        "Transmission [MW]",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )

    y_line = 0.20

    for x, value in zip(x_positions, flow_values):
        linewidth = float(flow_lw_mapper([value])[0])

        legend_ax.plot(
            [x - 0.09, x + 0.09],
            [y_line, y_line],
            color=line_color,
            linewidth=linewidth,
            solid_capstyle="round",
            zorder=3,
        )

        legend_ax.text(
            x,
            y_line - 0.08,
            f"{value}",
            ha="center",
            va="top",
            fontsize=12,
            color="0.45",
        )


def plot_bus_panel(
    ax,
    title,
    network,
    region_plot,
    centroid_proj,
    reservoir_proj,
    aggregated_carriers,
    lines_bus,
    dx,
    dy,
    r_layout,
    base_scale,
    agg_sizes,
    reservoir_sizes,
    lw_mapper,
    show_individual_reservoir_plants,
    bus_name,
    cap_col,
):
    arrow_angles = (
        lines_bus["angle_raw"].to_numpy()
        if DRAW_TRANSMISSION and not lines_bus.empty
        else np.array([])
    )

    n_icons = len(aggregated_carriers)
    r_carrier = CARRIER_RING_FACTOR * r_layout

    carrier_offset = choose_best_carrier_offset(
        n_icons,
        arrow_angles,
        margin_deg=CARRIER_ANGLE_MARGIN_DEG,
    )

    icon_xs, icon_ys, carrier_angles = get_carrier_positions(
        centroid_proj.x,
        centroid_proj.y,
        r_carrier,
        n_icons,
        carrier_offset,
    )

    region_plot.plot(
        ax=ax,
        facecolor="#f7f7f7",
        edgecolor="black",
        linewidth=1.2,
        zorder=1,
    )

    if show_individual_reservoir_plants and not reservoir_proj.empty:
        for geom in reservoir_proj.geometry:
            ax.plot(
                [geom.x, centroid_proj.x],
                [geom.y, centroid_proj.y],
                color=RESERVOIR_LINK_COLOR,
                linewidth=RESERVOIR_LINK_WIDTH,
                linestyle=RESERVOIR_LINK_STYLE,
                alpha=0.6,
                zorder=2,
            )

    if DRAW_TRANSMISSION and not lines_bus.empty:
        r_arrow_start = r_layout
        r_arrow_end = ARROW_END_FACTOR * r_layout
        arrow_head_width = 0.028 * base_scale
        arrow_head_length = 0.045 * base_scale

        lines_plot = lines_bus.sort_values("angle_raw").reset_index(drop=True)

        for _, row in lines_plot.iterrows():
            theta = float(row["angle_raw"])

            theta_adj = adjust_arrow_angle(
                theta,
                carrier_angles,
                margin_deg=CARRIER_ANGLE_MARGIN_DEG,
            )

            p_mean = float(row["power_mean"])
            p_abs = float(row["power_mean_abs"])
            linewidth = float(lw_mapper([p_abs])[0])

            start_x = centroid_proj.x + r_arrow_start * np.cos(theta_adj)
            start_y = centroid_proj.y + r_arrow_start * np.sin(theta_adj)
            end_x = centroid_proj.x + r_arrow_end * np.cos(theta_adj)
            end_y = centroid_proj.y + r_arrow_end * np.sin(theta_adj)

            bus_is_bus0 = str(row["bus0"]) == bus_name

            if bus_is_bus0:
                is_export = p_mean > 0
            else:
                is_export = p_mean < 0

            if is_export:
                base_x = start_x
                base_y = start_y
                dx_arrow = end_x - start_x
                dy_arrow = end_y - start_y
            else:
                base_x = end_x
                base_y = end_y
                dx_arrow = start_x - end_x
                dy_arrow = start_y - end_y

            ax.arrow(
                base_x,
                base_y,
                dx_arrow,
                dy_arrow,
                width=0.0,
                head_width=arrow_head_width,
                head_length=arrow_head_length,
                length_includes_head=True,
                fc=TRANSMISSION_COLOR,
                ec=TRANSMISSION_COLOR,
                linewidth=linewidth,
                alpha=0.90,
                zorder=3,
            )

            label_x = centroid_proj.x + ARROW_LABEL_DISTANCE_FACTOR * r_layout * np.cos(theta_adj)
            label_y = centroid_proj.y + ARROW_LABEL_DISTANCE_FACTOR * r_layout * np.sin(theta_adj)

            ax.text(
                label_x,
                label_y,
                f"{row['connected_bus']}",
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                zorder=4,
            )

    if show_individual_reservoir_plants and not reservoir_proj.empty:
        hydro_color = get_carrier_color(network, "hydro")

        ax.scatter(
            reservoir_proj.geometry.x,
            reservoir_proj.geometry.y,
            s=reservoir_sizes,
            color=hydro_color,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.9,
            zorder=4,
        )

    if (
        show_individual_reservoir_plants
        and SHOW_RESERVOIR_LABELS
        and not reservoir_proj.empty
    ):
        reservoir_label_df = reservoir_proj.nlargest(
            min(MAX_RESERVOIR_LABELS, len(reservoir_proj)),
            cap_col,
        )

        for _, row in reservoir_label_df.iterrows():
            plant_name = str(row["Name"]) if "Name" in row.index else "reservoir"

            if len(plant_name) > 18:
                plant_name = plant_name[:15] + "..."

            ax.text(
                row.geometry.x,
                row.geometry.y + RESERVOIR_LABEL_YOFFSET,
                plant_name,
                fontsize=6,
                ha="center",
                va="bottom",
                zorder=5,
            )

    ax.scatter(
        [centroid_proj.x],
        [centroid_proj.y],
        s=BUS_NODE_SIZE,
        color="white",
        edgecolor="black",
        linewidth=1.5,
        zorder=6,
    )

    ax.text(
        centroid_proj.x,
        centroid_proj.y,
        bus_name,
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=7,
    )

    if len(aggregated_carriers) > 0:
        for i, (carrier, value) in enumerate(aggregated_carriers.items()):
            x = icon_xs[i]
            y = icon_ys[i]
            size = agg_sizes[i]
            color = get_carrier_color(network, carrier)
            label = get_carrier_nice_name(network, carrier)

            ax.plot(
                [centroid_proj.x, x],
                [centroid_proj.y, y],
                color="gray",
                linewidth=1.0,
                zorder=2,
            )

            ax.scatter(
                [x],
                [y],
                s=size,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.7,
                zorder=5,
            )

            ax.text(
                x,
                y,
                label,
                fontsize=11,
                ha="center",
                va="center",
                zorder=6,
            )

    cx = centroid_proj.x
    cy = centroid_proj.y
    half_span = max(dx, dy) * (0.5 + PLOT_PADDING_FACTOR)

    ax.set_xlim(cx - half_span, cx + half_span)
    ax.set_ylim(cy - half_span, cy + half_span)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold")


def make_hydro_representation_figure(
    hydro_results_path,
    bus_name,
    output_path,
):
    pypsa_network_path = hydro_results_path / "PyPSA_2019" / "base_s_100_elec_.nc"
    efas_network_path = hydro_results_path / "EFAS-SABER_2019" / "base_s_100_elec_.nc"

    powerplants_path = hydro_results_path / "EFAS-SABER_2019" / "powerplants_s_100.csv"
    regions_path = hydro_results_path / "EFAS-SABER_2019" / "regions_onshore_base_s_100.geojson"

    if not powerplants_path.exists():
        raise FileNotFoundError(f"Missing powerplants file: {powerplants_path}")

    if not regions_path.exists():
        raise FileNotFoundError(f"Missing regions file: {regions_path}")

    n = load_network(pypsa_network_path)
    m = load_network(efas_network_path)

    plants = pd.read_csv(powerplants_path)
    regions = gpd.read_file(regions_path)

    bus_col = find_bus_column(regions, bus_name)

    region_bus = regions.loc[regions[bus_col].astype(str) == bus_name].copy()

    if region_bus.empty:
        raise ValueError(
            f"Bus '{bus_name}' not found in geojson. "
            f"Available bus column detected: '{bus_col}'"
        )

    region_bus = region_bus.to_crs(epsg=4326)

    if "bus" not in plants.columns:
        raise ValueError("Column 'bus' not found in powerplants CSV.")

    plants_bus = plants.loc[plants["bus"].astype(str) == bus_name].copy()
    cap_col = find_capacity_column(plants)

    reservoir_bus = plants_bus.loc[reservoir_mask(plants_bus)].copy()

    if not reservoir_bus.empty:
        reservoir_bus = reservoir_bus.dropna(subset=["lon", "lat"]).copy()

    if not reservoir_bus.empty:
        reservoir_gdf = gpd.GeoDataFrame(
            reservoir_bus,
            geometry=gpd.points_from_xy(
                reservoir_bus["lon"],
                reservoir_bus["lat"],
            ),
            crs="EPSG:4326",
        )
    else:
        reservoir_gdf = gpd.GeoDataFrame(
            reservoir_bus,
            geometry=[],
            crs="EPSG:4326",
        )

    aggregated_carriers_left = get_other_aggregated_carriers_for_bus(
        m,
        bus_name,
    ).drop(
        labels=["hydro", "Hydro"],
        errors="ignore",
    )

    aggregated_carriers_right = get_other_aggregated_carriers_for_bus(
        n,
        bus_name,
    )

    regions_proj = region_bus.to_crs(epsg=3035)
    region_geom = regions_proj.geometry.iloc[0]
    centroid_proj = region_geom.centroid

    minx, miny, maxx, maxy = regions_proj.total_bounds
    dx = maxx - minx
    dy = maxy - miny

    r_layout = get_layout_radius(region_geom, centroid_proj)
    base_scale = max(dx, dy, 2 * r_layout)

    if not reservoir_gdf.empty:
        reservoir_proj = reservoir_gdf.to_crs(epsg=3035)
    else:
        reservoir_proj = reservoir_gdf.copy()

    region_plot = regions_proj.copy()

    if DRAW_TRANSMISSION:
        _, centroid_df_all = build_bus_centroids_from_regions(regions, bus_col)
        centroid_map = centroid_df_all.set_index("bus")

        lines_bus_left = get_connected_lines_for_bus(m, bus_name)
        lines_bus_right = get_connected_lines_for_bus(n, bus_name)

        if not lines_bus_left.empty:
            lines_bus_left = lines_bus_left.loc[
                lines_bus_left["connected_bus"].isin(centroid_map.index)
            ].copy()

            lines_bus_left["neighbor_x"] = lines_bus_left["connected_bus"].map(
                centroid_map["x"]
            )

            lines_bus_left["neighbor_y"] = lines_bus_left["connected_bus"].map(
                centroid_map["y"]
            )

            lines_bus_left["angle_raw"] = np.arctan2(
                lines_bus_left["neighbor_y"] - centroid_proj.y,
                lines_bus_left["neighbor_x"] - centroid_proj.x,
            )

        if not lines_bus_right.empty:
            lines_bus_right = lines_bus_right.loc[
                lines_bus_right["connected_bus"].isin(centroid_map.index)
            ].copy()

            lines_bus_right["neighbor_x"] = lines_bus_right["connected_bus"].map(
                centroid_map["x"]
            )

            lines_bus_right["neighbor_y"] = lines_bus_right["connected_bus"].map(
                centroid_map["y"]
            )

            lines_bus_right["angle_raw"] = np.arctan2(
                lines_bus_right["neighbor_y"] - centroid_proj.y,
                lines_bus_right["neighbor_x"] - centroid_proj.x,
            )
    else:
        lines_bus_left = pd.DataFrame()
        lines_bus_right = pd.DataFrame()

    agg_values_left = (
        aggregated_carriers_left.values
        if len(aggregated_carriers_left)
        else np.array([])
    )

    agg_values_right = (
        aggregated_carriers_right.values
        if len(aggregated_carriers_right)
        else np.array([])
    )

    if not reservoir_proj.empty:
        reservoir_values = reservoir_proj[cap_col].fillna(0).values
    else:
        reservoir_values = np.array([])

    flow_values_left = (
        lines_bus_left["power_mean_abs"].values
        if DRAW_TRANSMISSION and not lines_bus_left.empty
        else np.array([])
    )

    flow_values_right = (
        lines_bus_right["power_mean_abs"].values
        if DRAW_TRANSMISSION and not lines_bus_right.empty
        else np.array([])
    )

    if USE_FIXED_GLOBAL_SCALES:
        capacity_ref_min = CAPACITY_REF_MIN
        capacity_ref_max = CAPACITY_REF_MAX

        flow_ref_min = FLOW_REF_MIN
        flow_ref_max = FLOW_REF_MAX
    else:
        capacity_values_all = (
            np.concatenate([agg_values_left, agg_values_right, reservoir_values])
            if (len(agg_values_left) + len(agg_values_right) + len(reservoir_values))
            else np.array([])
        )

        flow_values_all = (
            np.concatenate([flow_values_left, flow_values_right])
            if (len(flow_values_left) + len(flow_values_right))
            else np.array([])
        )

        capacity_ref_min = 0.0
        capacity_ref_max = (
            max(float(np.nanmax(capacity_values_all)), max(CAPACITY_LEGEND_VALUES))
            if capacity_values_all.size > 0
            else max(CAPACITY_LEGEND_VALUES)
        )

        flow_ref_min = 0.0
        flow_ref_max = (
            max(float(np.nanmax(flow_values_all)), max(FLOW_LEGEND_VALUES))
            if flow_values_all.size > 0
            else max(FLOW_LEGEND_VALUES)
        )

    capacity_size_mapper = make_size_mapper(
        capacity_ref_min,
        capacity_ref_max,
        CAPACITY_SIZE_MIN,
        CAPACITY_SIZE_MAX,
    )

    flow_lw_mapper = make_linewidth_mapper(
        flow_ref_min,
        flow_ref_max,
        MIN_LINEWIDTH,
        MAX_LINEWIDTH,
    )

    agg_sizes_left = capacity_size_mapper(agg_values_left)
    agg_sizes_right = capacity_size_mapper(agg_values_right)
    reservoir_sizes = capacity_size_mapper(reservoir_values)

    fig = plt.figure(figsize=(16, 8.5))

    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=[1.0, 0.32, 1.0],
        wspace=0.02,
    )

    ax_left = fig.add_subplot(gs[0, 0])
    legend_ax = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[0, 2])

    plot_bus_panel(
        ax=ax_left,
        title="aggregated hydro representation",
        network=n,
        region_plot=region_plot,
        centroid_proj=centroid_proj,
        reservoir_proj=reservoir_proj,
        aggregated_carriers=aggregated_carriers_right,
        lines_bus=lines_bus_right,
        dx=dx,
        dy=dy,
        r_layout=r_layout,
        base_scale=base_scale,
        agg_sizes=agg_sizes_right,
        reservoir_sizes=reservoir_sizes,
        lw_mapper=flow_lw_mapper,
        show_individual_reservoir_plants=False,
        bus_name=bus_name,
        cap_col=cap_col,
    )

    plot_bus_panel(
        ax=ax_right,
        title="individual reservoir plants",
        network=m,
        region_plot=region_plot,
        centroid_proj=centroid_proj,
        reservoir_proj=reservoir_proj,
        aggregated_carriers=aggregated_carriers_left,
        lines_bus=lines_bus_left,
        dx=dx,
        dy=dy,
        r_layout=r_layout,
        base_scale=base_scale,
        agg_sizes=agg_sizes_left,
        reservoir_sizes=reservoir_sizes,
        lw_mapper=flow_lw_mapper,
        show_individual_reservoir_plants=True,
        bus_name=bus_name,
        cap_col=cap_col,
    )

    draw_center_legend(
        legend_ax=legend_ax,
        capacity_values=CAPACITY_LEGEND_VALUES,
        capacity_size_mapper=capacity_size_mapper,
        flow_values=FLOW_LEGEND_VALUES,
        flow_lw_mapper=flow_lw_mapper,
        line_color=TRANSMISSION_COLOR,
        reservoir_color=get_carrier_color(m, "hydro"),
    )

    plt.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.92,
        bottom=0.06,
        wspace=0.02,
    )

    save_figure(fig, output_path)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    hydro_results_path = args.hydro_results
    bus_name = args.bus_name

    output_dir = args.output_dir or hydro_results_path / "images" / "representation"
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_bus_name = bus_name.replace(" ", "_").replace("/", "_")

    make_hydro_representation_figure(
        hydro_results_path=hydro_results_path,
        bus_name=bus_name,
        output_path=output_dir / f"hydro_representation_{safe_bus_name}.png",
    )

    print(f"Hydro representation figure saved to: {output_dir}")


if __name__ == "__main__":
    main()