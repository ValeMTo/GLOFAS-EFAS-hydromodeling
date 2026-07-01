#!/usr/bin/env python3

from pathlib import Path
import argparse
import logging
import os
import warnings

import matplotlib
matplotlib.use("Agg")

import geopandas as gpd
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

from hydro_inflow.utils import setup_logging


logger = logging.getLogger(__name__)


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

BASELINE_NAME = "PyPSA baseline"

COMPARISON_NAMES = [
    "PyPSA + GloFAS-SABER",
    "PyPSA + EFAS-SABER",
]

SCENARIO_COLORS = {
    "PyPSA baseline": "#D7263D",
    "PyPSA with no hydro grouping": "#8B1E2D",
    "PyPSA + GloFAS-SABER": "#1F77B4",
    "PyPSA + EFAS-SABER": "#2CA02C",
}

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


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 2050 PyPSA hydro post-processing figures."
    )

    parser.add_argument(
        "--hydro-results",
        type=Path,
        default=None,
        help="Path to hydro_results directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for generated 2050 figures.",
    )

    return parser.parse_args()


# ============================================================
# GENERAL HELPERS
# ============================================================

def save_figure(fig, output_path, bbox_inches=None, pad_inches=0.0):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
    )

    plt.close(fig)


def load_2050_networks(hydro_results_path):
    network_paths = {
        "PyPSA baseline": hydro_results_path / "PyPSA_2050" / "base_s_100___2050.nc",
        "PyPSA + GloFAS-SABER": hydro_results_path / "GloFAS-SABER_2050" / "base_s_100___2050.nc",
        "PyPSA + EFAS-SABER": hydro_results_path / "EFAS-SABER_2050" / "base_s_100___2050.nc",
    }

    missing = [
        path
        for path in network_paths.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing 2050 network files:\n"
            + "\n".join(str(path) for path in missing)
        )

    return {
        name: pypsa.Network(path, engine="netcdf4")
        for name, path in network_paths.items()
    }


def get_numeric_statistics(network):
    stats = network.statistics().copy()
    stats_num = stats.select_dtypes(include=[np.number]).copy()

    return stats_num


def get_country_from_bus(bus_name):
    return str(bus_name)[:2]


# ============================================================
# TECHNOLOGY NAME NORMALIZATION
# ============================================================

def normalize_technology_name(name):
    name = str(name).strip()

    replacements = {
        "battery": "Battery",
        "battery storage": "Battery",
        "home battery": "Home battery",
        "ev battery": "EV battery",
        "ev battery storage": "EV battery",
        "hydro": "Hydro",
        "ror": "Run-of-river",
        "run of river": "Run-of-river",
        "run-of-river": "Run-of-river",
        "onwind": "Onshore wind",
        "onshore wind": "Onshore wind",
        "offwind": "Offshore wind",
        "offshore wind": "Offshore wind",
        "offwind-ac": "Offshore wind AC",
        "offwind-dc": "Offshore wind DC",
        "offshore wind ac": "Offshore wind AC",
        "offshore wind dc": "Offshore wind DC",
        "solar": "Solar",
        "solar pv": "Solar",
        "solar rooftop": "Solar rooftop",
        "ocgt": "OCGT",
        "ccgt": "CCGT",
        "coal": "Coal",
        "lignite": "Lignite",
        "oil": "Oil",
        "gas": "Gas",
        "biomass": "Biomass",
        "nuclear": "Nuclear",
        "geothermal": "Geothermal",
        "hydrogen storage": "Hydrogen storage",
        "h2 storage": "Hydrogen storage",
        "h2": "Hydrogen",
        "fuel cell": "Fuel cell",
        "pumped hydro storage": "Pumped hydro storage",
        "phs": "Pumped hydro storage",
        "reservoir": "Reservoir hydro",
        "hydro reservoir": "Reservoir hydro",
        "solar hsat": "Solar Hsat",
        "solar-hsat": "Solar Hsat",
        "urban central heat vent": "Urban Central Heat Vent",
        "urban decentral heat vent": "Urban Decentral Heat Vent",
        "rural heat vent": "Rural Heat Vent",
        "oil primary": "Oil Primary",
        "offshore wind floating": "Offshore wind floating",
    }

    key = name.lower().replace("_", " ").replace("-", " ")
    key = " ".join(key.split())

    if key in replacements:
        return replacements[key]

    acronyms = {
        "ac": "AC",
        "dc": "DC",
        "h2": "H2",
        "co2": "CO2",
        "ev": "EV",
        "ocgt": "OCGT",
        "ccgt": "CCGT",
    }

    words = key.split()
    normalized_words = []

    for word in words:
        if word in acronyms:
            normalized_words.append(acronyms[word])
        else:
            normalized_words.append(word.capitalize())

    return " ".join(normalized_words)


def normalize_store_name(name):
    s = str(name).strip().lower().replace("_", " ").replace("-", " ")
    s = " ".join(s.split())

    mapping = {
        "h2 store": "Hydrogen storage",
        "hydrogen store": "Hydrogen storage",
        "battery storage": "Battery storage",
        "battery": "Battery storage",
        "ev battery": "EV battery",
        "home battery": "Home battery",
        "ammonia store": "Ammonia storage",
        "ammonia storage": "Ammonia storage",
        "methanol": "Methanol",
        "gas": "Gas storage",
        "gas store": "Gas storage",
        "oil": "Oil storage",
        "co2": "CO2",
        "co2 stored": "CO2 stored",
        "co2 sequestered": "CO2 sequestered",
        "urban central water pits": "Urban central water pits",
        "urban central water tanks": "Urban central water tanks",
        "urban decentral water tanks": "Urban decentral water tanks",
        "rural water tanks": "Rural water tanks",
        "non sequestered hvc": "Non-sequestered HVC",
    }

    if s in mapping:
        return mapping[s]

    acronyms = {
        "co2": "CO2",
        "h2": "H2",
        "ev": "EV",
        "hvc": "HVC",
    }

    words = s.split()
    words = [acronyms.get(word, word.capitalize()) for word in words]

    return " ".join(words)


# ============================================================
# CAPACITY / SUPPLY DIFFERENCE FIGURES
# ============================================================

def filter_components(df, components):
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(
            "Expected statistics index to be a MultiIndex with component as first level."
        )

    component_level = df.index.get_level_values(0)

    return df[component_level.isin(components)].copy()


def aggregate_by_technology(df, metric):
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in statistics columns.")

    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Expected statistics index to be a MultiIndex.")

    technology_level = 1

    series = df[metric].groupby(level=technology_level).sum(min_count=1)
    series.index = series.index.map(normalize_technology_name)
    series = series.groupby(series.index).sum(min_count=1)

    return series


def build_difference_plot_dataframe(
    diffs,
    metric,
    metric_conversion_factors,
    min_abs_value=0.1,
    top_n_technologies=None,
):
    plot_df = pd.DataFrame()

    for scenario_name, diff_df in diffs.items():
        plot_df[scenario_name] = aggregate_by_technology(diff_df, metric)

    plot_df = plot_df.fillna(0.0)

    conversion_factor = metric_conversion_factors[metric]
    plot_df = plot_df * conversion_factor

    if min_abs_value is not None:
        plot_df = plot_df.loc[plot_df.abs().max(axis=1) > min_abs_value]

    if top_n_technologies is not None:
        order = (
            plot_df
            .abs()
            .max(axis=1)
            .sort_values(ascending=False)
            .head(top_n_technologies)
            .index
        )
        plot_df = plot_df.loc[order]
    else:
        order = plot_df.abs().max(axis=1).sort_values(ascending=False).index
        plot_df = plot_df.loc[order]

    return plot_df


def build_baseline_absolute_series(
    stats_filtered,
    metric,
    technologies_index,
    metric_conversion_factors,
):
    baseline_series = aggregate_by_technology(
        stats_filtered[BASELINE_NAME],
        metric,
    ).fillna(0.0)

    baseline_series = baseline_series * metric_conversion_factors[metric]
    baseline_series = baseline_series.reindex(technologies_index).fillna(0.0)

    return baseline_series


def plot_grouped_bar_difference(
    plot_df,
    baseline_series,
    metric,
    metric_units,
    output_path,
    show_baseline_values_in_xticks=True,
):
    if plot_df.empty:
        return

    unit = metric_units[metric]

    colors = [
        SCENARIO_COLORS.get(col, None)
        for col in plot_df.columns
    ]

    fixed_figsize = (10, 7)

    fig = plt.figure(figsize=fixed_figsize, facecolor="white")
    ax = fig.add_axes([0.09, 0.24, 0.90, 0.70])

    plot_df.plot(
        kind="bar",
        ax=ax,
        width=0.8,
        edgecolor="black",
        linewidth=0.6,
        color=colors,
        zorder=3,
    )

    ax.axhline(0, linewidth=1.0, color="black", zorder=4)

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.35,
        zorder=0,
    )

    ax.set_axisbelow(True)

    if show_baseline_values_in_xticks:
        xtick_labels = [
            f"{tech}\n(Baseline {baseline_series.loc[tech]:.1f} {unit})"
            for tech in plot_df.index
        ]
    else:
        xtick_labels = list(plot_df.index)

    ax.set_xticklabels(
        xtick_labels,
        rotation=45,
        ha="right",
        fontsize=12,
    )

    ax.set_xlabel("")

    ax.set_ylabel(
        f"Difference in {metric} [{unit}]",
        fontsize=13,
    )

    ax.tick_params(axis="both", which="major", labelsize=12)

    ax.legend(
        title="Scenario",
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        fontsize=12,
        title_fontsize=12,
    )

    y_min = plot_df.min().min()
    y_max = plot_df.max().max()
    y_range = max(abs(y_min), abs(y_max), 1e-9)

    ax.set_ylim(
        y_min - 0.12 * y_range,
        y_max + 0.12 * y_range,
    )

    save_figure(fig, output_path)


def make_capacity_and_supply_figures(networks, output_dir):
    """Supply difference figure (absolute, TWh).

    Optimal Capacity is handled separately by make_optimal_capacity_pct_figure.
    """
    components_to_keep = ["Generator", "StorageUnit"]

    metrics_to_plot = [
        "Supply",
    ]

    metric_units = {
        "Supply": "TWh",
    }

    metric_conversion_factors = {
        "Supply": 1 / 1e6,
    }

    min_abs_value = 0.1
    min_baseline_value = 0.1
    top_n_technologies = None

    output_figures = {
        "Supply": output_dir / "supply_diff.png",
    }

    stats = {
        name: get_numeric_statistics(network)
        for name, network in networks.items()
    }

    full_index = stats[BASELINE_NAME].index
    full_columns = stats[BASELINE_NAME].columns

    for name in COMPARISON_NAMES:
        full_index = full_index.union(stats[name].index)
        full_columns = full_columns.union(stats[name].columns)

    for name in stats:
        stats[name] = stats[name].reindex(index=full_index, columns=full_columns)

    stats_filtered = {
        name: filter_components(df, components_to_keep)
        for name, df in stats.items()
    }

    diffs = {}

    for name in COMPARISON_NAMES:
        diffs[name] = stats_filtered[name] - stats_filtered[BASELINE_NAME]

    for metric in metrics_to_plot:
        plot_df = build_difference_plot_dataframe(
            diffs=diffs,
            metric=metric,
            metric_conversion_factors=metric_conversion_factors,
            min_abs_value=min_abs_value,
            top_n_technologies=top_n_technologies,
        )

        baseline_series = build_baseline_absolute_series(
            stats_filtered=stats_filtered,
            metric=metric,
            technologies_index=plot_df.index,
            metric_conversion_factors=metric_conversion_factors,
        )

        nonzero_baseline_mask = baseline_series.abs() > min_baseline_value

        plot_df = plot_df.loc[nonzero_baseline_mask]
        baseline_series = baseline_series.loc[nonzero_baseline_mask]

        if plot_df.empty:
            continue

        plot_df = plot_df.loc[
            plot_df.abs().max(axis=1).sort_values(ascending=False).index
        ]

        baseline_series = baseline_series.reindex(plot_df.index)

        plot_grouped_bar_difference(
            plot_df=plot_df,
            baseline_series=baseline_series,
            metric=metric,
            metric_units=metric_units,
            output_path=output_figures[metric],
            show_baseline_values_in_xticks=True,
        )


# ============================================================
# OPTIMAL CAPACITY DIFFERENCE FIGURE (percentage)
# ============================================================

def make_optimal_capacity_pct_figure(networks, output_path):
    """Optimal-capacity difference vs baseline, in percent, with absolute
    ('+X.X GW') labels on the bars."""
    components_to_keep = ["Generator", "StorageUnit"]

    metric = "Optimal Capacity"
    unit = "GW"
    conversion_factor = 1 / 1e3

    min_abs_value = 0.1
    min_baseline_value = 0.1
    show_baseline_values_in_xticks = True

    technologies_to_exclude = [
        "Rural Heat Vent",
        "Urban Decentral Heat Vent",
    ]

    stats = {
        name: get_numeric_statistics(network)
        for name, network in networks.items()
    }

    full_index = stats[BASELINE_NAME].index
    full_columns = stats[BASELINE_NAME].columns

    for name in COMPARISON_NAMES:
        full_index = full_index.union(stats[name].index)
        full_columns = full_columns.union(stats[name].columns)

    for name in stats:
        stats[name] = stats[name].reindex(index=full_index, columns=full_columns)

    stats_filtered = {
        name: filter_components(df, components_to_keep)
        for name, df in stats.items()
    }

    absolute_df = pd.DataFrame(
        {
            name: aggregate_by_technology(stats_filtered[name], metric)
            for name in [BASELINE_NAME] + COMPARISON_NAMES
        }
    ).fillna(0.0) * conversion_factor

    diff_df = pd.DataFrame(
        {
            name: aggregate_by_technology(stats_filtered[name], metric)
            - aggregate_by_technology(stats_filtered[BASELINE_NAME], metric)
            for name in COMPARISON_NAMES
        }
    ).fillna(0.0) * conversion_factor

    baseline_series = absolute_df[BASELINE_NAME]

    mask = diff_df.abs().max(axis=1) > min_abs_value
    diff_df = diff_df.loc[mask]

    mask2 = baseline_series.reindex(diff_df.index).abs() > min_baseline_value
    diff_df = diff_df.loc[mask2]

    diff_df = diff_df.loc[
        diff_df.abs().max(axis=1).sort_values(ascending=False).index
    ]

    if technologies_to_exclude:
        diff_df = diff_df.drop(
            index=[t for t in technologies_to_exclude if t in diff_df.index]
        )

    baseline_series = baseline_series.reindex(diff_df.index).fillna(0.0)

    pct_df = diff_df.divide(
        baseline_series.replace(0, np.nan),
        axis=0,
    ) * 100

    if pct_df.empty:
        return

    fixed_figsize = (10, 6.5)

    fig = plt.figure(figsize=fixed_figsize, facecolor="white")
    ax = fig.add_axes([0.09, 0.35, 0.90, 0.62])

    colors = [SCENARIO_COLORS.get(col, None) for col in pct_df.columns]

    pct_df.plot(
        kind="bar",
        ax=ax,
        width=0.8,
        edgecolor="black",
        linewidth=0.6,
        color=colors,
        zorder=3,
    )

    ax.axhline(0, color="black", linewidth=1.0, zorder=4)

    n_scenarios = len(pct_df.columns)
    bar_width = 0.8

    for i, tech in enumerate(pct_df.index):
        for j, scenario in enumerate(pct_df.columns):
            pct_val = pct_df.loc[tech, scenario]
            abs_val = diff_df.loc[tech, scenario]

            if pd.isna(pct_val) or pd.isna(abs_val):
                continue

            offset = (j - (n_scenarios - 1) / 2) * (bar_width / n_scenarios)
            x_pos = i + offset

            sign = "+" if abs_val >= 0 else ""
            label = f"{sign}{abs_val:.1f} {unit}"

            y_offset = 0.5 if pct_val >= 0 else -0.5
            va = "bottom" if pct_val >= 0 else "top"

            ax.text(
                x_pos,
                pct_val + y_offset,
                label,
                ha="center",
                va=va,
                fontsize=7.5,
                color="black",
                rotation=90,
            )

    if show_baseline_values_in_xticks:
        xtick_labels = [
            f"{tech}\n(Baseline {baseline_series.loc[tech]:.1f} {unit})"
            for tech in pct_df.index
        ]
    else:
        xtick_labels = list(pct_df.index)

    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=12)

    ax.set_xlabel("")
    ax.set_ylabel(f"Difference in {metric} [%]", fontsize=13)

    ax.tick_params(axis="both", which="major", labelsize=12)

    ax.legend(
        title="Scenario",
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        fontsize=12,
        title_fontsize=12,
    )

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    y_min = pct_df.min().min()
    y_max = pct_df.max().max()
    y_range = max(abs(y_min), abs(y_max), 1e-9)

    ax.set_ylim(
        y_min - 0.30 * y_range,
        y_max + 0.30 * y_range,
    )

    save_figure(fig, output_path)


# ============================================================
# COST DIFFERENCE FIGURE
# ============================================================

def get_component_mask(df, components):
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(
            "Expected statistics index to be a MultiIndex with component as first level."
        )

    return df.index.get_level_values(0).isin(components)


def prettify_cost_name(name):
    mapping = {
        "Capital Expenditure": "Installation cost",
        "Operational Expenditure": "Operation cost",
    }

    return mapping.get(name, str(name))


def find_cost_columns(df):
    preferred = []
    others = []

    for col in df.columns:
        col_str = str(col).strip()
        low = col_str.lower()

        if col_str in ["Capital Expenditure", "Operational Expenditure"]:
            preferred.append(col_str)
        elif ("expenditure" in low) or low.endswith(" cost") or low == "cost":
            others.append(col_str)

    ordered = []

    for col in preferred + others:
        if col not in ordered:
            ordered.append(col)

    return ordered


def summarize_system_costs(
    stats_df,
    network_components=("Line", "Link"),
    include_other_cost_columns=True,
):
    cost_columns = find_cost_columns(stats_df)

    if len(cost_columns) == 0:
        raise ValueError("No cost/expenditure columns found in n.statistics().")

    network_mask = get_component_mask(stats_df, network_components)
    non_network_mask = ~network_mask

    network_df = stats_df.loc[network_mask, cost_columns].copy()
    non_network_df = stats_df.loc[non_network_mask, cost_columns].copy()

    summary = pd.Series(dtype=float)

    capex_col = (
        "Capital Expenditure"
        if "Capital Expenditure" in cost_columns
        else None
    )

    opex_col = (
        "Operational Expenditure"
        if "Operational Expenditure" in cost_columns
        else None
    )

    if capex_col is not None:
        summary["Installation cost"] = non_network_df[capex_col].sum(skipna=True)
    else:
        summary["Installation cost"] = 0.0

    if opex_col is not None:
        summary["Operation cost"] = non_network_df[opex_col].sum(skipna=True)
    else:
        summary["Operation cost"] = 0.0

    summary["Network cost"] = network_df.sum(skipna=True).sum(skipna=True)

    if include_other_cost_columns:
        base_cols = {"Capital Expenditure", "Operational Expenditure"}
        extra_cols = [
            col
            for col in cost_columns
            if col not in base_cols
        ]

        for col in extra_cols:
            value = non_network_df[col].sum(skipna=True)

            if pd.notna(value) and abs(value) > 0:
                summary[prettify_cost_name(col)] = value

    summary = summary.fillna(0.0)

    return summary


def make_cost_difference_figure(networks, output_path):
    network_components = ["Line", "Link"]

    cost_conversion_factor = 1 / 1e9
    cost_unit = "B€"

    include_other_cost_columns = True

    cost_summary = {}

    raw_stats = {
        name: get_numeric_statistics(net)
        for name, net in networks.items()
    }

    for name, stats_df in raw_stats.items():
        cost_summary[name] = summarize_system_costs(
            stats_df,
            network_components=network_components,
            include_other_cost_columns=include_other_cost_columns,
        )

    plot_df = pd.DataFrame(cost_summary).T.fillna(0.0)

    main_order = ["Installation cost", "Operation cost", "Network cost"]
    extra_order = [
        col
        for col in plot_df.columns
        if col not in main_order
    ]

    plot_df = plot_df[
        [col for col in main_order if col in plot_df.columns]
        + extra_order
    ]

    plot_df = plot_df * cost_conversion_factor

    absolute_df = plot_df.copy()

    difference_df = absolute_df.subtract(
        absolute_df.loc[BASELINE_NAME],
        axis=1,
    )

    difference_plot_df = difference_df.loc[COMPARISON_NAMES].T

    main_cost_categories = [
        "Installation cost",
        "Operation cost",
        "Network cost",
    ]

    difference_plot_df = difference_plot_df.reindex(main_cost_categories)
    baseline_values = absolute_df.loc[BASELINE_NAME].reindex(main_cost_categories)

    valid_mask = difference_plot_df.notna().any(axis=1)
    difference_plot_df = difference_plot_df.loc[valid_mask]
    baseline_values = baseline_values.loc[valid_mask]

    fixed_figsize = (10, 7)

    fig = plt.figure(figsize=fixed_figsize, facecolor="white")
    ax = fig.add_axes([0.09, 0.24, 0.90, 0.70])

    colors = [
        SCENARIO_COLORS.get(col, None)
        for col in difference_plot_df.columns
    ]

    difference_plot_df.plot(
        kind="bar",
        ax=ax,
        width=0.75,
        edgecolor="black",
        linewidth=0.6,
        color=colors,
        zorder=3,
    )

    ax.axhline(
        0,
        color="black",
        linewidth=1.0,
        zorder=4,
    )

    xtick_labels = [
        f"{category}\n(Baseline: {baseline_values.loc[category]:.1f} {cost_unit})"
        for category in difference_plot_df.index
    ]

    ax.set_xticklabels(
        xtick_labels,
        rotation=35,
        ha="center",
        fontsize=12,
    )

    ax.set_xlabel("")

    ax.set_ylabel(
        f"Difference [{cost_unit}]",
        fontsize=13,
    )

    ax.tick_params(axis="both", which="major", labelsize=12)

    ax.legend(
        title="Scenario",
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        fontsize=12,
        title_fontsize=12,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.35,
        zorder=0,
    )

    ax.set_axisbelow(True)

    y_min = difference_plot_df.min().min()
    y_max = difference_plot_df.max().max()

    positive_margin = 0.2 * abs(y_max) if y_max > 0 else 0.5
    negative_margin = 0.3 * abs(y_min) if y_min < 0 else 0.5

    y_lower = (
        y_min - negative_margin
        if y_min < 0
        else -0.05 * max(abs(y_max), 1e-9)
    )

    y_upper = (
        y_max + positive_margin
        if y_max > 0
        else 0.05 * max(abs(y_min), 1e-9)
    )

    ax.set_ylim(y_lower, y_upper)

    save_figure(fig, output_path)


# ============================================================
# STORAGE DIFFERENCE FIGURE (percentage)
# ============================================================

def extract_store_capacity_by_technology(
    stats_df,
    metric="Optimal Capacity",
    component="Store",
):
    if metric not in stats_df.columns:
        raise ValueError(f"Metric '{metric}' not found in statistics columns.")

    if not isinstance(stats_df.index, pd.MultiIndex):
        raise ValueError("Expected statistics index to be a MultiIndex.")

    component_level = stats_df.index.get_level_values(0)
    df = stats_df.loc[component_level == component].copy()

    tech_series = df[metric].groupby(level=1).sum(min_count=1)

    normalized = {}

    for tech, value in tech_series.items():
        tech_norm = normalize_store_name(tech)
        normalized[tech_norm] = normalized.get(tech_norm, 0.0) + value

    result = pd.Series(normalized, dtype=float)

    return result


def make_storage_difference_figure(networks, output_path):
    """Storage installation difference vs baseline, in percent, with absolute
    ('+X.X TWh') labels on the bars."""
    component_to_keep = "Store"
    metric = "Optimal Capacity"

    conversion_factor = 1 / 1e6
    unit = "TWh"

    selected_technologies = [
        "Hydrogen storage",
        "Battery storage",
        "EV battery",
        "Home battery",
        "Ammonia storage",
        "Methanol",
        "Gas storage",
        "Urban central water pits",
        "Urban central water tanks",
        "Urban decentral water tanks",
        "Rural water tanks",
    ]

    drop_all_zero_differences = True
    drop_zero_pypsa_baseline = True
    sort_by_max_abs_difference = True

    min_abs_difference = 0.01
    min_baseline_value = 0.01

    absolute_df = pd.DataFrame()

    for scenario_name, network in networks.items():
        stats_df = get_numeric_statistics(network)

        series = extract_store_capacity_by_technology(
            stats_df,
            metric=metric,
            component=component_to_keep,
        )

        absolute_df[scenario_name] = series

    absolute_df = absolute_df.fillna(0.0)
    absolute_df = absolute_df * conversion_factor

    if selected_technologies is not None:
        absolute_df = absolute_df.reindex(selected_technologies, fill_value=0.0)

    difference_df = absolute_df.subtract(
        absolute_df[BASELINE_NAME],
        axis=0,
    )

    difference_df = difference_df.drop(columns=[BASELINE_NAME])

    baseline_col = absolute_df[BASELINE_NAME]
    pct_difference_df = difference_df.divide(
        baseline_col.replace(0, np.nan),
        axis=0,
    ) * 100

    if drop_all_zero_differences:
        mask = difference_df.abs().sum(axis=1) > 0
        difference_df = difference_df.loc[mask]
        pct_difference_df = pct_difference_df.loc[mask]

    if min_abs_difference is not None and min_abs_difference > 0:
        mask = difference_df.abs().max(axis=1) >= min_abs_difference
        difference_df = difference_df.loc[mask]
        pct_difference_df = pct_difference_df.loc[mask]

    baseline_values = absolute_df.loc[difference_df.index, BASELINE_NAME]

    if drop_zero_pypsa_baseline:
        nonzero_baseline_mask = baseline_values.abs() > min_baseline_value
        difference_df = difference_df.loc[nonzero_baseline_mask]
        pct_difference_df = pct_difference_df.loc[nonzero_baseline_mask]
        baseline_values = baseline_values.loc[nonzero_baseline_mask]

    if sort_by_max_abs_difference and not difference_df.empty:
        order = difference_df.abs().max(axis=1).sort_values(ascending=False).index
        difference_df = difference_df.loc[order]
        pct_difference_df = pct_difference_df.loc[order]
        baseline_values = baseline_values.loc[order]

    if pct_difference_df.empty:
        return

    fixed_figsize = (10, 6.5)

    fig = plt.figure(figsize=fixed_figsize, facecolor="white")
    ax = fig.add_axes([0.09, 0.35, 0.90, 0.62])

    colors = [SCENARIO_COLORS.get(col, None) for col in pct_difference_df.columns]

    pct_difference_df.plot(
        kind="bar",
        ax=ax,
        width=0.8,
        edgecolor="black",
        linewidth=0.6,
        color=colors,
        zorder=3,
    )

    ax.axhline(0, color="black", linewidth=1.0, zorder=4)

    n_scenarios = len(pct_difference_df.columns)
    bar_width = 0.8

    for i, tech in enumerate(pct_difference_df.index):
        for j, scenario in enumerate(pct_difference_df.columns):
            pct_val = pct_difference_df.loc[tech, scenario]
            abs_val = difference_df.loc[tech, scenario]

            if pd.isna(pct_val) or pd.isna(abs_val):
                continue

            offset = (j - (n_scenarios - 1) / 2) * (bar_width / n_scenarios)
            x_pos = i + offset

            sign = "+" if abs_val >= 0 else ""
            label = f"{sign}{abs_val:.1f} {unit}"

            y_offset = 0.5 if pct_val >= 0 else -0.5
            va = "bottom" if pct_val >= 0 else "top"

            ax.text(
                x_pos,
                pct_val + y_offset,
                label,
                ha="center",
                va=va,
                fontsize=7.5,
                color="black",
                rotation=90,
            )

    xtick_labels = [
        f"{tech}\n(Baseline: {baseline_values.loc[tech]:.1f} {unit})"
        for tech in pct_difference_df.index
    ]

    ax.set_xticklabels(
        xtick_labels,
        rotation=35,
        ha="right",
        fontsize=12,
    )

    ax.set_xlabel("")

    ax.set_ylabel(
        f"Difference in {metric} [%]",
        fontsize=13,
    )

    ax.tick_params(axis="both", which="major", labelsize=12)

    ax.legend(
        title="Scenario",
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        fontsize=12,
        title_fontsize=12,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.35,
        zorder=0,
    )

    ax.set_axisbelow(True)

    y_min = pct_difference_df.min().min()
    y_max = pct_difference_df.max().max()
    y_range = max(abs(y_min), abs(y_max), 1e-9)

    ax.set_ylim(
        y_min - 0.30 * y_range,
        y_max + 0.30 * y_range,
    )

    save_figure(fig, output_path)


# ============================================================
# HYDRO PRICE MAP (reservoir only)
# ============================================================

def infer_regions_country_column(regions_gdf):
    candidate_cols = [
        "name",
        "country",
        "iso2",
        "ISO2",
        "id",
        "ID",
        "nuts0",
        "NUTS_ID",
        "cntr_code",
        "CNTR_CODE",
    ]

    for col in candidate_cols:
        if col in regions_gdf.columns:
            values = regions_gdf[col].dropna().astype(str).str.upper()
            if (values.str.len() == 2).sum() >= max(1, int(0.5 * len(values))):
                return col

    raise ValueError(
        "Could not infer country code column in regions. "
        f"Available columns: {list(regions_gdf.columns)}"
    )


def weighted_average_price_by_country(generation_df, bus_map, marginal_prices):
    if generation_df.empty:
        return pd.Series(dtype=float)

    revenue_by_country = {}
    generation_by_country = {}

    common_assets = generation_df.columns.intersection(bus_map.index)

    for asset in common_assets:
        bus = bus_map.loc[asset]

        if bus not in marginal_prices.columns:
            continue

        country = get_country_from_bus(bus)

        generation = generation_df[asset].clip(lower=0.0)
        price = marginal_prices[bus]

        aligned = pd.concat(
            [generation.rename("generation"), price.rename("price")],
            axis=1,
        ).dropna()

        if aligned.empty:
            continue

        revenue = (aligned["generation"] * aligned["price"]).sum()
        total_generation = aligned["generation"].sum()

        revenue_by_country[country] = revenue_by_country.get(country, 0.0) + revenue
        generation_by_country[country] = generation_by_country.get(country, 0.0) + total_generation

    result = {}

    all_countries = sorted(set(revenue_by_country) | set(generation_by_country))

    for country in all_countries:
        generation = generation_by_country.get(country, 0.0)
        revenue = revenue_by_country.get(country, 0.0)

        result[country] = revenue / generation if generation > 0 else np.nan

    return pd.Series(result, dtype=float)


def extract_hydro_country_prices(network):
    if not hasattr(network, "buses_t") or not hasattr(network.buses_t, "marginal_price"):
        raise ValueError("network.buses_t.marginal_price is required.")

    marginal_prices = network.buses_t.marginal_price.copy()
    result = {}

    reservoir_units = network.storage_units[network.storage_units.carrier == "hydro"]

    if not reservoir_units.empty:
        reservoir_generation = network.storage_units_t.p[reservoir_units.index].clip(lower=0.0)
        reservoir_bus_map = reservoir_units["bus"]

        result["Reservoir & Dam"] = weighted_average_price_by_country(
            generation_df=reservoir_generation,
            bus_map=reservoir_bus_map,
            marginal_prices=marginal_prices,
        )
    else:
        result["Reservoir & Dam"] = pd.Series(dtype=float)

    return pd.DataFrame(result)


def build_map_dataframe(regions_gdf, price_df, regions_country_col):
    gdf = regions_gdf.copy()
    gdf["_country_code_"] = gdf[regions_country_col].astype(str).str.upper()

    merged = gdf.merge(
        price_df,
        how="left",
        left_on="_country_code_",
        right_index=True,
    )

    return merged


def plot_hydro_price_map_by_type(hydro_type, map_data_abs, norm, cmap, output_path):
    scenario_order = [
        "PyPSA baseline",
        "PyPSA + GloFAS-SABER",
        "PyPSA + EFAS-SABER",
    ]

    scenario_titles = {
        "PyPSA baseline": "PyPSA baseline",
        "PyPSA + GloFAS-SABER": "PyPSA + GloFAS-SABER",
        "PyPSA + EFAS-SABER": "PyPSA + EFAS-SABER",
    }

    fig = plt.figure(figsize=(18, 6))

    gs = fig.add_gridspec(
        nrows=1,
        ncols=4,
        width_ratios=[1, 1, 1, 0.045],
        wspace=0.05,
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
    ]

    cax = fig.add_subplot(gs[0, 3])

    for ax, scenario_name in zip(axes, scenario_order):
        merged = map_data_abs[scenario_name]

        merged.plot(
            column=hydro_type,
            ax=ax,
            cmap=cmap,
            norm=norm,
            linewidth=0.5,
            edgecolor="black",
            legend=False,
            missing_kwds={
                "color": "lightgrey",
                "edgecolor": "black",
                "hatch": "///",
                "label": "No hydro / no data",
            },
        )

        ax.set_title(
            scenario_titles[scenario_name],
            fontsize=15,
            fontweight="bold",
            pad=12,
        )

        ax.set_axis_off()

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        cax=cax,
        orientation="vertical",
    )

    cbar.set_label(
        "Average hydro selling price [€/MWh]",
        fontsize=15,
    )

    cbar.ax.tick_params(labelsize=12)

    save_figure(
        fig,
        output_path,
        bbox_inches="tight",
        pad_inches=0.05,
    )


def make_hydro_price_maps(networks, hydro_results_path, output_dir):
    shapes_year = 2050

    regions_path = (
        hydro_results_path
        / f"PyPSA_{shapes_year}"
        / "country_shapes.geojson"
    )

    if not regions_path.exists():
        raise FileNotFoundError(f"Missing country shapes file: {regions_path}")

    regions = gpd.read_file(regions_path).copy()
    regions_country_col = infer_regions_country_column(regions)

    country_price_results = {}

    for scenario_name, network in networks.items():
        country_price_results[scenario_name] = extract_hydro_country_prices(network)

    map_data_abs = {}

    for scenario_name, price_df in country_price_results.items():
        map_data_abs[scenario_name] = build_map_dataframe(
            regions_gdf=regions,
            price_df=price_df,
            regions_country_col=regions_country_col,
        )

    hydro_type = "Reservoir & Dam"

    all_abs_values = []

    for scenario_name in networks.keys():
        merged = map_data_abs[scenario_name]

        if hydro_type in merged.columns:
            all_abs_values.append(merged[hydro_type])

    all_abs_values = pd.concat(all_abs_values, axis=0)
    all_abs_values = all_abs_values.replace([np.inf, -np.inf], np.nan).dropna()

    if all_abs_values.empty:
        raise ValueError("No valid hydro selling price values found for mapping.")

    vmin = np.floor(all_abs_values.min())
    vmax = np.ceil(all_abs_values.max())

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis

    plot_hydro_price_map_by_type(
        hydro_type="Reservoir & Dam",
        map_data_abs=map_data_abs,
        norm=norm,
        cmap=cmap,
        output_path=output_dir / "hydro_price_reservoir.png",
    )


# ============================================================
# RESERVOIR SOC vs SYSTEM PRICE (mean-of-three background)
# ============================================================

def reservoir_soc_twh(network, carrier="hydro"):
    su = network.storage_units.index[network.storage_units.carrier == carrier]

    if len(su) == 0:
        raise ValueError(
            f"No StorageUnit with carrier '{carrier}'. "
            f"Available: {list(network.storage_units.carrier.unique())}"
        )

    return network.storage_units_t.state_of_charge[su].sum(axis=1) / 1e6


def reservoir_max_capacity_twh(network, carrier="hydro"):
    su = network.storage_units[network.storage_units.carrier == carrier]

    return (su["p_nom_opt"] * su["max_hours"]).sum() / 1e6


def system_price(network, carrier="AC"):
    buses = network.buses.index[network.buses.carrier == carrier]
    cols = [b for b in buses if b in network.buses_t.marginal_price.columns]

    return network.buses_t.marginal_price[cols].mean(axis=1)


def make_soc_price_figure(networks, output_path, resample="D"):
    """Reservoir SOC (absolute, TWh) for the three scenarios, over a background
    heatmap of the mean-of-three system marginal price."""
    soc_series = {
        name: reservoir_soc_twh(net)
        for name, net in networks.items()
    }
    max_caps = {
        name: reservoir_max_capacity_twh(net)
        for name, net in networks.items()
    }

    prices = {
        name: system_price(net).resample(resample).mean()
        for name, net in networks.items()
    }

    idx = list(prices.values())[0].index

    mean_price = pd.concat(prices.values(), axis=1).mean(axis=1).reindex(idx)

    soc_series = {
        name: series.resample(resample).mean().reindex(idx)
        for name, series in soc_series.items()
    }

    all_price_vals = np.concatenate([p.reindex(idx).values for p in prices.values()])
    vmin = np.nanpercentile(all_price_vals, 5)
    vmax = np.nanpercentile(all_price_vals, 95)

    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor="white")

    top = max(max_caps.values()) * 1.10

    t_num = mdates.date2num(idx.to_pydatetime())
    ax.imshow(
        mean_price.values.reshape(1, -1),
        aspect="auto",
        cmap="YlOrRd",
        extent=[t_num[0], t_num[-1], 0, top],
        alpha=0.45,
        zorder=0,
        vmin=vmin,
        vmax=vmax,
    )

    for name, soc in soc_series.items():
        ax.plot(
            idx, soc.values,
            label=name,
            color=SCENARIO_COLORS.get(name),
            linewidth=2.0,
            zorder=3,
        )

    for name, cap in max_caps.items():
        ax.axhline(
            cap,
            color=SCENARIO_COLORS.get(name),
            linestyle="--",
            linewidth=1.3,
            alpha=0.85,
            zorder=4,
            label=f"{name} — max ({cap:.0f} TWh)",
        )

    ax.set_ylabel("Reservoir State-of-Charge [TWh]", fontsize=13)
    ax.set_xlabel("")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(labelsize=11)
    ax.set_ylim(0, top)

    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        fontsize=10,
        ncol=1,
    )

    cbar = fig.colorbar(ax.images[0], ax=ax, pad=0.01, fraction=0.04)
    cbar.set_label("System marginal price [€/MWh]", fontsize=12)

    fig.tight_layout()

    save_figure(fig, output_path, bbox_inches="tight")


# ============================================================
# RESERVOIR SPILLAGE BY VOLUME CATEGORY
# ============================================================

def classify_volume_mm3(volume_mm3):
    if pd.isna(volume_mm3):
        return np.nan
    if volume_mm3 < 1:
        return "Small"
    if volume_mm3 < 3:
        return "Medium"
    if volume_mm3 <= 100:
        return "Large"
    return "Mega/Major"


def get_hydro_units_with_spill(model):
    if not hasattr(model, "storage_units_t") or not hasattr(model.storage_units_t, "spill"):
        return pd.Index([])

    if model.storage_units_t.spill.empty:
        return pd.Index([])

    hydro_units = model.storage_units.index[
        model.storage_units["carrier"]
        .astype(str)
        .str.lower()
        .str.contains("hydro", na=False)
    ]

    hydro_units = model.storage_units_t.spill.columns.intersection(hydro_units)

    return pd.Index(hydro_units.astype(str))


def apply_time_aggregation(series, rule=None, how="mean", rolling_window=None):
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()

    if rule is not None:
        if how == "mean":
            s = s.resample(rule).mean()
        elif how == "sum":
            s = s.resample(rule).sum()
        else:
            raise ValueError("resample_how must be 'mean' or 'sum'")

    if rolling_window is not None and rolling_window > 1:
        s = s.rolling(
            rolling_window,
            center=True,
            min_periods=1,
        ).mean()

    return s


def extract_plant_name_from_storage_unit(unit_name):
    s = str(unit_name).strip()
    parts = s.split()

    if "hydro" not in parts:
        return s

    hydro_pos = parts.index("hydro")

    if len(parts) > hydro_pos + 2:
        return " ".join(parts[hydro_pos + 2:])

    return s


def make_reservoir_spillage_figure(networks, hydro_results_path, output_path):
    powerplants_path = hydro_results_path / "PyPSA_2050" / "powerplants_s_100.csv"

    if not powerplants_path.exists():
        raise FileNotFoundError(f"Missing powerplants file: {powerplants_path}")

    fueltype_col = "Fueltype"
    technology_col = "Technology"
    volume_col = "Volume_Mm3"
    plant_name_col = "Name"

    fueltype_value = "Hydro"
    technology_value = "Reservoir"

    category_order = [
        "Small",
        "Medium",
        "Large",
        "Mega/Major",
    ]

    category_titles = {
        "Small": "Small (Volume < 1 Mm³)",
        "Medium": "Medium (1 Mm³ ≤ Volume < 3 Mm³)",
        "Large": "Large (3 Mm³ ≤ Volume ≤ 100 Mm³)",
        "Mega/Major": "Mega/Major (Volume > 100 Mm³)",
    }

    resample_rule = None
    resample_how = "mean"
    rolling_window = None

    spill_conversion_factor = 1.0
    spill_unit = "MW"
    month_tick_interval = 3

    powerplants = pd.read_csv(powerplants_path)

    required_cols = [
        plant_name_col,
        fueltype_col,
        technology_col,
        volume_col,
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in powerplants.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing columns in powerplants file: {missing_cols}\n"
            f"Available columns: {list(powerplants.columns)}"
        )

    reservoir_plants = powerplants[
        (powerplants[fueltype_col].astype(str).str.lower() == fueltype_value.lower())
        & (powerplants[technology_col].astype(str).str.lower() == technology_value.lower())
    ].copy()

    reservoir_plants[volume_col] = pd.to_numeric(
        reservoir_plants[volume_col],
        errors="coerce",
    )

    reservoir_plants = reservoir_plants.dropna(
        subset=[plant_name_col, volume_col],
    ).copy()

    reservoir_plants[plant_name_col] = reservoir_plants[plant_name_col].astype(str)
    reservoir_plants["volume_category"] = reservoir_plants[volume_col].apply(
        classify_volume_mm3
    )

    reservoir_plants = reservoir_plants.dropna(subset=["volume_category"]).copy()

    hydro_units_by_model = {}

    for model_name, model in networks.items():
        hydro_units = get_hydro_units_with_spill(model)

        if len(hydro_units) == 0:
            raise ValueError(f"{model_name}: no hydro storage units with spill data found.")

        hydro_units_by_model[model_name] = hydro_units

    common_units = pd.Index(
        sorted(
            set.intersection(
                *[
                    set(units.astype(str))
                    for units in hydro_units_by_model.values()
                ]
            )
        )
    )

    if len(common_units) == 0:
        raise ValueError("No common hydro storage units with spill data across models.")

    storage_unit_to_plant_name = pd.Series(
        {
            unit: extract_plant_name_from_storage_unit(unit)
            for unit in common_units
        },
        name="matched_plant_name",
    )

    plant_static_by_name = (
        reservoir_plants
        .drop_duplicates(subset=plant_name_col)
        .set_index(plant_name_col)
    )

    matched_records = []

    for unit, plant_name in storage_unit_to_plant_name.items():
        if plant_name in plant_static_by_name.index:
            row = plant_static_by_name.loc[plant_name].copy()
            row["storage_unit"] = unit
            row["matched_plant_name"] = plant_name
            matched_records.append(row)

    plant_static = pd.DataFrame(matched_records)

    if plant_static.empty:
        raise ValueError(
            "No matched reservoir plants after extracting plant names from storage unit index."
        )

    plant_static = plant_static.set_index("storage_unit")
    matched_units = pd.Index(plant_static.index.astype(str))

    units_by_category = {
        category: pd.Index(
            plant_static.index[
                plant_static["volume_category"] == category
            ].astype(str)
        )
        for category in category_order
    }

    mean_spill_by_model_and_category = {}

    for model_name, model in networks.items():
        spill_df = model.storage_units_t.spill.copy()
        spill_df.index = pd.to_datetime(spill_df.index)
        spill_df = spill_df.sort_index()
        spill_df.columns = spill_df.columns.astype(str)

        valid_units = spill_df.columns.intersection(matched_units)

        if len(valid_units) == 0:
            raise ValueError(
                f"{model_name}: no matched categorized hydro units in spill data."
            )

        mean_spill_by_model_and_category[model_name] = {}

        for category in category_order:
            units_in_cat = valid_units.intersection(units_by_category[category])

            if len(units_in_cat) == 0:
                mean_series = pd.Series(index=spill_df.index, dtype=float)
            else:
                mean_series = spill_df[units_in_cat].mean(axis=1)

            mean_series = mean_series * spill_conversion_factor

            mean_series = apply_time_aggregation(
                mean_series,
                rule=resample_rule,
                how=resample_how,
                rolling_window=rolling_window,
            )

            mean_spill_by_model_and_category[model_name][category] = mean_series

    fig, axes = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=(18, 4.2),
        sharex=True,
        sharey=True,
    )

    axes = axes.flatten()

    for ax, category in zip(axes, category_order):
        for model_name in networks.keys():
            s = mean_spill_by_model_and_category[model_name][category]

            if s.empty or s.dropna().empty:
                continue

            ax.plot(
                s.index,
                s.values,
                label=model_name,
                color=SCENARIO_COLORS.get(model_name, None),
                linewidth=1.8,
                zorder=3,
            )

        ax.set_title(
            category_titles[category],
            fontsize=15,
            fontweight="bold",
            pad=10,
        )

        ax.grid(
            axis="both",
            linestyle="--",
            linewidth=0.7,
            alpha=0.35,
            zorder=0,
        )

        ax.set_axisbelow(True)

        ax.xaxis.set_major_locator(
            mdates.MonthLocator(interval=month_tick_interval)
        )

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%b")
        )

        ax.tick_params(axis="x", labelsize=11, rotation=0)
        ax.tick_params(axis="y", labelsize=11)

    axes[0].set_ylabel(
        f"Mean spillage [{spill_unit}]",
        fontsize=13,
    )

    for ax in axes:
        ax.set_xlabel("")

    handles, labels = axes[0].get_legend_handles_labels()

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            frameon=True,
            framealpha=0.95,
            edgecolor="black",
            fontsize=16,
            title_fontsize=16,
            bbox_to_anchor=(0.5, 0.98),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.subplots_adjust(wspace=0.16)

    save_figure(
        fig,
        output_path,
        bbox_inches="tight",
        pad_inches=0.05,
    )


# ============================================================
# WORKFLOW
# ============================================================

def run_processing_2050(
    hydro_results_path: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    hydro_results_path = (
        hydro_results_path.resolve()
        if hydro_results_path is not None
        else get_hydro_results_root()
    )

    output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else hydro_results_path / "images" / "2050"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Hydro results path: %s", hydro_results_path)
    logger.info("2050 output directory: %s", output_dir)

    networks = load_2050_networks(hydro_results_path)

    make_cost_difference_figure(
        networks=networks,
        output_path=output_dir / "cost_difference.png",
    )

    make_optimal_capacity_pct_figure(
        networks=networks,
        output_path=output_dir / "optimal_capacity.png",
    )

    make_capacity_and_supply_figures(
        networks=networks,
        output_dir=output_dir,
    )

    make_storage_difference_figure(
        networks=networks,
        output_path=output_dir / "storage_diff.png",
    )

    make_hydro_price_maps(
        networks=networks,
        hydro_results_path=hydro_results_path,
        output_dir=output_dir,
    )

    make_soc_price_figure(
        networks=networks,
        output_path=output_dir / "soc_price.png",
    )

    make_reservoir_spillage_figure(
        networks=networks,
        hydro_results_path=hydro_results_path,
        output_path=output_dir / "reservoir_spillage_by_volume_category.png",
    )

    logger.info("2050 figures saved to: %s", output_dir)


def main() -> None:
    setup_logging()

    args = parse_args()

    run_processing_2050(
        hydro_results_path=args.hydro_results,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()