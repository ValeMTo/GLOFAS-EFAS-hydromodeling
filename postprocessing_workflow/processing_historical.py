#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import logging
import re
import unicodedata
import warnings

import matplotlib
matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import os
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from shapely.geometry import Point
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

NET_YEARS = [2015, 2016, 2017, 2018, 2019]
ENTSOE_YEARS = range(2015, 2020)

NETWORK_FILE = "base_s_100_elec_.nc"

NETWORK_FOLDERS = {
    "pypsa": "PyPSA_{year}",
    "glofas_saber": "GloFAS-SABER_{year}",
    "efas_saber": "EFAS-SABER_{year}",
}

CH_YEARS_WITHOUT_ELECTRICITY_MAPS = [2015, 2016]

COMPONENTS = ["Pumped", "RoR", "Reservoir"]

MIN_COMPONENT_COVERAGE = 0.50
MIN_COMPONENT_VALID_HOURS = 24 * 30
MIN_PERIOD_COVERAGE = 0.90

HYDRO_TYPE = "total_no_phs"
ENTSOE_SUFFIX = "Total"

ACTIVE_MODELS = [
    "PyPSA baseline",
    "PyPSA + GloFAS-SABER",
    "PyPSA + EFAS-SABER",
]

MODEL_COLORS = {
    "PyPSA baseline": "red",
    "PyPSA + GloFAS-SABER": "blue",
    "PyPSA + EFAS-SABER": "green",
}

MODEL_LINEWIDTHS = {
    "PyPSA baseline": 2.0,
    "PyPSA + GloFAS-SABER": 2.0,
    "PyPSA + EFAS-SABER": 2.0,
}

MODEL_ALPHA = {
    "PyPSA baseline": 0.6,
    "PyPSA + GloFAS-SABER": 0.6,
    "PyPSA + EFAS-SABER": 0.6,
}

PLOT_TITLE = "Hydro"

MIN_MONTHS_FOR_METRICS = 6
MIN_EUROPE_WEEKLY_COVERAGE = MIN_PERIOD_COVERAGE
MIN_REGIONAL_WEEKLY_COVERAGE = MIN_PERIOD_COVERAGE
MIN_COUNTRIES_PER_TIMESTEP = 1
MIN_MEAN_COUNTRIES_PER_WEEK = 1.0

# Figure 2 keeps the aggregated Nordic region. The old aggregated Alpine
# region has been replaced by the CH / AT / North-Italy 3-panel figure.
SELECTED_REGIONS = {
    "Nordic countries": ["FI", "SE", "NO"],
}

# ------------------------------------------------------------
# North-Italy (bidding-zone) pipeline settings
# ------------------------------------------------------------

GADM_NAME_COL = "NAME_1"

NORD_REGIONS = {
    "piemonte", "valle daosta", "liguria", "lombardia",
    "trentino alto adige", "veneto", "friuli venezia giulia", "emilia romagna",
}

GADM_SYNONYMS = {
    "valledaosta": "valle daosta",
    "aosta valley": "valle daosta",
    "trentino altoadige": "trentino alto adige",
    "trentino alto adige sudtirol": "trentino alto adige",
    "friuli veneziagiulia": "friuli venezia giulia",
    "sicily": "sicilia",
    "apulia": "puglia",
}

ALL_ITALIAN_REGIONS = NORD_REGIONS | {
    "toscana", "umbria", "marche", "lazio", "abruzzo", "campania",
    "molise", "puglia", "basilicata", "calabria", "sicilia", "sardegna",
}

# bus -> group manual override, e.g. {"IT2 1": "NORD"}
BUS_GROUP_OVERRIDE = {}

# Italian bidding zone used as North reference (ENTSO-E)
ITALY_NORTH_ZONE = "IT_North"
ITALY_ZONE_COMPONENT = "Total"   # Total = RoR + Reservoir (excludes PHS)

# ------------------------------------------------------------
# Supplementary multi-country panels (figure 5)
# ------------------------------------------------------------

COUNTRY_NAMES = {
    "NO": "Norway", "SE": "Sweden", "FI": "Finland",
    "CH": "Switzerland", "AT": "Austria", "IT": "Italy",
    "DE": "Germany", "FR": "France", "ES": "Spain",
    "PT": "Portugal", "NL": "Netherlands", "BE": "Belgium",
    "PL": "Poland", "CZ": "Czech Republic", "SK": "Slovakia",
    "HU": "Hungary", "RO": "Romania", "HR": "Croatia",
    "SI": "Slovenia", "GR": "Greece", "BG": "Bulgaria",
    "DK": "Denmark", "LT": "Lithuania", "LV": "Latvia",
    "EE": "Estonia", "GB": "Great Britain", "IE": "Ireland",
    "LU": "Luxembourg", "RS": "Serbia", "AL": "Albania",
    "BA": "Bosnia & Herz.", "ME": "Montenegro",
    "MK": "N. Macedonia",
}

SUPPLEMENTARY_PANEL_GROUPS = {
    "group1": ["NO", "SE", "FI", "MK"],
    "group2": ["FR", "CH", "AT", "IT"],
    "group3": ["ES", "PT", "GB", "IE"],
    "group4": ["DE", "PL", "CZ", "SK"],
    "group5": ["RO", "BG", "HU", "GR"],
    "group6": ["HR", "RS", "SI", "BA"],
    "group7": ["BE", "LT", "LV", "ME"],
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


def get_data_root():
    repo_root = get_repo_root()

    return Path(
        os.environ.get(
            "HYDRO_DATA_ROOT",
            repo_root / "data" / "hydro_workflow",
        )
    ).resolve()


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate historical hydro validation figures."
    )

    parser.add_argument(
        "--hydro-results",
        type=Path,
        default=None,
        help="Path to hydro_results directory.",
    )

    parser.add_argument(
        "--entsoe-production",
        type=Path,
        default=None,
        help="Path to ENTSO-E hydro production CSV files (per-country).",
    )

    parser.add_argument(
        "--electricity-maps",
        type=Path,
        default=None,
        help="Path to Electricity Maps JSON files.",
    )

    parser.add_argument(
        "--gadm-italy",
        type=Path,
        default=None,
        help="Path to GADM level-1 Italy file (gadm41_ITA_1.json).",
    )

    parser.add_argument(
        "--glohydrores",
        type=Path,
        default=None,
        help="Path to GloHydroRes CSV (GloHydroRes_vs1.csv).",
    )

    parser.add_argument(
        "--italy-zone-production",
        type=Path,
        default=None,
        help="Path to ENTSO-E Italian bidding-zone production CSV files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for generated figures.",
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


def model_key(model_name):
    """Canonical column-safe key for a model name.

    Must match the key used to build the KGESS map columns.
    'PyPSA + GloFAS-SABER' -> 'PyPSA_plus_GloFAS_SABER'.
    """
    return (
        model_name
        .replace("+", "plus")
        .replace("-", "_")
        .replace(" ", "_")
    )


# ============================================================
# NETWORK LOADING
# ============================================================

def load_networks(base_path, folder_template, years, network_file, engine="netcdf4"):
    networks = {}

    for year in years:
        path = base_path / folder_template.format(year=year) / network_file

        if not path.exists():
            raise FileNotFoundError(f"Network file not found: {path}")

        networks[year] = pypsa.Network(path, engine=engine)

    return networks


# ============================================================
# ENTSO-E LOADING AND PROCESSING (per-country)
# ============================================================

def load_ch_hydro_hourly_mw_from_electricity_maps(json_path, year):
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []

    for item in raw["data"]:
        timestamp = pd.to_datetime(item["datetime"]).tz_localize(None)
        hydro_mwh = item["mix"].get("hydro")

        records.append(
            {
                "datetime": timestamp,
                "CH_Total": hydro_mwh / 24 if hydro_mwh is not None else np.nan,
            }
        )

    daily_mw = (
        pd.DataFrame(records)
        .set_index("datetime")
        .sort_index()
    )

    daily_mw = daily_mw[daily_mw.index.year == year]

    hourly_index = pd.date_range(
        start=f"{year}-01-01 00:00:00",
        end=f"{year}-12-31 23:00:00",
        freq="h",
    )

    hourly_mw = daily_mw.reindex(hourly_index, method="ffill")

    return hourly_mw


def remove_country_columns(df, countries):
    columns_to_drop = [
        col
        for col in df.columns
        if any(col.startswith(f"{country}_") for country in countries)
    ]

    return df.drop(columns=columns_to_drop, errors="ignore")


def resample_mean_with_min_coverage(df, freq, min_coverage=0.90):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    period_mean = df.resample(freq).mean()
    valid_counts = df.resample(freq).count()
    expected_counts = df.resample(freq).size()

    coverage = valid_counts.div(expected_counts, axis=0)

    return period_mean.where(coverage >= min_coverage)


def get_countries_from_columns(df):
    return sorted(
        {
            col.rsplit("_", 1)[0]
            for col in df.columns
            if "_" in col
        }
    )


def component_is_usable(
    series,
    min_coverage=MIN_COMPONENT_COVERAGE,
    min_valid_hours=MIN_COMPONENT_VALID_HOURS,
):
    valid_hours = int(series.notna().sum())
    total_hours = len(series)
    coverage = valid_hours / total_hours if total_hours > 0 else 0.0

    return valid_hours >= min_valid_hours and coverage >= min_coverage


def build_nopumped_dataset(df_base, countries, year, ch_electricity_maps_files):
    output = pd.DataFrame(index=df_base.index)

    for country in countries:
        total_col = f"{country}_Total"
        ror_col = f"{country}_RoR"
        reservoir_col = f"{country}_Reservoir"

        if country == "CH" and year in ch_electricity_maps_files:
            if total_col in df_base.columns:
                output[total_col] = df_base[total_col]

            output[ror_col] = np.nan
            output[reservoir_col] = np.nan

            continue

        has_ror_col = ror_col in df_base.columns
        has_reservoir_col = reservoir_col in df_base.columns

        usable_ror = (
            has_ror_col
            and component_is_usable(df_base[ror_col])
        )

        usable_reservoir = (
            has_reservoir_col
            and component_is_usable(df_base[reservoir_col])
        )

        if usable_ror:
            output[ror_col] = df_base[ror_col]

        if usable_reservoir:
            output[reservoir_col] = df_base[reservoir_col]

        if usable_ror and usable_reservoir:
            output[total_col] = df_base[ror_col] + df_base[reservoir_col]
        elif usable_ror:
            output[total_col] = df_base[ror_col]
        elif usable_reservoir:
            output[total_col] = df_base[reservoir_col]

    ordered_cols = [
        f"{country}_{component}"
        for country in countries
        for component in ["RoR", "Reservoir", "Total"]
        if f"{country}_{component}" in output.columns
    ]

    return output[ordered_cols].copy()


def load_entsoe_hydro_data(entsoe_production_path, electricity_maps_path):
    ch_electricity_maps_files = {
        2017: electricity_maps_path / "CH_2017_electricity_maps.json",
        2018: electricity_maps_path / "CH_2018_electricity_maps.json",
        2019: electricity_maps_path / "CH_2019_electricity_maps.json",
    }

    hydro_data = {}

    for year in ENTSOE_YEARS:
        filepath = entsoe_production_path / f"Europe_Hydro_{year}.csv"

        if not filepath.exists():
            raise FileNotFoundError(f"Missing ENTSO-E file: {filepath}")

        df = pd.read_csv(filepath, index_col=0)
        df.index = pd.to_datetime(df.index)

        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

        df.index = df.index.floor("h")
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        if year in ch_electricity_maps_files:
            ch_path = ch_electricity_maps_files[year]

            if not ch_path.exists():
                raise FileNotFoundError(f"Missing Electricity Maps file: {ch_path}")

            ch_hourly = load_ch_hydro_hourly_mw_from_electricity_maps(
                json_path=ch_path,
                year=year,
            )

            ch_hourly = ch_hourly.reindex(df.index)

            df["CH_Total"] = ch_hourly["CH_Total"]

            for component in COMPONENTS:
                df[f"CH_{component}"] = np.nan

        elif year in CH_YEARS_WITHOUT_ELECTRICITY_MAPS:
            df = remove_country_columns(df, ["CH"])

        hydro_data[year] = df

    hydro_data_nopumped = {}
    hydro_data_ror = {}
    hydro_data_reservoir = {}

    for year, df in hydro_data.items():
        df_base = df.copy()
        countries = get_countries_from_columns(df_base)

        ror_cols = [
            f"{country}_RoR"
            for country in countries
            if f"{country}_RoR" in df_base.columns
        ]

        reservoir_cols = [
            f"{country}_Reservoir"
            for country in countries
            if f"{country}_Reservoir" in df_base.columns
        ]

        hydro_data_ror[year] = df_base[ror_cols].copy()
        hydro_data_reservoir[year] = df_base[reservoir_cols].copy()

        hydro_data_nopumped[year] = build_nopumped_dataset(
            df_base=df_base,
            countries=countries,
            year=year,
            ch_electricity_maps_files=ch_electricity_maps_files,
        )

    return hydro_data_nopumped, hydro_data_ror, hydro_data_reservoir


# ============================================================
# MODEL HYDRO EXTRACTION (per-country)
# ============================================================

def get_country_from_bus(bus_name):
    return str(bus_name)[:2]


def extract_hydro_by_country(network, hydro_type="total_no_phs"):
    valid_hydro_types = ["total_no_phs", "ror", "reservoir"]

    if hydro_type not in valid_hydro_types:
        raise ValueError(f"hydro_type must be one of {valid_hydro_types}")

    extracted = []

    if hydro_type in ["total_no_phs", "ror"]:
        ror_units = network.generators[network.generators.carrier == "ror"]

        if not ror_units.empty:
            ror_country_map = ror_units.bus.map(get_country_from_bus)

            ror_by_country = (
                network.generators_t.p[ror_units.index]
                .clip(lower=0.0)
                .T.groupby(ror_country_map)
                .sum()
                .T
            )

            extracted.append(ror_by_country)

    if hydro_type in ["total_no_phs", "reservoir"]:
        reservoir_units = network.storage_units[
            network.storage_units.carrier == "hydro"
        ]

        if not reservoir_units.empty:
            reservoir_country_map = reservoir_units.bus.map(get_country_from_bus)

            reservoir_by_country = (
                network.storage_units_t.p[reservoir_units.index]
                .clip(lower=0.0)
                .T.groupby(reservoir_country_map)
                .sum()
                .T
            )

            extracted.append(reservoir_by_country)

    if not extracted:
        return pd.DataFrame(index=network.snapshots)

    hydro_by_country = pd.concat(extracted, axis=1)
    hydro_by_country = hydro_by_country.T.groupby(level=0).sum().T
    hydro_by_country = hydro_by_country.clip(lower=0.0)

    return hydro_by_country


def extract_hydro_timeseries_from_networks(networks, years, hydro_type="total_no_phs"):
    if len(networks) != len(years):
        raise ValueError("networks and years must have the same length.")

    yearly_data = []

    for network, year in zip(networks, years):
        df_year = extract_hydro_by_country(
            network,
            hydro_type=hydro_type,
        ).copy()

        df_year.index = pd.to_datetime(df_year.index)
        df_year = df_year.sort_index()

        yearly_data.append(df_year)

    common_countries = sorted(set().union(*(df.columns for df in yearly_data)))

    yearly_data = [
        df.reindex(columns=common_countries, fill_value=0.0)
        for df in yearly_data
    ]

    hydro_final = pd.concat(yearly_data, axis=0)
    hydro_final = hydro_final.sort_index()

    return hydro_final


# ============================================================
# METRICS
# ============================================================

def kge0(obs, sim, min_points=5, eps=1e-12, constant_corr=0.0):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]

    if obs.size < min_points:
        return np.nan

    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)

    std_obs = np.std(obs, ddof=0)
    std_sim = np.std(sim, ddof=0)

    if abs(mean_obs) < eps:
        return np.nan

    beta = mean_sim / mean_obs

    if std_obs < eps or std_sim < eps:
        corr = constant_corr
    else:
        corr = np.corrcoef(obs, sim)[0, 1]

    if not np.isfinite(corr):
        corr = constant_corr

    if abs(mean_sim) < eps or std_obs < eps:
        return np.nan

    cv_obs = std_obs / mean_obs
    cv_sim = std_sim / mean_sim

    if abs(cv_obs) < eps:
        return np.nan

    gamma = cv_sim / cv_obs

    if not np.isfinite(beta) or not np.isfinite(gamma):
        return np.nan

    return 1.0 - np.sqrt(
        (corr - 1.0) ** 2
        + (beta - 1.0) ** 2
        + (gamma - 1.0) ** 2
    )


def kgess(obs, sim, min_points=5, eps=1e-12):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    mask = np.isfinite(obs) & np.isfinite(sim)
    obs_valid = obs[mask]
    sim_valid = sim[mask]

    if obs_valid.size < min_points:
        return np.nan

    kge_model = kge0(
        obs_valid,
        sim_valid,
        min_points=min_points,
        eps=eps,
        constant_corr=0.0,
    )

    benchmark = np.full_like(obs_valid, np.nanmean(obs_valid), dtype=float)

    kge_benchmark = kge0(
        obs_valid,
        benchmark,
        min_points=min_points,
        eps=eps,
        constant_corr=0.0,
    )

    if not np.isfinite(kge_model) or not np.isfinite(kge_benchmark):
        return np.nan

    denominator = 1.0 - kge_benchmark

    if abs(denominator) < eps:
        return np.nan

    return (kge_model - kge_benchmark) / denominator


# ============================================================
# DATASET BUILDERS
# ============================================================

def build_entsoe_all(entsoe_data):
    entsoe_list = []

    for year in ENTSOE_YEARS:
        if year not in entsoe_data:
            continue

        df = entsoe_data[year].copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        entsoe_list.append(df)

    if not entsoe_list:
        raise ValueError("No ENTSO-E data available.")

    entsoe_all = pd.concat(entsoe_list, axis=0).sort_index()
    entsoe_all = entsoe_all[~entsoe_all.index.duplicated(keep="first")]

    return entsoe_all


def build_model_all(model_datasets):
    model_all = {}

    for model_name in ACTIVE_MODELS:
        if model_name not in model_datasets:
            raise ValueError(f"Unknown model selected: {model_name}")

        df = model_datasets[model_name].copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        model_all[model_name] = df

    return model_all


def prepare_entsoe_country_dataframe(entsoe_all):
    entsoe_df = entsoe_all.filter(like=f"_{ENTSOE_SUFFIX}").copy()

    entsoe_df.columns = entsoe_df.columns.str.replace(
        f"_{ENTSOE_SUFFIX}",
        "",
        regex=False,
    )

    entsoe_df.index = pd.to_datetime(entsoe_df.index)
    entsoe_df = entsoe_df.sort_index()
    entsoe_df = entsoe_df[~entsoe_df.index.duplicated(keep="first")]

    return entsoe_df


def get_available_years(entsoe_df, model_all):
    year_sets = [set(entsoe_df.index.year)]

    for df_model in model_all.values():
        year_sets.append(set(df_model.index.year))

    available_years = sorted(set.intersection(*year_sets))

    if not available_years:
        raise ValueError("No common years found across ENTSO-E and selected models.")

    return available_years


# ============================================================
# WEEKLY COUNTRY METRICS FOR MAP
# ============================================================

def compute_country_weekly_metrics(entsoe_all, model_all):
    entsoe_countries = {
        col.replace(f"_{ENTSOE_SUFFIX}", "")
        for col in entsoe_all.columns
        if col.endswith(f"_{ENTSOE_SUFFIX}")
    }

    model_country_sets = [
        set(df.columns)
        for df in model_all.values()
    ]

    common_countries_weekly = sorted(
        set.intersection(entsoe_countries, *model_country_sets)
    )

    results_weekly = []

    for country in common_countries_weekly:
        entsoe_col = f"{country}_{ENTSOE_SUFFIX}"

        series_to_compare = [
            entsoe_all[entsoe_col].rename("ENTSOE")
        ]

        for model_name, df_model in model_all.items():
            series_to_compare.append(
                df_model[country].rename(model_name)
            )

        df_compare = pd.concat(
            series_to_compare,
            axis=1,
            join="inner",
        ).dropna(how="all")

        df_compare = df_compare.dropna(subset=["ENTSOE"])

        if df_compare.empty:
            continue

        df_weekly = df_compare.resample("1W").mean()

        metric_columns = ["ENTSOE"] + ACTIVE_MODELS
        df_weekly_metric = df_weekly.dropna(subset=metric_columns)

        if len(df_weekly_metric) < MIN_MONTHS_FOR_METRICS:
            continue

        country_results = {
            "Country": country,
            "N_WEEKS_ENTSOE": int(df_weekly["ENTSOE"].notna().sum()),
            "N_WEEKS_COMMON_METRIC": len(df_weekly_metric),
        }

        for model_name in ACTIVE_MODELS:
            kgess_model = kgess(
                df_weekly_metric["ENTSOE"],
                df_weekly_metric[model_name],
            )

            metric_name = model_key(model_name)

            country_results[f"KGESS_{metric_name}"] = kgess_model
            country_results[f"N_WEEKS_{metric_name}"] = int(
                df_weekly[model_name].notna().sum()
            )

        results_weekly.append(country_results)

    if not results_weekly:
        raise ValueError(
            "No weekly country results available. Check common countries and time overlap."
        )

    results_df_weekly = pd.DataFrame(results_weekly).set_index("Country")

    sort_column = f"KGESS_{model_key(ACTIVE_MODELS[0])}"

    if sort_column in results_df_weekly.columns:
        results_df_weekly = results_df_weekly.sort_values(
            sort_column,
            ascending=False,
        )

    return results_df_weekly


# ============================================================
# EUROPE WEEKLY FIGURE
# ============================================================

def build_europe_weekly_series(entsoe_all, model_all):
    entsoe_df = prepare_entsoe_country_dataframe(entsoe_all)
    available_years = get_available_years(entsoe_df, model_all)

    model_eu_all = {}

    for model_name in ACTIVE_MODELS:
        df_model = model_all[model_name].copy()
        df_model.index = pd.to_datetime(df_model.index)
        df_model = df_model.sort_index()
        df_model = df_model[~df_model.index.duplicated(keep="first")]

        model_eu_all[model_name] = df_model

    europe_yearly_series = []

    for year in available_years:
        entsoe_year = entsoe_df.loc[entsoe_df.index.year == year]

        model_years = {
            model_name: df_model.loc[df_model.index.year == year]
            for model_name, df_model in model_eu_all.items()
        }

        if entsoe_year.empty or any(df.empty for df in model_years.values()):
            continue

        country_sets_year = [set(entsoe_year.columns)]

        for df_model_year in model_years.values():
            country_sets_year.append(set(df_model_year.columns))

        common_countries_year = sorted(set.intersection(*country_sets_year))

        if not common_countries_year:
            continue

        country_triplets = []

        for country in common_countries_year:
            series_for_country = [
                entsoe_year[country].rename((country, "ENTSO-E"))
            ]

            for model_name, df_model_year in model_years.items():
                series_for_country.append(
                    df_model_year[country].rename((country, model_name))
                )

            df_country = pd.concat(
                series_for_country,
                axis=1,
                join="inner",
            )

            df_country.columns = pd.MultiIndex.from_tuples(df_country.columns)

            required_columns = [(country, "ENTSO-E")] + [
                (country, model_name)
                for model_name in ACTIVE_MODELS
            ]

            valid_mask = df_country[required_columns].notna().all(axis=1)
            df_country.loc[~valid_mask, :] = np.nan

            if int(valid_mask.sum()) > 0:
                country_triplets.append(df_country)

        if not country_triplets:
            continue

        df_country_year = pd.concat(country_triplets, axis=1).sort_index()

        entsoe_country_matrix = df_country_year.xs(
            "ENTSO-E",
            axis=1,
            level=1,
        )

        n_countries_used_ts = entsoe_country_matrix.notna().sum(axis=1)

        series_to_compare = []

        entsoe_europe = (
            df_country_year
            .xs("ENTSO-E", axis=1, level=1)
            .sum(axis=1, min_count=MIN_COUNTRIES_PER_TIMESTEP)
            / 1e3
        ).rename("ENTSO-E")

        series_to_compare.append(entsoe_europe)

        for model_name in ACTIVE_MODELS:
            model_europe = (
                df_country_year
                .xs(model_name, axis=1, level=1)
                .sum(axis=1, min_count=MIN_COUNTRIES_PER_TIMESTEP)
                / 1e3
            ).rename(model_name)

            series_to_compare.append(model_europe)

        df_europe_year = pd.concat(
            series_to_compare,
            axis=1,
            join="inner",
        )

        df_europe_year["N_COUNTRIES_USED"] = n_countries_used_ts

        df_europe_year = df_europe_year.dropna(
            subset=["ENTSO-E"] + ACTIVE_MODELS
        )

        if not df_europe_year.empty:
            europe_yearly_series.append(df_europe_year)

    if not europe_yearly_series:
        raise ValueError("No European yearly series could be built.")

    df_europe = pd.concat(europe_yearly_series, axis=0).sort_index()
    df_europe = df_europe[~df_europe.index.duplicated(keep="first")]

    df_weekly_eu = resample_mean_with_min_coverage(
        df_europe[["ENTSO-E"] + ACTIVE_MODELS],
        freq="1W",
        min_coverage=MIN_EUROPE_WEEKLY_COVERAGE,
    )

    metric_columns = ["ENTSO-E"] + ACTIVE_MODELS

    df_weekly_eu_metric = df_weekly_eu.dropna(
        subset=metric_columns
    )

    if df_weekly_eu_metric.empty:
        raise ValueError("No valid weekly data available for Europe metrics.")

    kgess_values_eu = {}

    for model_name in ACTIVE_MODELS:
        kgess_values_eu[model_name] = kgess(
            df_weekly_eu_metric["ENTSO-E"],
            df_weekly_eu_metric[model_name],
        )

    return df_weekly_eu, kgess_values_eu, available_years


def plot_weekly_series(df_weekly, kgess_values, title_prefix, output_path, countries_label=None):
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        df_weekly.index,
        df_weekly["ENTSO-E"],
        color="black",
        linewidth=3.5,
        label="ENTSO-E",
    )

    for model_name in ACTIVE_MODELS:
        ax.plot(
            df_weekly.index,
            df_weekly[model_name],
            color=MODEL_COLORS.get(model_name),
            linewidth=MODEL_LINEWIDTHS.get(model_name, 2.0),
            alpha=MODEL_ALPHA.get(model_name, 0.8),
            label=model_name,
        )

    plot_years = sorted(df_weekly.index.year.unique())

    for year in plot_years:
        ax.axvline(
            pd.Timestamp(f"{year}-01-01"),
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.35,
        )

    kgess_text = " | ".join(
        f"{model_name}={kgess_values[model_name]:.2f}"
        for model_name in ACTIVE_MODELS
    )

    plot_start_year = df_weekly.index.min().year
    plot_end_year = df_weekly.index.max().year

    if countries_label is None:
        title = (
            f"{title_prefix} — Weekly {PLOT_TITLE} "
            f"{plot_start_year}-{plot_end_year - 1}\n"
            f"KGESS → {kgess_text}"
        )
    else:
        title = (
            f"{title_prefix} — Weekly {PLOT_TITLE} "
            f"{plot_start_year}-{plot_end_year - 1}\n"
            f"Countries: {countries_label}\n"
            f"KGESS → {kgess_text}"
        )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=14,
    )

    ax.set_ylabel("Power (GW)", fontsize=13)

    ax.legend(
        fontsize=12,
        title_fontsize=12,
        loc="best",
        frameon=True,
    )

    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    save_figure(fig, output_path)


# ============================================================
# REGIONAL WEEKLY FIGURES (aggregated)
# ============================================================

def build_region_weekly_series(entsoe_all, model_all, requested_countries):
    entsoe_df = prepare_entsoe_country_dataframe(entsoe_all)
    available_years = get_available_years(entsoe_df, model_all)

    entsoe_df = entsoe_df[entsoe_df.index.year.isin(available_years)]

    model_region_all = {}

    for model_name in ACTIVE_MODELS:
        df_model = model_all[model_name].copy()
        df_model.index = pd.to_datetime(df_model.index)
        df_model = df_model.sort_index()
        df_model = df_model[~df_model.index.duplicated(keep="first")]
        df_model = df_model[df_model.index.year.isin(available_years)]

        model_region_all[model_name] = df_model

    country_blocks = []

    for country in requested_countries:
        country_available = (
            country in entsoe_df.columns
            and all(
                country in df_model.columns
                for df_model in model_region_all.values()
            )
        )

        if not country_available:
            continue

        country_series = [
            entsoe_df[country].rename((country, "ENTSO-E"))
        ]

        for model_name, df_model in model_region_all.items():
            country_series.append(
                df_model[country].rename((country, model_name))
            )

        tmp = pd.concat(
            country_series,
            axis=1,
            join="inner",
        )

        tmp.columns = pd.MultiIndex.from_tuples(tmp.columns)

        valid_columns = [(country, "ENTSO-E")] + [
            (country, model_name)
            for model_name in ACTIVE_MODELS
        ]

        valid_mask = tmp[valid_columns].notna().all(axis=1)
        tmp.loc[~valid_mask, :] = np.nan

        if int(valid_mask.sum()) > 0:
            country_blocks.append(tmp)

    if not country_blocks:
        raise ValueError(
            f"No countries with valid common timesteps for region: {requested_countries}"
        )

    df_country_region = pd.concat(country_blocks, axis=1).sort_index()

    countries_used = sorted(
        df_country_region.columns.get_level_values(0).unique()
    )

    entsoe_country_matrix = df_country_region.xs(
        "ENTSO-E",
        axis=1,
        level=1,
    )

    n_countries_used_ts = entsoe_country_matrix.notna().sum(axis=1)

    regional_series = {}

    regional_series["ENTSO-E"] = (
        df_country_region
        .xs("ENTSO-E", axis=1, level=1)
        .sum(axis=1, min_count=MIN_COUNTRIES_PER_TIMESTEP)
        / 1e3
    )

    for model_name in ACTIVE_MODELS:
        regional_series[model_name] = (
            df_country_region
            .xs(model_name, axis=1, level=1)
            .sum(axis=1, min_count=MIN_COUNTRIES_PER_TIMESTEP)
            / 1e3
        )

    df_region = pd.concat(
        [regional_series["ENTSO-E"].rename("ENTSO-E")]
        + [
            regional_series[model_name].rename(model_name)
            for model_name in ACTIVE_MODELS
        ],
        axis=1,
        join="inner",
    )

    df_region["N_COUNTRIES_USED"] = n_countries_used_ts

    df_region = df_region.dropna(subset=["ENTSO-E"] + ACTIVE_MODELS)

    if df_region.empty:
        raise ValueError("No common timesteps found after regional aggregation.")

    df_weekly_region = resample_mean_with_min_coverage(
        df_region[["ENTSO-E"] + ACTIVE_MODELS],
        freq="1W",
        min_coverage=MIN_REGIONAL_WEEKLY_COVERAGE,
    )

    df_weekly_region_countries = df_region[["N_COUNTRIES_USED"]].resample("1W").mean()

    low_country_weeks = (
        df_weekly_region_countries["N_COUNTRIES_USED"]
        < MIN_MEAN_COUNTRIES_PER_WEEK
    )

    df_weekly_region.loc[low_country_weeks, :] = np.nan

    metric_columns = ["ENTSO-E"] + ACTIVE_MODELS

    df_weekly_region_metric = df_weekly_region.dropna(
        subset=metric_columns
    )

    if df_weekly_region_metric.empty:
        raise ValueError("No valid weekly data for regional metrics.")

    kgess_values = {}

    for model_name in ACTIVE_MODELS:
        kgess_values[model_name] = kgess(
            df_weekly_region_metric["ENTSO-E"],
            df_weekly_region_metric[model_name],
        )

    return df_weekly_region, kgess_values, countries_used


# ============================================================
# NORTH-ITALY (BIDDING ZONE) PIPELINE
# ============================================================

def _normalize_region_name(name):
    s = str(name).split("/")[0]
    s = "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )
    s = s.lower().strip().replace("'", "")
    s = re.sub(r"[-_.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return GADM_SYNONYMS.get(s, s)


def build_nord_polygon(gadm_italy_path):
    """Dissolved polygon of the Italian 'North' bidding-zone regions."""
    if not gadm_italy_path.exists():
        raise FileNotFoundError(f"Missing GADM Italy file: {gadm_italy_path}")

    regions = gpd.read_file(gadm_italy_path).to_crs(4326)
    regions["norm"] = regions[GADM_NAME_COL].map(_normalize_region_name)

    unmatched = set(regions["norm"]) - ALL_ITALIAN_REGIONS
    if unmatched:
        logger.warning("Unrecognised GADM region names: %s", unmatched)

    regions["is_nord"] = regions["norm"].isin(NORD_REGIONS)

    n_found = int(regions["is_nord"].sum())
    logger.info("North-Italy regions found: %d / %d", n_found, len(NORD_REGIONS))

    nord_poly = regions[regions["is_nord"]].dissolve().geometry.iloc[0]

    return nord_poly


def build_bus_group(network, nord_poly):
    """Map each Italian bus to 'NORD' or 'RESTO' by point-in-polygon."""
    buses = network.buses

    if "country" in buses.columns:
        it_buses = buses[buses["country"] == "IT"]
    else:
        it_buses = buses[buses.index.astype(str).str.startswith("IT")]

    def assign_group(x, y):
        return "NORD" if nord_poly.contains(Point(x, y)) else "RESTO"

    bus_group = pd.Series(
        {
            b: assign_group(it_buses.loc[b, "x"], it_buses.loc[b, "y"])
            for b in it_buses.index
        },
        name="group",
    )

    for b, g in BUS_GROUP_OVERRIDE.items():
        if b in bus_group.index:
            bus_group[b] = g

    logger.info("Bus group counts: %s", bus_group.value_counts().to_dict())

    return bus_group, it_buses


def _plant_name_from_index(idx):
    m = re.search(r"hydro\s+\d+\s+(.+)$", str(idx))
    return m.group(1).strip() if m else None


def _norm_plant_name(s):
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def build_reservoir_group(network, glohydrores_path, nord_poly, bus_group, it_buses):
    """Map each Italian reservoir storage unit to 'NORD' / 'RESTO'.

    Assignment is by GloHydroRes plant coordinates when the plant name can be
    matched, otherwise it falls back to the group of the unit's bus.
    """
    if not glohydrores_path.exists():
        raise FileNotFoundError(f"Missing GloHydroRes file: {glohydrores_path}")

    su = network.storage_units
    res = su[
        (su["carrier"] == "hydro") & (su["bus"].isin(it_buses.index))
    ].copy()

    res["plant_name"] = res.index.to_series().map(_plant_name_from_index)

    ghr = pd.read_csv(glohydrores_path)
    ghr_it = ghr[ghr["country"] == "Italy"].copy()

    exact_lookup = ghr_it.set_index("name")[["plant_lat", "plant_lon"]]
    norm_lookup = (
        ghr_it.assign(_n=ghr_it["name"].map(_norm_plant_name))
        .drop_duplicates("_n")
        .set_index("_n")[["plant_lat", "plant_lon"]]
    )

    def match_plant(name):
        if name is None:
            return (None, None, "no_name")
        if name in exact_lookup.index:
            r = exact_lookup.loc[name]
            return (float(r["plant_lat"]), float(r["plant_lon"]), "exact")
        nn = _norm_plant_name(name)
        if nn in norm_lookup.index:
            r = norm_lookup.loc[nn]
            return (float(r["plant_lat"]), float(r["plant_lon"]), "norm")
        return (None, None, "unmatched")

    matched = res["plant_name"].map(match_plant)
    res["lat"] = [m[0] for m in matched]
    res["lon"] = [m[1] for m in matched]
    res["match"] = [m[2] for m in matched]

    def assign(row):
        if pd.notna(row["lat"]) and pd.notna(row["lon"]):
            g = "NORD" if nord_poly.contains(Point(row["lon"], row["lat"])) else "RESTO"
            return pd.Series([g, "coord"])
        return pd.Series([bus_group.get(row["bus"], "RESTO"), "bus_fallback"])

    res[["group", "group_src"]] = res.apply(assign, axis=1)

    logger.info("Reservoir group counts: %s", res["group"].value_counts().to_dict())
    logger.info(
        "Reservoir assignment source: %s",
        res["group_src"].value_counts().to_dict(),
    )

    reservoir_group = res[
        ["bus", "plant_name", "lat", "lon", "match", "group", "group_src"]
    ]

    return reservoir_group


def extract_group_production(net, hydro_type, bus_group, reservoir_group):
    """Hourly production per group (NORD / RESTO) from one network."""
    parts = []

    if hydro_type in ("ror", "total_no_phs"):
        gen = net.generators
        ror = gen[(gen["carrier"] == "ror") & (gen["bus"].isin(bus_group.index))]

        if not ror.empty:
            g = ror["bus"].map(bus_group)
            keep = g.notna()
            ror, g = ror[keep.values], g[keep]

            if not ror.empty:
                p = net.generators_t.p[ror.index].clip(lower=0.0)
                parts.append(p.T.groupby(g).sum().T)

    if hydro_type in ("reservoir", "total_no_phs"):
        su = net.storage_units
        res = su[(su["carrier"] == "hydro") & (su["bus"].isin(bus_group.index))]

        if not res.empty:
            g = res.index.to_series().map(reservoir_group["group"])
            miss = g.isna()

            if miss.any():
                g.loc[miss] = res.loc[miss, "bus"].map(bus_group)

            keep = g.notna()
            res, g = res[keep.values], g[keep]

            if not res.empty:
                p = net.storage_units_t.p[res.index].clip(lower=0.0)
                parts.append(p.T.groupby(g).sum().T)

    if not parts:
        return pd.DataFrame(index=net.snapshots)

    out = pd.concat(parts, axis=1)
    out = out.T.groupby(level=0).sum().T
    out = out.clip(lower=0.0)

    return out


def build_model_group_series(networks_by_year, years, hydro_type, bus_group, reservoir_group):
    """Concatenated NORD/RESTO production series across years for one model."""
    yearly = []

    for year in years:
        df = extract_group_production(
            networks_by_year[year],
            hydro_type,
            bus_group,
            reservoir_group,
        ).copy()

        df.index = pd.to_datetime(df.index)
        yearly.append(df.sort_index())

    full = pd.concat(yearly, axis=0).sort_index()

    for col in ["NORD", "RESTO"]:
        if col not in full.columns:
            full[col] = 0.0

    return full[["NORD", "RESTO"]]


def build_all_model_group_series(model_networks, bus_group, reservoir_group):
    model_groups = {}

    for model_name, networks_by_year in model_networks.items():
        model_groups[model_name] = build_model_group_series(
            networks_by_year=networks_by_year,
            years=NET_YEARS,
            hydro_type=HYDRO_TYPE,
            bus_group=bus_group,
            reservoir_group=reservoir_group,
        )

    return model_groups


def load_italy_north_reference(italy_zone_path, zone=ITALY_NORTH_ZONE, component=ITALY_ZONE_COMPONENT):
    """ENTSO-E hourly reference for one Italian bidding zone, across all years.

    For component 'Total' the series is rebuilt as RoR + Reservoir, excluding
    pumped hydro (PHS), to match the model 'total_no_phs' definition.
    """
    frames = []

    for year in ENTSOE_YEARS:
        fp = italy_zone_path / f"Italy_{zone}_{year}.csv"

        if not fp.exists():
            continue

        df = pd.read_csv(fp, index_col=0)
        df.index = (
            pd.to_datetime(df.index, utc=True)
            .tz_convert(None)
            .floor("h")
        )
        df = df[~df.index.duplicated(keep="first")]

        if component == "Total":
            cols = [c for c in ["RoR", "Reservoir"] if c in df.columns]
            if cols:
                frames.append(df[cols].astype(float).sum(axis=1, min_count=1))
        else:
            if component in df.columns:
                frames.append(df[component].astype(float))

    if not frames:
        return None

    return pd.concat(frames).sort_index()


# ============================================================
# CH / AT / NORTH-ITALY 3-PANEL FIGURE  (figure 3)
# ============================================================

def _country_panel_sources(country_code, hydro_data_nopumped, country_model_frames):
    obs_parts = [
        df[f"{country_code}_Total"]
        for df in hydro_data_nopumped.values()
        if f"{country_code}_Total" in df.columns
    ]
    obs = pd.concat(obs_parts).sort_index() if obs_parts else None

    models = {
        name: frame[country_code]
        for name, frame in country_model_frames.items()
        if country_code in frame.columns
    }

    return obs, models


def _north_italy_panel_sources(italy_north_obs, model_groups):
    models = {name: model_groups[name]["NORD"] for name in model_groups}
    return italy_north_obs, models


def _build_panel_weekly(obs, models):
    if obs is None:
        return None

    cols = {"ENTSOE": obs}
    for name in ACTIVE_MODELS:
        if name in models:
            cols[name] = models[name]

    df = pd.concat(cols, axis=1, join="outer").dropna(subset=["ENTSOE"])

    if df.empty:
        return None

    return df.resample("W").mean()


def plot_country_model_panels(panels, output_path, per_row_height=2.6, fixed_width=14.0):
    """Stacked weekly panels (one per location). Values in GW."""
    n = len(panels)

    fig, axes = plt.subplots(
        n, 1,
        figsize=(fixed_width, per_row_height * n),
        sharex=True,
        facecolor="white",
    )

    if n == 1:
        axes = [axes]

    weekly_all = {
        label: _build_panel_weekly(obs, models)
        for (label, obs, models) in panels
    }

    all_dates = pd.DatetimeIndex([])
    for dfw in weekly_all.values():
        if dfw is not None:
            all_dates = all_dates.union(dfw.index)

    for ax, (label, obs, models) in zip(axes, panels):
        dfw = weekly_all[label]

        if dfw is None:
            ax.text(
                0.5, 0.5, f"{label}: no data",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray",
            )
            ax.set_ylabel("Power (GW)", fontsize=11)
            continue

        ax.plot(
            dfw.index, dfw["ENTSOE"] / 1e3,
            color="black", linewidth=2.5, label="ENTSO-E", zorder=5,
        )

        kgess_parts = []
        for name in ACTIVE_MODELS:
            if name not in dfw.columns:
                continue

            valid = dfw[["ENTSOE", name]].dropna()
            if len(valid) >= 4:
                kg = kgess(valid["ENTSOE"].values, valid[name].values)
                kgess_parts.append(f"{name} = {kg:.2f}")
            else:
                kgess_parts.append(f"{name} = n/a")

            ax.plot(
                dfw.index, dfw[name] / 1e3,
                color=MODEL_COLORS.get(name, "#999999"),
                linewidth=MODEL_LINEWIDTHS.get(name, 1.8),
                alpha=MODEL_ALPHA.get(name, 0.85),
                label=name, zorder=4,
            )

        for yr in sorted(dfw.index.year.unique()):
            ax.axvline(
                pd.Timestamp(f"{yr}-01-01"),
                color="gray", linestyle="--",
                linewidth=0.7, alpha=0.35, zorder=1,
            )

        ax.set_title(
            f"{label}    KGESS → " + "  |  ".join(kgess_parts),
            fontsize=12, fontweight="bold", loc="left", pad=6,
        )
        ax.set_ylabel("Power (GW)", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

        if len(all_dates):
            ax.set_xlim(all_dates.min(), all_dates.max())

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%b"))
    fig.autofmt_xdate(rotation=30, ha="right")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(ACTIVE_MODELS) + 1,
        fontsize=11, frameon=True, framealpha=0.95, edgecolor="black",
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.16, hspace=0.30)

    save_figure(fig, output_path)


# ============================================================
# SUPPLEMENTARY MULTI-COUNTRY PANELS  (figure 5)
# ============================================================

def build_country_weekly_df(country, entsoe_all, model_all):
    entsoe_col = f"{country}_{ENTSOE_SUFFIX}"

    if entsoe_col not in entsoe_all.columns:
        return None

    series_list = [entsoe_all[entsoe_col].rename("ENTSOE")]

    for model_name in ACTIVE_MODELS:
        df_model = model_all[model_name]
        if country in df_model.columns:
            series_list.append(df_model[country].rename(model_name))
        else:
            series_list.append(
                pd.Series(pd.NA, index=entsoe_all.index, name=model_name)
            )

    df = pd.concat(series_list, axis=1, join="outer").dropna(how="all")
    df = df.dropna(subset=["ENTSOE"])

    if df.empty:
        return None

    return df.resample("W").mean()


def plot_supplementary_group(group_name, countries, entsoe_all, model_all, output_path,
                             per_row_height=3.8, fixed_width=14.0):
    n = len(countries)

    fig, axes = plt.subplots(
        n, 1,
        figsize=(fixed_width, per_row_height * n),
        sharex=True,
        facecolor="white",
    )

    if n == 1:
        axes = [axes]

    weekly_dfs = {
        country: build_country_weekly_df(country, entsoe_all, model_all)
        for country in countries
    }

    all_dates = pd.DatetimeIndex([])
    for dfw in weekly_dfs.values():
        if dfw is not None:
            all_dates = all_dates.union(dfw.index)

    for ax, country in zip(axes, countries):
        dfw = weekly_dfs[country]

        if dfw is None:
            ax.text(
                0.5, 0.5, f"{country}: no data",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray",
            )
            ax.set_ylabel("Power (GW)", fontsize=11)
            continue

        ax.plot(
            dfw.index, dfw["ENTSOE"] / 1e3,
            color="black", linewidth=2.5, label="ENTSO-E", zorder=5,
        )

        kgess_parts = []
        for model_name in ACTIVE_MODELS:
            if model_name not in dfw.columns:
                continue

            valid = dfw[["ENTSOE", model_name]].dropna()
            if len(valid) >= 4:
                kg = kgess(valid["ENTSOE"].values, valid[model_name].values)
                kgess_parts.append(f"{model_name} = {kg:.2f}")
            else:
                kgess_parts.append(f"{model_name} = n/a")

            ax.plot(
                dfw.index, dfw[model_name] / 1e3,
                color=MODEL_COLORS.get(model_name, "#999999"),
                linewidth=MODEL_LINEWIDTHS.get(model_name, 1.8),
                alpha=MODEL_ALPHA.get(model_name, 0.85),
                label=model_name, zorder=4,
            )

        for yr in sorted(dfw.index.year.unique()):
            ax.axvline(
                pd.Timestamp(f"{yr}-01-01"),
                color="gray", linestyle="--",
                linewidth=0.7, alpha=0.35, zorder=1,
            )

        country_label = COUNTRY_NAMES.get(country, country)
        ax.set_title(
            f"{country_label}    KGESS → " + "  |  ".join(kgess_parts),
            fontsize=12, fontweight="bold", loc="left", pad=6,
        )
        ax.set_ylabel("Power (GW)", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

        if len(all_dates):
            ax.set_xlim(all_dates.min(), all_dates.max())

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%b"))
    fig.autofmt_xdate(rotation=30, ha="right")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(ACTIVE_MODELS) + 1,
        fontsize=11, frameon=True, framealpha=0.95, edgecolor="black",
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.subplots_adjust(left=0.07, right=0.97, top=0.97, bottom=0.10, hspace=0.18)

    save_figure(fig, output_path)


# ============================================================
# COUNTRY KGESS MAP  (figure 4 — red/blue diverging, version 2)
# ============================================================

def plot_country_kgess_map(results_df_weekly, available_years, hydro_results_path, output_path):
    shapes_year = 2019

    regions_path = (
        hydro_results_path
        / f"PyPSA_{shapes_year}"
        / "country_shapes.geojson"
    )

    if not regions_path.exists():
        raise FileNotFoundError(f"Missing country shapes file: {regions_path}")

    regions = gpd.read_file(regions_path)

    if "name" not in regions.columns:
        raise ValueError(
            f"Column 'name' not found in {regions_path}. "
            f"Available columns: {list(regions.columns)}"
        )

    df = results_df_weekly.copy()
    df = df.drop(index="Europe", errors="ignore").reset_index()

    regions["Country"] = regions["name"]
    gdf = regions.merge(df, on="Country", how="left")

    # Centroids in a projected CRS, converted back for label placement.
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    centroids = gdf.geometry.to_crs(3035).centroid.to_crs(4326)
    gdf["centroid_x"] = centroids.x
    gdf["centroid_y"] = centroids.y

    metric_cols = [f"KGESS_{model_key(m)}" for m in ACTIVE_MODELS]
    subplot_titles = list(ACTIVE_MODELS)

    missing_cols = [col for col in metric_cols if col not in gdf.columns]
    if missing_cols:
        raise ValueError(f"Missing metric columns in results_df_weekly: {missing_cols}")

    nodata_color = "#d9d9d9"
    europe_xlim = (-12, 35)
    europe_ylim = (34, 72)
    label_fontsize = 8

    # Diverging red-white-blue colormap, symmetric around 0.
    rb_cmap = LinearSegmentedColormap.from_list(
        "red_white_blue",
        [
            "#d73027", "#f46d43", "#fdae61", "#ffffff",
            "#9ecae1", "#3182bd", "#08519c",
        ],
    )

    all_vals = gdf[metric_cols].stack().dropna()
    abs_max = max(abs(all_vals.min()), abs(all_vals.max()), 0.01)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    plt.subplots_adjust(wspace=0.05, right=0.88, left=0.02, top=0.93, bottom=0.02)

    for ax, metric_col, title in zip(axes, metric_cols, subplot_titles):
        gdf_nodata = gdf[gdf[metric_col].isna()]
        gdf_valid = gdf[gdf[metric_col].notna()]

        gdf_nodata.plot(
            ax=ax,
            color=nodata_color,
            edgecolor="black",
            linewidth=0.5,
        )

        gdf_valid.plot(
            ax=ax,
            column=metric_col,
            cmap=rb_cmap,
            norm=norm,
            edgecolor="black",
            linewidth=0.5,
            missing_kwds={"color": nodata_color},
        )

        for _, row in gdf.iterrows():
            val = row[metric_col]
            if isinstance(val, float) and np.isnan(val):
                continue
            try:
                label = f"{float(val):.2f}"
                norm_val = norm(float(val))
                text_color = "white" if norm_val > 0.85 or norm_val < 0.15 else "black"
                ax.annotate(
                    label,
                    xy=(row["centroid_x"], row["centroid_y"]),
                    ha="center", va="center",
                    fontsize=label_fontsize,
                    color=text_color,
                    fontweight="bold",
                    clip_on=True,
                )
            except (ValueError, TypeError):
                pass

        ax.set_xlim(*europe_xlim)
        ax.set_ylim(*europe_ylim)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=6)
        ax.axis("off")

    sm = ScalarMappable(norm=norm, cmap=rb_cmap)
    sm.set_array([])

    cax = fig.add_axes([0.90, 0.18, 0.016, 0.64])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("KGESS", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    patch = mpatches.Patch(color=nodata_color, label="No data")
    fig.legend(
        handles=[patch],
        loc="lower right",
        bbox_to_anchor=(0.995, 0.02),
        fontsize=10,
        frameon=True,
        edgecolor="black",
    )

    save_figure(fig, output_path)


# ============================================================
# WORKFLOW
# ============================================================

def run_processing_historical(
    hydro_results_path: Path | None = None,
    entsoe_production_path: Path | None = None,
    electricity_maps_path: Path | None = None,
    gadm_italy_path: Path | None = None,
    glohydrores_path: Path | None = None,
    italy_zone_path: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    hydro_results_path = (
        hydro_results_path.resolve()
        if hydro_results_path is not None
        else get_hydro_results_root()
    )

    data_root = get_data_root()

    entsoe_production_path = (
        entsoe_production_path.resolve()
        if entsoe_production_path is not None
        else data_root / "hydro_global" / "ENTSOE" / "Production"
    )

    electricity_maps_path = (
        electricity_maps_path.resolve()
        if electricity_maps_path is not None
        else data_root / "hydro_global" / "ENTSOE" / "ElectricityMaps"
    )

    gadm_italy_path = (
        gadm_italy_path.resolve()
        if gadm_italy_path is not None
        else data_root / "hydro_global" / "GADM" / "gadm41_ITA_1.json"
    )

    glohydrores_path = (
        glohydrores_path.resolve()
        if glohydrores_path is not None
        else data_root.parent / "pypsa" / "GloHydroRes_vs1.csv"
    )

    italy_zone_path = (
        italy_zone_path.resolve()
        if italy_zone_path is not None
        else data_root / "hydro_global" / "ENTSOE" / "Italia_BiddingZone"
    )

    output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else hydro_results_path / "images" / "historical"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    supplementary_dir = output_dir / "supplementary"
    supplementary_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Hydro results path: %s", hydro_results_path)
    logger.info("ENTSO-E production path: %s", entsoe_production_path)
    logger.info("Electricity Maps path: %s", electricity_maps_path)
    logger.info("GADM Italy path: %s", gadm_italy_path)
    logger.info("GloHydroRes path: %s", glohydrores_path)
    logger.info("Italy bidding-zone production path: %s", italy_zone_path)
    logger.info("Historical output directory: %s", output_dir)

    # --- networks ---
    pypsa_networks = load_networks(
        base_path=hydro_results_path,
        folder_template=NETWORK_FOLDERS["pypsa"],
        years=NET_YEARS,
        network_file=NETWORK_FILE,
    )

    glofas_saber_networks = load_networks(
        base_path=hydro_results_path,
        folder_template=NETWORK_FOLDERS["glofas_saber"],
        years=NET_YEARS,
        network_file=NETWORK_FILE,
    )

    efas_saber_networks = load_networks(
        base_path=hydro_results_path,
        folder_template=NETWORK_FOLDERS["efas_saber"],
        years=NET_YEARS,
        network_file=NETWORK_FILE,
    )

    # --- per-country ENTSO-E and model timeseries ---
    hydro_data_nopumped, _, _ = load_entsoe_hydro_data(
        entsoe_production_path=entsoe_production_path,
        electricity_maps_path=electricity_maps_path,
    )

    pypsa_hydro = extract_hydro_timeseries_from_networks(
        networks=[pypsa_networks[year] for year in NET_YEARS],
        years=list(NET_YEARS),
        hydro_type=HYDRO_TYPE,
    )

    glofas_saber_hydro = extract_hydro_timeseries_from_networks(
        networks=[glofas_saber_networks[year] for year in NET_YEARS],
        years=list(NET_YEARS),
        hydro_type=HYDRO_TYPE,
    )

    efas_saber_hydro = extract_hydro_timeseries_from_networks(
        networks=[efas_saber_networks[year] for year in NET_YEARS],
        years=list(NET_YEARS),
        hydro_type=HYDRO_TYPE,
    )

    model_datasets = {
        "PyPSA baseline": pypsa_hydro,
        "PyPSA + GloFAS-SABER": glofas_saber_hydro,
        "PyPSA + EFAS-SABER": efas_saber_hydro,
    }

    entsoe_all = build_entsoe_all(hydro_data_nopumped)
    model_all = build_model_all(model_datasets)

    # --- weekly country metrics (used by the KGESS map) ---
    results_df_weekly = compute_country_weekly_metrics(
        entsoe_all=entsoe_all,
        model_all=model_all,
    )

    # ---------------------------------------------------------
    # Figure 1 — Europe weekly
    # ---------------------------------------------------------
    df_weekly_eu, kgess_values_eu, available_years = build_europe_weekly_series(
        entsoe_all=entsoe_all,
        model_all=model_all,
    )

    plot_weekly_series(
        df_weekly=df_weekly_eu,
        kgess_values=kgess_values_eu,
        title_prefix="Europe",
        output_path=output_dir / "historical_europe_weekly_hydro.png",
    )

    # ---------------------------------------------------------
    # Figure 2 — aggregated regions (Nordic)
    # ---------------------------------------------------------
    for region_name, requested_countries in SELECTED_REGIONS.items():
        df_weekly_region, kgess_values_region, countries_used = build_region_weekly_series(
            entsoe_all=entsoe_all,
            model_all=model_all,
            requested_countries=requested_countries,
        )

        output_name = (
            "historical_region_"
            + region_name.lower().replace(" ", "_").replace("+", "plus")
            + "_weekly_hydro.png"
        )

        plot_weekly_series(
            df_weekly=df_weekly_region,
            kgess_values=kgess_values_region,
            title_prefix=region_name,
            countries_label=", ".join(countries_used),
            output_path=output_dir / output_name,
        )

    # ---------------------------------------------------------
    # Figure 3 — CH / AT / North-Italy 3-panel
    # ---------------------------------------------------------
    nord_poly = build_nord_polygon(gadm_italy_path)

    bus_group, it_buses = build_bus_group(pypsa_networks[NET_YEARS[0]], nord_poly)

    reservoir_group = build_reservoir_group(
        network=pypsa_networks[NET_YEARS[0]],
        glohydrores_path=glohydrores_path,
        nord_poly=nord_poly,
        bus_group=bus_group,
        it_buses=it_buses,
    )

    model_networks = {
        "PyPSA baseline": pypsa_networks,
        "PyPSA + GloFAS-SABER": glofas_saber_networks,
        "PyPSA + EFAS-SABER": efas_saber_networks,
    }

    model_groups = build_all_model_group_series(
        model_networks=model_networks,
        bus_group=bus_group,
        reservoir_group=reservoir_group,
    )

    italy_north_obs = load_italy_north_reference(italy_zone_path)
    if italy_north_obs is None:
        logger.warning(
            "No ENTSO-E bidding-zone data found in %s; North-Italy panel will be empty.",
            italy_zone_path,
        )

    country_model_frames = {
        "PyPSA baseline": pypsa_hydro,
        "PyPSA + GloFAS-SABER": glofas_saber_hydro,
        "PyPSA + EFAS-SABER": efas_saber_hydro,
    }

    ch_obs, ch_models = _country_panel_sources(
        "CH", hydro_data_nopumped, country_model_frames
    )
    at_obs, at_models = _country_panel_sources(
        "AT", hydro_data_nopumped, country_model_frames
    )
    it_north_obs, it_north_models = _north_italy_panel_sources(
        italy_north_obs, model_groups
    )

    panels = [
        ("Switzerland", ch_obs, ch_models),
        ("Austria", at_obs, at_models),
        ("North Italy", it_north_obs, it_north_models),
    ]

    plot_country_model_panels(
        panels=panels,
        output_path=output_dir / "historical_ch_at_northitaly_weekly_hydro.png",
    )

    # ---------------------------------------------------------
    # Figure 4 — KGESS country map (red/blue diverging)
    # ---------------------------------------------------------
    plot_country_kgess_map(
        results_df_weekly=results_df_weekly,
        available_years=available_years,
        hydro_results_path=hydro_results_path,
        output_path=output_dir / "historical_kgess_country_map.png",
    )

    # ---------------------------------------------------------
    # Figure 5 — supplementary multi-country panels
    # ---------------------------------------------------------
    for group_name, countries in SUPPLEMENTARY_PANEL_GROUPS.items():
        plot_supplementary_group(
            group_name=group_name,
            countries=countries,
            entsoe_all=entsoe_all,
            model_all=model_all,
            output_path=supplementary_dir / f"weekly_comparison_{group_name}.png",
        )

    logger.info("Historical figures saved to: %s", output_dir)


def main() -> None:
    setup_logging()

    args = parse_args()

    run_processing_historical(
        hydro_results_path=args.hydro_results,
        entsoe_production_path=args.entsoe_production,
        electricity_maps_path=args.electricity_maps,
        gadm_italy_path=args.gadm_italy,
        glohydrores_path=args.glohydrores,
        italy_zone_path=args.italy_zone_production,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()