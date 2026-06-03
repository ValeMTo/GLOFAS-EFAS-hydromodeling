#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import logging
import warnings

import matplotlib
matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import os
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize


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

SELECTED_REGIONS = {
    "Nordic countries": ["FI", "SE", "NO"],
    "Alpine countries": ["IT", "AT", "CH"],
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


DEFAULT_HYDRO_RESULTS_PATH = get_hydro_results_root()

DEFAULT_ENTSOE_PRODUCTION_PATH = (
    get_data_root()
    / "hydro_global"
    / "ENTSOE"
    / "Production"
)

DEFAULT_ELECTRICITY_MAPS_PATH = (
    get_data_root()
    / "hydro_global"
    / "ENTSOE"
    / "ElectricityMaps"
)

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
        default=DEFAULT_HYDRO_RESULTS_PATH,
        help="Path to hydro_results directory.",
    )

    parser.add_argument(
        "--entsoe-production",
        type=Path,
        default=DEFAULT_ENTSOE_PRODUCTION_PATH,
        help="Path to ENTSO-E hydro production CSV files.",
    )

    parser.add_argument(
        "--electricity-maps",
        type=Path,
        default=DEFAULT_ELECTRICITY_MAPS_PATH,
        help="Path to Electricity Maps JSON files.",
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


def clean_metric_name(model_name):
    return model_name.replace("-", "_").replace(" ", "_")


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
# ENTSO-E LOADING AND PROCESSING
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
# MODEL HYDRO EXTRACTION
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

            metric_name = clean_metric_name(model_name)

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

    sort_column = f"KGESS_{clean_metric_name(ACTIVE_MODELS[0])}"

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
# REGIONAL WEEKLY FIGURES
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
# COUNTRY KGESS MAP
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
            f"Column 'name' not found in {regions_path}. Available columns: {list(regions.columns)}"
        )

    df = results_df_weekly.copy()
    df = df.drop(index="Europe", errors="ignore").reset_index()

    regions["Country"] = regions["name"]
    gdf = regions.merge(df, on="Country", how="left")

    metric_cols = [
        "KGESS_PyPSA_baseline",
        "KGESS_PyPSA_+_GloFAS_SABER",
        "KGESS_PyPSA_+_EFAS_SABER",
    ]

    titles = [
        f"PyPSA Baseline by Country ({min(available_years)}-{max(available_years)})",
        f"PyPSA + GloFAS-SABER by Country ({min(available_years)}-{max(available_years)})",
        f"PyPSA + EFAS-SABER by Country ({min(available_years)}-{max(available_years)})",
    ]

    blue_cmap = LinearSegmentedColormap.from_list(
        "white_to_blue",
        [
            "#ffffff",
            "#deebf7",
            "#9ecae1",
            "#3182bd",
            "#08519c",
        ],
    )

    vmin = 0.0
    vmax = 1.0
    nodata_color = "#bdbdbd"

    missing_cols = [col for col in metric_cols if col not in gdf.columns]

    if missing_cols:
        raise ValueError(f"Missing metric columns in results_df_weekly: {missing_cols}")

    for metric_col in metric_cols:
        gdf[f"{metric_col}_plot"] = gdf[metric_col].copy()

        gdf.loc[gdf[metric_col].notna(), f"{metric_col}_plot"] = (
            gdf.loc[gdf[metric_col].notna(), metric_col]
            .clip(lower=vmin, upper=vmax)
        )

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    plt.subplots_adjust(wspace=0.08, right=0.88)

    for ax, metric_col, title in zip(axes, metric_cols, titles):
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
            column=f"{metric_col}_plot",
            cmap=blue_cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.axis("off")

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=blue_cmap)
    sm.set_array([])

    cax = fig.add_axes([0.90, 0.18, 0.018, 0.64])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("KGESS", fontsize=14)
    cbar.ax.tick_params(labelsize=13)

    save_figure(fig, output_path)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    hydro_results_path = args.hydro_results
    output_dir = args.output_dir or hydro_results_path / "images" / "historical"
    output_dir.mkdir(parents=True, exist_ok=True)

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

    hydro_data_nopumped, _, _ = load_entsoe_hydro_data(
        entsoe_production_path=args.entsoe_production,
        electricity_maps_path=args.electricity_maps,
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

    results_df_weekly = compute_country_weekly_metrics(
        entsoe_all=entsoe_all,
        model_all=model_all,
    )

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

    plot_country_kgess_map(
        results_df_weekly=results_df_weekly,
        available_years=available_years,
        hydro_results_path=hydro_results_path,
        output_path=output_dir / "historical_kgess_country_map.png",
    )

    print(f"Historical figures saved to: {output_dir}")


if __name__ == "__main__":
    main()