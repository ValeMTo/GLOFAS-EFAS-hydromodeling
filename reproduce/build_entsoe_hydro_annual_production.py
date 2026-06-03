from __future__ import annotations

from pathlib import Path
import json
import logging
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.hydro_inflow.hydro_config import get_config


logger = logging.getLogger(__name__)

COMPONENTS = ["Pumped", "RoR", "Reservoir"]

# A component is considered usable only if it has enough valid data.
# This avoids sparse components destroying Reservoir totals.
MIN_COMPONENT_COVERAGE = 0.50
MIN_COMPONENT_VALID_HOURS = 24 * 30

# Used later for weekly/monthly resampling.
MIN_PERIOD_COVERAGE = 0.90

CH_YEARS_WITHOUT_ELECTRICITY_MAPS = [2015, 2016]


def read_hourly_hydro_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)

    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    df.index = df.index.floor("h")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    return df


def load_ch_hydro_hourly_mw_from_electricity_maps(
    json_path: Path,
    year: int,
) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

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


def get_countries_from_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        {
            col.rsplit("_", 1)[0]
            for col in df.columns
            if "_" in col
        }
    )


def remove_country_columns(df: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    columns_to_drop = [
        col
        for col in df.columns
        if any(col.startswith(f"{country}_") for country in countries)
    ]

    return df.drop(columns=columns_to_drop, errors="ignore")


def resample_mean_with_min_coverage(
    df: pd.DataFrame,
    freq: str,
    min_coverage: float = MIN_PERIOD_COVERAGE,
) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    period_mean = df.resample(freq).mean()
    valid_counts = df.resample(freq).count()
    expected_counts = df.resample(freq).size()

    coverage = valid_counts.div(expected_counts, axis=0)

    return period_mean.where(coverage >= min_coverage)


def component_is_usable(
    series: pd.Series,
    min_coverage: float = MIN_COMPONENT_COVERAGE,
    min_valid_hours: int = MIN_COMPONENT_VALID_HOURS,
) -> bool:
    valid_hours = int(series.notna().sum())
    total_hours = len(series)
    coverage = valid_hours / total_hours if total_hours > 0 else 0.0

    return valid_hours >= min_valid_hours and coverage >= min_coverage


def build_nopumped_dataset(
    df_base: pd.DataFrame,
    countries: list[str],
    year: int,
    ch_electricity_maps_years: set[int],
    min_component_coverage: float = MIN_COMPONENT_COVERAGE,
    min_component_valid_hours: int = MIN_COMPONENT_VALID_HOURS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build no-pumped ENTSO-E reference.

    Rules:
    - CH 2017-2019 uses Electricity Maps CH_Total.
    - For other countries, use only hydro components with sufficient coverage.
    - If only RoR is usable, Total = RoR.
    - If only Reservoir is usable, Total = Reservoir.
    - If both are usable, Total = RoR + Reservoir.
    - Sparse components with only isolated values are ignored.
    """

    output = pd.DataFrame(index=df_base.index)
    component_report_rows = []

    for country in countries:
        total_col = f"{country}_Total"
        ror_col = f"{country}_RoR"
        reservoir_col = f"{country}_Reservoir"

        if country == "CH" and year in ch_electricity_maps_years:
            if total_col in df_base.columns:
                output[total_col] = df_base[total_col]

            output[ror_col] = np.nan
            output[reservoir_col] = np.nan

            component_report_rows.append(
                {
                    "year": year,
                    "country": country,
                    "ror_valid_hours": 0,
                    "reservoir_valid_hours": 0,
                    "ror_coverage": 0.0,
                    "reservoir_coverage": 0.0,
                    "usable_ror": False,
                    "usable_reservoir": False,
                    "rule": "CH Electricity Maps Total",
                }
            )

            continue

        has_ror_col = ror_col in df_base.columns
        has_reservoir_col = reservoir_col in df_base.columns

        ror_valid_hours = int(df_base[ror_col].notna().sum()) if has_ror_col else 0
        reservoir_valid_hours = (
            int(df_base[reservoir_col].notna().sum())
            if has_reservoir_col
            else 0
        )

        ror_coverage = df_base[ror_col].notna().mean() if has_ror_col else 0.0
        reservoir_coverage = (
            df_base[reservoir_col].notna().mean()
            if has_reservoir_col
            else 0.0
        )

        usable_ror = (
            has_ror_col
            and component_is_usable(
                df_base[ror_col],
                min_coverage=min_component_coverage,
                min_valid_hours=min_component_valid_hours,
            )
        )

        usable_reservoir = (
            has_reservoir_col
            and component_is_usable(
                df_base[reservoir_col],
                min_coverage=min_component_coverage,
                min_valid_hours=min_component_valid_hours,
            )
        )

        if usable_ror:
            output[ror_col] = df_base[ror_col]

        if usable_reservoir:
            output[reservoir_col] = df_base[reservoir_col]

        if usable_ror and usable_reservoir:
            output[total_col] = df_base[ror_col] + df_base[reservoir_col]
            rule = "RoR + Reservoir"
        elif usable_ror:
            output[total_col] = df_base[ror_col]
            rule = "RoR only"
        elif usable_reservoir:
            output[total_col] = df_base[reservoir_col]
            rule = "Reservoir only"
        else:
            rule = "No usable component"

        component_report_rows.append(
            {
                "year": year,
                "country": country,
                "ror_valid_hours": ror_valid_hours,
                "reservoir_valid_hours": reservoir_valid_hours,
                "ror_coverage": ror_coverage,
                "reservoir_coverage": reservoir_coverage,
                "usable_ror": usable_ror,
                "usable_reservoir": usable_reservoir,
                "rule": rule,
            }
        )

    ordered_cols = [
        f"{country}_{component}"
        for country in countries
        for component in ["RoR", "Reservoir", "Total"]
        if f"{country}_{component}" in output.columns
    ]

    component_report = pd.DataFrame(component_report_rows)

    return output[ordered_cols].copy(), component_report


def build_annual_totals_from_hourly(
    entsoe_hourly_dir: Path,
    electricity_maps_ch_files: dict[int, Path],
    years: list[int],
    output_fn: Path,
    ch_years_without_electricity_maps: list[int],
    min_component_coverage: float = MIN_COMPONENT_COVERAGE,
    min_component_valid_hours: int = MIN_COMPONENT_VALID_HOURS,
) -> None:
    rows = []
    component_reports = []

    ch_electricity_maps_years = set(electricity_maps_ch_files)

    for year in years:
        hourly_fn = entsoe_hourly_dir / f"Europe_Hydro_{year}.csv"

        if not hourly_fn.exists():
            raise FileNotFoundError(f"Missing ENTSO-E hourly file: {hourly_fn}")

        df = read_hourly_hydro_csv(hourly_fn)

        if year in electricity_maps_ch_files:
            ch_path = electricity_maps_ch_files[year]

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

            logger.info(
                "%s: CH replaced with Electricity Maps. CH_Total non-null hours: %s",
                year,
                int(df["CH_Total"].notna().sum()),
            )

        elif year in ch_years_without_electricity_maps:
            df = remove_country_columns(df, ["CH"])
            logger.info(
                "%s: CH removed because Electricity Maps replacement is unavailable.",
                year,
            )

        countries = get_countries_from_columns(df)

        hydro_nopumped, component_report = build_nopumped_dataset(
            df_base=df,
            countries=countries,
            year=year,
            ch_electricity_maps_years=ch_electricity_maps_years,
            min_component_coverage=min_component_coverage,
            min_component_valid_hours=min_component_valid_hours,
        )

        component_reports.append(component_report)

        expected_hours = len(
            pd.date_range(
                start=f"{year}-01-01 00:00:00",
                end=f"{year}-12-31 23:00:00",
                freq="h",
            )
        )

        for country in countries:
            total_col = f"{country}_Total"

            if total_col not in hydro_nopumped.columns:
                logger.warning(
                    "Skipping %s %s because no no-pumped total was generated.",
                    country,
                    year,
                )
                continue

            series = hydro_nopumped[total_col].astype(float)
            valid_hours = int(series.notna().sum())
            coverage = valid_hours / expected_hours if expected_hours > 0 else np.nan

            hydro_generation_mwh = float(series.sum())

            if country == "CH" and year in ch_electricity_maps_years:
                source = "ELECTRICITY_MAPS_CH"
            else:
                source = "ENTSOE_NOPUMPED"

            rows.append(
                {
                    "year": year,
                    "country": country,
                    "hydro_generation_mwh": hydro_generation_mwh,
                    "hydro_generation_twh": hydro_generation_mwh / 1e6,
                    "valid_hours": valid_hours,
                    "expected_hours": expected_hours,
                    "coverage": coverage,
                    "source": source,
                }
            )

        logger.info(
            "%s processed. Countries in raw data: %s. Countries in annual output so far: %s",
            year,
            len(countries),
            len(rows),
        )

    annual = pd.DataFrame(rows)

    if annual.empty:
        raise RuntimeError("No annual hydro production totals were generated.")

    annual = annual.sort_values(["year", "country"])

    output_fn.parent.mkdir(parents=True, exist_ok=True)
    annual.to_csv(output_fn, index=False)

    logger.info("Saved annual hydro production totals to %s", output_fn)

    if component_reports:
        component_report_df = pd.concat(component_reports, ignore_index=True)
        report_fn = output_fn.with_name("hydro_annual_production_component_report.csv")
        component_report_df = component_report_df.sort_values(["year", "country"])
        component_report_df.to_csv(report_fn, index=False)
        logger.info("Saved component selection report to %s", report_fn)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )

    cfg = get_config()

    years = [
        int(year)
        for year in cfg.get(
            "entsoe_hydro_years",
            [2015, 2016, 2017, 2018, 2019],
        )
    ]

    electricity_maps_ch_files = {
        int(year): Path(path)
        for year, path in cfg.get("electricity_maps_ch_files", {}).items()
    }

    build_annual_totals_from_hourly(
        entsoe_hourly_dir=Path(cfg["entsoe_hydro_hourly_dir"]),
        electricity_maps_ch_files=electricity_maps_ch_files,
        years=years,
        output_fn=Path(cfg["entsoe_hydro_annual_production_path"]),
        ch_years_without_electricity_maps=list(
            cfg.get(
                "entsoe_ch_years_without_electricity_maps",
                CH_YEARS_WITHOUT_ELECTRICITY_MAPS,
            )
        ),
        min_component_coverage=float(
            cfg.get("entsoe_min_component_coverage", MIN_COMPONENT_COVERAGE)
        ),
        min_component_valid_hours=int(
            cfg.get("entsoe_min_component_valid_hours", MIN_COMPONENT_VALID_HOURS)
        ),
    )


if __name__ == "__main__":
    main()