from __future__ import annotations

from pathlib import Path
import argparse
import logging
import os

import numpy as np
import pandas as pd

from logging_utils import setup_logging


logger = logging.getLogger(__name__)


PYPSA_EUR_COUNTRIES = [
    "AL", "AT", "BA", "BE", "BG", "CH", "CZ", "DE", "DK", "EE",
    "ES", "FI", "FR", "GB", "GR", "HR", "HU", "IE", "IT", "LT",
    "LU", "LV", "ME", "MK", "NL", "NO", "PL", "PT", "RO", "RS",
    "SE", "SI", "SK", "XK",
]


PPL_COLUMNS = [
    "Name",
    "Fueltype",
    "Technology",
    "Set",
    "Country",
    "Capacity",
    "Efficiency",
    "DateIn",
    "DateRetrofit",
    "DateOut",
    "lat",
    "lon",
    "Duration",
    "Volume_Mm3",
    "DamHeight_m",
    "StorageCapacity_MWh",
    "EIC",
    "projectID",
    "bus",
    "source_dataset",
    "source_id",
]


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_data_root() -> Path:
    return Path(
        os.environ.get(
            "HYDRO_DATA_ROOT",
            get_repo_root() / "data" / "hydro_workflow",
        )
    )


def load_and_prepare_glohydrores(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    for col in ["name", "country", "plant_type"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    required = [
        "name",
        "country",
        "plant_type",
        "capacity_mw",
        "year",
        "plant_lat",
        "plant_lon",
        "res_vol_km3",
        "head_m",
        "plant_source_id",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required GloHydroRes columns: {missing}")

    out = pd.DataFrame(index=df.index)

    technology_mapping = {
        "ROR": "Run-Of-River",
        "STO": "Reservoir",
        "PS": "Pumped Storage",
        "Canal": "Run-Of-River",
    }

    out["Name"] = df["name"]
    out["Fueltype"] = "Hydro"
    out["Technology"] = df["plant_type"].map(technology_mapping)
    out["Technology"] = (
        out["Technology"]
        .replace({"": np.nan, "nan": np.nan})
        .fillna("Run-Of-River")
    )

    out["Set"] = np.where(df["plant_type"] == "PS", "Store", "PP")
    out["Country"] = df["country"]

    out["Capacity"] = pd.to_numeric(df["capacity_mw"], errors="coerce")
    out["Efficiency"] = np.nan
    out["DateIn"] = pd.to_numeric(df["year"], errors="coerce")
    out["DateRetrofit"] = pd.to_numeric(df["year"], errors="coerce")
    out["DateOut"] = pd.to_numeric(df["year"], errors="coerce") + 150

    out["lat"] = pd.to_numeric(df["plant_lat"], errors="coerce")
    out["lon"] = pd.to_numeric(df["plant_lon"], errors="coerce")

    out["Duration"] = np.nan
    out["Volume_Mm3"] = pd.to_numeric(df["res_vol_km3"], errors="coerce") * 1000.0
    out["DamHeight_m"] = pd.to_numeric(df["head_m"], errors="coerce")
    out["StorageCapacity_MWh"] = np.nan

    out["EIC"] = ""
    out["projectID"] = df["plant_source_id"]
    out["bus"] = ""

    out["source_dataset"] = "glohydrores"
    out["source_id"] = (df.index + 1).astype(str)

    for col in PPL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    out = out.reindex(columns=PPL_COLUMNS)

    return out


def convert_glohydrores_country_to_iso2(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Country"] = out["Country"].astype(str).str.strip()

    country_to_code = {
        "Albania": "AL",
        "Austria": "AT",
        "Bosnia and Herzegovina": "BA",
        "Belgium": "BE",
        "Bulgaria": "BG",
        "Switzerland": "CH",
        "Czech Republic": "CZ",
        "Germany": "DE",
        "Denmark": "DK",
        "Estonia": "EE",
        "Spain": "ES",
        "Finland": "FI",
        "France": "FR",
        "United Kingdom": "GB",
        "Greece": "GR",
        "Croatia": "HR",
        "Hungary": "HU",
        "Ireland": "IE",
        "Italy": "IT",
        "Lithuania": "LT",
        "Luxembourg": "LU",
        "Latvia": "LV",
        "Montenegro": "ME",
        "North Macedonia": "MK",
        "Netherlands": "NL",
        "Norway": "NO",
        "Poland": "PL",
        "Portugal": "PT",
        "Romania": "RO",
        "Serbia": "RS",
        "Sweden": "SE",
        "Slovenia": "SI",
        "Slovakia": "SK",
        "Kosovo": "XK",
    }

    out["Country"] = out["Country"].map(country_to_code)

    missing_after = out["Country"].isna().sum()
    logger.info("Plants with missing ISO-2 country after conversion: %s", missing_after)

    return out


def apply_manual_glohydrores_corrections(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    name = out["Name"].astype(str).str.lower()

    dossi_mask = (
        (out["Country"] == "IT")
        & name.str.contains("dossi", na=False)
    )

    out.loc[dossi_mask, "DamHeight_m"] = 1000.0

    remove_by_name_mask = (
        (out["Country"] == "CH")
        & (
            name.str.contains("innertkirchen 2", na=False)
            | name.str.contains("albignawerk löbbia", na=False)
        )
    )

    remove_tierfehd_duplicate_mask = (
        (out["Country"] == "CH")
        & out["source_id"].astype(str).eq("6121")
    )

    remove_mask = remove_by_name_mask | remove_tierfehd_duplicate_mask

    logger.info("Manual GloHydroRes corrections.")
    logger.info("Dossi DamHeight_m corrected: %s", int(dossi_mask.sum()))
    logger.info("Removed plants: %s", int(remove_mask.sum()))

    if remove_mask.any():
        logger.debug(
            "Removed plants table:\n%s",
            out.loc[
                remove_mask,
                ["Name", "Country", "Capacity", "Technology", "source_id"],
            ].to_string(index=False),
        )

    return out.loc[~remove_mask].copy()

def filter_to_pypsa_countries(df: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["Country"] = out["Country"].astype(str).str.strip().str.upper()
    countries = [country.upper() for country in countries]

    out = out[out["Country"].isin(countries)].copy()

    logger.info("Total plants before filtering: %s", len(df))
    logger.info("Total plants after filtering: %s", len(out))

    return out


def fill_hydro_nan_parameters(
    glohydro_df: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    out = glohydro_df.copy()

    required_cols = ["Country", "Technology"]

    missing_required = [
        col for col in required_cols
        if col not in out.columns
    ]

    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    out["Country"] = out["Country"].astype(str).str.strip()
    out["Technology"] = out["Technology"].astype(str).str.strip()

    for col in ["DamHeight_m", "Volume_Mm3", "Capacity", "lat", "lon"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    def _fill_column_for_subset(
        df: pd.DataFrame,
        subset_mask: pd.Series,
        target_col: str,
    ) -> pd.DataFrame:
        work = df.copy()

        if target_col not in work.columns:
            logger.debug("Column not found, skipping: %s", target_col)
            return work

        eligible = work.loc[subset_mask].copy()

        if eligible.empty:
            logger.debug("No eligible rows for column: %s", target_col)
            return work

        country_tech_median = (
            eligible.groupby(["Country", "Technology"])[target_col]
            .median()
        )

        tech_median = (
            eligible.groupby("Technology")[target_col]
            .median()
        )

        global_median = eligible[target_col].median()

        missing_mask = subset_mask & work[target_col].isna()

        for idx in work.index[missing_mask]:
            country = work.at[idx, "Country"]
            tech = work.at[idx, "Technology"]

            value = np.nan

            if (country, tech) in country_tech_median.index:
                value = country_tech_median.loc[(country, tech)]

            if pd.isna(value) and tech in tech_median.index:
                value = tech_median.loc[tech]

            if pd.isna(value):
                value = global_median

            work.at[idx, target_col] = value

        return work

    ror_mask = out["Technology"] == "Run-Of-River"

    if verbose:
        logger.debug("--- ROR ---")
        if "DamHeight_m" in out.columns:
            logger.debug(
                "Missing DamHeight before: %s",
                out.loc[ror_mask, "DamHeight_m"].isna().sum(),
            )

    out = _fill_column_for_subset(
        df=out,
        subset_mask=ror_mask,
        target_col="DamHeight_m",
    )

    if verbose:
        if "DamHeight_m" in out.columns:
            logger.debug(
                "Missing DamHeight after: %s",
                out.loc[ror_mask, "DamHeight_m"].isna().sum(),
            )

    storage_mask = out["Technology"].isin(["Reservoir", "Pumped Storage"])

    if verbose:
        logger.debug("--- STORAGE ---")
        if "DamHeight_m" in out.columns:
            logger.debug(
                "Missing DamHeight before: %s",
                out.loc[storage_mask, "DamHeight_m"].isna().sum(),
            )
        if "Volume_Mm3" in out.columns:
            logger.debug(
                "Missing Volume before: %s",
                out.loc[storage_mask, "Volume_Mm3"].isna().sum(),
            )

    out = _fill_column_for_subset(
        df=out,
        subset_mask=storage_mask,
        target_col="DamHeight_m",
    )

    out = _fill_column_for_subset(
        df=out,
        subset_mask=storage_mask,
        target_col="Volume_Mm3",
    )

    if verbose:
        if "DamHeight_m" in out.columns:
            logger.debug(
                "Missing DamHeight after: %s",
                out.loc[storage_mask, "DamHeight_m"].isna().sum(),
            )
        if "Volume_Mm3" in out.columns:
            logger.debug(
                "Missing Volume after: %s",
                out.loc[storage_mask, "Volume_Mm3"].isna().sum(),
            )

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the GloHydroRes hydropower plant table for the hydro workflow."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input GloHydroRes CSV. Default: data/hydro/GloHydroRes_vs1.csv.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output prepared CSV. Default: "
            "HYDRO_DATA_ROOT/hydro_global/Eur_custom_ppls_GloHydroRes_filled.csv."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed filling diagnostics at DEBUG log level.",
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()

    args = parse_args()

    repo_root = get_repo_root()
    data_root = get_data_root()

    input_path = args.input or repo_root / "data" / "hydro" / "GloHydroRes_vs1.csv"
    output_path = (
        args.output
        or data_root / "hydro_global" / "Eur_custom_ppls_GloHydroRes_filled.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Input GloHydroRes file not found: {input_path}")

    logger.info("Reading GloHydroRes source: %s", input_path)

    df = load_and_prepare_glohydrores(input_path=input_path)
    logger.info("Loaded and converted plants: %s", len(df))

    df = convert_glohydrores_country_to_iso2(df)
    df = filter_to_pypsa_countries(df, countries=PYPSA_EUR_COUNTRIES)
    df = apply_manual_glohydrores_corrections(df)

    df = fill_hydro_nan_parameters(
        glohydro_df=df,
        verbose=args.verbose,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df.index = df["source_id"].astype(str)
    df.index.name = "plant_index"
    df.to_csv(output_path, index=True)

    logger.info("Prepared hydropower plant file saved: %s", output_path)
    logger.info("Rows saved: %s", len(df))
    logger.info("Columns saved: %s", len(df.columns))


if __name__ == "__main__":
    main()