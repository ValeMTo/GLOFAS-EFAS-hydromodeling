#!/usr/bin/env python3
# Build PyPSA-ready hydropower inflow time series from SABER-corrected outputs.
# Corrected SABER series are used where available; all remaining plants keep their
# original raw GLoFAS/EFAS series from hindcast_plants.zarr.

from __future__ import annotations

from pathlib import Path
import logging
import pickle

import numpy as np
import pandas as pd
import xarray as xr

from hydro_config import get_config, log_config_summary
from logging_utils import setup_logging


logger = logging.getLogger(__name__)

CFG = get_config()

# ============================================================
# CONFIGURATION
# ============================================================

CORRECTED_DIR = CFG["corrected_dir"]

RIVID_MAP_PATH = CFG["rivid_map_output_path"]
HINDCAST_PLANTS_ZARR = CFG["zarr_output_path"]
OUTPUT_PICKLE_PATH = CFG["pypsa_inflow_pickle_path"]
OUTPUT_NETCDF_DIR = CFG["pypsa_inflow_netcdf_dir"]
OUTPUT_NETCDF_TEMPLATE = CFG["pypsa_inflow_netcdf_template"]
REPORT_PATH = CFG["pypsa_inflow_report_path"]

YEARS = CFG["pypsa_inflow_years"]

DO_SAVE_PICKLE = True
DO_SAVE_NETCDF = True

RAW_VARIABLE_NAME = "Qsim"
CORRECTED_VARIABLE_NAME = "Qmod"

REMOVE_LEAP_DAY = True
FILL_NA_WITH_ZERO = True


# ============================================================
# HELPERS
# ============================================================

def normalize_string_id(value) -> str:
    if pd.isna(value):
        return ""

    value = str(value).strip()

    if value.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""

    return value


def remove_leap_day_from_series(s: pd.Series) -> pd.Series:
    if not REMOVE_LEAP_DAY:
        return s

    return s[~((s.index.month == 2) & (s.index.day == 29))]


def read_rivid_map(path: Path) -> pd.DataFrame:
    rivid_map = pd.read_csv(path, dtype=str)

    required_cols = {"plant", "rivid"}
    missing = required_cols - set(rivid_map.columns)

    if missing:
        raise ValueError(f"Missing required columns in rivid map: {sorted(missing)}")

    rivid_map = rivid_map.copy()
    rivid_map["plant"] = rivid_map["plant"].map(normalize_string_id)
    rivid_map["rivid"] = pd.to_numeric(rivid_map["rivid"], errors="coerce")

    if "model_id" not in rivid_map.columns:
        rivid_map["model_id"] = ""

    rivid_map["model_id"] = rivid_map["model_id"].map(normalize_string_id)

    rivid_map = rivid_map[
        (rivid_map["plant"] != "")
        & rivid_map["rivid"].notna()
    ].copy()

    rivid_map["rivid"] = rivid_map["rivid"].astype(int)

    duplicated_plants = rivid_map["plant"].duplicated(keep=False).sum()
    if duplicated_plants > 0:
        logger.warning(
            "Duplicated plant rows in rivid_map: %s. Keeping first occurrence.",
            duplicated_plants,
        )
        rivid_map = rivid_map.drop_duplicates(subset=["plant"], keep="first").copy()

    logger.debug("Rivid map loaded: %s", path)
    logger.info("Plants: %s", rivid_map["plant"].nunique())
    logger.debug("Rows: %s", len(rivid_map))
    logger.info("Plants with model_id: %s", int((rivid_map["model_id"] != "").sum()))

    return rivid_map


def build_model_to_plants(rivid_map: pd.DataFrame) -> dict[str, list[str]]:
    valid = rivid_map[
        (rivid_map["model_id"] != "")
        & (rivid_map["plant"] != "")
    ].copy()

    model_to_plants = (
        valid
        .groupby("model_id")["plant"]
        .apply(lambda x: list(pd.unique(x)))
        .to_dict()
    )

    logger.debug("Model-to-plant map built.")
    logger.debug("N model_id: %s", len(model_to_plants))
    logger.debug("N linked plants: %s", sum(len(v) for v in model_to_plants.values()))

    return model_to_plants


def load_raw_inflows_by_plant(
    rivid_map: pd.DataFrame,
    hindcast_plants_zarr: Path,
) -> dict[str, pd.Series]:
    ds = xr.open_zarr(hindcast_plants_zarr)

    if RAW_VARIABLE_NAME not in ds.data_vars:
        raise ValueError(
            f"Variable '{RAW_VARIABLE_NAME}' not found in {hindcast_plants_zarr}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    if "time" not in ds.coords:
        raise ValueError("hindcast_plants.zarr must contain a 'time' coordinate.")

    if "rivid" not in ds.coords:
        raise ValueError("hindcast_plants.zarr must contain a 'rivid' coordinate.")

    logger.info("Using raw discharge variable from zarr: %s", RAW_VARIABLE_NAME)

    time_index = pd.to_datetime(ds["time"].values)
    available_rivids = set(pd.Series(ds["rivid"].values).astype(int).tolist())

    raw_inflows = {}
    missing_rivids = []

    for _, row in rivid_map.iterrows():
        plant = normalize_string_id(row["plant"])
        rivid = int(row["rivid"])

        if plant == "":
            continue

        if rivid not in available_rivids:
            missing_rivids.append({"plant": plant, "rivid": rivid})
            continue

        s = pd.Series(
            ds[RAW_VARIABLE_NAME].sel(rivid=rivid).values,
            index=time_index,
            name=plant,
        ).astype(float)

        s = remove_leap_day_from_series(s)
        raw_inflows[plant] = s

    ds.close()

    logger.info("Raw inflows loaded.")
    logger.debug("Raw plants recovered: %s", len(raw_inflows))
    logger.info("Missing rivids: %s", len(missing_rivids))

    if missing_rivids:
        logger.debug(
            "First missing rivid rows:\n%s",
            pd.DataFrame(missing_rivids).head(20).to_string(index=False),
        )

    return raw_inflows


def load_corrected_inflows_by_plant(
    corrected_dir: Path,
    model_to_plants: dict[str, list[str]],
) -> dict[str, pd.Series]:
    corrected_inflows = {}

    if not corrected_dir.exists():
        logger.warning("Corrected directory not found: %s", corrected_dir)
        return corrected_inflows

    files = sorted(corrected_dir.glob("*.csv"))

    logger.debug("Corrected SABER files found: %s", len(files))

    used_files = 0
    skipped_no_model = 0
    skipped_missing_qmod = 0

    for path in files:
        model_id = path.stem

        if model_id not in model_to_plants:
            skipped_no_model += 1
            continue

        df = pd.read_csv(path, index_col=0, parse_dates=True)

        if CORRECTED_VARIABLE_NAME not in df.columns:
            skipped_missing_qmod += 1
            continue

        s = df[CORRECTED_VARIABLE_NAME].astype(float)
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        s = remove_leap_day_from_series(s)

        for plant in model_to_plants[model_id]:
            corrected_inflows[str(plant)] = s.copy()

        used_files += 1

    logger.info("Corrected SABER inflows loaded.")
    logger.debug("Used corrected files: %s", used_files)
    logger.debug("Corrected plants: %s", len(corrected_inflows))
    logger.debug("Skipped files without model match: %s", skipped_no_model)
    logger.debug("Skipped files without Qmod: %s", skipped_missing_qmod)

    return corrected_inflows


def merge_raw_and_corrected_inflows(
    raw_inflows: dict[str, pd.Series],
    corrected_inflows: dict[str, pd.Series],
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    final_inflows = {}
    rows = []

    all_plants = sorted(set(raw_inflows) | set(corrected_inflows), key=str)

    for plant in all_plants:
        has_raw = plant in raw_inflows
        has_corrected = plant in corrected_inflows

        if has_corrected:
            final_inflows[plant] = corrected_inflows[plant].copy()
            source = "corrected_saber"
        elif has_raw:
            final_inflows[plant] = raw_inflows[plant].copy()
            source = "raw_hydrological_model"
        else:
            continue

        s = final_inflows[plant]

        rows.append(
            {
                "plant": plant,
                "source": source,
                "has_raw": has_raw,
                "has_corrected": has_corrected,
                "n_values": int(len(s)),
                "n_valid": int(np.isfinite(s.to_numpy(dtype=float)).sum()),
                "start": s.index.min(),
                "end": s.index.max(),
            }
        )

    report = pd.DataFrame(rows)

    logger.info("Final inflows merged.")
    logger.info("Final plants: %s", len(final_inflows))

    if not report.empty:
        logger.debug(
            "Corrected plants: %s",
            int((report["source"] == "corrected_saber").sum()),
        )
        logger.debug(
            "Raw fallback plants: %s",
            int((report["source"] == "raw_hydrological_model").sum()),
        )

    return final_inflows, report


def save_final_inflows_pickle(
    final_inflows: dict[str, pd.Series],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as file:
        pickle.dump(final_inflows, file)

    logger.debug("Final inflows pickle saved: %s", output_path)


def build_daily_dataarray_for_year(
    final_inflows: dict[str, pd.Series],
    year: int,
) -> xr.DataArray:
    daily_index = pd.date_range(
        start=f"{year}-01-01",
        end=f"{year}-12-31",
        freq="D",
    )

    if REMOVE_LEAP_DAY:
        daily_index = daily_index[
            ~((daily_index.month == 2) & (daily_index.day == 29))
        ]

    plants = sorted(final_inflows.keys(), key=str)
    data = []
    missing_or_bad = []

    for plant in plants:
        s = final_inflows[plant].copy()
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        s = remove_leap_day_from_series(s)

        s_year = s.loc[f"{year}-01-01":f"{year}-12-31"]
        s_year = s_year.reindex(daily_index)

        if s_year.isna().any():
            missing_or_bad.append(
                {
                    "plant": plant,
                    "year": year,
                    "missing_days": int(s_year.isna().sum()),
                }
            )

        data.append(s_year.to_numpy(dtype=float))

    if missing_or_bad:
        logger.warning(
            "Missing daily values found for year %s in %s plants.",
            year,
            len(missing_or_bad),
        )
        logger.debug(
            "First missing daily value rows:\n%s",
            pd.DataFrame(missing_or_bad).head(20).to_string(index=False),
        )

    arr = np.asarray(data, dtype="float32")

    da = xr.DataArray(
        data=arr,
        dims=["plant", "valid_time"],
        coords={
            "plant": np.asarray(plants, dtype=str),
            "valid_time": daily_index,
        },
        name="inflow_Glofas_d",
    )

    return da


def daily_to_hourly_inflow(
    da_daily: xr.DataArray,
    year: int,
) -> xr.DataArray:
    time_hourly = pd.date_range(
        start=f"{year}-01-01 00:00:00",
        end=f"{year}-12-31 23:00:00",
        freq="h",
    )

    if REMOVE_LEAP_DAY:
        time_hourly = time_hourly[
            ~((time_hourly.month == 2) & (time_hourly.day == 29))
        ]

    daily_index = pd.date_range(
        start=f"{year}-01-01",
        end=f"{year}-12-31",
        freq="D",
    )

    if REMOVE_LEAP_DAY:
        daily_index = daily_index[
            ~((daily_index.month == 2) & (daily_index.day == 29))
        ]

    if da_daily.sizes["valid_time"] != len(daily_index):
        raise ValueError(
            f"Mismatch for year {year}: da_daily has "
            f"{da_daily.sizes['valid_time']} days, expected {len(daily_index)}."
        )

    expanded = []

    for i in range(da_daily.sizes["plant"]):
        daily_series = pd.Series(
            da_daily.isel(plant=i).values,
            index=daily_index,
        )

        extended_index = (
            pd.DatetimeIndex([time_hourly[0]])
            .append(daily_series.index)
            .append(pd.DatetimeIndex([time_hourly[-1]]))
        )

        extended_values = np.concatenate(
            [
                [daily_series.iloc[0]],
                daily_series.values,
                [daily_series.iloc[-1]],
            ]
        )

        extended_series = pd.Series(
            extended_values,
            index=extended_index,
        ).loc[~extended_index.duplicated()]

        hourly_series = (
            extended_series
            .reindex(time_hourly)
            .interpolate(method="linear")
        )

        expanded.append(hourly_series.values)

    arr = np.asarray(expanded, dtype="float32")

    da_hourly = xr.DataArray(
        data=arr,
        dims=["plant", "time"],
        coords={
            "plant": da_daily["plant"].values,
            "time": time_hourly,
        },
        name="inflow_Glofas_h",
    )

    if FILL_NA_WITH_ZERO:
        da_hourly = da_hourly.fillna(0.0)

    return da_hourly


def save_pypsa_inflow_for_year(
    final_inflows: dict[str, pd.Series],
    year: int,
    output_dir: Path,
    output_template: str,
) -> Path:
    da_daily = build_daily_dataarray_for_year(
        final_inflows=final_inflows,
        year=year,
    )

    da_hourly = daily_to_hourly_inflow(
        da_daily=da_daily,
        year=year,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_template.format(year=year)

    da_hourly.to_netcdf(output_path)

    logger.info("PyPSA-ready hourly inflow saved for %s: %s", year, output_path)
    logger.debug("Hourly inflow DataArray for %s:\n%s", year, da_hourly)

    return output_path


def log_sanity_check(
    final_inflows: dict[str, pd.Series],
    report: pd.DataFrame,
) -> None:
    if not final_inflows:
        logger.warning("Final inflows are empty.")
        return

    periods = {
        (s.index.min(), s.index.max())
        for s in final_inflows.values()
    }

    logger.debug("Final sanity check")
    logger.debug("Final plants: %s", len(final_inflows))
    logger.debug("Unique plant keys: %s", len(set(final_inflows)))

    if not report.empty:
        logger.debug(
            "Source counts:\n%s",
            report["source"].value_counts(dropna=False).to_string(),
        )

    logger.debug("Periods:")
    for start, end in sorted(periods):
        logger.debug("  %s -> %s", start, end)

    k0 = sorted(final_inflows.keys(), key=str)[0]
    s0 = final_inflows[k0]

    logger.debug("Sample plant: %s", k0)
    logger.debug("Sample type: %s", type(s0))
    logger.debug("Sample index type: %s", type(s0.index))
    logger.debug("Sample period: %s -> %s", s0.index.min(), s0.index.max())
    logger.debug("Sample head:\n%s", s0.head().to_string())


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    setup_logging()
    log_config_summary(CFG)

    rivid_map = read_rivid_map(RIVID_MAP_PATH)

    model_to_plants = build_model_to_plants(rivid_map)

    raw_inflows = load_raw_inflows_by_plant(
        rivid_map=rivid_map,
        hindcast_plants_zarr=HINDCAST_PLANTS_ZARR,
    )

    corrected_inflows = load_corrected_inflows_by_plant(
        corrected_dir=CORRECTED_DIR,
        model_to_plants=model_to_plants,
    )

    final_inflows, report = merge_raw_and_corrected_inflows(
        raw_inflows=raw_inflows,
        corrected_inflows=corrected_inflows,
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)
    logger.info("Final inflow report saved: %s", REPORT_PATH)

    log_sanity_check(
        final_inflows=final_inflows,
        report=report,
    )

    if DO_SAVE_PICKLE:
        save_final_inflows_pickle(
            final_inflows=final_inflows,
            output_path=OUTPUT_PICKLE_PATH,
        )

    if DO_SAVE_NETCDF:
        for year in YEARS:
            save_pypsa_inflow_for_year(
                final_inflows=final_inflows,
                year=int(year),
                output_dir=OUTPUT_NETCDF_DIR,
                output_template=OUTPUT_NETCDF_TEMPLATE,
            )

    logger.info("SABER-to-PyPSA inflow conversion completed.")


if __name__ == "__main__":
    main()