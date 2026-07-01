from __future__ import annotations

from pathlib import Path
import argparse
import gc
import glob
import logging
import os
import shutil
import tempfile
import zipfile

import cdsapi
import pandas as pd
import xarray as xr

from hydro_inflow.utils import get_data_root, setup_logging


logger = logging.getLogger(__name__)


DATASET = "efas-historical"

LAT_MIN = 32.0
LON_MAX = 40.0

DAYS = [f"{day:02d}" for day in range(1, 32)]


def get_efas_output_dir(data_root: Path | None = None) -> Path:
    if data_root is None:
        data_root = get_data_root()

    output_dir = data_root / "efas"
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def parse_years(years_arg: str) -> list[int]:
    if ":" in years_arg:
        start, end = years_arg.split(":", maxsplit=1)
        return list(range(int(start), int(end) + 1))

    return [int(item.strip()) for item in years_arg.split(",") if item.strip()]


def download_efas_month(
    client: cdsapi.Client,
    input_folder: Path,
    year: int,
    month: int,
) -> Path:
    month_str = f"{month:02d}"
    output_file = input_folder / f"efas_historical_{year}_{month_str}.zip"

    if output_file.exists():
        logger.debug("Already exists, skipping monthly ZIP: %s", output_file)
        return output_file

    request = {
        "system_version": ["version_5_0"],
        "variable": ["river_discharge_in_the_last_6_hours"],
        "model_levels": "surface_level",
        "hyear": [str(year)],
        "hmonth": [month_str],
        "hday": DAYS,
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "data_format": "netcdf",
        "download_format": "zip",
    }

    logger.info("Downloading EFAS %s-%s to %s", year, month_str, output_file)
    client.retrieve(DATASET, request).download(str(output_file))
    logger.info("Completed EFAS monthly download: %s", output_file)

    return output_file


def download_efas_year(
    client: cdsapi.Client,
    input_folder: Path,
    year: int,
) -> None:
    logger.info("Downloading EFAS year %s", year)

    for month in range(1, 13):
        download_efas_month(
            client=client,
            input_folder=input_folder,
            year=year,
            month=month,
        )


def open_cut_from_zip(
    zip_path: Path,
    temp_nc_dir: Path,
) -> tuple[xr.Dataset, xr.Dataset]:
    zip_name = zip_path.stem
    extract_dir = temp_nc_dir / zip_name
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        nc_files = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".nc")
        ]

        if not nc_files:
            raise FileNotFoundError(f"No .nc file found inside {zip_path}")

        if len(nc_files) > 1:
            logger.warning(
                "More than one NetCDF found in %s; using the first one: %s",
                zip_path,
                nc_files[0],
            )

        nc_path = Path(archive.extract(nc_files[0], path=extract_dir))

    ds = xr.open_dataset(nc_path)

    if "valid_time" not in ds.coords:
        ds.close()
        raise ValueError(f"'valid_time' coordinate not found in {zip_path}")

    if "dis06" not in ds.data_vars:
        ds.close()
        raise ValueError(f"'dis06' variable not found in {zip_path}")

    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        ds.close()
        raise ValueError(f"'latitude'/'longitude' coordinates not found in {zip_path}")

    lat_descending = float(ds["latitude"][0]) > float(ds["latitude"][-1])

    if lat_descending:
        ds_cut = ds.sel(
            latitude=slice(float(ds["latitude"].max()), LAT_MIN),
            longitude=slice(float(ds["longitude"].min()), LON_MAX),
        )
    else:
        ds_cut = ds.sel(
            latitude=slice(LAT_MIN, float(ds["latitude"].max())),
            longitude=slice(float(ds["longitude"].min()), LON_MAX),
        )

    return ds, ds_cut


def process_efas_year(
    input_folder: Path,
    year: int,
) -> Path:
    logger.info("=" * 60)
    logger.info("Processing EFAS year %s", year)
    logger.info("=" * 60)

    temp_root = Path(tempfile.mkdtemp(prefix=f"efas_{year}_daily_cut_"))
    temp_nc_dir = temp_root / "extracted_nc"
    temp_daily_dir = temp_root / "monthly_daily_nc"

    temp_nc_dir.mkdir(parents=True, exist_ok=True)
    temp_daily_dir.mkdir(parents=True, exist_ok=True)

    monthly_daily_files: list[Path] = []

    try:
        for month in range(1, 13):
            month_str = f"{month:02d}"
            current_zip = input_folder / f"efas_historical_{year}_{month_str}.zip"

            if not current_zip.exists():
                raise FileNotFoundError(f"Missing current month ZIP: {current_zip}")

            logger.info("Processing EFAS month %s-%s", year, month_str)

            ds_current, ds_current_cut = open_cut_from_zip(
                zip_path=current_zip,
                temp_nc_dir=temp_nc_dir,
            )

            try:
                dis06 = ds_current_cut["dis06"].sortby("valid_time")

                logger.debug(
                    "Original monthly time range for %s-%s: %s -> %s",
                    year,
                    month_str,
                    str(dis06["valid_time"].values[0]),
                    str(dis06["valid_time"].values[-1]),
                )

                dis06_shifted = dis06.assign_coords(
                    valid_time=dis06["valid_time"] - pd.Timedelta(hours=6)
                )

                daily = dis06_shifted.resample(valid_time="1D").mean(keep_attrs=True)

                daily = daily.sel(
                    valid_time=(
                        (daily["valid_time"].dt.year == year)
                        & (daily["valid_time"].dt.month == month)
                    )
                )

                if daily.sizes["valid_time"] == 0:
                    raise RuntimeError(
                        f"No daily timesteps after filtering {year}-{month_str}"
                    )

                daily.attrs["long_name"] = "Daily mean discharge"
                daily.attrs["aggregation"] = (
                    "Daily mean computed from EFAS dis06 shifted back by 6 hours. "
                    "For day D, the intended set is D 06:00, D 12:00, D 18:00 "
                    "and D+1 00:00. EFAS monthly files include D+1 00:00 "
                    "at the end of each month."
                )
                daily.attrs["source_variable"] = "dis06"

                daily_ds = daily.to_dataset(name="discharge_daily_mean")
                daily_ds = daily_ds.rename({"valid_time": "time"})

                logger.debug(
                    "Daily monthly shape for %s-%s: %s",
                    year,
                    month_str,
                    daily_ds["discharge_daily_mean"].shape,
                )
                logger.debug(
                    "Daily monthly time range for %s-%s: %s -> %s",
                    year,
                    month_str,
                    str(daily_ds["time"].values[0]),
                    str(daily_ds["time"].values[-1]),
                )

                monthly_daily_path = (
                    temp_daily_dir / f"efas_historical_{year}_{month_str}_daily_cut.nc"
                )

                encoding = {
                    "discharge_daily_mean": {
                        "zlib": True,
                        "complevel": 4,
                        "dtype": "float32",
                    }
                }

                daily_ds.to_netcdf(monthly_daily_path, encoding=encoding)
                monthly_daily_files.append(monthly_daily_path)

                daily_ds.close()

                logger.debug("Saved daily monthly cut file: %s", monthly_daily_path)

            finally:
                ds_current.close()
                ds_current_cut.close()
                dis06 = dis06_shifted = daily = daily_ds = None
                gc.collect()

        logger.info("Merging monthly daily cut files into annual EFAS file.")

        ds_year = xr.open_mfdataset(
            [str(path) for path in monthly_daily_files],
            combine="by_coords",
        ).sortby("time")

        try:
            expected_days = 366 if pd.Timestamp(f"{year}-12-31").is_leap_year else 365
            actual_days = ds_year.sizes["time"]

            if actual_days != expected_days:
                raise RuntimeError(
                    f"Unexpected number of daily timesteps for {year}: "
                    f"{actual_days}, expected {expected_days}"
                )

            output_file = input_folder / f"efas_historical_{year}_daily_cut.nc"
            temp_output_file = input_folder / f"efas_historical_{year}_daily_cut.nc.tmp"

            encoding = {
                "discharge_daily_mean": {
                    "zlib": True,
                    "complevel": 4,
                    "dtype": "float32",
                }
            }

            ds_year.to_netcdf(temp_output_file, encoding=encoding)
            os.replace(temp_output_file, output_file)

            logger.info("Annual EFAS daily cut file saved: %s", output_file)

            return output_file

        finally:
            ds_year.close()

    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
        logger.debug("Temporary EFAS processing directory removed: %s", temp_root)


def delete_monthly_zip_files(
    input_folder: Path,
    year: int,
) -> None:
    monthly_zip_pattern = str(input_folder / f"efas_historical_{year}_*.zip")
    monthly_zip_files = sorted(glob.glob(monthly_zip_pattern))

    logger.info("Deleting monthly EFAS ZIP files for %s.", year)

    for zip_path in monthly_zip_files:
        Path(zip_path).unlink()
        logger.debug("Deleted monthly ZIP: %s", zip_path)


def download_and_merge_efas(
    years: str = "1992:2025",
    data_root: Path | None = None,
    keep_monthly_zips: bool = False,
    overwrite: bool = False,
) -> list[Path]:
    data_root = data_root or get_data_root()
    input_folder = get_efas_output_dir(data_root)
    parsed_years = parse_years(years)

    logger.info("HYDRO_DATA_ROOT: %s", data_root)
    logger.info("EFAS output directory: %s", input_folder)
    logger.info("EFAS years: %s", parsed_years)

    client = cdsapi.Client()
    output_files: list[Path] = []

    for year in parsed_years:
        output_file = input_folder / f"efas_historical_{year}_daily_cut.nc"

        if output_file.exists() and not overwrite:
            logger.info(
                "Skipping %s: annual daily file already exists: %s",
                year,
                output_file,
            )
            output_files.append(output_file)
            continue

        if output_file.exists() and overwrite:
            logger.info("Overwriting existing annual EFAS file: %s", output_file)
            output_file.unlink()

        try:
            download_efas_year(
                client=client,
                input_folder=input_folder,
                year=year,
            )

            produced_file = process_efas_year(
                input_folder=input_folder,
                year=year,
            )

            if not keep_monthly_zips:
                delete_monthly_zip_files(
                    input_folder=input_folder,
                    year=year,
                )

            output_files.append(produced_file)

            logger.info("EFAS year %s completed successfully.", year)

        except Exception:
            logger.exception(
                "Error while processing EFAS year %s. "
                "Stopping here. Monthly files are kept for inspection/restart.",
                year,
            )
            raise

    logger.info("EFAS download and processing completed.")

    return output_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, cut, aggregate and merge EFAS historical discharge files."
    )

    parser.add_argument(
        "--years",
        default="1992:2025",
        help="Years to process. Use '1992:2025' or '1992,1993,2021'.",
    )

    parser.add_argument(
        "--keep-monthly-zips",
        action="store_true",
        help="Keep monthly ZIP files after the annual NetCDF has been created.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing annual daily NetCDF files.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Use DEBUG logging.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    download_and_merge_efas(
        years=args.years,
        keep_monthly_zips=args.keep_monthly_zips,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
