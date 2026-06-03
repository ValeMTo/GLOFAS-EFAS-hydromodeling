from __future__ import annotations

from pathlib import Path
import argparse
import logging
import os
import sys
import zipfile

import cdsapi


sys.path.append(str(Path(__file__).resolve().parents[1]))

from logging_utils import setup_logging


logger = logging.getLogger(__name__)


DATASET = "cems-glofas-historical"

MONTHS = [
    "01", "02", "03", "04", "05", "06",
    "07", "08", "09", "10", "11", "12",
]

DAYS = [
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31",
]


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_data_root() -> Path:
    return Path(
        os.environ.get(
            "HYDRO_DATA_ROOT",
            get_repo_root() / "data" / "hydro_workflow",
        )
    )


def extract_single_netcdf(zip_path: Path, output_nc_path: Path) -> None:
    output_nc_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        nc_members = [
            member
            for member in archive.namelist()
            if member.lower().endswith(".nc")
        ]

        if not nc_members:
            raise FileNotFoundError(f"No NetCDF file found inside {zip_path}")

        if len(nc_members) > 1:
            logger.warning(
                "More than one NetCDF found in %s; using %s",
                zip_path,
                nc_members[0],
            )

        member = nc_members[0]

        with archive.open(member) as source, output_nc_path.open("wb") as target:
            target.write(source.read())

    logger.debug("Extracted NetCDF: %s", output_nc_path)


def download_year(
    client: cdsapi.Client,
    year: int,
    out_dir: Path,
    overwrite: bool,
    keep_zip: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir / f"glofas_eu_{year}.zip"
    nc_path = out_dir / f"glofas_eu_{year}.nc"

    if nc_path.exists() and not overwrite:
        logger.info("Skipping %s: NetCDF already exists: %s", year, nc_path)
        return

    if not zip_path.exists() or overwrite:
        logger.info("Downloading GloFAS %s to %s", year, zip_path)

        request = {
            "system_version": ["version_4_0"],
            "hydrological_model": ["lisflood"],
            "product_type": ["consolidated"],
            "variable": ["river_discharge_in_the_last_24_hours"],
            "hyear": [str(year)],
            "hmonth": MONTHS,
            "hday": DAYS,
            "data_format": "netcdf",
            "download_format": "zip",
            "area": [71.075, -9.375, 36.375, 31.325],
        }

        client.retrieve(DATASET, request).download(str(zip_path))
    else:
        logger.debug("Using existing ZIP for %s: %s", year, zip_path)

    extract_single_netcdf(zip_path=zip_path, output_nc_path=nc_path)

    if not keep_zip:
        zip_path.unlink(missing_ok=True)
        logger.debug("Deleted ZIP: %s", zip_path)


def parse_years(years_arg: str) -> list[int]:
    if ":" in years_arg:
        start, end = years_arg.split(":", maxsplit=1)
        return list(range(int(start), int(end) + 1))

    return [
        int(item.strip())
        for item in years_arg.split(",")
        if item.strip()
    ]


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Download and extract annual GloFAS Europe NetCDF files."
    )

    parser.add_argument(
        "--years",
        default="1980:2025",
        help="Years to download. Use '1980:2025' or '1980,1981,2021'.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing ZIP/NetCDF files.",
    )

    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep downloaded ZIP files after extracting NetCDF.",
    )

    args = parser.parse_args()

    data_root = get_data_root()
    out_dir = data_root / "glofas_europe"
    years = parse_years(args.years)

    logger.debug("HYDRO_DATA_ROOT: %s", data_root)
    logger.debug("GloFAS output directory: %s", out_dir)
    logger.debug("Years: %s-%s (%s years)", years[0], years[-1], len(years))

    client = cdsapi.Client()

    for year in years:
        download_year(
            client=client,
            year=year,
            out_dir=out_dir,
            overwrite=args.overwrite,
            keep_zip=args.keep_zip,
        )

    logger.info("GloFAS download completed.")


if __name__ == "__main__":
    main()