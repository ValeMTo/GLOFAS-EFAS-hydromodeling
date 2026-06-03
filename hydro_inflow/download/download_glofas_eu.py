from __future__ import annotations

from pathlib import Path
import argparse
import logging
import zipfile

import cdsapi

from hydro_inflow.utils import get_data_root, setup_logging


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


def parse_years(years_arg: str) -> list[int]:
    if ":" in years_arg:
        start, end = years_arg.split(":", maxsplit=1)
        return list(range(int(start), int(end) + 1))

    return [
        int(item.strip())
        for item in years_arg.split(",")
        if item.strip()
    ]


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


def download_glofas_year(
    client: cdsapi.Client,
    year: int,
    out_dir: Path,
    overwrite: bool = False,
    keep_zip: bool = False,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir / f"glofas_eu_{year}.zip"
    nc_path = out_dir / f"glofas_eu_{year}.nc"

    if nc_path.exists() and not overwrite:
        logger.info("Skipping %s: NetCDF already exists: %s", year, nc_path)
        return nc_path

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

    extract_single_netcdf(
        zip_path=zip_path,
        output_nc_path=nc_path,
    )

    if not keep_zip:
        zip_path.unlink(missing_ok=True)
        logger.debug("Deleted ZIP: %s", zip_path)

    return nc_path


def download_glofas_eu(
    years: str = "1980:2025",
    data_root: Path | None = None,
    overwrite: bool = False,
    keep_zip: bool = False,
) -> list[Path]:
    data_root = data_root or get_data_root()
    out_dir = data_root / "glofas_europe"
    parsed_years = parse_years(years)

    logger.info("HYDRO_DATA_ROOT: %s", data_root)
    logger.info("GloFAS output directory: %s", out_dir)
    logger.info("GloFAS years: %s", parsed_years)

    client = cdsapi.Client()
    output_files: list[Path] = []

    for year in parsed_years:
        output_file = download_glofas_year(
            client=client,
            year=year,
            out_dir=out_dir,
            overwrite=overwrite,
            keep_zip=keep_zip,
        )
        output_files.append(output_file)

    logger.info("GloFAS download completed.")

    return output_files


def parse_args() -> argparse.Namespace:
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

    download_glofas_eu(
        years=args.years,
        overwrite=args.overwrite,
        keep_zip=args.keep_zip,
    )


if __name__ == "__main__":
    main()