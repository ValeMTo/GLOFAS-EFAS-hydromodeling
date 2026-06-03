from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

from logging_utils import setup_logging


logger = logging.getLogger(__name__)


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    setup_logging()

    repo_root = get_repo_root()
    saber_path = repo_root / "external" / "saber_hbc"

    if saber_path.exists():
        sys.path.insert(0, str(saber_path))

    packages = [
        "pypsa",
        "atlite",
        "pandas",
        "numpy",
        "xarray",
        "geopandas",
        "rasterio",
        "rioxarray",
        "matplotlib",
        "plotly",
        "h5netcdf",
        "zarr",
        "netCDF4",
        "scipy",
        "shapely",
        "pyproj",
        "fiona",
        "yaml",
        "ruamel.yaml",
        "hydrostats",
        "kneed",
        "contextily",
        "mercantile",
        "saber",
    ]

    failed = []

    logger.info("Checking Python environment packages.")

    for package in packages:
        try:
            importlib.import_module(package)
            logger.info("OK: %s", package)
        except Exception as error:
            logger.error("FAILED: %s -> %s", package, error)
            failed.append(package)

    if failed:
        raise SystemExit(f"Missing or broken packages: {failed}")

    logger.info("Environment check passed.")


if __name__ == "__main__":
    main()