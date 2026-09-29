from __future__ import annotations

from pathlib import Path
import argparse
import logging

from hydro_inflow.utils import download_file, get_data_root, require_file, setup_logging


logger = logging.getLogger(__name__)


# Same GloFAS v4.0 auxiliary-data page as uparea_glofas_v4_0.nc (see
# prepare_hydro_inputs.py:GLOFAS_UPAREA_URL), different attachment.
# https://confluence.ecmwf.int/pages/viewpage.action?pageId=242067380
ELEVATION_URL = (
    "https://confluence.ecmwf.int/download/attachments/242067380/"
    "elevation_glofas_v4_0.nc"
)


def download_glofas_elevation(data_root: Path | None = None, overwrite: bool = False) -> Path:
    data_root = data_root or get_data_root()
    destination = data_root / "hydro_global" / "elevation_glofas_v4_0.nc"

    logger.info("HYDRO_DATA_ROOT: %s", data_root)
    logger.info("Destination: %s", destination)

    download_file(url=ELEVATION_URL, destination=destination, overwrite=overwrite)
    require_file(destination, "GloFAS elevation")

    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the GloFAS v4.0 elevation static map."
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing file.")
    parser.add_argument("--verbose", action="store_true", help="Use DEBUG logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    download_glofas_elevation(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
