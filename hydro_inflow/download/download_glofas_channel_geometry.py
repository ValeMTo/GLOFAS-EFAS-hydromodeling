from __future__ import annotations

from pathlib import Path
import argparse
import logging

from hydro_inflow.utils import download_file, get_data_root, require_file, setup_logging


logger = logging.getLogger(__name__)


# Source: CEMS_SurfaceFields_2022 / LISFLOOD static and parameter maps for
# GloFAS v1.1.1 (OS-LISFLOOD v4.x), global 3 arcmin fields, used operationally
# for GloFASv4/EFASv5 -- same static-map family as uparea_glofas_v4_0.nc
# already used by this pipeline. Served directly (no WAF) from JEODPP:
# https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/LISFLOOD_static_and_parameter_maps_for_GloFAS/v1.1.1_OS-LISFLOOD-v4.x/Catchments_morphology_and_river_network/
_BASE_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/"
    "LISFLOOD_static_and_parameter_maps_for_GloFAS/v1.1.1_OS-LISFLOOD-v4.x/"
    "Catchments_morphology_and_river_network"
)

CHANNEL_GEOMETRY_URLS: dict[str, str] = {
    "chanbnkf": f"{_BASE_URL}/chanbnkf_Global_03min.nc",     # bankfull channel depth [m]
    "chanbw": f"{_BASE_URL}/chanbw_Global_03min.nc",         # channel bottom width [m]
    "changrad": f"{_BASE_URL}/changrad_Global_03min.nc",     # channel gradient [m/m]
    "chanman": f"{_BASE_URL}/chanman_Global_03min.nc",       # Manning's roughness [m^-1/3 s]
    "chans": f"{_BASE_URL}/chans_Global_03min.nc",           # channel side slope [m/m]
    "chanflpn": f"{_BASE_URL}/chanflpn_Global_03min.nc",     # floodplain width [m]
    "chanlength": f"{_BASE_URL}/chanlength_Global_03min.nc", # channel length [m]
}


def download_glofas_channel_geometry(
    data_root: Path | None = None,
    overwrite: bool = False,
    variables: list[str] | None = None,
) -> dict[str, Path]:
    data_root = data_root or get_data_root()
    out_dir = data_root / "hydro_global" / "channel_geometry"

    logger.info("HYDRO_DATA_ROOT: %s", data_root)
    logger.info("Channel geometry output directory: %s", out_dir)

    selected = variables or list(CHANNEL_GEOMETRY_URLS)
    missing_urls = [name for name in selected if not CHANNEL_GEOMETRY_URLS.get(name)]
    if missing_urls:
        raise ValueError(
            "Missing direct download URL(s) for: "
            f"{', '.join(missing_urls)}. Open the CEMS_SurfaceFields_2022 "
            "dataset page in a browser (JRC Data Catalogue) and fill in "
            "CHANNEL_GEOMETRY_URLS -- the WAF blocks automated lookup."
        )

    output_files: dict[str, Path] = {}
    for name in selected:
        destination = out_dir / f"{name}_glofas_v4_0.nc"

        download_file(
            url=CHANNEL_GEOMETRY_URLS[name],
            destination=destination,
            overwrite=overwrite,
        )
        require_file(destination, f"GloFAS channel geometry: {name}")
        output_files[name] = destination

    logger.info("Channel geometry download completed.")

    return output_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download GloFAS v4.0 channel geometry static maps "
        "(CEMS_SurfaceFields_2022)."
    )

    parser.add_argument(
        "--variables",
        default=None,
        help="Comma-separated subset of variables to download "
        f"(default: all of {list(CHANNEL_GEOMETRY_URLS)}).",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files.",
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

    variables = (
        [item.strip() for item in args.variables.split(",") if item.strip()]
        if args.variables
        else None
    )

    download_glofas_channel_geometry(
        overwrite=args.overwrite,
        variables=variables,
    )


if __name__ == "__main__":
    main()
