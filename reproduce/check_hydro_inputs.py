from __future__ import annotations

from pathlib import Path
from logging_utils import setup_logging
import logging
import os
import sys


logger = logging.getLogger(__name__)


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_data_root() -> Path:
    return Path(
        os.environ.get(
            "HYDRO_DATA_ROOT",
            get_repo_root() / "data" / "hydro_workflow",
        )
    )


def load_hydro_config(repo_root: Path) -> dict:
    sys.path.insert(0, str(repo_root))

    from scripts.hydro_inflow.hydro_config import get_config

    return get_config()


def check_paths(paths: list[Path], label: str) -> list[Path]:
    missing = []

    for path in paths:
        if path.exists():
            logger.info("OK %s: %s", label, path)
        else:
            logger.warning("Missing %s: %s", label, path)
            missing.append(path)

    return missing


def main() -> None:
    setup_logging()

    repo_root = get_repo_root()
    data_root = get_data_root()

    os.environ["HYDRO_REPO_ROOT"] = str(repo_root)
    os.environ["HYDRO_DATA_ROOT"] = str(data_root)

    cfg = load_hydro_config(repo_root=repo_root)

    logger.info("Checking required hydro input files.")
    logger.info("Repository root = %s", repo_root)
    logger.info("HYDRO_DATA_ROOT = %s", data_root)

    required_files = [
        data_root / "hydro_global" / "uparea_glofas_v4_0.nc",
        data_root / "hydro_global" / "uparea_5.0_cut.nc",
        data_root / "hydro_global" / "Eur_custom_ppls_GloHydroRes_filled.csv",
        data_root / "hydro_global" / "GRanD_reservoirs_v1_3.shp",
        data_root / "Stations" / "GRDC_Europe.nc",
        data_root / "glofas_europe" / "glofas_eu_2021.nc",
        data_root / "efas" / "efas_historical_2021_daily_cut.nc",
    ]

    required_dirs = [
        data_root / "hydro_global" / "Hydrobasins" / "hybas_eu_",
        Path(cfg["entsoe_hydro_dir"]),
        Path(cfg["entsoe_hydro_hourly_dir"]),
        Path(cfg["electricity_maps_dir"]),
    ]

    for year in cfg["entsoe_hydro_years"]:
        required_files.append(
            Path(cfg["entsoe_hydro_hourly_dir"]) / f"Europe_Hydro_{year}.csv"
        )

    for path in cfg["electricity_maps_ch_files"].values():
        required_files.append(Path(path))

    required_files.append(Path(cfg["entsoe_hydro_annual_production_path"]))

    missing = []
    missing.extend(check_paths(required_files, label="file"))
    missing.extend(check_paths(required_dirs, label="directory"))

    if missing:
        logger.error("Missing required hydro input files/directories.")

        for path in missing:
            logger.error("Missing: %s", path)

        logger.info("Some files can be downloaded automatically with:")
        logger.info("python reproduce/prepare_hydro_inputs.py")

        logger.info("For ENTSO-E hydro production, set the API token first:")
        logger.info("export ENTSOE_API_TOKEN='<your-token>'")

        logger.info("Then run:")
        logger.info(
            "python reproduce/prepare_hydro_inputs.py "
            "--skip-glofas --skip-efas --entsoe-years 2015:2019"
        )

        logger.info("Files that require manual access must be placed manually.")

        logger.info(
            "Expected GRDC path: %s",
            data_root / "Stations" / "GRDC_Europe.nc",
        )

        logger.info(
            "Expected GRanD path: %s",
            data_root / "hydro_global" / "GRanD_reservoirs_v1_3.shp",
        )

        logger.info(
            "Expected EFAS uparea path: %s",
            data_root / "hydro_global" / "uparea_5.0_cut.nc",
        )

        logger.info(
            "Expected Electricity Maps CH directory: %s",
            Path(cfg["electricity_maps_dir"]),
        )

        for year, path in cfg["electricity_maps_ch_files"].items():
            logger.info(
                "Expected Electricity Maps CH file %s: %s",
                year,
                path,
            )

        logger.info(
            "After ENTSO-E hourly CSVs and Electricity Maps JSONs are available, "
            "the annual reference can be built automatically by:"
        )

        logger.info(
            "python reproduce/prepare_hydro_inputs.py "
            "--skip-glofas --skip-efas --skip-entsoe"
        )

        logger.info("Alternatively, build it directly with:")
        logger.info("python reproduce/build_entsoe_hydro_annual_production.py")

        raise SystemExit(1)

    logger.info("All required hydro input files/directories are present.")


if __name__ == "__main__":
    main()