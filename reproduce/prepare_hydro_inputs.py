from __future__ import annotations

from pathlib import Path
import argparse
import logging
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

from logging_utils import setup_logging


GLOHYDRORES_URL = (
    "https://zenodo.org/records/14526360/files/GloHydroRes_vs1.csv?download=1"
)

GLOFAS_UPAREA_URL = (
    "https://confluence.ecmwf.int/download/attachments/242067380/"
    "uparea_glofas_v4_0.nc?version=2&modificationDate=1668604690076&api=v2"
)

HYDROBASINS_EU_URL = (
    "https://data.hydrosheds.org/file/hydrobasins/standard/"
    "hybas_eu_lev01-12_v1c.zip"
)


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


def ensure_directories(data_root: Path, repo_root: Path) -> None:
    directories = [
        repo_root / "data" / "hydro",
        data_root / "hydro_global",
        data_root / "hydro_global" / "Hydrobasins",
        data_root / "hydro_global" / "Hydrobasins" / "hybas_eu_",
        data_root / "hydro_global" / "ENTSOE",
        data_root / "hydro_global" / "ENTSOE" / "Production",
        data_root / "hydro_global" / "ENTSOE" / "ElectricityMaps",
        data_root / "Stations",
        data_root / "glofas_europe",
        data_root / "efas",
        data_root / "saber_global" / "glofas",
        data_root / "saber_global" / "efas",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info("Created or found: %s", directory)


def download_file(url: str, destination: Path, overwrite: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        logger.info("Already exists, skipping: %s", destination)
        return

    temporary_destination = destination.with_suffix(destination.suffix + ".tmp")

    logger.info("Downloading: %s", url)
    logger.info("Destination: %s", destination)

    try:
        urllib.request.urlretrieve(url, temporary_destination)
        temporary_destination.replace(destination)
    except Exception:
        temporary_destination.unlink(missing_ok=True)
        logger.exception("Download failed for %s", destination)
        raise

    logger.info("Downloaded: %s", destination)


def download_glohydrores(repo_root: Path, overwrite: bool = False) -> None:
    destination = repo_root / "data" / "hydro" / "GloHydroRes_vs1.csv"

    download_file(
        url=GLOHYDRORES_URL,
        destination=destination,
        overwrite=overwrite,
    )


def run_python_script(script_path: Path, args: list[str], env: dict[str, str]) -> None:
    command = [sys.executable, str(script_path), *args]

    logger.info("=" * 80)
    logger.info("Running: %s", " ".join(command))
    logger.info("=" * 80)

    subprocess.run(command, check=True, env=env)


def prepare_hydro_plants(repo_root: Path, env: dict[str, str]) -> None:
    script_path = (
        repo_root
        / "scripts"
        / "hydro_inflow"
        / "prepare_hydro_plants.py"
    )

    run_python_script(
        script_path=script_path,
        args=[],
        env=env,
    )


def download_glofas_uparea(data_root: Path, overwrite: bool = False) -> None:
    destination = data_root / "hydro_global" / "uparea_glofas_v4_0.nc"

    download_file(
        url=GLOFAS_UPAREA_URL,
        destination=destination,
        overwrite=overwrite,
    )


def copy_shapefile_family(source_shp: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)

    destination_stem = destination_dir / source_shp.stem

    for source_file in source_shp.parent.glob(source_shp.stem + ".*"):
        destination_file = destination_stem.with_suffix(source_file.suffix)
        shutil.copy2(source_file, destination_file)
        logger.debug("Copied: %s -> %s", source_file, destination_file)


def download_and_extract_hydrobasins(data_root: Path, overwrite: bool = False) -> None:
    hydrobasins_root = data_root / "hydro_global" / "Hydrobasins"
    expected_dir = hydrobasins_root / "hybas_eu_"
    expected_shp = expected_dir / "hybas_eu_lev03_v1c.shp"

    if expected_shp.exists() and not overwrite:
        logger.info("Already exists, skipping HydroBASINS: %s", expected_shp)
        return

    zip_path = hydrobasins_root / "hybas_eu_lev01-12_v1c.zip"

    download_file(
        url=HYDROBASINS_EU_URL,
        destination=zip_path,
        overwrite=overwrite,
    )

    extract_dir = hydrobasins_root / "_hydrobasins_extract_tmp"

    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    extract_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting HydroBASINS ZIP: %s", zip_path)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)

    found = list(extract_dir.rglob("hybas_eu_lev03_v1c.shp"))

    if not found:
        raise FileNotFoundError(
            "Could not find hybas_eu_lev03_v1c.shp inside the HydroBASINS ZIP."
        )

    copy_shapefile_family(
        source_shp=found[0],
        destination_dir=expected_dir,
    )

    shutil.rmtree(extract_dir, ignore_errors=True)

    logger.info("HydroBASINS level 03 ready at: %s", expected_shp)


def download_glofas(repo_root: Path, years: str, env: dict[str, str]) -> None:
    script_path = (
        repo_root
        / "scripts"
        / "hydro_inflow"
        / "download"
        / "download_glofas_eu.py"
    )

    run_python_script(
        script_path=script_path,
        args=["--years", years],
        env=env,
    )


def download_efas(repo_root: Path, years: str, env: dict[str, str]) -> None:
    script_path = (
        repo_root
        / "scripts"
        / "hydro_inflow"
        / "download"
        / "download_and_merge_efas.py"
    )

    run_python_script(
        script_path=script_path,
        args=["--years", years],
        env=env,
    )


def download_entsoe_hydro(repo_root: Path, years: str, env: dict[str, str]) -> None:
    script_path = (
        repo_root
        / "scripts"
        / "hydro_inflow"
        / "download"
        / "download_entsoe.py"
    )

    if not script_path.exists():
        raise FileNotFoundError(
            f"Missing ENTSO-E download script: {script_path}"
        )

    run_python_script(
        script_path=script_path,
        args=["--years", years],
        env=env,
    )
    
def build_entsoe_hydro_annual_production(
    repo_root: Path,
    cfg: dict,
    env: dict[str, str],
) -> None:
    output_path = Path(cfg["entsoe_hydro_annual_production_path"])

    if output_path.exists():
        logger.info(
            "ENTSO-E annual hydro production CSV already exists, skipping: %s",
            output_path,
        )
        return

    script_path = repo_root / "reproduce" / "build_entsoe_hydro_annual_production.py"

    if not script_path.exists():
        raise FileNotFoundError(
            f"Missing ENTSO-E annual production builder: {script_path}"
        )

    missing_hourly_files = []

    for year in cfg["entsoe_hydro_years"]:
        hourly_file = (
            Path(cfg["entsoe_hydro_hourly_dir"])
            / f"Europe_Hydro_{year}.csv"
        )

        if not hourly_file.exists():
            missing_hourly_files.append(hourly_file)

    missing_electricity_maps_files = [
        Path(path)
        for path in cfg["electricity_maps_ch_files"].values()
        if not Path(path).exists()
    ]

    if missing_hourly_files:
        logger.warning(
            "Skipping ENTSO-E annual production build because hourly ENTSO-E files are missing."
        )
        for path in missing_hourly_files:
            logger.warning("Missing ENTSO-E hourly file: %s", path)
        return

    if missing_electricity_maps_files:
        logger.warning(
            "Skipping ENTSO-E annual production build because Electricity Maps files are missing."
        )
        for path in missing_electricity_maps_files:
            logger.warning("Missing Electricity Maps file: %s", path)
        return

    run_python_script(
        script_path=script_path,
        args=[],
        env=env,
    )

def log_input_status(data_root: Path, repo_root: Path, cfg: dict) -> None:
    logger.info("Hydro input status")

    required = [
        (
            "GloHydroRes source CSV",
            repo_root / "data" / "hydro" / "GloHydroRes_vs1.csv",
        ),
        (
            "GloFAS uparea",
            data_root / "hydro_global" / "uparea_glofas_v4_0.nc",
        ),
        (
            "EFAS uparea",
            data_root / "hydro_global" / "uparea_5.0_cut.nc",
        ),
        (
            "Processed hydropower plant list",
            data_root
            / "hydro_global"
            / "Eur_custom_ppls_GloHydroRes_filled.csv",
        ),
        (
            "GRanD reservoirs shapefile",
            data_root / "hydro_global" / "GRanD_reservoirs_v1_3.shp",
        ),
        (
            "HydroBASINS level 03 shapefile",
            data_root
            / "hydro_global"
            / "Hydrobasins"
            / "hybas_eu_"
            / "hybas_eu_lev03_v1c.shp",
        ),
        (
            "GRDC Europe stations",
            data_root / "Stations" / "GRDC_Europe.nc",
        ),
        (
            "GloFAS example annual file",
            data_root / "glofas_europe" / "glofas_eu_2021.nc",
        ),
        (
            "EFAS example annual file",
            data_root / "efas" / "efas_historical_2021_daily_cut.nc",
        ),
        (
            "ENTSO-E hydro hourly directory",
            Path(cfg["entsoe_hydro_hourly_dir"]),
        ),
        (
            "Electricity Maps CH directory",
            Path(cfg["electricity_maps_dir"]),
        ),
        (
            "ENTSO-E annual hydro production CSV",
            Path(cfg["entsoe_hydro_annual_production_path"]),
        ),
    ]

    for year in cfg["entsoe_hydro_years"]:
        required.append(
            (
                f"ENTSO-E hydro hourly production {year}",
                Path(cfg["entsoe_hydro_hourly_dir"])
                / f"Europe_Hydro_{year}.csv",
            )
        )

    for year, path in cfg["electricity_maps_ch_files"].items():
        required.append(
            (
                f"Electricity Maps CH hydro {year}",
                Path(path),
            )
        )

    for label, path in required:
        status = "OK" if path.exists() else "MISSING"
        log_func = logger.info if path.exists() else logger.warning
        log_func("%-8s %s | %s", status, label, path)


def log_manual_input_instructions(data_root: Path, cfg: dict) -> None:
    logger.info("Manual or not-yet-automated input files")

    logger.info(
        "GRDC_Europe.nc requires manual GRDC access and must be placed in: %s",
        data_root / "Stations" / "GRDC_Europe.nc",
    )

    logger.info(
        "GRanD reservoir shapefile must be placed at: %s",
        data_root / "hydro_global" / "GRanD_reservoirs_v1_3.shp",
    )

    logger.info(
        "EFAS upstream area file must be placed at: %s",
        data_root / "hydro_global" / "uparea_5.0_cut.nc",
    )

    logger.info(
        "ENTSO-E hydro production can be downloaded automatically if "
        "ENTSOE_API_TOKEN is set."
    )

    logger.info(
        "Example ENTSO-E command: "
        "ENTSOE_API_TOKEN='<token>' python reproduce/prepare_hydro_inputs.py "
        "--skip-glofas --skip-efas --entsoe-years 2015:2019"
    )

    logger.info(
        "Electricity Maps CH JSON files must be placed manually in: %s",
        Path(cfg["electricity_maps_dir"]),
    )

    for year, path in cfg["electricity_maps_ch_files"].items():
        logger.info(
            "Expected Electricity Maps CH file %s: %s",
            year,
            path,
        )

    logger.info(
        "After ENTSO-E hourly CSVs and Electricity Maps JSONs are available, run: %s",
        "python reproduce/build_entsoe_hydro_annual_production.py",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare hydro input folders and download hydro source data."
    )

    parser.add_argument(
        "--glofas-years",
        default="1980:2025",
        help="GloFAS years to download. Example: '1980:2025' or '2021'.",
    )

    parser.add_argument(
        "--efas-years",
        default="1992:2025",
        help="EFAS years to download/process. Example: '1992:2025' or '2021'.",
    )

    parser.add_argument(
        "--entsoe-years",
        default="2015:2019",
        help="ENTSO-E hydro production years to download. Example: '2015:2019' or '2019'.",
    )

    parser.add_argument(
        "--skip-static",
        action="store_true",
        help="Do not download static input files such as HydroBASINS and GloFAS uparea.",
    )

    parser.add_argument(
        "--skip-glofas",
        action="store_true",
        help="Do not run the GloFAS download script.",
    )

    parser.add_argument(
        "--skip-efas",
        action="store_true",
        help="Do not run the EFAS download/process script.",
    )

    parser.add_argument(
        "--skip-entsoe",
        action="store_true",
        help="Do not run the ENTSO-E hydro production download script.",
    )

    parser.add_argument(
        "--overwrite-static",
        action="store_true",
        help="Overwrite static downloaded files if they already exist.",
    )

    parser.add_argument(
        "--overwrite-glohydrores",
        action="store_true",
        help="Overwrite data/hydro/GloHydroRes_vs1.csv if it already exists.",
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()

    args = parse_args()

    repo_root = get_repo_root()
    data_root = get_data_root()

    env = os.environ.copy()
    env["HYDRO_REPO_ROOT"] = str(repo_root)
    env["HYDRO_DATA_ROOT"] = str(data_root)

    cfg = load_hydro_config(repo_root=repo_root)

    logger.info("Repository root: %s", repo_root)
    logger.info("HYDRO_DATA_ROOT: %s", data_root)

    ensure_directories(data_root=data_root, repo_root=repo_root)

    download_glohydrores(
        repo_root=repo_root,
        overwrite=args.overwrite_glohydrores,
    )

    prepare_hydro_plants(
        repo_root=repo_root,
        env=env,
    )

    if not args.skip_static:
        download_glofas_uparea(
            data_root=data_root,
            overwrite=args.overwrite_static,
        )

        download_and_extract_hydrobasins(
            data_root=data_root,
            overwrite=args.overwrite_static,
        )

    if not args.skip_glofas:
        download_glofas(
            repo_root=repo_root,
            years=args.glofas_years,
            env=env,
        )

    if not args.skip_efas:
        download_efas(
            repo_root=repo_root,
            years=args.efas_years,
            env=env,
        )

    if not args.skip_entsoe:
        if "ENTSOE_API_TOKEN" not in env:
            logger.warning(
                "Skipping ENTSO-E download because ENTSOE_API_TOKEN is not set."
            )
            logger.warning(
                "Set it with: export ENTSOE_API_TOKEN='<your-token>'"
            )
        else:
            download_entsoe_hydro(
                repo_root=repo_root,
                years=args.entsoe_years,
                env=env,
            )
        
    build_entsoe_hydro_annual_production(
        repo_root=repo_root,
        cfg=cfg,
        env=env,
    )

    log_input_status(data_root=data_root, repo_root=repo_root, cfg=cfg)
    log_manual_input_instructions(data_root=data_root, cfg=cfg)

    logger.info("Next check: python reproduce/check_hydro_inputs.py")


if __name__ == "__main__":
    main()
