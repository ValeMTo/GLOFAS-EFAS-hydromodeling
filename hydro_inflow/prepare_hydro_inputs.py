from __future__ import annotations

from pathlib import Path
import logging
import os
import shutil
import zipfile

from hydro_inflow.hydro_config import get_config
from hydro_inflow.prepare_hydro_plants import prepare_hydro_plants
from hydro_inflow.download.download_glofas_eu import download_glofas_eu
from hydro_inflow.download.download_and_merge_efas import download_and_merge_efas
from hydro_inflow.download.download_entsoe import download_entsoe_hydro
from hydro_inflow.utils import download_file, get_data_root, get_repo_root, require_file
from hydro_inflow.build_entsoe_hydro_annual_production import (
    build_entsoe_hydro_annual_production,
)


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


def download_glohydrores(repo_root: Path, overwrite: bool = False) -> Path:
    destination = repo_root / "data" / "pypsa" / "GloHydroRes_vs1.csv"

    download_file(
        url=GLOHYDRORES_URL,
        destination=destination,
        overwrite=overwrite,
    )

    require_file(destination, "GloHydroRes source CSV")
    return destination


def download_glofas_uparea(data_root: Path, overwrite: bool = False) -> Path:
    destination = data_root / "hydro_global" / "uparea_glofas_v4_0.nc"

    download_file(
        url=GLOFAS_UPAREA_URL,
        destination=destination,
        overwrite=overwrite,
    )

    require_file(destination, "GloFAS upstream area")
    return destination


def copy_shapefile_family(source_shp: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_stem = destination_dir / source_shp.stem

    for source_file in source_shp.parent.glob(source_shp.stem + ".*"):
        destination_file = destination_stem.with_suffix(source_file.suffix)
        shutil.copy2(source_file, destination_file)
        logger.debug("Copied: %s -> %s", source_file, destination_file)


def download_and_extract_hydrobasins(data_root: Path, overwrite: bool = False) -> Path:
    hydrobasins_root = data_root / "hydro_global" / "Hydrobasins"
    expected_dir = hydrobasins_root / "hybas_eu_"
    expected_shp = expected_dir / "hybas_eu_lev03_v1c.shp"

    if expected_shp.exists() and not overwrite:
        logger.info("Already exists, skipping HydroBASINS: %s", expected_shp)
        return expected_shp

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

    require_file(expected_shp, "HydroBASINS level 03 shapefile")
    logger.info("HydroBASINS level 03 ready at: %s", expected_shp)

    return expected_shp


def prepare_and_check_hydro_inputs(
    *,
    glofas_years: str = "1980:2025",
    efas_years: str = "1992:2025",
    entsoe_years: str = "2015:2019",
    skip_static: bool = False,
    skip_glofas: bool = False,
    skip_efas: bool = False,
    skip_entsoe: bool = False,
    overwrite_static: bool = False,
    overwrite_glohydrores: bool = False,
) -> None:
    repo_root = get_repo_root()
    data_root = get_data_root(repo_root)
    cfg = get_config()

    logger.info("Repository root: %s", repo_root)
    logger.info("Hydro data root: %s", data_root)

    download_glohydrores(
        repo_root=repo_root,
        overwrite=overwrite_glohydrores,
    )

    prepare_hydro_plants()

    if not skip_static:
        download_glofas_uparea(
            data_root=data_root,
            overwrite=overwrite_static,
        )

        download_and_extract_hydrobasins(
            data_root=data_root,
            overwrite=overwrite_static,
        )

    if not skip_glofas:
        download_glofas_eu(
            years=glofas_years,
            data_root=data_root,
        )

    if not skip_efas:
        download_and_merge_efas(
            years=efas_years,
            data_root=data_root,
        )

    if not skip_entsoe:
        if "ENTSOE_API_TOKEN" not in os.environ:
            logger.warning(
                "Skipping ENTSO-E download because ENTSOE_API_TOKEN is not set."
            )
        else:
            download_entsoe_hydro(
                years=entsoe_years,
            )

    try:
        build_entsoe_hydro_annual_production(cfg=cfg)
    except FileNotFoundError as exc:
        logger.warning(
            "Skipping ENTSO-E annual production build because required inputs are missing: %s",
            exc,
        )

    logger.info("Hydro input preparation completed.")