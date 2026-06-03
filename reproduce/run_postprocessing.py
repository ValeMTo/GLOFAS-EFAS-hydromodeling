#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import argparse
import logging
import os
import shutil
import subprocess
import sys


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ============================================================
# ROOT PATHS
# ============================================================

def get_repo_root() -> Path:
    """
    Return repository root.

    Default:
        repo root inferred from this file:
        repo/reproduce/run_postprocessing.py -> repo

    Optional override:
        HYDRO_REPO_ROOT=/path/to/repo
    """

    return Path(
        os.environ.get(
            "HYDRO_REPO_ROOT",
            Path(__file__).resolve().parents[1],
        )
    ).resolve()


def get_hydro_results_root(repo_root: Path) -> Path:
    """
    Return hydro_results directory.

    Optional override:
        HYDRO_RESULTS_ROOT=/path/to/hydro_results
    """

    return Path(
        os.environ.get(
            "HYDRO_RESULTS_ROOT",
            repo_root / "hydro_results",
        )
    ).resolve()


# ============================================================
# SETTINGS
# ============================================================

HISTORICAL_YEARS = [2015, 2016, 2017, 2018, 2019]

HISTORICAL_NETWORK_FILE = "base_s_100_elec_.nc"
FUTURE_NETWORK_FILE = "base_s_100___2050.nc"

SCENARIOS_HISTORICAL = {
    "pypsa": {
        "target_prefix": "PyPSA",
        "source_candidates": [
            "Europe_{year}",
        ],
    },
    "glofas_saber": {
        "target_prefix": "GloFAS-SABER",
        "source_candidates": [
            "Europe_{year}_GloFAS_SABER",
        ],
    },
    "efas_saber": {
        "target_prefix": "EFAS-SABER",
        "source_candidates": [
            "Europe_{year}_EFAS_SABER",
        ],
    },
}

SCENARIOS_2050 = {
    "pypsa": {
        "target_dir": "PyPSA_2050",
        "source_candidates": [
            "PyPSA_Europe_2050",
        ],
    },
    "glofas_saber": {
        "target_dir": "GloFAS-SABER_2050",
        "source_candidates": [
            "GloFAS_Europe_2050",
        ],
    },
    "efas_saber": {
        "target_dir": "EFAS-SABER_2050",
        "source_candidates": [
            "EFAS_Europe_2050",
        ],
    },
}


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect PyPSA network outputs and intermediate files into "
            "hydro_results, then generate all paper figures."
        )
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=(
            "Repository root. Default: inferred from this script location, "
            "or HYDRO_REPO_ROOT if set."
        ),
    )

    parser.add_argument(
        "--hydro-results",
        type=Path,
        default=None,
        help=(
            "hydro_results directory. Default: repo_root/hydro_results, "
            "or HYDRO_RESULTS_ROOT if set."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files already present in hydro_results.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without copying files or running scripts.",
    )

    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Do not copy input files, only run processing scripts.",
    )

    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Only collect input files, do not run processing scripts.",
    )

    return parser.parse_args()


# ============================================================
# FILE HELPERS
# ============================================================

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_file(
    source: Path,
    destination: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing source file: {source}")

    if destination.exists() and not overwrite:
        logger.info("Already exists, skipping: %s", destination)
        return

    ensure_parent(destination)

    if dry_run:
        action = "overwrite" if destination.exists() else "copy"
        logger.info("[dry-run:%s] %s -> %s", action, source, destination)
        return

    shutil.copy2(source, destination)
    logger.info("Copied: %s -> %s", source, destination)


def find_first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def copy_from_candidates(
    candidates: list[Path],
    destination: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    if destination.exists() and not overwrite:
        logger.info("Already exists in hydro_results, skipping: %s", destination)
        return

    source = find_first_existing(candidates)

    if source is None:
        candidate_list = "\n".join(str(path) for path in candidates)

        raise FileNotFoundError(
            f"No valid source found for target:\n"
            f"{destination}\n\n"
            f"Tried:\n{candidate_list}"
        )

    copy_file(
        source=source,
        destination=destination,
        overwrite=overwrite,
        dry_run=dry_run,
    )


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


# ============================================================
# SOURCE CANDIDATES
# ============================================================

def build_network_source_candidates(
    repo_root: Path,
    scenario_folder_candidates: list[str],
    network_filename: str,
) -> list[Path]:
    results_dir = repo_root / "results"

    return [
        results_dir / folder / "networks" / network_filename
        for folder in scenario_folder_candidates
    ]


def build_resource_source_candidates(
    repo_root: Path,
    scenario_folder_candidates: list[str],
    filename: str,
) -> list[Path]:
    resources_dir = repo_root / "resources"
    results_dir = repo_root / "results"

    candidates = []

    for folder in scenario_folder_candidates:
        candidates.extend(
            [
                resources_dir / folder / filename,
                results_dir / folder / filename,
                results_dir / folder / "resources" / filename,
            ]
        )

    return candidates


# ============================================================
# COLLECT HISTORICAL INPUTS
# ============================================================

def collect_historical_networks(
    repo_root: Path,
    hydro_results: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    for year in HISTORICAL_YEARS:
        for _, scenario_cfg in SCENARIOS_HISTORICAL.items():
            source_folders = [
                template.format(year=year)
                for template in scenario_cfg["source_candidates"]
            ]

            target_file = (
                hydro_results
                / f"{scenario_cfg['target_prefix']}_{year}"
                / HISTORICAL_NETWORK_FILE
            )

            candidates = build_network_source_candidates(
                repo_root=repo_root,
                scenario_folder_candidates=source_folders,
                network_filename=HISTORICAL_NETWORK_FILE,
            )

            copy_from_candidates(
                candidates=candidates,
                destination=target_file,
                overwrite=overwrite,
                dry_run=dry_run,
            )


def collect_historical_intermediate_files(
    repo_root: Path,
    hydro_results: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    year = 2019

    pypsa_folders = [
        template.format(year=year)
        for template in SCENARIOS_HISTORICAL["pypsa"]["source_candidates"]
    ]

    efas_folders = [
        template.format(year=year)
        for template in SCENARIOS_HISTORICAL["efas_saber"]["source_candidates"]
    ]

    copy_from_candidates(
        candidates=build_resource_source_candidates(
            repo_root=repo_root,
            scenario_folder_candidates=pypsa_folders,
            filename="country_shapes.geojson",
        ),
        destination=hydro_results / "PyPSA_2019" / "country_shapes.geojson",
        overwrite=overwrite,
        dry_run=dry_run,
    )

    copy_from_candidates(
        candidates=build_resource_source_candidates(
            repo_root=repo_root,
            scenario_folder_candidates=efas_folders,
            filename="powerplants_s_100.csv",
        ),
        destination=hydro_results / "EFAS-SABER_2019" / "powerplants_s_100.csv",
        overwrite=overwrite,
        dry_run=dry_run,
    )

    copy_from_candidates(
        candidates=build_resource_source_candidates(
            repo_root=repo_root,
            scenario_folder_candidates=efas_folders,
            filename="regions_onshore_base_s_100.geojson",
        ),
        destination=(
            hydro_results
            / "EFAS-SABER_2019"
            / "regions_onshore_base_s_100.geojson"
        ),
        overwrite=overwrite,
        dry_run=dry_run,
    )


# ============================================================
# COLLECT 2050 INPUTS
# ============================================================

def collect_2050_networks(
    repo_root: Path,
    hydro_results: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    for _, scenario_cfg in SCENARIOS_2050.items():
        target_file = (
            hydro_results
            / scenario_cfg["target_dir"]
            / FUTURE_NETWORK_FILE
        )

        candidates = build_network_source_candidates(
            repo_root=repo_root,
            scenario_folder_candidates=scenario_cfg["source_candidates"],
            network_filename=FUTURE_NETWORK_FILE,
        )

        copy_from_candidates(
            candidates=candidates,
            destination=target_file,
            overwrite=overwrite,
            dry_run=dry_run,
        )


def collect_2050_intermediate_files(
    repo_root: Path,
    hydro_results: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    pypsa_folders = SCENARIOS_2050["pypsa"]["source_candidates"]

    copy_from_candidates(
        candidates=build_resource_source_candidates(
            repo_root=repo_root,
            scenario_folder_candidates=pypsa_folders,
            filename="country_shapes.geojson",
        ),
        destination=hydro_results / "PyPSA_2050" / "country_shapes.geojson",
        overwrite=overwrite,
        dry_run=dry_run,
    )

    copy_from_candidates(
        candidates=build_resource_source_candidates(
            repo_root=repo_root,
            scenario_folder_candidates=pypsa_folders,
            filename="powerplants_s_100.csv",
        ),
        destination=hydro_results / "PyPSA_2050" / "powerplants_s_100.csv",
        overwrite=overwrite,
        dry_run=dry_run,
    )


# ============================================================
# COLLECT ALL INPUTS
# ============================================================

def collect_all_inputs(
    repo_root: Path,
    hydro_results: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    logger.info("=" * 100)
    logger.info("Collecting historical network files")
    logger.info("=" * 100)

    collect_historical_networks(
        repo_root=repo_root,
        hydro_results=hydro_results,
        overwrite=overwrite,
        dry_run=dry_run,
    )

    logger.info("=" * 100)
    logger.info("Collecting historical intermediate files")
    logger.info("=" * 100)

    collect_historical_intermediate_files(
        repo_root=repo_root,
        hydro_results=hydro_results,
        overwrite=overwrite,
        dry_run=dry_run,
    )

    logger.info("=" * 100)
    logger.info("Collecting 2050 network files")
    logger.info("=" * 100)

    collect_2050_networks(
        repo_root=repo_root,
        hydro_results=hydro_results,
        overwrite=overwrite,
        dry_run=dry_run,
    )

    logger.info("=" * 100)
    logger.info("Collecting 2050 intermediate files")
    logger.info("=" * 100)

    collect_2050_intermediate_files(
        repo_root=repo_root,
        hydro_results=hydro_results,
        overwrite=overwrite,
        dry_run=dry_run,
    )


# ============================================================
# PROCESSING SCRIPTS
# ============================================================

def get_processing_scripts(hydro_results: Path) -> list[Path]:
    return [
        hydro_results / "processing_historical.py",
        hydro_results / "processing_2050.py",
        hydro_results / "processing_hydro_representation.py",
    ]


def run_processing_script(
    script_path: Path,
    hydro_results: Path,
    repo_root: Path,
    dry_run: bool = False,
) -> None:
    require_file(script_path, "processing script")

    command = [
        sys.executable,
        str(script_path),
        "--hydro-results",
        str(hydro_results),
    ]

    env = os.environ.copy()
    env["HYDRO_REPO_ROOT"] = str(repo_root)
    env["HYDRO_RESULTS_ROOT"] = str(hydro_results)

    if dry_run:
        logger.info("[dry-run:run] %s", " ".join(command))
        return

    logger.info("=" * 100)
    logger.info("Running: %s", script_path.name)
    logger.info("=" * 100)

    subprocess.run(
        command,
        check=True,
        cwd=str(repo_root),
        env=env,
    )


def run_all_processing_scripts(
    repo_root: Path,
    hydro_results: Path,
    dry_run: bool = False,
) -> None:
    scripts = get_processing_scripts(hydro_results)

    for script in scripts:
        require_file(script, "processing script")

    for script in scripts:
        run_processing_script(
            script_path=script,
            hydro_results=hydro_results,
            repo_root=repo_root,
            dry_run=dry_run,
        )


# ============================================================
# STATUS
# ============================================================

def log_final_outputs(hydro_results: Path) -> None:
    expected_dirs = [
        hydro_results / "images" / "historical",
        hydro_results / "images" / "2050",
        hydro_results / "images" / "representation",
    ]

    logger.info("=" * 100)
    logger.info("Post-processing completed")
    logger.info("=" * 100)

    for directory in expected_dirs:
        logger.info("Output directory: %s", directory)

        if directory.exists():
            for image in sorted(directory.glob("*.png")):
                logger.info("  - %s", image.name)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    setup_logging()

    args = parse_args()

    repo_root = (
        args.repo_root.resolve()
        if args.repo_root is not None
        else get_repo_root()
    )

    hydro_results = (
        args.hydro_results.resolve()
        if args.hydro_results is not None
        else get_hydro_results_root(repo_root)
    )

    require_file(repo_root, "repository root")

    hydro_results.mkdir(parents=True, exist_ok=True)

    logger.info("Repository root: %s", repo_root)
    logger.info("Hydro results root: %s", hydro_results)

    if not args.skip_copy:
        collect_all_inputs(
            repo_root=repo_root,
            hydro_results=hydro_results,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

    if not args.skip_processing:
        run_all_processing_scripts(
            repo_root=repo_root,
            hydro_results=hydro_results,
            dry_run=args.dry_run,
        )

    if not args.dry_run:
        log_final_outputs(hydro_results)


if __name__ == "__main__":
    main()