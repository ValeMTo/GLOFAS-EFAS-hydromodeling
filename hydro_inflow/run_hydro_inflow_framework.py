from __future__ import annotations

from pathlib import Path
import argparse
import logging
import os
import subprocess
import sys

from hydro_inflow.utils import get_repo_root, setup_logging


logger = logging.getLogger(__name__)


DATASET_STEPS = {
    "glofas": [
        "glofas_to_drainage_network.py",
        "extract_glofas_series_for_hydro_plants.py",
        "grdc_to_drain_glofas.py",
        "GRanD_to_drain.py",
        "saber_preprocessing.py",
        "write_saber_config.py",
        "run_saber.py",
        "saber_to_pypsa.py",
    ],
    "efas": [
        "efas_to_drainage_network.py",
        "extract_efas_series_for_hydro_plants.py",
        "grdc_to_drain_efas.py",
        "GRanD_to_drain.py",
        "saber_preprocessing.py",
        "write_saber_config.py",
        "run_saber.py",
        "saber_to_pypsa.py",
    ],
}


def get_hydro_scripts_dir(repo_root: Path | None = None) -> Path:
    if repo_root is None:
        repo_root = get_repo_root()

    return repo_root / "hydro_inflow"


def get_saber_hbc_dir(repo_root: Path | None = None) -> Path:
    if repo_root is None:
        repo_root = get_repo_root()

    return repo_root / "external" / "saber_hbc"


def build_environment(dataset: str, repo_root: Path | None = None) -> dict[str, str]:
    if repo_root is None:
        repo_root = get_repo_root()

    hydro_scripts_dir = get_hydro_scripts_dir(repo_root)
    saber_hbc_dir = get_saber_hbc_dir(repo_root)

    env = os.environ.copy()

    env["HYDRO_DATASET"] = dataset
    env["HYDRO_REPO_ROOT"] = str(repo_root)
    env["HYDRO_SCRIPTS_DIR"] = str(hydro_scripts_dir)

    pythonpath_parts = [str(repo_root)]

    if saber_hbc_dir.exists():
        pythonpath_parts.append(str(saber_hbc_dir))
    else:
        logger.warning("SABER package directory not found: %s", saber_hbc_dir)

    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    logger.debug("HYDRO_DATASET = %s", env["HYDRO_DATASET"])
    logger.debug("HYDRO_REPO_ROOT = %s", env["HYDRO_REPO_ROOT"])
    logger.debug("HYDRO_SCRIPTS_DIR = %s", env["HYDRO_SCRIPTS_DIR"])
    logger.debug("PYTHONPATH = %s", env.get("PYTHONPATH", ""))

    return env


def check_required_files(dataset: str, repo_root: Path | None = None) -> None:
    if repo_root is None:
        repo_root = get_repo_root()

    hydro_scripts_dir = get_hydro_scripts_dir(repo_root)

    missing = []

    if not hydro_scripts_dir.exists():
        missing.append(hydro_scripts_dir)

    for script_name in DATASET_STEPS[dataset]:
        script_path = hydro_scripts_dir / script_name

        if not script_path.exists():
            missing.append(script_path)

    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing required hydro inflow framework files/directories:\n"
            f"{missing_text}"
        )

    logger.debug("All framework scripts are present for dataset: %s", dataset)


def select_steps(
    dataset: str,
    start_from: str | None,
    stop_after: str | None,
) -> list[str]:
    steps = DATASET_STEPS[dataset]

    if start_from is not None:
        if start_from not in steps:
            available = "\n".join(f"  - {step}" for step in steps)
            raise ValueError(
                f"Unknown start step: {start_from}\n"
                f"Available steps for {dataset}:\n{available}"
            )

        steps = steps[steps.index(start_from):]

    if stop_after is not None:
        if stop_after not in steps:
            available = "\n".join(f"  - {step}" for step in steps)
            raise ValueError(
                f"Unknown stop step: {stop_after}\n"
                f"Available steps for selected range:\n{available}"
            )

        steps = steps[: steps.index(stop_after) + 1]

    return steps


def run_step(
    script_name: str,
    dataset: str,
    repo_root: Path | None = None,
    dry_run: bool = False,
) -> None:
    if repo_root is None:
        repo_root = get_repo_root()

    hydro_scripts_dir = get_hydro_scripts_dir(repo_root)
    script_path = hydro_scripts_dir / script_name
    env = build_environment(dataset=dataset, repo_root=repo_root)

    command = [sys.executable, str(script_path)]

    logger.info("=" * 90)
    logger.info("Step: %s", script_name)
    logger.info("Command: %s", " ".join(command))
    logger.info("Working directory: %s", hydro_scripts_dir)
    logger.info("HYDRO_DATASET: %s", dataset)
    logger.info("=" * 90)

    if dry_run:
        logger.info("Dry run: step not executed.")
        return

    try:
        subprocess.run(
            command,
            cwd=hydro_scripts_dir,
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        logger.error("Step failed: %s", script_name)
        logger.error("Return code: %s", error.returncode)
        raise


def run_hydro_inflow_framework(
    dataset: str = "all",
    start_from: str | None = None,
    stop_after: str | None = None,
    dry_run: bool = False,
) -> None:
    datasets = ["glofas", "efas"] if dataset == "all" else [dataset]
    repo_root = get_repo_root()

    logger.info("Selected hydro datasets: %s", datasets)

    for current_dataset in datasets:
        logger.info("")
        logger.info("#" * 90)
        logger.info("Starting hydro inflow workflow for dataset: %s", current_dataset)
        logger.info("#" * 90)

        check_required_files(
            dataset=current_dataset,
            repo_root=repo_root,
        )

        selected_steps = select_steps(
            dataset=current_dataset,
            start_from=start_from,
            stop_after=stop_after,
        )

        logger.info("Hydro inflow framework")
        logger.info("Dataset: %s", current_dataset)
        logger.info("Repository root: %s", repo_root)
        logger.info("Hydro scripts directory: %s", get_hydro_scripts_dir(repo_root))

        logger.info("Selected steps:")

        for index, step in enumerate(selected_steps, start=1):
            logger.info("  %s. %s", index, step)

        if dry_run:
            logger.info("Dry run mode: commands will be printed but not executed.")

        for step in selected_steps:
            run_step(
                script_name=step,
                dataset=current_dataset,
                repo_root=repo_root,
                dry_run=dry_run,
            )

        logger.info("Hydro inflow framework completed for dataset: %s", current_dataset)

    logger.info("Selected hydro inflow workflows completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the hydro inflow preprocessing framework."
    )

    parser.add_argument(
        "--dataset",
        choices=["glofas", "efas", "all"],
        default="all",
        help="Hydrological dataset to process.",
    )

    parser.add_argument(
        "--start-from",
        default=None,
        help="Optional script name to start from.",
    )

    parser.add_argument(
        "--stop-after",
        default=None,
        help="Optional script name to stop after.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected commands without executing them.",
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

    run_hydro_inflow_framework(
        dataset=args.dataset,
        start_from=args.start_from,
        stop_after=args.stop_after,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()