from __future__ import annotations

from pathlib import Path
import argparse
import logging
import os
import subprocess
import sys

from hydro_inflow.logging_utils import setup_logging


logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[1]
HYDRO_SCRIPTS_DIR = REPO_ROOT / "scripts" / "hydro_inflow"
SABER_HBC_DIR = REPO_ROOT / "external" / "saber_hbc"


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


def build_environment(dataset: str) -> dict[str, str]:
    env = os.environ.copy()

    env["HYDRO_DATASET"] = dataset
    env["HYDRO_REPO_ROOT"] = str(REPO_ROOT)
    env["HYDRO_SCRIPTS_DIR"] = str(HYDRO_SCRIPTS_DIR)

    pythonpath_parts = []

    if SABER_HBC_DIR.exists():
        pythonpath_parts.append(str(SABER_HBC_DIR))
    else:
        logger.warning("SABER package directory not found: %s", SABER_HBC_DIR)

    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    if pythonpath_parts:
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    logger.debug("HYDRO_DATASET = %s", env["HYDRO_DATASET"])
    logger.debug("HYDRO_REPO_ROOT = %s", env["HYDRO_REPO_ROOT"])
    logger.debug("HYDRO_SCRIPTS_DIR = %s", env["HYDRO_SCRIPTS_DIR"])
    logger.debug("PYTHONPATH = %s", env.get("PYTHONPATH", ""))

    return env


def check_required_files(dataset: str) -> None:
    missing = []

    if not HYDRO_SCRIPTS_DIR.exists():
        missing.append(HYDRO_SCRIPTS_DIR)

    for script_name in DATASET_STEPS[dataset]:
        script_path = HYDRO_SCRIPTS_DIR / script_name
        if not script_path.exists():
            missing.append(script_path)

    if missing:
        logger.error("Missing required files or directories.")
        for path in missing:
            logger.error("Missing: %s", path)
        raise SystemExit(1)

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
            raise SystemExit(
                f"Unknown --start-from step: {start_from}\n"
                f"Available steps for {dataset}:\n{available}"
            )
        steps = steps[steps.index(start_from):]

    if stop_after is not None:
        if stop_after not in steps:
            available = "\n".join(f"  - {step}" for step in steps)
            raise SystemExit(
                f"Unknown --stop-after step: {stop_after}\n"
                f"Available steps for selected range:\n{available}"
            )
        steps = steps[: steps.index(stop_after) + 1]

    return steps


def run_step(script_name: str, env: dict[str, str], dry_run: bool) -> None:
    script_path = HYDRO_SCRIPTS_DIR / script_name
    command = [sys.executable, str(script_path)]

    logger.info("=" * 90)
    logger.info("Step: %s", script_name)
    logger.info("Command: %s", " ".join(command))
    logger.info("Working directory: %s", HYDRO_SCRIPTS_DIR)
    logger.info("HYDRO_DATASET: %s", env["HYDRO_DATASET"])
    logger.info("=" * 90)

    if dry_run:
        logger.info("Dry run: step not executed.")
        return

    try:
        subprocess.run(
            command,
            cwd=HYDRO_SCRIPTS_DIR,
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        logger.error("Step failed: %s", script_name)
        logger.error("Return code: %s", error.returncode)
        raise


def run_framework(
    dataset: str,
    start_from: str | None,
    stop_after: str | None,
    dry_run: bool,
) -> None:
    check_required_files(dataset)
    selected_steps = select_steps(dataset, start_from, stop_after)
    env = build_environment(dataset)

    logger.info("Hydro inflow framework")
    logger.info("Dataset: %s", dataset)
    logger.info("Repository root: %s", REPO_ROOT)
    logger.info("Hydro scripts directory: %s", HYDRO_SCRIPTS_DIR)

    logger.info("Selected steps:")
    for index, step in enumerate(selected_steps, start=1):
        logger.info("  %s. %s", index, step)

    if dry_run:
        logger.info("Dry run mode: commands will be printed but not executed.")

    for step in selected_steps:
        run_step(step, env=env, dry_run=dry_run)

    logger.info("Hydro inflow framework completed for dataset: %s", dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the hydro inflow preprocessing framework."
    )

    parser.add_argument(
        "--dataset",
        choices=["glofas", "efas", "all"],
        default="all",
        help=(
            "Hydrological dataset to process. "
            "Use 'all' to run GloFAS and EFAS sequentially. "
            "Default: all."
        ),
    )

    parser.add_argument(
        "--start-from",
        default=None,
        help=(
            "Optional script name to start from. "
            "When --dataset all is used, this is applied to both datasets."
        ),
    )

    parser.add_argument(
        "--stop-after",
        default=None,
        help=(
            "Optional script name to stop after. "
            "When --dataset all is used, this is applied to both datasets."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected commands without executing them.",
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()

    args = parse_args()

    datasets = ["glofas", "efas"] if args.dataset == "all" else [args.dataset]

    logger.info("Selected hydro datasets: %s", datasets)

    for dataset in datasets:
        logger.info("")
        logger.info("#" * 90)
        logger.info("Starting hydro inflow workflow for dataset: %s", dataset)
        logger.info("#" * 90)

        run_framework(
            dataset=dataset,
            start_from=args.start_from,
            stop_after=args.stop_after,
            dry_run=args.dry_run,
        )

    logger.info("Selected hydro inflow workflows completed.")


if __name__ == "__main__":
    main()