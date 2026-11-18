from __future__ import annotations

from pathlib import Path
import argparse
import logging
import os
from collections.abc import Callable

from hydro_inflow.utils import get_repo_root, setup_logging

from hydro_inflow.glofas_to_drainage_network import build_glofas_drainage_network
from hydro_inflow.efas_to_drainage_network import build_efas_drainage_network
from hydro_inflow.extract_glofas_series_for_hydro_plants import (
    extract_glofas_series_for_hydro_plants,
)
from hydro_inflow.extract_efas_series_for_hydro_plants import (
    extract_efas_series_for_hydro_plants,
)
from hydro_inflow.grdc_to_drain_glofas import run_grdc_to_drain_glofas
from hydro_inflow.grdc_to_drain_efas import run_grdc_to_drain_efas
from hydro_inflow.GRanD_to_drain import run_grand_to_drain
from hydro_inflow.saber_preprocessing import run_saber_preprocessing
from hydro_inflow.write_saber_config import write_saber_config
from hydro_inflow.run_saber import run_saber_pipeline
from hydro_inflow.saber_to_pypsa import run_saber_to_pypsa


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

EXPANDING_STEPS = [
    "saber_preprocessing.py",
    "write_saber_config.py",
    "run_saber.py",
    "saber_to_pypsa.py",
]


PYTHON_STEP_FUNCTIONS: dict[str, dict[str, Callable[[], object]]] = {
    "glofas": {
        "glofas_to_drainage_network.py": build_glofas_drainage_network,
        "extract_glofas_series_for_hydro_plants.py": extract_glofas_series_for_hydro_plants,
        "grdc_to_drain_glofas.py": run_grdc_to_drain_glofas,
        "GRanD_to_drain.py": run_grand_to_drain,
        "saber_preprocessing.py": run_saber_preprocessing,
        "write_saber_config.py": write_saber_config,
        "run_saber.py": run_saber_pipeline,
        "saber_to_pypsa.py": run_saber_to_pypsa,
    },
    "efas": {
        "efas_to_drainage_network.py": build_efas_drainage_network,
        "extract_efas_series_for_hydro_plants.py": extract_efas_series_for_hydro_plants,
        "grdc_to_drain_efas.py": run_grdc_to_drain_efas,
        "GRanD_to_drain.py": run_grand_to_drain,
        "saber_preprocessing.py": run_saber_preprocessing,
        "write_saber_config.py": write_saber_config,
        "run_saber.py": run_saber_pipeline,
        "saber_to_pypsa.py": run_saber_to_pypsa,
    },
}


def get_hydro_scripts_dir(repo_root: Path | None = None) -> Path:
    if repo_root is None:
        repo_root = get_repo_root()

    return repo_root / "hydro_inflow"


def build_environment(dataset: str, repo_root: Path | None = None) -> dict[str, str]:
    if repo_root is None:
        repo_root = get_repo_root()

    hydro_scripts_dir = get_hydro_scripts_dir(repo_root)

    env = os.environ.copy()
    env["HYDRO_DATASET"] = dataset
    env["HYDRO_REPO_ROOT"] = str(repo_root)
    env["HYDRO_SCRIPTS_DIR"] = str(hydro_scripts_dir)

    existing_pythonpath = env.get("PYTHONPATH")
    pythonpath_parts = [str(repo_root)]

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

        if script_name not in PYTHON_STEP_FUNCTIONS[dataset]:
            missing.append(Path(f"Missing Python function mapping for {script_name}"))

    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing required hydro inflow framework files/functions:\n"
            f"{missing_text}"
        )

    logger.debug("All framework scripts/functions are present for dataset: %s", dataset)


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

    env = build_environment(dataset=dataset, repo_root=repo_root)
    step_function = PYTHON_STEP_FUNCTIONS[dataset][script_name]

    logger.info("=" * 90)
    logger.info("Step: %s", script_name)
    logger.info("Mode: Python function")
    logger.info("Function: %s.%s", step_function.__module__, step_function.__name__)
    logger.info("HYDRO_DATASET: %s", dataset)
    logger.info("=" * 90)

    if dry_run:
        logger.info("Dry run: function not executed.")
        return

    previous_dataset = os.environ.get("HYDRO_DATASET")
    previous_repo_root = os.environ.get("HYDRO_REPO_ROOT")
    previous_scripts_dir = os.environ.get("HYDRO_SCRIPTS_DIR")
    previous_pythonpath = os.environ.get("PYTHONPATH")

    os.environ.update(env)

    try:
        step_function()
    except Exception:
        logger.exception("Step failed: %s", script_name)
        raise
    finally:
        restore_environment_variable("HYDRO_DATASET", previous_dataset)
        restore_environment_variable("HYDRO_REPO_ROOT", previous_repo_root)
        restore_environment_variable("HYDRO_SCRIPTS_DIR", previous_scripts_dir)
        restore_environment_variable("PYTHONPATH", previous_pythonpath)


def config_years(dataset: str, key: str, env_name: str) -> list[int]:
    """Years from hydro_config, overridden by a space/comma list in env_name.

    An env_name set to an empty string gives no years, which skips the pass.
    """
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return [int(y) for y in env_value.replace(",", " ").split()]

    from hydro_inflow.hydro_config import get_config

    prev_dataset = os.environ.get("HYDRO_DATASET")
    os.environ["HYDRO_DATASET"] = dataset
    try:
        return [int(y) for y in get_config()[key]]
    finally:
        restore_environment_variable("HYDRO_DATASET", prev_dataset)


def run_saber_steps(
    dataset: str,
    repo_root: Path,
    calibration_end: int | None,
    inflow_years: list[int],
    output_suffix: str = "",
) -> None:
    """Run the four SABER steps with the calibration cutoff and output years
    passed through SABER_CALIBRATION_END / SABER_INFLOW_YEARS /
    SABER_OUTPUT_SUFFIX. calibration_end=None calibrates on the full period."""
    settings = {
        "SABER_CALIBRATION_END": None if calibration_end is None else str(calibration_end),
        "SABER_INFLOW_YEARS": " ".join(str(y) for y in inflow_years),
        "SABER_OUTPUT_SUFFIX": output_suffix or None,
    }
    previous = {name: os.environ.get(name) for name in settings}

    for name, value in settings.items():
        restore_environment_variable(name, value)

    try:
        for step in EXPANDING_STEPS:
            run_step(script_name=step, dataset=dataset, repo_root=repo_root)
    finally:
        for name, value in previous.items():
            restore_environment_variable(name, value)


def run_fullcal_saber(
    dataset: str,
    repo_root: Path,
    dry_run: bool = False,
) -> None:
    """Full-period SABER calibration for the 2050 climate years.

    The 2050 scenarios are projections, not validation, so their inflows use
    every observed year. Files are written as
    Europe_inflow_<year>_<DATASET>_SABER_FULLCAL.nc. This pass runs before the
    expanding-window one, so the shared SABER intermediates end up in the same
    state as without it.
    """
    years = config_years(dataset, "fullcal_inflow_years", "SABER_FULLCAL_YEARS")

    if not years:
        logger.info("No FULLCAL years selected, skipping the full-period calibration.")
        return

    logger.info("")
    logger.info("=" * 90)
    logger.info("Full-period calibration (FULLCAL) for years: %s", years)
    logger.info("=" * 90)

    if dry_run:
        logger.info(
            "[dry-run] Would run %s without calibration cutoff, "
            "SABER_INFLOW_YEARS=%s, SABER_OUTPUT_SUFFIX=_FULLCAL",
            EXPANDING_STEPS, years,
        )
        return

    run_saber_steps(
        dataset=dataset,
        repo_root=repo_root,
        calibration_end=None,
        inflow_years=years,
        output_suffix="_FULLCAL",
    )


def run_expanding_saber(
    dataset: str,
    repo_root: Path,
    dry_run: bool = False,
) -> None:
    inflow_years = config_years(dataset, "pypsa_inflow_years", "SABER_EXPANDING_YEARS")

    logger.info("Expanding-window calibration for years: %s", inflow_years)

    for year in inflow_years:
        calibration_end = year - 1

        logger.info("")
        logger.info("=" * 90)
        logger.info(
            "Expanding window: year %s (calibration <= %s)",
            year, calibration_end,
        )
        logger.info("=" * 90)

        if dry_run:
            logger.info(
                "[dry-run] Would run %s with SABER_CALIBRATION_END=%s, "
                "SABER_INFLOW_YEARS=%s",
                EXPANDING_STEPS, calibration_end, year,
            )
            continue

        run_saber_steps(
            dataset=dataset,
            repo_root=repo_root,
            calibration_end=calibration_end,
            inflow_years=[year],
        )


def restore_environment_variable(name: str, previous_value: str | None) -> None:
    if previous_value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous_value


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
            logger.info("Dry run mode: functions will be printed but not executed.")

        expanding_done = False
        for step in selected_steps:
            if step in EXPANDING_STEPS:
                if not expanding_done:
                    run_fullcal_saber(
                        dataset=current_dataset,
                        repo_root=repo_root,
                        dry_run=dry_run,
                    )
                    run_expanding_saber(
                        dataset=current_dataset,
                        repo_root=repo_root,
                        dry_run=dry_run,
                    )
                    expanding_done = True
                continue

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
        help="Print the selected functions without executing them.",
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