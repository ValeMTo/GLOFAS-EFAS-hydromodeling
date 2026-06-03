from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import logging
import shutil
import subprocess


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CutoutScenario:
    name: str
    config: Path
    build_target: str
    build_file: Path
    archive_file: Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def backup_active_config(active_config: Path, dryrun: bool = False) -> Path | None:
    if not active_config.exists():
        logger.warning("No active config found to back up: %s", active_config)
        return None

    backup = active_config.with_suffix(
        active_config.suffix + ".before_prepare_cutouts.bak"
    )

    logger.info("Backing up active config:")
    logger.info("  from: %s", active_config)
    logger.info("  to:   %s", backup)

    if not dryrun:
        shutil.copy2(active_config, backup)

    return backup


def restore_active_config(
    backup_config: Path | None,
    active_config: Path,
    dryrun: bool = False,
) -> None:
    if dryrun:
        logger.info("Dryrun mode: active config was not modified, skipping restore.")
        return

    if backup_config is None:
        return

    if not backup_config.exists():
        logger.warning("Backup config does not exist, cannot restore: %s", backup_config)
        return

    logger.info("Restoring original active config:")
    logger.info("  from: %s", backup_config)
    logger.info("  to:   %s", active_config)

    shutil.copy2(backup_config, active_config)


def copy_config_to_active_config(
    scenario_config: Path,
    active_config: Path,
    dryrun: bool = False,
) -> None:
    if not scenario_config.exists():
        raise FileNotFoundError(f"Missing cutout config: {scenario_config}")

    active_config.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Activating cutout config:")
    logger.info("  from: %s", scenario_config)
    logger.info("  to:   %s", active_config)

    if not dryrun:
        shutil.copy2(scenario_config, active_config)


def build_cutout_scenarios(repo_root: Path) -> list[CutoutScenario]:
    cutout_config_dir = repo_root / "config" / "cutouts"

    build_dir = repo_root / "data" / "cutout" / "build" / "unknown"
    archive_dir = repo_root / "data" / "cutout" / "archive" / "v1.0"

    scenarios: list[CutoutScenario] = []

    for year in range(2015, 2020):
        filename = f"europe-{year}-sarah3-era5.nc"

        scenarios.append(
            CutoutScenario(
                name=f"cutout_{year}",
                config=cutout_config_dir / f"cutout_{year}.yaml",
                build_target=f"data/cutout/build/unknown/{filename}",
                build_file=build_dir / filename,
                archive_file=archive_dir / filename,
            )
        )

    return scenarios


def run_command(command: list[str], cwd: Path, dryrun: bool = False) -> None:
    logger.info("=" * 80)
    logger.info("Running command: %s", " ".join(command))
    logger.info("=" * 80)

    if dryrun:
        return

    subprocess.run(command, cwd=cwd, check=True)
    
def ensure_cutout_tmpdir(repo_root: Path, dryrun: bool = False) -> None:
    tmpdir = repo_root / "cutouts_tmp"

    logger.info("Ensuring cutout temporary directory exists: %s", tmpdir)

    if not dryrun:
        tmpdir.mkdir(parents=True, exist_ok=True)

def copy_cutout_to_archive(
    build_file: Path,
    archive_file: Path,
    overwrite: bool = False,
    dryrun: bool = False,
) -> None:
    if not build_file.exists():
        raise FileNotFoundError(f"Built cutout does not exist: {build_file}")

    archive_file.parent.mkdir(parents=True, exist_ok=True)

    if archive_file.exists() and not overwrite:
        logger.info("Archived cutout already exists, keeping: %s", archive_file)
        return

    logger.info("Copying cutout to archive:")
    logger.info("  from: %s", build_file)
    logger.info("  to:   %s", archive_file)

    if not dryrun:
        shutil.copy2(build_file, archive_file)
        
def remove_file(path: Path, dryrun: bool = False) -> None:
    if not path.exists():
        logger.info("File already absent, nothing to clean: %s", path)
        return

    logger.info("Removing temporary file: %s", path)

    if not dryrun:
        path.unlink()


def clean_cutout_tmpdir(repo_root: Path, dryrun: bool = False) -> None:
    tmpdir = repo_root / "cutouts_tmp"

    if not tmpdir.exists():
        logger.info("Temporary cutout directory already absent: %s", tmpdir)
        return

    logger.info("Removing temporary cutout directory: %s", tmpdir)

    if not dryrun:
        shutil.rmtree(tmpdir)


def run_cutout_scenario(
    scenario: CutoutScenario,
    repo_root: Path,
    active_config: Path,
    cores: int,
    force: bool,
    dryrun: bool,
    snakemake_dryrun: bool,
) -> None:
    logger.info("")
    logger.info("#" * 80)
    logger.info("Cutout scenario: %s", scenario.name)
    logger.info("Config:       %s", scenario.config)
    logger.info("Build target: %s", scenario.build_target)
    logger.info("Build file:   %s", scenario.build_file)
    logger.info("Archive file: %s", scenario.archive_file)
    logger.info("#" * 80)

    if scenario.archive_file.exists() and not force:
        logger.info(
            "Archived cutout already exists, skipping scenario: %s",
            scenario.archive_file,
        )
        return

    copy_config_to_active_config(
        scenario_config=scenario.config,
        active_config=active_config,
        dryrun=dryrun,
    )

    command = [
        "snakemake",
        scenario.build_target,
        "--cores",
        str(cores),
        "--rerun-incomplete",
        "--printshellcmds",
    ]

    if snakemake_dryrun:
        command.append("--dryrun")
        
    ensure_cutout_tmpdir(
        repo_root=repo_root,
        dryrun=dryrun,
    )

    run_command(
        command=command,
        cwd=repo_root,
        dryrun=dryrun,
    )

    if snakemake_dryrun:
        logger.info("Snakemake dryrun enabled, skipping archive copy.")
        return

    copy_cutout_to_archive(
        build_file=scenario.build_file,
        archive_file=scenario.archive_file,
        overwrite=force,
        dryrun=dryrun,
    )

    remove_file(
        path=scenario.build_file,
        dryrun=dryrun,
    )

    clean_cutout_tmpdir(
        repo_root=repo_root,
        dryrun=dryrun,
    )


def filter_scenarios(
    scenarios: list[CutoutScenario],
    only: str | None,
) -> list[CutoutScenario]:
    if only is None:
        return scenarios

    selected = [scenario for scenario in scenarios if scenario.name == only]

    if not selected:
        available = ", ".join(scenario.name for scenario in scenarios)
        raise ValueError(f"Unknown cutout scenario {only!r}. Available: {available}")

    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build PyPSA-Eur cutouts sequentially and copy them from "
            "data/cutout/build/unknown to data/cutout/archive/v1.0."
        )
    )

    parser.add_argument(
        "--cores",
        type=int,
        default=16,
        help="Number of Snakemake cores.",
    )

    parser.add_argument(
        "--active-config",
        default="config/config.yaml",
        help="Path to active PyPSA-Eur config file.",
    )

    parser.add_argument(
        "--only",
        default=None,
        help="Run only one cutout scenario, e.g. cutout_2017.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild and recopy even if the archived cutout already exists.",
    )

    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Only print actions. Do not copy configs or run Snakemake.",
    )

    parser.add_argument(
        "--snakemake-dryrun",
        action="store_true",
        help="Pass --dryrun to Snakemake.",
    )

    parser.add_argument(
        "--no-restore-config",
        action="store_true",
        help="Do not restore the original config at the end.",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )

    args = parse_args()

    repo_root = get_repo_root()
    active_config = repo_root / args.active_config

    scenarios = filter_scenarios(
        scenarios=build_cutout_scenarios(repo_root),
        only=args.only,
    )

    logger.info("Repository root: %s", repo_root)
    logger.info("Active config: %s", active_config)
    logger.info("Selected cutout scenarios: %s", [scenario.name for scenario in scenarios])

    backup_config = backup_active_config(
        active_config=active_config,
        dryrun=args.dryrun,
    )

    try:
        for scenario in scenarios:
            run_cutout_scenario(
                scenario=scenario,
                repo_root=repo_root,
                active_config=active_config,
                cores=args.cores,
                force=args.force,
                dryrun=args.dryrun,
                snakemake_dryrun=args.snakemake_dryrun,
            )
    finally:
        if args.dryrun:
            logger.info("Dryrun mode: active config was not modified, skipping restore.")
        elif args.no_restore_config:
            logger.info(
                "Leaving last cutout config active because --no-restore-config was set."
            )
        else:
            restore_active_config(
                backup_config=backup_config,
                active_config=active_config,
                dryrun=False,
            )

    logger.info("Cutout preparation completed.")


if __name__ == "__main__":
    main()