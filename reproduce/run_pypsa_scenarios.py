from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import logging
import shutil
import subprocess
import sys


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Scenario:
    name: str
    config: Path
    target: str
    mode: str


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_command(command: list[str], cwd: Path, dryrun: bool = False) -> None:
    logger.info("=" * 80)
    logger.info("Running command: %s", " ".join(command))
    logger.info("=" * 80)

    if dryrun:
        return

    subprocess.run(command, cwd=cwd, check=True)

def prepare_cutouts(
    repo_root: Path,
    cores: int,
    dryrun: bool,
    snakemake_dryrun: bool,
) -> None:
    script_path = repo_root / "reproduce" / "prepare_cutouts.py"

    if not script_path.exists():
        raise FileNotFoundError(f"Missing cutout preparation script: {script_path}")

    command = [
        sys.executable,
        str(script_path),
        "--cores",
        str(cores),
    ]

    if dryrun:
        command.append("--dryrun")

    if snakemake_dryrun:
        command.append("--snakemake-dryrun")

    run_command(
        command=command,
        cwd=repo_root,
        dryrun=False,
    )

def copy_config_to_active_config(
    scenario_config: Path,
    active_config: Path,
    dryrun: bool = False,
) -> None:
    if not scenario_config.exists():
        raise FileNotFoundError(f"Missing scenario config: {scenario_config}")

    active_config.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Activating scenario config:")
    logger.info("  from: %s", scenario_config)
    logger.info("  to:   %s", active_config)

    if dryrun:
        return

    shutil.copy2(scenario_config, active_config)


def backup_active_config(active_config: Path, dryrun: bool = False) -> Path | None:
    if not active_config.exists():
        logger.warning("No active config found to back up: %s", active_config)
        return None

    backup = active_config.with_suffix(active_config.suffix + ".before_run_pypsa_scenarios.bak")

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
    if backup_config is None:
        return

    if not backup_config.exists():
        logger.warning("Backup config does not exist, cannot restore: %s", backup_config)
        return

    logger.info("Restoring original active config:")
    logger.info("  from: %s", backup_config)
    logger.info("  to:   %s", active_config)

    if not dryrun:
        shutil.copy2(backup_config, active_config)


def build_scenarios(repo_root: Path) -> list[Scenario]:
    scenarios: list[Scenario] = []

    # Historical electricity-only runs.
    for year in range(2015, 2020):
        scenarios.append(
            Scenario(
                name=f"pypsa_{year}",
                config=repo_root / "config" / "historical" / f"pypsa_{year}.yaml",
                target=f"results/Europe_{year}/networks/base_s_100_elec_.nc",
                mode="historical",
            )
        )

    for year in range(2015, 2020):
        scenarios.append(
            Scenario(
                name=f"glofas_{year}",
                config=repo_root / "config" / "historical" / f"glofas_{year}.yaml",
                target=f"results/Europe_{year}_GloFAS_SABER/networks/base_s_100_elec_.nc",
                mode="historical",
            )
        )

    for year in range(2015, 2020):
        scenarios.append(
            Scenario(
                name=f"efas_{year}",
                config=repo_root / "config" / "historical" / f"efas_{year}.yaml",
                target=f"results/Europe_{year}_EFAS_SABER/networks/base_s_100_elec_.nc",
                mode="historical",
            )
        )

    # 2050 sector-coupled runs.
    scenarios.extend(
        [
            Scenario(
                name="pypsa_2050",
                config=repo_root / "config" / "planning_2050" / "pypsa_2050.yaml",
                target="results/PyPSA_Europe_2050/networks/base_s_100___2050.nc",
                mode="sector_2050",
            ),
            Scenario(
                name="glofas_2050",
                config=repo_root / "config" / "planning_2050" / "glofas_2050.yaml",
                target="results/GloFAS_Europe_2050/networks/base_s_100___2050.nc",
                mode="sector_2050",
            ),
            Scenario(
                name="efas_2050",
                config=repo_root / "config" / "planning_2050" / "efas_2050.yaml",
                target="results/EFAS_Europe_2050/networks/base_s_100___2050.nc",
                mode="sector_2050",
            ),
        ]
    )

    return scenarios


def filter_scenarios(
    scenarios: list[Scenario],
    only: str | None,
    group: str,
) -> list[Scenario]:
    if only:
        selected = [scenario for scenario in scenarios if scenario.name == only]

        if not selected:
            available = ", ".join(scenario.name for scenario in scenarios)
            raise ValueError(
                f"Unknown scenario {only!r}. Available scenarios: {available}"
            )

        return selected

    if group == "historical":
        return [scenario for scenario in scenarios if scenario.mode == "historical"]

    if group == "2050":
        return [scenario for scenario in scenarios if scenario.mode == "sector_2050"]

    if group == "all":
        return scenarios

    raise ValueError(f"Unknown group: {group}")


def build_snakemake_command(
    scenario: Scenario,
    cores: int,
    snakemake_dryrun: bool,
    keep_going: bool,
    rerun_incomplete: bool,
    printshellcmds: bool,
) -> list[str]:
    command = [
        "snakemake",
        "--cores",
        str(cores),
    ]

    if rerun_incomplete:
        command.append("--rerun-incomplete")

    if printshellcmds:
        command.append("--printshellcmds")

    if keep_going:
        command.append("--keep-going")

    if scenario.mode == "historical":
        command.append(scenario.target)
    elif scenario.mode == "sector_2050":
        command.append("solve_sector_networks")
    else:
        raise ValueError(f"Unknown scenario mode: {scenario.mode}")

    if snakemake_dryrun:
        command.append("--dryrun")

    return command


def run_scenario(
    scenario: Scenario,
    repo_root: Path,
    active_config: Path,
    cores: int,
    force: bool,
    snakemake_dryrun: bool,
    keep_going: bool,
    rerun_incomplete: bool,
    printshellcmds: bool,
    dryrun: bool,
) -> None:
    target_path = repo_root / scenario.target

    logger.info("")
    logger.info("#" * 80)
    logger.info("Scenario: %s", scenario.name)
    logger.info("Config:   %s", scenario.config)
    logger.info("Target:   %s", target_path)
    logger.info("Mode:     %s", scenario.mode)
    logger.info("#" * 80)

    if target_path.exists() and not force:
        logger.info("Target already exists, skipping scenario: %s", target_path)
        return

    copy_config_to_active_config(
        scenario_config=scenario.config,
        active_config=active_config,
        dryrun=dryrun,
    )

    command = build_snakemake_command(
        scenario=scenario,
        cores=cores,
        snakemake_dryrun=snakemake_dryrun,
        keep_going=keep_going,
        rerun_incomplete=rerun_incomplete,
        printshellcmds=printshellcmds,
    )

    run_command(
        command=command,
        cwd=repo_root,
        dryrun=dryrun,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PyPSA-Eur historical and 2050 scenarios sequentially."
    )

    parser.add_argument(
        "--prepare-cutouts",
        action="store_true",
        help="Run reproduce/prepare_cutouts.py before launching PyPSA-Eur scenarios.",
    )

    parser.add_argument(
        "--cutout-cores",
        type=int,
        default=16,
        help="Number of cores used for cutout preparation.",
    )
    
    parser.add_argument(
        "--group",
        choices=["historical", "2050", "all"],
        default="historical",
        help="Scenario group to run.",
    )

    parser.add_argument(
        "--only",
        default=None,
        help="Run only one scenario, e.g. pypsa_2015, glofas_2017, efas_2050.",
    )

    parser.add_argument(
        "--cores",
        type=int,
        default=8,
        help="Number of Snakemake cores.",
    )

    parser.add_argument(
        "--active-config",
        default="config/config.yaml",
        help="Path to the active PyPSA-Eur config file to overwrite.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Run scenario even if the expected target already exists.",
    )

    parser.add_argument(
        "--snakemake-dryrun",
        action="store_true",
        help="Pass --dryrun to Snakemake.",
    )

    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Only print what would be done. Do not copy configs or run Snakemake.",
    )

    parser.add_argument(
        "--no-restore-config",
        action="store_true",
        help="Do not restore the original active config at the end.",
    )

    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Pass --keep-going to Snakemake.",
    )

    parser.add_argument(
        "--no-rerun-incomplete",
        action="store_true",
        help="Do not pass --rerun-incomplete to Snakemake.",
    )

    parser.add_argument(
        "--no-printshellcmds",
        action="store_true",
        help="Do not pass --printshellcmds to Snakemake.",
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

    scenarios = build_scenarios(repo_root)
    selected_scenarios = filter_scenarios(
        scenarios=scenarios,
        only=args.only,
        group=args.group,
    )

    logger.info("Repository root: %s", repo_root)
    logger.info("Active config: %s", active_config)
    logger.info("Selected scenarios: %s", [scenario.name for scenario in selected_scenarios])

    if args.prepare_cutouts:
        prepare_cutouts(
            repo_root=repo_root,
            cores=args.cutout_cores,
            dryrun=args.dryrun,
            snakemake_dryrun=args.snakemake_dryrun,
        )

    backup_config = backup_active_config(
        active_config=active_config,
        dryrun=args.dryrun,
    )

    try:
        for scenario in selected_scenarios:
            run_scenario(
                scenario=scenario,
                repo_root=repo_root,
                active_config=active_config,
                cores=args.cores,
                force=args.force,
                snakemake_dryrun=args.snakemake_dryrun,
                keep_going=args.keep_going,
                rerun_incomplete=not args.no_rerun_incomplete,
                printshellcmds=not args.no_printshellcmds,
                dryrun=args.dryrun,
            )
    finally:
        if args.dryrun:
            logger.info("Dryrun mode: active config was not modified, skipping restore.")
        elif args.no_restore_config:
            logger.info("Leaving last scenario config active because --no-restore-config was set.")
        else:
            restore_active_config(
                backup_config=backup_config,
                active_config=active_config,
                dryrun=False,
            )

    logger.info("Selected PyPSA-Eur scenarios completed.")


if __name__ == "__main__":
    main()