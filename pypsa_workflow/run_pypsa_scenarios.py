from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import logging
import shutil
import subprocess

from hydro_inflow.utils import get_repo_root, setup_logging
from pypsa_workflow.prepare_cutouts import prepare_cutouts


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Scenario:
    name: str
    config: Path
    target: str
    mode: str


def get_pypsa_root(repo_root: Path | None = None) -> Path:
    if repo_root is None:
        repo_root = get_repo_root()

    return repo_root / "external" / "pypsa-eur-hydro"


def run_command(command: list[str], cwd: Path, dryrun: bool = False) -> None:
    logger.info("=" * 80)
    logger.info("Running command: %s", " ".join(command))
    logger.info("Working directory: %s", cwd)
    logger.info("=" * 80)

    if dryrun:
        return

    subprocess.run(command, cwd=cwd, check=True)


def copy_config_to_active_config(
    scenario_config: Path,
    active_config: Path,
    dryrun: bool = False,
) -> None:
    if not scenario_config.exists():
        raise FileNotFoundError(f"Missing scenario config: {scenario_config}")

    active_config.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Activating PyPSA scenario config:")
    logger.info("  from: %s", scenario_config)
    logger.info("  to:   %s", active_config)

    if dryrun:
        return

    shutil.copy2(scenario_config, active_config)


def backup_active_config(active_config: Path, dryrun: bool = False) -> Path | None:
    if not active_config.exists():
        logger.warning("No active PyPSA config found to back up: %s", active_config)
        return None

    backup = active_config.with_suffix(
        active_config.suffix + ".before_run_pypsa_scenarios.bak"
    )

    logger.info("Backing up active PyPSA config:")
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

    logger.info("Restoring original active PyPSA config:")
    logger.info("  from: %s", backup_config)
    logger.info("  to:   %s", active_config)

    if not dryrun:
        shutil.copy2(backup_config, active_config)


def build_scenarios(repo_root: Path) -> list[Scenario]:
    scenarios: list[Scenario] = []

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
        "--configfile",
        str(scenario.config),
        "--nolock",
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
    pypsa_root: Path,
    active_config: Path,
    cores: int,
    force: bool,
    snakemake_dryrun: bool,
    keep_going: bool,
    rerun_incomplete: bool,
    printshellcmds: bool,
    dryrun: bool,
) -> None:
    target_path = pypsa_root / scenario.target

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

    if not scenario.config.exists():
        raise FileNotFoundError(f"Missing scenario config: {scenario.config}")

    logger.info("Using scenario config through Snakemake --configfile:")
    logger.info("  configfile: %s", scenario.config)
    logger.info("  active config is not modified")

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
        cwd=pypsa_root,
        dryrun=dryrun,
    )


def run_pypsa_scenarios(
    group: str = "historical",
    only: str | None = None,
    cores: int = 8,
    active_config_relative_path: str = "config/config.yaml",
    force: bool = False,
    snakemake_dryrun: bool = False,
    dryrun: bool = False,
    no_restore_config: bool = False,
    keep_going: bool = False,
    rerun_incomplete: bool = True,
    printshellcmds: bool = True,
    prepare_cutouts_first: bool = False,
    cutout_cores: int = 16,
) -> None:
    repo_root = get_repo_root()
    pypsa_root = get_pypsa_root(repo_root)

    if not pypsa_root.exists():
        raise FileNotFoundError(f"Missing PyPSA-Eur repository: {pypsa_root}")

    active_config = pypsa_root / active_config_relative_path

    scenarios = build_scenarios(repo_root)
    selected_scenarios = filter_scenarios(
        scenarios=scenarios,
        only=only,
        group=group,
    )

    logger.info("Repository root: %s", repo_root)
    logger.info("PyPSA-Eur root: %s", pypsa_root)
    logger.info("Active PyPSA config: %s", active_config)
    logger.info("Selected scenarios: %s", [scenario.name for scenario in selected_scenarios])

    if prepare_cutouts_first:
        prepare_cutouts(
            cores=cutout_cores,
            dryrun=dryrun,
            snakemake_dryrun=snakemake_dryrun,
        )

    backup_config = None
    logger.info(
        "Scenario configs are passed with Snakemake --configfile; active PyPSA config will not be modified."
    )

    try:
        for scenario in selected_scenarios:
            run_scenario(
                scenario=scenario,
                pypsa_root=pypsa_root,
                active_config=active_config,
                cores=cores,
                force=force,
                snakemake_dryrun=snakemake_dryrun,
                keep_going=keep_going,
                rerun_incomplete=rerun_incomplete,
                printshellcmds=printshellcmds,
                dryrun=dryrun,
            )
    finally:
        logger.info("No active PyPSA config restore needed; active config was not modified.")

    logger.info("Selected PyPSA-Eur scenarios completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PyPSA-Eur historical and 2050 scenarios sequentially."
    )

    parser.add_argument(
        "--prepare-cutouts",
        action="store_true",
        help="Run pypsa_workflow.prepare_cutouts before launching PyPSA-Eur scenarios.",
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
        help="Path to active PyPSA-Eur config relative to external/pypsa-eur-hydro.",
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
        help="Do not restore the original active PyPSA-Eur config at the end.",
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

    run_pypsa_scenarios(
        group=args.group,
        only=args.only,
        cores=args.cores,
        active_config_relative_path=args.active_config,
        force=args.force,
        snakemake_dryrun=args.snakemake_dryrun,
        dryrun=args.dryrun,
        no_restore_config=args.no_restore_config,
        keep_going=args.keep_going,
        rerun_incomplete=not args.no_rerun_incomplete,
        printshellcmds=not args.no_printshellcmds,
        prepare_cutouts_first=args.prepare_cutouts,
        cutout_cores=args.cutout_cores,
    )


if __name__ == "__main__":
    main()