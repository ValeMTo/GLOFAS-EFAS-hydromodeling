from __future__ import annotations

import argparse
import logging
from pathlib import Path

from hydro_inflow.prepare_hydro_inputs import prepare_and_check_hydro_inputs
from hydro_inflow.run_hydro_inflow_framework import run_hydro_inflow_framework
from hydro_inflow.utils import setup_logging
from postprocessing_workflow.run_postprocessing import run_postprocessing
from pypsa_workflow.prepare_cutouts import prepare_cutouts
from pypsa_workflow.run_pypsa_scenarios import run_pypsa_scenarios


logger = logging.getLogger(__name__)


WORKFLOW_STEPS = [
    "prepare-hydro-inputs",
    "run-hydro-inflow",
    "prepare-cutouts",
    "run-pypsa-scenarios",
    "run-postprocessing",
]


def select_workflow_steps(
    from_step: str | None,
    to_step: str | None,
) -> list[str]:
    steps = WORKFLOW_STEPS

    if from_step is not None:
        if from_step not in steps:
            available = "\n".join(f"  - {step}" for step in steps)
            raise ValueError(
                f"Unknown --from-step: {from_step}\n"
                f"Available steps:\n{available}"
            )

        steps = steps[steps.index(from_step):]

    if to_step is not None:
        if to_step not in steps:
            available = "\n".join(f"  - {step}" for step in steps)
            raise ValueError(
                f"Unknown --to-step: {to_step}\n"
                f"Available steps in selected range:\n{available}"
            )

        steps = steps[: steps.index(to_step) + 1]

    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete GloFAS/EFAS hydromodeling workflow: "
            "prepare hydro inputs, run hydro inflow preprocessing, run PyPSA-Eur, "
            "and postprocess the results."
        )
    )

    # ============================================================
    # General workflow control
    # ============================================================

    parser.add_argument(
        "--from-step",
        choices=WORKFLOW_STEPS,
        default=None,
        help="Optional workflow step to start from.",
    )

    parser.add_argument(
        "--to-step",
        choices=WORKFLOW_STEPS,
        default=None,
        help="Optional workflow step to stop after.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print workflow actions without executing external/expensive steps.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Use DEBUG logging.",
    )

    # ============================================================
    # Hydro input preparation options
    # ============================================================

    parser.add_argument(
        "--glofas-years",
        default="1980:2025",
        help="GloFAS years to download. Example: '1980:2025', '2021' or '2020,2021'.",
    )

    parser.add_argument(
        "--efas-years",
        default="1992:2025",
        help="EFAS years to download/process. Example: '1992:2025', '2021' or '2020,2021'.",
    )

    parser.add_argument(
        "--entsoe-years",
        default="2015:2019",
        help="ENTSO-E hydro production years to download. Example: '2015:2019', '2019' or '2018,2019'.",
    )

    parser.add_argument(
        "--skip-static",
        action="store_true",
        help="Debug option: skip static hydro inputs such as HydroBASINS and GloFAS uparea.",
    )

    parser.add_argument(
        "--skip-glofas",
        action="store_true",
        help="Debug option: skip GloFAS download.",
    )

    parser.add_argument(
        "--skip-efas",
        action="store_true",
        help="Debug option: skip EFAS download/process.",
    )

    parser.add_argument(
        "--skip-entsoe",
        action="store_true",
        help="Debug option: skip ENTSO-E hydro production download.",
    )

    parser.add_argument(
        "--overwrite-static",
        action="store_true",
        help="Overwrite static downloaded files if they already exist.",
    )

    parser.add_argument(
        "--overwrite-glohydrores",
        action="store_true",
        help="Overwrite the downloaded GloHydroRes source CSV if it already exists.",
    )

    # ============================================================
    # Hydro inflow framework options
    # ============================================================

    parser.add_argument(
        "--hydro-dataset",
        choices=["glofas", "efas", "all"],
        default="all",
        help="Hydrological dataset to process in the hydro inflow framework.",
    )

    parser.add_argument(
        "--hydro-start-from",
        default=None,
        help="Optional hydro inflow step script name to start from.",
    )

    parser.add_argument(
        "--hydro-stop-after",
        default=None,
        help="Optional hydro inflow step script name to stop after.",
    )

    # ============================================================
    # PyPSA cutouts and scenario options
    # ============================================================

    parser.add_argument(
        "--pypsa-group",
        choices=["historical", "2050", "all"],
        default="all",
        help="PyPSA scenario group to run.",
    )

    parser.add_argument(
        "--pypsa-only",
        default=None,
        help="Debug option: run only one PyPSA scenario or cutout, e.g. pypsa_2015 or cutout_2015.",
    )

    parser.add_argument(
        "--pypsa-cores",
        type=int,
        default=8,
        help="Number of Snakemake cores for PyPSA scenarios.",
    )

    parser.add_argument(
        "--cutout-cores",
        type=int,
        default=16,
        help="Number of Snakemake cores for cutout preparation.",
    )

    parser.add_argument(
        "--pypsa-force",
        action="store_true",
        help="Run PyPSA scenario/cutout even if expected output already exists.",
    )

    parser.add_argument(
        "--pypsa-snakemake-dryrun",
        action="store_true",
        help="Pass --dryrun to Snakemake for PyPSA workflow.",
    )

    parser.add_argument(
        "--pypsa-no-restore-config",
        action="store_true",
        help="Do not restore the original PyPSA-Eur active config at the end.",
    )

    parser.add_argument(
        "--pypsa-keep-going",
        action="store_true",
        help="Pass --keep-going to Snakemake for PyPSA scenarios.",
    )

    parser.add_argument(
        "--pypsa-no-rerun-incomplete",
        action="store_true",
        help="Do not pass --rerun-incomplete to Snakemake for PyPSA scenarios.",
    )

    parser.add_argument(
        "--pypsa-no-printshellcmds",
        action="store_true",
        help="Do not pass --printshellcmds to Snakemake for PyPSA scenarios.",
    )

    # ============================================================
    # Postprocessing options
    # ============================================================

    parser.add_argument(
        "--postprocessing-hydro-results",
        type=Path,
        default=None,
        help="Optional hydro_results directory. Default: repo_root/hydro_results.",
    )

    parser.add_argument(
        "--postprocessing-overwrite",
        action="store_true",
        help="Overwrite files already present in hydro_results during postprocessing.",
    )

    parser.add_argument(
        "--postprocessing-skip-copy",
        action="store_true",
        help="Debug option: do not copy PyPSA outputs, only run postprocessing scripts.",
    )

    parser.add_argument(
        "--postprocessing-skip-processing",
        action="store_true",
        help="Debug option: only collect PyPSA outputs, do not run postprocessing scripts.",
    )

    return parser.parse_args()


def run_complete_workflow(args: argparse.Namespace) -> None:
    selected_steps = select_workflow_steps(
        from_step=args.from_step,
        to_step=args.to_step,
    )

    logger.info("Selected workflow steps:")

    for index, step in enumerate(selected_steps, start=1):
        logger.info("  %s. %s", index, step)

    if args.dry_run:
        logger.info("Dry-run mode enabled.")

    for step in selected_steps:
        logger.info("")
        logger.info("#" * 100)
        logger.info("Workflow step: %s", step)
        logger.info("#" * 100)

        if step == "prepare-hydro-inputs":
            if args.dry_run:
                logger.info(
                    "[dry-run] Would prepare hydro inputs "
                    "(GloFAS years=%s, EFAS years=%s, ENTSO-E years=%s).",
                    args.glofas_years,
                    args.efas_years,
                    args.entsoe_years,
                )
                continue

            prepare_and_check_hydro_inputs(
                glofas_years=args.glofas_years,
                efas_years=args.efas_years,
                entsoe_years=args.entsoe_years,
                skip_static=args.skip_static,
                skip_glofas=args.skip_glofas,
                skip_efas=args.skip_efas,
                skip_entsoe=args.skip_entsoe,
                overwrite_static=args.overwrite_static,
                overwrite_glohydrores=args.overwrite_glohydrores,
            )

        elif step == "run-hydro-inflow":
            run_hydro_inflow_framework(
                dataset=args.hydro_dataset,
                start_from=args.hydro_start_from,
                stop_after=args.hydro_stop_after,
                dry_run=args.dry_run,
            )

        elif step == "prepare-cutouts":
            prepare_cutouts(
                cores=args.cutout_cores,
                only=args.pypsa_only,
                force=args.pypsa_force,
                dryrun=args.dry_run,
                snakemake_dryrun=args.pypsa_snakemake_dryrun,
                no_restore_config=args.pypsa_no_restore_config,
            )

        elif step == "run-pypsa-scenarios":
            run_pypsa_scenarios(
                group=args.pypsa_group,
                only=args.pypsa_only,
                cores=args.pypsa_cores,
                force=args.pypsa_force,
                snakemake_dryrun=args.pypsa_snakemake_dryrun,
                dryrun=args.dry_run,
                no_restore_config=args.pypsa_no_restore_config,
                keep_going=args.pypsa_keep_going,
                rerun_incomplete=not args.pypsa_no_rerun_incomplete,
                printshellcmds=not args.pypsa_no_printshellcmds,
                prepare_cutouts_first=False,
                cutout_cores=args.cutout_cores,
            )

        elif step == "run-postprocessing":
            run_postprocessing(
                overwrite=args.postprocessing_overwrite,
                dry_run=args.dry_run,
                skip_copy=args.postprocessing_skip_copy,
                skip_processing=args.postprocessing_skip_processing,
                hydro_results=args.postprocessing_hydro_results,
            )

        else:
            raise RuntimeError(f"Unhandled workflow step: {step}")

    logger.info("")
    logger.info("Selected workflow steps completed.")


def main() -> None:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    run_complete_workflow(args)


if __name__ == "__main__":
    main()