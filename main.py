from __future__ import annotations

import argparse
import logging

from hydro_inflow.prepare_hydro_inputs import prepare_and_check_hydro_inputs
from hydro_inflow.run_hydro_inflow_framework import run_hydro_inflow_framework
from hydro_inflow.utils import setup_logging
from pypsa_workflow.prepare_cutouts import prepare_cutouts
from pypsa_workflow.run_pypsa_scenarios import run_pypsa_scenarios


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GloFAS/EFAS hydromodeling workflow."
    )

    # ============================================================
    # General
    # ============================================================

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Use DEBUG logging.",
    )

    # ============================================================
    # Hydro input preparation
    # ============================================================

    parser.add_argument(
        "--prepare-hydro-inputs",
        action="store_true",
        help="Prepare and check hydro input data.",
    )

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
        help="Skip static hydro inputs such as HydroBASINS and GloFAS uparea.",
    )

    parser.add_argument(
        "--skip-glofas",
        action="store_true",
        help="Skip GloFAS download.",
    )

    parser.add_argument(
        "--skip-efas",
        action="store_true",
        help="Skip EFAS download/process.",
    )

    parser.add_argument(
        "--skip-entsoe",
        action="store_true",
        help="Skip ENTSO-E hydro production download.",
    )

    parser.add_argument(
        "--skip-manual-check",
        action="store_true",
        help=(
            "Do not require manually provided files such as GRDC, GRanD, "
            "EFAS uparea and Electricity Maps CH files in the final check."
        ),
    )

    parser.add_argument(
        "--skip-final-check",
        action="store_true",
        help="Skip the final hydro input availability check.",
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
    # Hydro inflow framework
    # ============================================================

    parser.add_argument(
        "--run-hydro-inflow",
        action="store_true",
        help="Run the hydro inflow framework.",
    )

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

    parser.add_argument(
        "--hydro-dry-run",
        action="store_true",
        help="Print selected hydro inflow functions without executing them.",
    )

    # ============================================================
    # PyPSA cutouts and scenarios
    # ============================================================

    parser.add_argument(
        "--prepare-cutouts",
        action="store_true",
        help="Prepare PyPSA-Eur cutouts.",
    )

    parser.add_argument(
        "--run-pypsa-scenarios",
        action="store_true",
        help="Run PyPSA-Eur scenarios.",
    )

    parser.add_argument(
        "--pypsa-group",
        choices=["historical", "2050", "all"],
        default="historical",
        help="PyPSA scenario group to run.",
    )

    parser.add_argument(
        "--pypsa-only",
        default=None,
        help="Run only one PyPSA scenario or cutout, e.g. pypsa_2015 or cutout_2015.",
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
        "--pypsa-dryrun",
        action="store_true",
        help="Only print PyPSA workflow actions. Do not copy configs or run Snakemake.",
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    if args.prepare_hydro_inputs:
        prepare_and_check_hydro_inputs(
            glofas_years=args.glofas_years,
            efas_years=args.efas_years,
            entsoe_years=args.entsoe_years,
            skip_static=args.skip_static,
            skip_glofas=args.skip_glofas,
            skip_efas=args.skip_efas,
            skip_entsoe=args.skip_entsoe,
            skip_manual_check=args.skip_manual_check,
            skip_final_check=args.skip_final_check,
            overwrite_static=args.overwrite_static,
            overwrite_glohydrores=args.overwrite_glohydrores,
        )

    if args.run_hydro_inflow:
        run_hydro_inflow_framework(
            dataset=args.hydro_dataset,
            start_from=args.hydro_start_from,
            stop_after=args.hydro_stop_after,
            dry_run=args.hydro_dry_run,
        )

    if args.prepare_cutouts:
        prepare_cutouts(
            cores=args.cutout_cores,
            only=args.pypsa_only,
            force=args.pypsa_force,
            dryrun=args.pypsa_dryrun,
            snakemake_dryrun=args.pypsa_snakemake_dryrun,
            no_restore_config=args.pypsa_no_restore_config,
        )

    if args.run_pypsa_scenarios:
        run_pypsa_scenarios(
            group=args.pypsa_group,
            only=args.pypsa_only,
            cores=args.pypsa_cores,
            force=args.pypsa_force,
            snakemake_dryrun=args.pypsa_snakemake_dryrun,
            dryrun=args.pypsa_dryrun,
            no_restore_config=args.pypsa_no_restore_config,
            keep_going=args.pypsa_keep_going,
            rerun_incomplete=not args.pypsa_no_rerun_incomplete,
            printshellcmds=not args.pypsa_no_printshellcmds,
            prepare_cutouts_first=False,
            cutout_cores=args.cutout_cores,
        )

    if (
        not args.prepare_hydro_inputs
        and not args.run_hydro_inflow
        and not args.prepare_cutouts
        and not args.run_pypsa_scenarios
    ):
        logger.info(
            "No workflow step selected. Use --prepare-hydro-inputs, "
            "--run-hydro-inflow, --prepare-cutouts and/or --run-pypsa-scenarios."
        )


if __name__ == "__main__":
    main()