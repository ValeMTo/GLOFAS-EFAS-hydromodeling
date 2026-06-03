from __future__ import annotations

import argparse
import logging

from hydro_inflow.prepare_hydro_inputs import prepare_and_check_hydro_inputs
from hydro_inflow.utils import setup_logging


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GloFAS/EFAS hydromodeling workflow."
    )

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
    else:
        logger.info("No workflow step selected. Use --prepare-hydro-inputs.")


if __name__ == "__main__":
    main()