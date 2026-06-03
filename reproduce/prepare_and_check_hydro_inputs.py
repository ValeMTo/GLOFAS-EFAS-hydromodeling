from __future__ import annotations

from pathlib import Path
import argparse
import logging
import subprocess
import sys


logger = logging.getLogger(__name__)


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_command(command: list[str], cwd: Path) -> None:
    logger.info("=" * 80)
    logger.info("Running command: %s", " ".join(command))
    logger.info("=" * 80)

    subprocess.run(command, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare hydro inputs and check that required files are available."
    )

    parser.add_argument(
        "--skip-static",
        action="store_true",
        help="Forward --skip-static to prepare_hydro_inputs.py.",
    )

    parser.add_argument(
        "--skip-glofas",
        action="store_true",
        help="Forward --skip-glofas to prepare_hydro_inputs.py.",
    )

    parser.add_argument(
        "--skip-efas",
        action="store_true",
        help="Forward --skip-efas to prepare_hydro_inputs.py.",
    )

    parser.add_argument(
        "--skip-entsoe",
        action="store_true",
        help="Forward --skip-entsoe to prepare_hydro_inputs.py.",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )

    args = parse_args()
    repo_root = get_repo_root()

    prepare_cmd = [
        sys.executable,
        "reproduce/prepare_hydro_inputs.py",
    ]

    for flag in ["skip_static", "skip_glofas", "skip_efas", "skip_entsoe"]:
        if getattr(args, flag):
            prepare_cmd.append("--" + flag.replace("_", "-"))

    check_cmd = [
        sys.executable,
        "reproduce/check_hydro_inputs.py",
    ]

    run_command(prepare_cmd, cwd=repo_root)
    run_command(check_cmd, cwd=repo_root)

    logger.info("Hydro input preparation and check completed.")


if __name__ == "__main__":
    main()