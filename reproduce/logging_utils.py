from __future__ import annotations

import logging
import os

def setup_logging(default_level: str = "INFO") -> None:
    level_name = os.environ.get("HYDRO_LOG_LEVEL", default_level).upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )