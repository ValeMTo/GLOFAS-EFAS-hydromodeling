from __future__ import annotations

from pathlib import Path
import logging

import yaml

from hydro_inflow.hydro_config import get_config
from hydro_inflow.utils import setup_logging


logger = logging.getLogger(__name__)


def build_saber_config(cfg: dict | None = None) -> dict:
    cfg = cfg or get_config()

    return {
        "workdir": str(cfg["saber_workdir"]),
        "cluster_data": str(cfg["cluster_data_path"]),
        "drain_table": str(cfg["drain_table_path"]),
        "gauge_table": str(cfg["gauge_table_path"]),
        "regulate_table": str(cfg["regulate_table_output_path"]),
        "drain_gis": str(cfg["drain_gis_output_path"]),
        "gauge_gis": str(cfg["gauge_gis_path"]),
        "gauge_data": str(cfg["gauge_data_dir"]),
        "hindcast_zarr": str(cfg["target_hindcast_zarr_path"]),
        "n_processes": int(cfg["saber_n_processes"]),
    }


def write_saber_config(cfg: dict | None = None) -> Path:
    cfg = cfg or get_config()

    saber_config = build_saber_config(cfg)

    out_path = Path(cfg["saber_config_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(saber_config, file, sort_keys=False)

    logger.info("SABER config written: %s", out_path)
    logger.debug("SABER config content: %s", saber_config)

    return out_path


def main() -> None:
    setup_logging()
    write_saber_config()


if __name__ == "__main__":
    main()