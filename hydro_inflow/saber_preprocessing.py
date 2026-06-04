# Final preprocessing of hydro plants and GRDC stations for SABER.
# Target points are defined by unique model_id, they are stations + plants.
# Cluster data are built from simulated/model discharge time series.

from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
import pandas as pd
import xarray as xr

from hydro_inflow.hydro_config import get_config, log_config_summary
from hydro_inflow.utils import setup_logging


logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

N_PCTL = 100
MIN_VALID_VALUES_FOR_CLUSTER = 100

# ============================================================
# HELPERS
# ============================================================

def normalize_model_id(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()

    fake_nulls = {
        "",
        "nan",
        "NaN",
        "None",
        "NONE",
        "null",
        "NULL",
        "<NA>",
    }

    return s.mask(s.isin(fake_nulls), pd.NA)


def get_discharge_variable(ds: xr.Dataset) -> str:
    """
    Return the available discharge variable from an input hindcast dataset.

    Upstream plant preprocessing may still save Qsim, while SABER/CYBER expects
    the final target hindcast to expose Qout. This function only selects the
    source variable; the final output is written as Qout.
    """
    if "Qout" in ds.data_vars:
        return "Qout"

    if "Qsim" in ds.data_vars:
        logger.warning("Plant zarr contains Qsim. Using it as source for final Qout.")
        return "Qsim"

    raise ValueError("Neither 'Qout' nor 'Qsim' found in plant hindcast zarr.")


def zscore_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    mean = np.nanmean(values)
    std = np.nanstd(values)

    if not np.isfinite(std) or std == 0:
        return np.zeros_like(values, dtype=float)

    return (values - mean) / std


# ============================================================
# TARGET MODEL MAP
# ============================================================

def build_target_model_map(
    plant_map_path: Path,
    gauge_table_path: Path,
    station_model_map_path: Path,
) -> pd.DataFrame:
    plants = pd.read_csv(
        plant_map_path,
        dtype=str,
        keep_default_na=False,
    )

    if "model_id" not in plants.columns:
        raise ValueError(f"'model_id' missing in plant map: {plant_map_path}")

    plants = plants[["model_id"]].copy()
    plants["model_id"] = normalize_model_id(plants["model_id"])

    plants = plants.dropna(subset=["model_id"])
    plants = plants.drop_duplicates(subset=["model_id"])

    plants["type"] = "plant"
    plants["gauge_id"] = pd.NA
    plants["station_name"] = pd.NA

    logger.debug("Plant target unique model_id: %s", len(plants))

    gauge_table = pd.read_csv(
        gauge_table_path,
        dtype=str,
        keep_default_na=False,
    )

    if not {"gauge_id", "model_id"}.issubset(gauge_table.columns):
        raise ValueError(
            f"gauge_table must contain gauge_id and model_id: {gauge_table_path}"
        )

    gauge_table = gauge_table[["gauge_id", "model_id"]].copy()
    gauge_table["gauge_id"] = gauge_table["gauge_id"].astype(str)
    gauge_table["model_id"] = normalize_model_id(gauge_table["model_id"])

    gauge_table = gauge_table.dropna(subset=["model_id"])

    station_map = pd.read_csv(
        station_model_map_path,
        dtype=str,
        keep_default_na=False,
    )

    metadata_cols = [
        col
        for col in [
            "gauge_id",
            "station_name",
            "dist_m",
            "area_rel_diff",
            "min_obs_per_month",
        ]
        if col in station_map.columns
    ]

    station_meta = station_map[metadata_cols].copy()

    if "gauge_id" in station_meta.columns:
        station_meta["gauge_id"] = station_meta["gauge_id"].astype(str)

    gauges = gauge_table.merge(
        station_meta,
        on="gauge_id",
        how="left",
    )

    if "station_name" not in gauges.columns:
        gauges["station_name"] = gauges["gauge_id"]

    gauges["station_name"] = gauges["station_name"].where(
        gauges["station_name"].astype(str).str.strip() != "",
        gauges["gauge_id"],
    )

    if "dist_m" in gauges.columns:
        gauges["dist_m"] = pd.to_numeric(gauges["dist_m"], errors="coerce")
    else:
        gauges["dist_m"] = np.nan

    if "area_rel_diff" in gauges.columns:
        gauges["area_rel_diff"] = pd.to_numeric(gauges["area_rel_diff"], errors="coerce")
    else:
        gauges["area_rel_diff"] = np.nan

    if "min_obs_per_month" in gauges.columns:
        gauges["min_obs_per_month"] = pd.to_numeric(
            gauges["min_obs_per_month"],
            errors="coerce",
        )
    else:
        gauges["min_obs_per_month"] = np.nan

    gauges = gauges.sort_values(
        ["model_id", "dist_m", "area_rel_diff", "gauge_id"],
        ascending=[True, True, True, True],
    )

    duplicate_gauge_model_ids = gauges["model_id"].duplicated(keep=False).sum()

    if duplicate_gauge_model_ids > 0:
        logger.warning(
            "Gauge rows sharing model_id before dedup: %s",
            int(duplicate_gauge_model_ids),
        )

    gauges = gauges.drop_duplicates(subset=["model_id"], keep="first")
    gauges["type"] = "gauge"

    gauges = gauges[
        [
            "model_id",
            "type",
            "gauge_id",
            "station_name",
        ]
    ].copy()

    logger.debug(
        "Gauge target unique model_id before plant overlap removal: %s",
        len(gauges),
    )

    plant_model_ids = set(plants["model_id"].astype(str))

    gauges = gauges[
        ~gauges["model_id"].astype(str).isin(plant_model_ids)
    ].copy()

    logger.debug("Gauge target after removing plant overlaps: %s", len(gauges))

    targets = pd.concat(
        [
            plants[["model_id", "type", "gauge_id", "station_name"]],
            gauges[["model_id", "type", "gauge_id", "station_name"]],
        ],
        ignore_index=True,
    )

    targets = targets.reset_index(drop=True)
    targets["rivid"] = np.arange(len(targets), dtype=int)

    targets = targets[
        [
            "rivid",
            "type",
            "model_id",
            "gauge_id",
            "station_name",
        ]
    ].copy()

    logger.debug("Target model map check")
    logger.debug("Total targets: %s", len(targets))
    logger.debug("Unique model_id: %s", targets["model_id"].nunique())
    logger.debug("Target type counts:\n%s", targets["type"].value_counts().to_string())

    assert targets["model_id"].nunique() == len(targets), (
        "ERROR: target model_id are not unique."
    )

    return targets


# ============================================================
# HINDCAST TARGET ZARR
# ============================================================

def build_hindcast_target_dataset(
    targets: pd.DataFrame,
    hindcast_plants_zarr: Path,
    plant_map_path: Path,
    gauge_qsim_dir: Path,
) -> tuple[xr.Dataset, pd.DataFrame]:
    targets = targets.sort_values("rivid").reset_index(drop=True)

    ds_plants = xr.open_zarr(hindcast_plants_zarr)
    plant_var = get_discharge_variable(ds_plants)

    Qp = ds_plants[plant_var]
    time = pd.to_datetime(Qp["time"].values)

    plant_map = pd.read_csv(
        plant_map_path,
        dtype=str,
        keep_default_na=False,
    )

    plant_map["model_id"] = normalize_model_id(plant_map["model_id"])
    plant_map = plant_map.dropna(subset=["model_id", "rivid"])

    plant_model_to_rivid = dict(
        zip(
            plant_map["model_id"].astype(str),
            plant_map["rivid"].astype(int),
        )
    )

    qout_list = []
    missing_gauge_qsim = []
    missing_plant_qsim = []

    for _, row in targets.iterrows():
        model_id = str(row["model_id"])
        target_type = str(row["type"])

        if target_type == "plant":
            plant_rivid = plant_model_to_rivid.get(model_id)

            if plant_rivid is None:
                q = np.full(len(time), np.nan, dtype="float32")
                missing_plant_qsim.append(
                    {
                        "model_id": model_id,
                        "reason": "model_id_not_found_in_plant_map",
                    }
                )
            else:
                q = Qp.sel(rivid=plant_rivid).values.astype("float32")

        elif target_type == "gauge":
            gauge_id = str(row["gauge_id"])

            if gauge_id in {"", "nan", "NaN", "None", "<NA>"}:
                q = np.full(len(time), np.nan, dtype="float32")
                missing_gauge_qsim.append(
                    {
                        "model_id": model_id,
                        "gauge_id": gauge_id,
                        "reason": "missing_gauge_id",
                    }
                )
            else:
                csv_path = gauge_qsim_dir / f"{gauge_id}.csv"

                if not csv_path.exists():
                    q = np.full(len(time), np.nan, dtype="float32")
                    missing_gauge_qsim.append(
                        {
                            "model_id": model_id,
                            "gauge_id": gauge_id,
                            "reason": f"missing_csv:{gauge_id}.csv",
                        }
                    )
                else:
                    df = pd.read_csv(csv_path, index_col=0)
                    df.index = pd.to_datetime(df.index)

                    if "Qsim" not in df.columns:
                        q = np.full(len(time), np.nan, dtype="float32")
                        missing_gauge_qsim.append(
                            {
                                "model_id": model_id,
                                "gauge_id": gauge_id,
                                "reason": "missing_Qsim_column",
                            }
                        )
                    else:
                        s = df["Qsim"].reindex(time)
                        q = s.values.astype("float32")

        else:
            raise ValueError(f"Unknown target type: {target_type}")

        qout_list.append(q)

    Qout = np.stack(qout_list, axis=1).astype("float32")

    ds_target = xr.Dataset(
        data_vars={
            "Qout": (("time", "rivid"), Qout),
        },
        coords={
            "time": time.values,
            "rivid": targets["rivid"].to_numpy(dtype=int),
            "model_id": ("rivid", targets["model_id"].astype(str).to_numpy()),
            "type": ("rivid", targets["type"].astype(str).to_numpy()),
            "gauge_id": (
                "rivid",
                targets["gauge_id"]
                .astype("string")
                .fillna("")
                .astype(str)
                .to_numpy(),
            ),
        },
    )

    missing_rows = []

    if missing_gauge_qsim:
        missing_rows.extend(missing_gauge_qsim)

    if missing_plant_qsim:
        missing_rows.extend(missing_plant_qsim)

    missing_df = pd.DataFrame(missing_rows)

    logger.debug("Hindcast target check")
    logger.debug("Target dataset dimensions: %s", dict(ds_target.sizes))
    logger.info("Missing gauge Qsim: %s", len(missing_gauge_qsim))
    logger.info("Missing plant Qsim: %s", len(missing_plant_qsim))
    logger.debug("Target dataset detail:\n%s", ds_target)

    ds_plants.close()

    return ds_target, missing_df


# ============================================================
# CLUSTER DATA
# ============================================================

def build_cluster_data(
    hindcast_target_zarr: Path,
    cluster_data_path: Path,
    n_pctl: int,
    min_valid_values: int,
) -> pd.DataFrame:
    percentiles = np.linspace(0, 100, n_pctl)

    logger.debug("Loading hindcast target: %s", hindcast_target_zarr)

    ds = xr.open_zarr(hindcast_target_zarr)

    if "Qout" not in ds.data_vars:
        raise ValueError("Qout not found in hindcast target zarr.")

    Q = ds["Qout"]
    model_ids = ds["model_id"].values.astype(str)

    rows = []
    index = []

    logger.info("Building FDC cluster data from Qout.")

    for i in range(Q.sizes["rivid"]):
        q = Q.isel(rivid=i).values
        q = q[np.isfinite(q)]

        if len(q) < min_valid_values:
            continue

        fdc = np.percentile(q, percentiles)
        fdc_z = zscore_1d(fdc)

        rows.append(fdc_z)
        index.append(model_ids[i])

    cluster_df = pd.DataFrame(
        rows,
        index=index,
        columns=[f"p{int(p)}" for p in percentiles],
    )

    cluster_df.index.name = "model_id"

    cluster_data_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_df.to_parquet(cluster_data_path, engine="pyarrow")

    ds.close()

    logger.debug("Cluster data shape: %s", cluster_df.shape)
    logger.info("Cluster data saved: %s", cluster_data_path)

    return cluster_df


# ============================================================
# SAVE
# ============================================================

def save_outputs(
    targets: pd.DataFrame,
    ds_target: xr.Dataset,
    missing_df: pd.DataFrame,
    workdir_tables: Path,
    output_base_dir: Path,
    target_model_map_path: Path,
    hindcast_target_zarr: Path,
    missing_gauge_qsim_path: Path,
) -> None:
    workdir_tables.mkdir(parents=True, exist_ok=True)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    targets.to_csv(target_model_map_path, index=False)
    logger.info("Target model map saved: %s", target_model_map_path)

    ds_target.to_zarr(hindcast_target_zarr, mode="w")
    logger.info("Hindcast target zarr saved: %s", hindcast_target_zarr)

    if not missing_df.empty:
        missing_df.to_csv(missing_gauge_qsim_path, index=False)
        logger.warning("Missing Qsim report saved: %s", missing_gauge_qsim_path)
    else:
        logger.info("No missing Qsim report needed.")


# ============================================================
# WORKFLOW
# ============================================================

def run_saber_preprocessing(cfg: dict | None = None) -> None:
    cfg = cfg or get_config()

    log_config_summary(cfg)

    plant_map_path = cfg["rivid_map_output_path"]
    hindcast_plants_zarr = cfg["zarr_output_path"]

    output_base_dir = cfg["base_output_dir"]
    workdir_tables = cfg["saber_workdir"] / "tables"

    station_model_map_path = cfg["station_model_map_path"]
    gauge_table_path = cfg["gauge_table_path"]
    gauge_qsim_dir = cfg["station_qsim_dir"]

    target_model_map_path = cfg["target_model_map_path"]
    missing_gauge_qsim_path = cfg["missing_gauge_qsim_path"]
    cluster_data_path = cfg["cluster_data_path"]
    hindcast_target_zarr = cfg["target_hindcast_zarr_path"]

    targets = build_target_model_map(
        plant_map_path=plant_map_path,
        gauge_table_path=gauge_table_path,
        station_model_map_path=station_model_map_path,
    )

    ds_target, missing_df = build_hindcast_target_dataset(
        targets=targets,
        hindcast_plants_zarr=hindcast_plants_zarr,
        plant_map_path=plant_map_path,
        gauge_qsim_dir=gauge_qsim_dir,
    )

    save_outputs(
        targets=targets,
        ds_target=ds_target,
        missing_df=missing_df,
        workdir_tables=workdir_tables,
        output_base_dir=output_base_dir,
        target_model_map_path=target_model_map_path,
        hindcast_target_zarr=hindcast_target_zarr,
        missing_gauge_qsim_path=missing_gauge_qsim_path,
    )

    build_cluster_data(
        hindcast_target_zarr=hindcast_target_zarr,
        cluster_data_path=cluster_data_path,
        n_pctl=N_PCTL,
        min_valid_values=MIN_VALID_VALUES_FOR_CLUSTER,
    )

    logger.info("SABER preprocessing completed.")


def main() -> None:
    setup_logging()
    run_saber_preprocessing()


if __name__ == "__main__":
    main()