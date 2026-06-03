from __future__ import annotations

from pathlib import Path
import logging
import os


logger = logging.getLogger(__name__)


def get_repo_root() -> Path:
    return Path(
        os.environ.get(
            "HYDRO_REPO_ROOT",
            Path(__file__).resolve().parents[2],
        )
    )


def get_dataset() -> str:
    dataset = os.environ.get("HYDRO_DATASET", "glofas").lower()

    if dataset not in {"glofas", "efas"}:
        raise ValueError(
            f"Invalid HYDRO_DATASET={dataset!r}. Expected 'glofas' or 'efas'."
        )

    return dataset


def get_data_root() -> Path:
    return Path(
        os.environ.get(
            "HYDRO_DATA_ROOT",
            get_repo_root() / "data" / "hydro_workflow",
        )
    )


def ensure_directories(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def build_saber_paths(
    saber_global_dir: Path,
    station_qsim_dir_name: str,
) -> dict[str, Path | int | str]:
    base_output_dir = saber_global_dir / "output_grdc_to_drain"
    workdir = base_output_dir / "workdir"
    workdir_tables = workdir / "tables"
    diagnostic_plots_dir = base_output_dir / "diagnostic_plots"

    return {
        "base_output_dir": base_output_dir,
        "saber_workdir": workdir,
        "saber_config_path": workdir / "saber_config.yaml",
        "station_model_map_path": base_output_dir / "station_model_map.csv",
        "gauge_table_path": workdir_tables / "gauge_table_all.csv",
        "gauge_gis_path": base_output_dir / "gauge_gis_all.gpkg",
        "gauge_data_dir": base_output_dir / "gauge_data_grdc",
        "corrected_dir": base_output_dir / "gauge_data_grdc" / "corrected",
        "station_qsim_dir": base_output_dir / station_qsim_dir_name,
        "station_qsim_dir_name": station_qsim_dir_name,
        "regulate_table_output_path": workdir_tables / "regulate_table.csv",
        "target_model_map_path": workdir_tables / "target_model_map.csv",
        "missing_gauge_qsim_path": (
            workdir_tables / "missing_gauge_qsim_in_hindcast_target.csv"
        ),
        "cluster_data_path": workdir_tables / "cluster_data.parquet",
        "target_hindcast_zarr_path": base_output_dir / "hindcast_target.zarr",
        "diagnostic_plots_dir": diagnostic_plots_dir,
        "saber_n_processes": 4,
    }


def get_config() -> dict:
    dataset = get_dataset()
    repo_root = get_repo_root()
    data_root = get_data_root()

    hydro_global_dir = data_root / "hydro_global"
    saber_global_root_dir = data_root / "saber_global"
    saber_global_dir = saber_global_root_dir / dataset

    entsoe_hydro_dir = hydro_global_dir / "ENTSOE"
    entsoe_hydro_hourly_dir = entsoe_hydro_dir / "Production"
    electricity_maps_dir = entsoe_hydro_dir / "ElectricityMaps"

    pypsa_inflow_netcdf_dir = repo_root / "data" / "hydro"
    paper_figures_dir = repo_root / "hydro_results" / "images"

    common = {
        "dataset": dataset,
        "repo_root": repo_root,
        "data_root": data_root,
        "hydro_global_dir": hydro_global_dir,
        "saber_global_root_dir": saber_global_root_dir,
        "saber_global_dir": saber_global_dir,
        "paper_figures_dir": paper_figures_dir,
        "hydrobasins_path": (
            hydro_global_dir
            / "Hydrobasins"
            / "hybas_eu_"
            / "hybas_eu_lev03_v1c.shp"
        ),
        "plants_path": hydro_global_dir / "Eur_custom_ppls_GloHydroRes_filled.csv",
        "grdc_path": data_root / "Stations" / "GRDC_Europe.nc",
        "grand_path": hydro_global_dir / "GRanD_reservoirs_v1_3.shp",
        "pypsa_inflow_netcdf_dir": pypsa_inflow_netcdf_dir,
        "pypsa_inflow_years": [2015, 2016, 2017, 2018, 2019],

        # ENTSO-E / Electricity Maps hydro production reference
        "entsoe_hydro_years": [2015, 2016, 2017, 2018, 2019],
        "entsoe_hydro_dir": entsoe_hydro_dir,
        "entsoe_hydro_hourly_dir": entsoe_hydro_hourly_dir,
        "entsoe_hydro_annual_production_path": (
            entsoe_hydro_dir / "hydro_annual_production.csv"
        ),
        "electricity_maps_dir": electricity_maps_dir,
        "electricity_maps_ch_files": {
            2017: electricity_maps_dir / "CH_2017_electricity_maps.json",
            2018: electricity_maps_dir / "CH_2018_electricity_maps.json",
            2019: electricity_maps_dir / "CH_2019_electricity_maps.json",
        },
        "entsoe_ch_years_without_electricity_maps": [2015, 2016],
        "entsoe_min_component_coverage": 0.50,
        "entsoe_min_component_valid_hours": 24 * 30,
        "entsoe_min_total_coverage": 0.80,
    }

    if dataset == "glofas":
        saber_paths = build_saber_paths(
            saber_global_dir=saber_global_dir,
            station_qsim_dir_name="station_qsim_glofas",
        )

        dataset_config = {
            "lon_min": -9.375,
            "lon_max": 31.325,
            "lat_min": 36.375,
            "lat_max": 71.075,
            "uparea_path": hydro_global_dir / "uparea_glofas_v4_0.nc",
            "uparea_variable_name": "uparea",
            "threshold_uparea_km2": 250.0,
            "grid_step": 0.05,
            "tolerance_factor": 1.2,
            "river_reference_path": data_root / "glofas_europe" / "glofas_eu_2021.nc",
            "hydro_data_dir": data_root / "glofas_europe",
            "hydro_file_template": "glofas_eu_{year}.nc",
            "hydro_var_name": "dis24",
            "years": list(range(1980, 2026)),
            "drain_csv_output_path": saber_global_dir / "glofas_drain_lv3_europe.csv",
            "drain_parquet_output_path": (
                saber_global_dir / "glofas_drain_lv3_europe_full.parquet"
            ),
            "drain_gis_output_path": (
                saber_global_dir / "glofas_drain_gis_lv3_europe.gpkg"
            ),
            "drain_full_path": saber_global_dir / "glofas_drain_lv3_europe_full.parquet",
            "drain_table_path": saber_global_dir / "glofas_drain_lv3_europe.csv",
            "final_matches_output_path": (
                saber_global_dir / "final_hydro_plant_glofas_points.parquet"
            ),
            "series_output_path": saber_global_dir / "all_years_series_clean_europe.pkl",
            "rivid_map_output_path": saber_global_dir / "rivid_model_map.csv",
            "zarr_output_path": saber_global_dir / "hindcast_plants.zarr",
            "match_plot_html_output_path": (
                saber_paths["diagnostic_plots_dir"]
                / "hydro_plants_glofas_matches_drain_network.html"
            ),
            "grdc_data_folder": data_root / "glofas_europe",
            "grdc_file_template": "glofas_eu_{year}.nc",
            "pypsa_inflow_pickle_path": (
                hydro_global_dir / "corrected_inflows_saber_Europe_glofas.pkl"
            ),
            "pypsa_inflow_netcdf_template": "Europe_inflow_{year}_GloFAS_SABER.nc",
            "pypsa_inflow_report_path": (
                hydro_global_dir / "saber_final_inflow_report_Europe_glofas.csv"
            ),
            **saber_paths,
        }

    else:
        saber_paths = build_saber_paths(
            saber_global_dir=saber_global_dir,
            station_qsim_dir_name="station_qsim_efas",
        )

        dataset_config = {
            "lon_min": -25.241666666666667,
            "lon_max": 39.99166666666666,
            "lat_min": 32.00833333333333,
            "lat_max": 72.24166666666667,
            "uparea_path": hydro_global_dir / "uparea_5.0_cut.nc",
            "uparea_variable_name": "ec_upArea",
            "threshold_uparea_km2": 100.0,
            "grid_step": 0.016666667,
            "tolerance_factor": 1.1,
            "river_reference_path": (
                data_root / "efas" / "efas_historical_2021_daily_cut.nc"
            ),
            "hydro_data_dir": data_root / "efas",
            "hydro_file_template": "efas_historical_{year}_daily_cut.nc",
            "hydro_var_name": "discharge_daily_mean",
            "years": list(range(1992, 2026)),
            "drain_csv_output_path": saber_global_dir / "efas_drain_lv3_europe.csv",
            "drain_parquet_output_path": (
                saber_global_dir / "efas_df_drain_lv3_europe_full.parquet"
            ),
            "drain_gis_output_path": (
                saber_global_dir / "efas_drain_gis_lv3_europe.gpkg"
            ),
            "drain_full_path": saber_global_dir / "efas_df_drain_lv3_europe_full.parquet",
            "drain_table_path": saber_global_dir / "efas_drain_lv3_europe.csv",
            "final_matches_output_path": (
                saber_global_dir / "final_hydro_plant_efas_points.parquet"
            ),
            "series_output_path": saber_global_dir / "all_years_series_clean_europe.pkl",
            "rivid_map_output_path": saber_global_dir / "rivid_model_map.csv",
            "zarr_output_path": saber_global_dir / "hindcast_plants.zarr",
            "match_plot_html_output_path": (
                saber_paths["diagnostic_plots_dir"]
                / "hydro_plants_efas_matches_drain_network.html"
            ),
            "grdc_data_folder": data_root / "efas",
            "grdc_file_template": "efas_historical_{year}_daily_cut.nc",
            "pypsa_inflow_pickle_path": (
                hydro_global_dir / "corrected_inflows_saber_Europe_efas.pkl"
            ),
            "pypsa_inflow_netcdf_template": "Europe_inflow_{year}_EFAS_SABER.nc",
            "pypsa_inflow_report_path": (
                hydro_global_dir / "saber_final_inflow_report_Europe_efas.csv"
            ),
            **saber_paths,
        }

    cfg = common | dataset_config

    ensure_directories(
        [
            hydro_global_dir,
            hydro_global_dir / "Hydrobasins",
            hydro_global_dir / "Hydrobasins" / "hybas_eu_",
            entsoe_hydro_dir,
            entsoe_hydro_hourly_dir,
            electricity_maps_dir,
            data_root / "Stations",
            data_root / "glofas_europe",
            data_root / "efas",
            saber_global_root_dir,
            saber_global_dir,
            cfg["base_output_dir"],
            cfg["saber_workdir"],
            cfg["saber_workdir"] / "tables",
            cfg["gauge_data_dir"],
            cfg["corrected_dir"],
            cfg["station_qsim_dir"],
            cfg["diagnostic_plots_dir"],
            paper_figures_dir,
            pypsa_inflow_netcdf_dir,
        ]
    )

    return cfg


def log_config_summary(cfg: dict) -> None:
    logger.info("Hydro configuration loaded.")
    logger.info("Dataset: %s", cfg["dataset"])
    logger.info("Repository root: %s", cfg["repo_root"])
    logger.info("Data root: %s", cfg["data_root"])
    logger.info("Hydro global directory: %s", cfg["hydro_global_dir"])
    logger.info("SABER global directory: %s", cfg["saber_global_dir"])
    logger.info("Diagnostic plots directory: %s", cfg["diagnostic_plots_dir"])
    logger.info("Paper figures directory: %s", cfg["paper_figures_dir"])
    logger.debug("Full hydro configuration: %s", cfg)