#!/usr/bin/env python3
"""One-off FULL-period SABER calibration for GloFAS, bypassing the
expanding-window dispatch that hydro_inflow/run_hydro_inflow_framework.py
now always applies to the 4 SABER steps.

Not part of the tracked pipeline: calls the step functions directly, with
SABER_CALIBRATION_END/SABER_INFLOW_YEARS left unset, which is exactly what
each step already treats as "no cutoff, use pypsa_inflow_years as-is".

This overwrites, for every year in pypsa_inflow_years:
- data/pypsa/Europe_inflow_<year>_GloFAS_SABER.nc
- the shared intermediate tables under
  data/hydro_workflow/saber_global/glofas/output_grdc_to_drain/

Back both up before running this, and restore them afterwards: the causal
(expanding-window) inflows are what the historical validation runs need, while
the FULLCAL outputs it writes are only appropriate for the 2050 scenarios.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path("/work/cmcc/ds07226/GLOFAS-EFAS-hydromodeling")

os.environ["HYDRO_DATASET"] = "glofas"
os.environ["HYDRO_REPO_ROOT"] = str(REPO_ROOT)
os.environ["HYDRO_SCRIPTS_DIR"] = str(REPO_ROOT / "hydro_inflow")
os.environ["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

# Explicitly ensure no expanding-window env vars leak in from the caller.
os.environ.pop("SABER_CALIBRATION_END", None)
os.environ.pop("SABER_INFLOW_YEARS", None)
os.environ.pop("SABER_EXPANDING_YEARS", None)

from hydro_inflow.saber_preprocessing import run_saber_preprocessing
from hydro_inflow.write_saber_config import write_saber_config
from hydro_inflow.run_saber import run_saber_pipeline
from hydro_inflow.saber_to_pypsa import run_saber_to_pypsa


def main() -> None:
    print("=== FULL calibration (no expanding window) for GloFAS ===")
    print("SABER_CALIBRATION_END:", os.environ.get("SABER_CALIBRATION_END"))
    print("SABER_INFLOW_YEARS:", os.environ.get("SABER_INFLOW_YEARS"))

    print("\n--- saber_preprocessing.py ---")
    run_saber_preprocessing()

    print("\n--- write_saber_config.py ---")
    write_saber_config()

    print("\n--- run_saber.py ---")
    run_saber_pipeline()

    print("\n--- saber_to_pypsa.py ---")
    run_saber_to_pypsa()

    print("\n=== FULL calibration run complete ===")


if __name__ == "__main__":
    main()
