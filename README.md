# GLOFAS-EFAS-hydromodeling

Workflow behind the paper *Hydrologically Constrained Hydropower Representation
in PyPSA-Eur: Implications for European Energy System Planning*
([doi:10.2139/ssrn.7096701](https://doi.org/10.2139/ssrn.7096701)).

It turns GloFAS and EFAS river discharge (ERA5-forced) into plant-level hydro
inflows for PyPSA-Eur. The discharge is bias-corrected with SABER against GRDC
observations. The workflow then compares the result with the default PyPSA-Eur
hydro treatment in two settings:

- **historical validation**: hourly, electricity-only runs for 2015–2023,
  checked against ENTSO-E hydro generation;
- **2050 scenarios**: sector-coupled, multi-year runs over climate years
  2010 + 2011.

---

## Workflow at a glance

`python main.py` runs five steps in sequence:

| Step | What it does | Main output |
|---|---|---|
| `prepare-hydro-inputs` | downloads static data, GloFAS/EFAS, ENTSO-E; builds the plant table and the annual hydro reference | `data/hydro_workflow/`, `data/pypsa/` |
| `run-hydro-inflow` | matches plants to the GloFAS/EFAS drainage network and runs the SABER correction | `data/pypsa/Europe_inflow_<year>_<GloFAS\|EFAS>_SABER[_FULLCAL].nc` |
| `prepare-cutouts` | builds the PyPSA-Eur weather cutouts | `external/pypsa-eur-hydro/cutouts/` |
| `run-pypsa-scenarios` | solves the PyPSA-Eur scenarios (below) | `external/pypsa-eur-hydro/results/` |
| `run-postprocessing` | collects networks and produces figures and tables | `hydro_results/` |

### Scenarios

| Group | Scenarios | Config | Resolution | Cost per run |
|---|---|---|---|---|
| historical | `pypsa_`, `glofas_`, `efas_` × 2015–2023 (27 runs) | `config/historical/` | 1 h, electricity only | ~2 h, ~88 GB |
| 2050 | `pypsa_2050_multiyear`, `glofas_2050_multiyear`, `efas_2050_multiyear` | `config/planning_2050/` | 4 h, sector-coupled | ~11 h, ~400 GB, 32 threads |

`pypsa_*` uses the default PyPSA-Eur hydro inflows. `glofas_*` and `efas_*` use
the SABER-corrected inflows of the same year.

---

## Setup

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/ValeMTo/GLOFAS-EFAS-hydromodeling.git
cd GLOFAS-EFAS-hydromodeling
# if already cloned without submodules:
git submodule update --init --recursive
```

Submodules: `external/pypsa-eur-hydro/` (modified PyPSA-Eur) and
`external/saber-hbc/` (SABER).

### 2. Environment

```bash
conda env create -f environment.yml
conda activate pypsa-eur-hydro
```

PyPSA-Eur is solved with Gurobi, so a valid licence is required. If Gurobi does
not find it, set `GRB_LICENSE_FILE=/path/to/gurobi.lic`.

### 3. Credentials

| Source | Needed for | How |
|---|---|---|
| Copernicus CDS | GloFAS / EFAS download | `~/.cdsapirc`, and accept the dataset terms on the portal |
| ENTSO-E Transparency | hourly hydro generation | `export ENTSOE_API_TOKEN=...` |
| Electricity Maps | Swiss hydro generation (not on ENTSO-E), 2017–2023 | `export EM_TOKEN=...` |

Without a token the corresponding download is skipped. The files are written in
a format specific to this workflow, so a manual export from the ENTSO-E website
cannot replace them. An ENTSO-E token is free: register on the Transparency
Platform and request API access. Then either rerun `prepare-hydro-inputs` or
call the download script on its own:

```bash
python -m hydro_inflow.download.download_entsoe --years 2015:2023
```

Switzerland is excluded in 2015–2016, which Electricity Maps does not cover.

### 4. Inputs to obtain manually

These cannot be redistributed or downloaded automatically:

| File | Source |
|---|---|
| `data/hydro_workflow/Stations/GRDC_Europe.nc` | GRDC data portal (GRDC terms of use) |
| `data/hydro_workflow/hydro_global/GRanD_reservoirs_v1_3.shp` | GRanD v1.3, or ask the authors |
| `data/hydro_workflow/hydro_global/uparea_5.0_cut.nc` | EFAS upstream area, or ask the authors |
| `data/hydro_workflow/hydro_global/GADM/gadm41_ITA_1.json` | gadm.org, level 1 Italy (used for the IT-North bidding zone) |

---

## Running

```bash
python main.py --dry-run          # print the full plan, run nothing
python main.py                    # full workflow
```

Partial runs use `--from-step` / `--to-step`:

```bash
# hydro inflows only
python main.py --from-step prepare-hydro-inputs --to-step run-hydro-inflow
# PyPSA only, inflows already available
python main.py --from-step prepare-cutouts --to-step run-pypsa-scenarios --pypsa-group historical   # or 2050, all
# a single scenario (an existing target network is skipped unless --pypsa-force is given)
python main.py --from-step run-pypsa-scenarios --to-step run-pypsa-scenarios --pypsa-only glofas_2019
# postprocessing only
python main.py --from-step run-postprocessing --to-step run-postprocessing
```

Other useful options:
- `--hydro-dataset glofas|efas|all`;
- `--pypsa-snakemake-dryrun`;
- `--verbose`.

The full list is in `python main.py --help`.

Scenario configs are passed to Snakemake with `--configfile`, so
`external/pypsa-eur-hydro/config/config.yaml` is never modified.

---

## Outputs

| Where | What |
|---|---|
| `data/pypsa/Europe_inflow_<year>_<DATASET>_SABER.nc` | corrected inflows, historical (expanding-window calibration) |
| `data/pypsa/Europe_inflow_<year>_<DATASET>_SABER_FULLCAL.nc` | corrected inflows, full-period calibration, used by the 2050 runs |
| `hydro_results/<Scenario>_<year>/`, `hydro_results/<Scenario>_2050/` | collected PyPSA networks |
| `hydro_results/images/` | paper figures |
| `hydro_results/images/historical/supplementary/` | supplementary tables (LaTeX) and figures |

The figures can be regenerated without rerunning anything:

```bash
python -m postprocessing_workflow.processing_historical --hydro-results hydro_results --years 2015:2023
python -m postprocessing_workflow.processing_2050 --hydro-results hydro_results
```

Both accept `--output-dir`.

---

## Notes for interpreting the results

- **Multi-year 2050 networks** cover two climate years. Before comparing them
  with a single-year run, divide extensive quantities (energy, costs, emissions)
  by `nyears`. The postprocessing already does this.
- **Two SABER calibrations.** Historical inflows use an expanding window (year
  *Y* is corrected with coefficients fitted on years before *Y* only), so the
  comparison with ENTSO-E is out of sample. The 2050 climate years (2010, 2011)
  use the full observed period and are written with the `_FULLCAL` suffix.
- **Reservoirs** stay as individual units
  (`GloFAS_ERA5.reservoir_subclustering: false`). Turning aggregation on makes
  the GloFAS/EFAS runs incomparable with the baseline.

---

## Repository layout

```text
main.py                     single entry point
hydro_inflow/               hydro preprocessing and SABER correction
pypsa_workflow/             PyPSA-Eur wrapper (cutouts, scenarios)
postprocessing_workflow/    collection, figures, tables
analysis/                   one-off analyses outside the pipeline (see analysis/README.md)
config/                     historical/, planning_2050/, cutouts/
external/                   pypsa-eur-hydro, saber-hbc (submodules)
data/, hydro_results/       generated, not tracked by git
```
