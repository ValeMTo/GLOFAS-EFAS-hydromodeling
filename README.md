# GLOFAS-EFAS-hydromodeling

This repository provides a reproducible hydrological preprocessing and PyPSA-Eur workflow based on GloFAS/EFAS-ERA5, GRDC observations, GRanD reservoirs and SABER bias correction.

The workflow is organised around a single entry point:

```bash
python main.py
```

Running `main.py` executes the complete workflow sequence:

1. prepare hydro input data;
2. run the GloFAS/EFAS hydro inflow preprocessing framework;
3. prepare PyPSA-Eur weather cutouts;
4. run PyPSA-Eur historical and 2050 scenarios;
5. collect and postprocess the final results.

---

## 1. Clone the repository

Clone the repository including submodules:

```bash
git clone --recurse-submodules https://github.com/ValeMTo/GLOFAS-EFAS-hydromodeling.git
cd GLOFAS-EFAS-hydromodeling
```

If the repository was already cloned without submodules, initialise them with:

```bash
git submodule update --init --recursive
```

The main external submodules are:

```text
external/pypsa-eur-hydro/
external/saber-hbc/
```

`external/pypsa-eur-hydro/` contains the modified PyPSA-Eur workflow used to run the energy-system simulations.

`external/saber-hbc/` contains the SABER package used by the hydrological correction step.

---

## 2. Create the conda environment

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate pypsa-eur-hydro
```

The workflow uses Gurobi as optimisation solver for PyPSA-Eur. The conda environment installs the Python interface, but users need a valid Gurobi installation and license.

If Gurobi does not find the license automatically, set:

```bash
export GRB_LICENSE_FILE=/path/to/gurobi.lic
```

On Windows PowerShell:

```powershell
$env:GRB_LICENSE_FILE="C:\path\to\gurobi.lic"
```

---

## 3. Configure external credentials

### Copernicus / ECMWF API

GloFAS and EFAS data are downloaded through the Copernicus/ECMWF API using `cdsapi`.

Before running the workflow, configure the CDS API credentials in your home directory and accept the required dataset terms of use on the Copernicus portal.

If API access is not available, the corresponding NetCDF files can be provided manually in the expected folders described below.

### ENTSO-E Transparency Platform

Downloading ENTSO-E hydro generation data requires an API token.

On Linux/macOS:

```bash
export ENTSOE_API_TOKEN="<your-entsoe-token>"
```

On Windows PowerShell:

```powershell
$env:ENTSOE_API_TOKEN="<your-entsoe-token>"
```

If the token is not available, the workflow skips the ENTSO-E download step. In that case, the hourly ENTSO-E CSV files must be placed manually in:

data/hydro_workflow/hydro_global/ENTSOE/Production/

Expected files are:

Europe_Hydro_2015.csv
Europe_Hydro_2016.csv
Europe_Hydro_2017.csv
Europe_Hydro_2018.csv
Europe_Hydro_2019.csv

These hourly files are used to build the annual no-pumped hydro production reference:

data/hydro_workflow/hydro_global/ENTSOE/hydro_annual_production.csv
Electricity Maps Switzerland data

For Switzerland, the annual hydro production reference uses Electricity Maps data for 2017–2019. These files are not downloaded automatically and must be placed manually in:

data/hydro_workflow/hydro_global/ENTSOE/ElectricityMaps/

Expected files are:

CH_2017_electricity_maps.json
CH_2018_electricity_maps.json
CH_2019_electricity_maps.json

During the annual reference build, Switzerland is replaced with Electricity Maps total hydro generation for the years where these files are available. For 2015–2016, Switzerland is excluded because the corresponding Electricity Maps replacement files are not used in this workflow.

---

## 4. Data folders

By default, hydro input and intermediate data are stored under:

```text
data/hydro_workflow/
```

PyPSA-ready hydro inputs are stored under:

```text
data/pypsa/
```

The main hydro workflow uses:

```text
data/hydro_workflow/hydro_global/
data/hydro_workflow/saber_global/
```

`hydro_global/` contains common static and observational input data, including plant data, HydroBASINS, GRanD, GRDC and ENTSO-E inputs.

`saber_global/` contains intermediate GloFAS/EFAS matching and SABER correction outputs.

Postprocessed PyPSA outputs are written locally to:

```text
hydro_results/
```

---

## 5. Run the complete workflow

From the repository root, run:

```bash
python main.py
```

This executes the full sequence:

```text
prepare-hydro-inputs
run-hydro-inflow
prepare-cutouts
run-pypsa-scenarios
run-postprocessing
```

To check the full sequence without executing expensive operations:

```bash
python main.py --dry-run
```

The dry-run prints the complete planned workflow, including hydro preprocessing functions, PyPSA-Eur Snakemake commands and postprocessing collection targets.

---

## 6. Main workflow options

The workflow is controlled from `main.py`.

Useful options include:

```bash
python main.py --dry-run
python main.py --verbose
```

To run only part of the workflow during development or after an interrupted run:

```bash
python main.py --from-step run-hydro-inflow
python main.py --from-step prepare-cutouts
python main.py --from-step run-postprocessing --to-step run-postprocessing
```

Available step names are:

```text
prepare-hydro-inputs
run-hydro-inflow
prepare-cutouts
run-pypsa-scenarios
run-postprocessing
```

Hydro dataset selection:

```bash
python main.py --hydro-dataset glofas
python main.py --hydro-dataset efas
python main.py --hydro-dataset all
```

PyPSA scenario group selection:

```bash
python main.py --pypsa-group historical
python main.py --pypsa-group 2050
python main.py --pypsa-group all
```

Run only one PyPSA scenario or cutout:

```bash
python main.py --from-step prepare-cutouts --to-step prepare-cutouts --pypsa-only cutout_2019
python main.py --from-step run-pypsa-scenarios --to-step run-pypsa-scenarios --pypsa-only glofas_2019
```

Use a Snakemake dry-run inside the PyPSA-Eur workflow:

```bash
python main.py --from-step run-pypsa-scenarios --to-step run-pypsa-scenarios --pypsa-snakemake-dryrun
```

---

## 7. Hydro input preparation

The first workflow step prepares hydro inputs:

```text
prepare-hydro-inputs
```

It performs the following tasks:

* downloads `GloHydroRes_vs1.csv` from Zenodo into `data/pypsa/`;
* builds the PyPSA-ready hydropower plant table;
* downloads the GloFAS upstream-area file;
* downloads and extracts HydroBASINS Europe;
* downloads GloFAS data through the Copernicus/ECMWF API;
* downloads and processes EFAS data through the Copernicus/ECMWF API;
* downloads ENTSO-E hourly hydro production data if `ENTSOE_API_TOKEN` is set;
* downloads ENTSO-E hourly hydro production for the Italian "North" bidding zone (IT_North) if `ENTSOE_API_TOKEN` is set;
* builds the annual ENTSO-E/Electricity Maps hydro production reference when the required inputs are available.

The annual hydro production reference is written to:

```text
data/hydro_workflow/hydro_global/ENTSOE/hydro_annual_production.csv
```

This file is used by the modified PyPSA-Eur `build_hydro_profile.py` workflow.

---

## 8. Manual input files

Some files cannot be automatically redistributed or downloaded by this repository. They must be obtained by the user or requested from the authors.

### GRDC river discharge observations

Expected path:

```text
data/hydro_workflow/Stations/GRDC_Europe.nc
```

GRDC data must be obtained from the GRDC data portal and are subject to GRDC terms of use.

### GRanD reservoir data

Expected path:

```text
data/hydro_workflow/hydro_global/GRanD_reservoirs_v1_3.shp
```

The workflow was prepared using GRanD reservoir data version 1.3. If the exact file cannot be retrieved, please request the original input from the authors.

### EFAS upstream-area auxiliary data

Expected path:

```text
data/hydro_workflow/hydro_global/uparea_5.0_cut.nc
```

This auxiliary file is required for the EFAS drainage workflow.

### Electricity Maps Switzerland data

Expected directory:

```text
data/hydro_workflow/hydro_global/ENTSOE/ElectricityMaps/
```

Expected files:

```text
CH_2017_electricity_maps.json
CH_2018_electricity_maps.json
CH_2019_electricity_maps.json
```

These files are used to replace Switzerland hydro generation values for 2017–2019 in the annual hydro production reference.

### GADM administrative boundaries (Italy)

Expected path:

data/hydro_workflow/hydro_global/GADM/gadm41_ITA_1.json

GADM level-1 administrative boundaries for Italy are used by the historical
postprocessing (`processing_historical.py`) to build the Italian "North"
bidding-zone polygon. The file must be downloaded from the GADM portal
(https://gadm.org) and is subject to GADM terms of use.

---

## 9. Hydro inflow framework

The second workflow step runs the hydrological preprocessing framework:

```text
run-hydro-inflow
```

By default, both datasets are processed sequentially:

```text
glofas
efas
```

The GloFAS sequence is:

```text
glofas_to_drainage_network.py
extract_glofas_series_for_hydro_plants.py
grdc_to_drain_glofas.py
GRanD_to_drain.py
saber_preprocessing.py
write_saber_config.py
run_saber.py
saber_to_pypsa.py
```

The EFAS sequence is:

```text
efas_to_drainage_network.py
extract_efas_series_for_hydro_plants.py
grdc_to_drain_efas.py
GRanD_to_drain.py
saber_preprocessing.py
write_saber_config.py
run_saber.py
saber_to_pypsa.py
```

The workflow is implemented as importable Python functions inside:

```text
hydro_inflow/
```

The final PyPSA-ready files are written to:

```text
data/pypsa/
```

Examples:

```text
data/pypsa/Europe_inflow_2015_GloFAS_SABER.nc
data/pypsa/Europe_inflow_2015_EFAS_SABER.nc
data/pypsa/Europe_inflow_2019_GloFAS_SABER.nc
data/pypsa/Europe_inflow_2019_EFAS_SABER.nc
```

Diagnostic plots are disabled by default. They can be enabled with:

```bash
HYDRO_MAKE_DIAGNOSTIC_PLOTS=1 python main.py --from-step run-hydro-inflow --to-step run-hydro-inflow
```

---

## 10. PyPSA-Eur cutouts and scenarios

The third and fourth workflow steps run the modified PyPSA-Eur workflow:

```text
prepare-cutouts
run-pypsa-scenarios
```

The modified PyPSA-Eur repository is included as a submodule:

```text
external/pypsa-eur-hydro/
```

Scenario-specific configs are stored in:

```text
config/historical/
config/planning_2050/
config/cutouts/
```

During a run, the wrapper copies the selected config into:

```text
external/pypsa-eur-hydro/config/config.yaml
```

and then launches the appropriate Snakemake target inside:

```text
external/pypsa-eur-hydro/
```

Historical simulations are electricity-only runs for 2015–2019:

```text
pypsa_2015 ... pypsa_2019
glofas_2015 ... glofas_2019
efas_2015 ... efas_2019
```

The 2050 sector-coupled scenarios are run at 4-hour temporal resolution
(`resolution_sector: 4h` in the `config/planning_2050/*.yaml` files), which is
the configuration used for the published results.

```text
pypsa_2050
glofas_2050
efas_2050
```

Expected historical network outputs include:

```text
external/pypsa-eur-hydro/results/Europe_2015/networks/base_s_100_elec_.nc
external/pypsa-eur-hydro/results/Europe_2015_GloFAS_SABER/networks/base_s_100_elec_.nc
external/pypsa-eur-hydro/results/Europe_2015_EFAS_SABER/networks/base_s_100_elec_.nc
```

Expected 2050 network outputs include:

```text
external/pypsa-eur-hydro/results/PyPSA_Europe_2050/networks/base_s_100___2050.nc
external/pypsa-eur-hydro/results/GloFAS_Europe_2050/networks/base_s_100___2050.nc
external/pypsa-eur-hydro/results/EFAS_Europe_2050/networks/base_s_100___2050.nc
```

Hydro inflow selection is inferred from the scenario name:

* GloFAS scenarios use `data/pypsa/Europe_inflow_<year>_GloFAS_SABER.nc`;
* EFAS scenarios use `data/pypsa/Europe_inflow_<year>_EFAS_SABER.nc`;
* baseline PyPSA scenarios use the default PyPSA hydro treatment.

---

## 11. Postprocessing

The final workflow step is:

```text
run-postprocessing
```

It collects selected PyPSA-Eur results and intermediate files into:

```text
hydro_results/
```

Then it runs the postprocessing modules stored in:

```text
postprocessing_workflow/
```

The postprocessing modules are:

```text
postprocessing_workflow/processing_historical.py
postprocessing_workflow/processing_2050.py
postprocessing_workflow/processing_hydro_representation.py
```

The expected local output folders include:

```text
hydro_results/PyPSA_2015/
hydro_results/GloFAS-SABER_2015/
hydro_results/EFAS-SABER_2015/
...
hydro_results/PyPSA_2050/
hydro_results/GloFAS-SABER_2050/
hydro_results/EFAS-SABER_2050/
hydro_results/images/
```

`hydro_results/` is a generated local output directory and is ignored by Git.

A postprocessing-only dry-run can be launched with:

```bash
python main.py --from-step run-postprocessing --to-step run-postprocessing --dry-run
```

---

## 12. Repository structure

Main entry point:

```text
main.py
```

Hydro preprocessing package:

```text
hydro_inflow/
```

PyPSA wrapper package:

```text
pypsa_workflow/
```

Postprocessing wrapper and analysis scripts:

```text
postprocessing_workflow/
```

Scenario and cutout configuration files:

```text
config/historical/
config/planning_2050/
config/cutouts/
```

External submodules:

```text
external/pypsa-eur-hydro/
external/saber-hbc/
```

Generated local data and outputs:

```text
data/hydro_workflow/
data/pypsa/
hydro_results/
```

These generated folders are not intended to be committed to the repository.

---

## 13. Typical commands

Full dry-run:

```bash
python main.py --dry-run
```

Full workflow:

```bash
python main.py
```

Run only the hydro preprocessing framework:

```bash
python main.py --from-step prepare-hydro-inputs --to-step run-hydro-inflow
```

Run only PyPSA-Eur scenarios after hydro inflows are already available:

```bash
python main.py --from-step prepare-cutouts --to-step run-pypsa-scenarios
```

Run only postprocessing after PyPSA-Eur outputs are already available:

```bash
python main.py --from-step run-postprocessing --to-step run-postprocessing
```

Run only one PyPSA scenario:

```bash
python main.py --from-step run-pypsa-scenarios --to-step run-pypsa-scenarios --pypsa-only glofas_2019
```

Run a Snakemake dry-run for PyPSA scenarios:

```bash
python main.py --from-step run-pypsa-scenarios --to-step run-pypsa-scenarios --pypsa-snakemake-dryrun
```
