# One-off analyses

Exploratory and sensitivity work kept for reference. **None of this is part of
the pipeline**: nothing here is called by `main.py` or by the Snakemake
workflow, and nothing here needs to run to reproduce the published results.

Each script writes into its own `*_out/` directory next to it, so run them from
this folder:

```bash
cd analysis
python ror_plant_level_audit.py
```

They read the shared inputs under `../data/`, which they reference by absolute
path; adjust those constants if the repository moves.

## Run-of-river diagnostics

| Script | Output | What it looks at |
|---|---|---|
| `ror_plant_level_audit.py` | `ror_audit_out/` | Per-plant capacity factors against GloFAS discharge; finds over- and under-producers |
| `ror_diagnose.py` | `ror_headsens_out/` | Inventory reconciliation, `qmax` diagnostics and head sensitivity |
| `ror_recoverability_test.py` | `ror_recover_out/` | Whether badly matched plants have a better nearby GloFAS cell |
| `ror_match_prototype.py` | `ror_match_prototype_out/` | Prototype of an alternative plant-to-cell matching rule |

`ror_head_sensitivity_summary.md` is the write-up of the head-sensitivity
results.

## Reservoir diagnostics

`reservoir_audit.py` → `reservoir_audit_out/`: reservoir matching quality and
the distance between the assigned cell and the best available one.

## SABER leave-one-out validation

`saber_loo_extremes.py` → `saber_loo_out/` runs the leave-one-out experiment on
extreme quantiles; `plot_loo_extremes.py` draws the figures from the metrics CSV
it writes. Both need the SABER intermediates under
`../data/hydro_workflow/saber_global/`.

## Hydrological year index

`build_hydrological_year_index.py` → `hydro_year_index_out/` builds the wet/dry
year ranking used to pick the climate years for the 2050 scenarios, and compares
it against EIA capacity factors. `compare_hydro_prod.py` compares modelled and
observed hydro production for single years.

---

Recovered from the `prova-claude-code` branch (`3b98b7e308d3f1f5905c708fa60e35775888d2a4`), deleted after this
was extracted. Three unused experimental configs
(`config/historical/pypsa_2019_{iberia,italy,hourly_country}.yaml`) were left
behind; `git show 3b98b7e308d3f1f5905c708fa60e35775888d2a4:<path>` still retrieves them.
