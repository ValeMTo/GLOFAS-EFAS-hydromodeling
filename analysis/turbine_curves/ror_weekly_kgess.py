"""Europe weekly RoR-only production 2015-2023: each conversion version vs
ENTSO-E observed RoR, with KGESS -- the RoR counterpart of
hydro_results/images/historical/historical_europe_weekly_hydro.png.

Modelled RoR is computed directly from the GloFAS-SABER inflow of each plant:

    P(t) = Pnom * P/Pmax(q(t))      q = Q/Qmax

i.e. the RoR availability profile PyPSA receives (before any curtailment),
summed per country. The Europe aggregation replicates
processing_historical.build_europe_weekly_series(): per year, only countries
present in both ENTSO-E (usable RoR component) and the model, only hours where
the observation is valid, summed over countries, weekly mean with the same
minimum coverage, and the same KGESS (benchmark = mean of the observations).

Versions in the figure:
    Paper ramp                       (q - 0.2)/0.8, head constant
    Turbine by head                  digitized curve by head class, head constant
    Turbine + head                   + variable net head, alpha = 0, mode full
    Turbine + head, no flood loss    + variable net head, alpha = 0, part load only
All (mode, alpha) combinations are scored in kgess_all_versions.csv.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO))
import ror_diagnose as rd  # noqa: E402
from turbine_type_energy import assign_turbine  # noqa: E402
from head_alpha_energy import (ALPHAS, HEAD_MODES, head_ingredients,  # noqa: E402
                               head_table, load_inputs, turbine_power)
from postprocessing_workflow import processing_historical as ph  # noqa: E402

YEARS = list(range(2015, 2024))
OUTDIR = HERE / "out_weekly"
ENTSOE_DIR = REPO / "data/hydro_workflow/hydro_global/ENTSOE"

PLOT_VERSIONS = {
    "Paper ramp": ("red", ("paper", None, None)),
    "Turbine by head": ("orange", ("turbine", None, None)),
    "Turbine + head": ("blue", ("head", 0.0, "full")),
    "Turbine + head, no flood loss": ("green", ("head", 0.0, "no_flood_loss")),
}


def observed_ror_by_year():
    """ENTSO-E RoR per country and year (MW, hourly), only components passing
    the same usability test processing_historical applies to build no-PHS."""
    _, ror_by_year, _ = ph.load_entsoe_hydro_data(
        ENTSOE_DIR / "Production", ENTSOE_DIR / "ElectricityMaps"
    )
    out = {}
    for year, df in ror_by_year.items():
        usable = [c for c in df.columns if ph.component_is_usable(df[c])]
        out[year] = df[usable].rename(columns=lambda c: c[: -len("_RoR")])
    return out


def europe_weekly(obs_by_year, model_by_name):
    """Replicates build_europe_weekly_series for RoR: returns the weekly
    Europe frame (GW) and the KGESS of each model."""
    names = list(model_by_name)
    yearly = []
    for year, obs_y in obs_by_year.items():
        models_y = {n: m.loc[m.index.year == year] for n, m in model_by_name.items()}
        if obs_y.empty or any(m.empty for m in models_y.values()):
            continue
        common = sorted(set(obs_y.columns).intersection(*[set(m.columns) for m in models_y.values()]))
        if not common:
            continue
        idx = obs_y.index.intersection(next(iter(models_y.values())).index)
        o = obs_y.loc[idx, common]
        valid = o.notna()
        frame = {"ENTSO-E": o.sum(axis=1, min_count=ph.MIN_COUNTRIES_PER_TIMESTEP) / 1e3}
        for n, m in models_y.items():
            frame[n] = (m.loc[idx, common].where(valid)
                        .sum(axis=1, min_count=ph.MIN_COUNTRIES_PER_TIMESTEP) / 1e3)
        yearly.append(pd.DataFrame(frame).dropna())

    df = pd.concat(yearly).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    weekly = ph.resample_mean_with_min_coverage(df, freq="1W",
                                                min_coverage=ph.MIN_EUROPE_WEEKLY_COVERAGE)
    metric = weekly.dropna(subset=["ENTSO-E"] + names)
    scores = {n: ph.kgess(metric["ENTSO-E"], metric[n]) for n in names}
    return weekly, scores


def plot(weekly, scores, path):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(weekly.index, weekly["ENTSO-E"], color="black", lw=3.5, label="ENTSO-E (RoR)")
    for name, (color, _) in PLOT_VERSIONS.items():
        ax.plot(weekly.index, weekly[name], color=color, lw=2.0, alpha=0.6, label=name)
    for year in sorted(weekly.index.year.unique()):
        ax.axvline(pd.Timestamp(f"{year}-01-01"), color="gray", ls="--", lw=0.8, alpha=0.35)
    start = weekly.index.min().year
    end = (weekly.index.max() - pd.Timedelta(days=6)).year
    items = [f"{n}={scores[n]:.2f}" for n in PLOT_VERSIONS]
    ax.set_title(f"Europe — Weekly Run-of-River {start}-{end} (GloFAS-SABER inflow)\n"
                 f"KGESS → {' | '.join(items[:2])}\n{' | '.join(items[2:])}",
                 fontsize=15, fontweight="bold", pad=14)
    ax.set_ylabel("Power (GW)", fontsize=13)
    ax.legend(fontsize=11, loc="best", frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    OUTDIR.mkdir(exist_ok=True)
    curves = pd.read_csv(HERE / "power_curves.csv")

    ror, Q, _, used, times = load_inputs(YEARS)
    times = pd.DatetimeIndex(times)
    if times.tz is not None:
        times = times.tz_convert(None)
    times = times.floor("h")
    qmax = ror.qmax.values.astype("float64")
    pnom = ror.p_nom.values.astype("float64")
    head = ror.head_m.values.astype("float64")
    q = Q / qmax[None, :].astype("float32")
    del Q

    countries = ror.country.fillna("??").values
    country_list = sorted(set(countries))
    weights = np.zeros((len(ror), len(country_list)))
    for j, c in enumerate(country_list):
        weights[countries == c, j] = pnom[countries == c]

    def to_country_mw(p):
        return pd.DataFrame(np.asarray(p, dtype="float32") @ weights.astype("float32"),
                            index=times, columns=country_list)

    p_turb = turbine_power(q, assign_turbine(head), curves)
    dy_down, spill, geom_ok = head_ingredients(ror, qmax)

    models = {
        "Paper ramp": to_country_mw(rd.p_V0(q)),
        "Turbine by head": to_country_mw(p_turb),
    }
    for mode in HEAD_MODES:
        for alpha in ALPHAS:
            h = rd.h_manning_series(q, head_table(dy_down, spill, head, alpha, mode))
            p = np.clip(p_turb * h, 0.0, 1.0)
            p[:, ~geom_ok] = p_turb[:, ~geom_ok]
            models[f"head|{mode}|{alpha:g}"] = to_country_mw(p)
            del h, p
    models["Turbine + head"] = models["head|full|0"]
    models["Turbine + head, no flood loss"] = models["head|no_flood_loss|0"]

    obs = observed_ror_by_year()
    obs = {y: d for y, d in obs.items() if y in used}

    weekly, scores = europe_weekly(obs, models)
    weekly[["ENTSO-E", *PLOT_VERSIONS]].to_csv(OUTDIR / "europe_weekly_ror.csv")

    table = pd.DataFrame([
        {"version": n, "KGESS": s,
         "mean_GW": weekly[n].mean(), "ENTSOE_mean_GW": weekly["ENTSO-E"].mean()}
        for n, s in scores.items()])
    table["bias_pct"] = 100 * (table.mean_GW / table.ENTSOE_mean_GW - 1)
    table.round(3).to_csv(OUTDIR / "kgess_all_versions.csv", index=False)

    plot(weekly, scores, OUTDIR / "europe_weekly_ror.png")
    print(f"years: {used}")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
