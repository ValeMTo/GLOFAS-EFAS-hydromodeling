#!/usr/bin/env python3
"""Compare hydropower production (RoR + Reservoir) between two PyPSA networks.

Produces:
  - a per-country table of annual hydro production (TWh) with absolute and
    percentage differences, plus the European total;
  - a time-series plot of the total European hydro production (RoR + Reservoir)
    for both networks over the year.

Pumped hydro storage (PHS) is excluded: only generators with carrier "ror" and
storage units with carrier "hydro" are considered, consistent with the
"total_no_phs" definition used in the historical postprocessing.
"""
import sys
import pypsa
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def country_of(bus):
    """Return the 2-letter country code from a bus name."""
    return str(bus)[:2]


def hydro_timeseries_total(net):
    """Total European hydro power time series (MW): RoR + Reservoir, no PHS.

    Returns a pandas Series indexed by snapshot, summed over all buses.
    """
    parts = []

    # Run-of-river: generators with carrier "ror"
    ror = net.generators[net.generators.carrier == "ror"]
    if not ror.empty:
        parts.append(net.generators_t.p[ror.index].clip(lower=0.0).sum(axis=1))

    # Reservoir: storage units with carrier "hydro"
    res = net.storage_units[net.storage_units.carrier == "hydro"]
    if not res.empty:
        parts.append(net.storage_units_t.p[res.index].clip(lower=0.0).sum(axis=1))

    if not parts:
        return pd.Series(dtype=float, index=net.snapshots)

    # Sum RoR + Reservoir at each snapshot
    return pd.concat(parts, axis=1).sum(axis=1)


def hydro_by_country(net):
    """Annual hydro production per country (MWh): RoR + Reservoir, no PHS."""
    parts = []

    # Run-of-river (generators), grouped by country, summed over time
    ror = net.generators[net.generators.carrier == "ror"]
    if not ror.empty:
        cmap = ror.bus.map(country_of)
        p = net.generators_t.p[ror.index].clip(lower=0.0)
        parts.append(p.T.groupby(cmap).sum().T.sum())

    # Reservoir (storage units), grouped by country, summed over time
    res = net.storage_units[net.storage_units.carrier == "hydro"]
    if not res.empty:
        cmap = res.bus.map(country_of)
        p = net.storage_units_t.p[res.index].clip(lower=0.0)
        parts.append(p.T.groupby(cmap).sum().T.sum())

    if not parts:
        return pd.Series(dtype=float)

    # Sum RoR + Reservoir per country
    return pd.concat(parts, axis=1).sum(axis=1)


def plot_total_timeseries(ts_new, ts_old, out_path, label_new, label_old):
    """Plot the two total European hydro production time series (weekly mean)."""
    # Resample to weekly mean for readability (MW)
    ts_new_w = ts_new.resample("1W").mean()
    ts_old_w = ts_old.resample("1W").mean()

    fig, ax = plt.subplots(figsize=(14, 6))

    # Convert MW -> GW
    ax.plot(ts_old_w.index, ts_old_w.values / 1e3, color="black",
            linewidth=2.2, alpha=0.85, label=label_old)
    ax.plot(ts_new_w.index, ts_new_w.values / 1e3, color="red",
            linewidth=2.2, alpha=0.85, label=label_new)

    ax.set_title("Total European hydro production (RoR + Reservoir), weekly mean",
                 fontsize=15, fontweight="bold")
    ax.set_ylabel("Power (GW)", fontsize=12)
    ax.set_xlabel("Time", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nTime-series plot saved to: {out_path}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_hydro_prod.py <network_new.nc> <network_old.nc> "
              "[output_plot.png]")
        sys.exit(1)

    path_new, path_old = sys.argv[1], sys.argv[2]
    out_plot = sys.argv[3] if len(sys.argv) > 3 else "hydro_prod_comparison.png"

    new = pypsa.Network(path_new)
    old = pypsa.Network(path_old)

    # --- per-country annual production (MWh) ---
    prod_new = hydro_by_country(new)
    prod_old = hydro_by_country(old)

    countries = sorted(set(prod_new.index) | set(prod_old.index))
    rows = []
    for c in countries:
        n = prod_new.get(c, 0.0) / 1e6  # TWh
        o = prod_old.get(c, 0.0) / 1e6
        diff = n - o
        pct = (diff / o * 100) if abs(o) > 1e-9 else np.nan
        rows.append({"country": c, "new_TWh": n, "old_TWh": o,
                     "diff_TWh": diff, "diff_pct": pct})

    df = pd.DataFrame(rows).set_index("country")

    # European total
    tot_new = df["new_TWh"].sum()
    tot_old = df["old_TWh"].sum()
    tot_pct = (tot_new - tot_old) / tot_old * 100 if tot_old else np.nan

    pd.set_option("display.float_format", lambda x: f"{x:8.3f}")
    print("\n=== Hydro production (RoR + Reservoir) per country, TWh ===")
    print(df.sort_values("diff_pct", key=abs, ascending=False).to_string())
    print("\n=== EUROPEAN TOTAL ===")
    print(f"new={tot_new:.2f} TWh  old={tot_old:.2f} TWh  "
          f"diff={tot_new - tot_old:+.2f} TWh  ({tot_pct:+.2f}%)")

    # --- total European time series plot ---
    ts_new = hydro_timeseries_total(new)
    ts_old = hydro_timeseries_total(old)
    plot_total_timeseries(
        ts_new=ts_new,
        ts_old=ts_old,
        out_path=out_plot,
        label_new="pre-2015 calibration",
        label_old="full-period (official)",
    )


if __name__ == "__main__":
    main()