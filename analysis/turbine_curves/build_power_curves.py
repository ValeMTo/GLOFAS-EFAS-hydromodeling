"""From the digitized efficiency curves of Sinagra et al. Fig. 2b, build the
relative power curve of each turbine type,

    P/Pmax = q * eta(q) / eta(1)        q = Q/Qmax, head constant,

and compare it with the linear ramp used in the paper, (q - 0.2) / 0.8.
eta(1) is the efficiency at the design flow Qmax, because Pmax is the power
delivered at Qmax -- not the peak of the curve, which for Kaplan/Francis/
Pelton sits slightly below Qmax.

Outputs (this folder):
    efficiency_curves_clean.csv   eta(q) on a 0.01 grid, per turbine
    power_curves.csv              P/Pmax(q) on the same grid, per turbine
    linear_fit.csv                best linear ramp (q - q0)/(1 - q0) per turbine
    fig_efficiency_power.png      eta(q) and P/Pmax(q), with the paper ramp
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
Q_GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)
PAPER_QMIN = 0.20
COLORS = {"crossflow": "#7b3294", "pelton": "#1a9850", "kaplan": "#d73027",
          "francis": "#fc8d59", "propeller": "#2166ac"}
LABELS = {"crossflow": "Cross-flow", "pelton": "Pelton", "kaplan": "Kaplan",
          "francis": "Francis", "propeller": "Propeller"}


def clean_curve(g):
    """Drop pixels that jump off the curve (colour mixing where two lines
    cross), average per grid step, fill the gaps left by masked labels."""
    g = g.sort_values("q")
    med = g["eta"].rolling(21, center=True, min_periods=5).median()
    g = g[(g["eta"] - med).abs() < 0.03]
    q_start = g["q"].min()
    binned = g.groupby(np.round(g["q"], 2))["eta"].mean()
    eta = np.interp(Q_GRID, binned.index.values, binned.values)
    eta[Q_GRID < q_start] = 0.0          # below its minimum flow, the turbine is off
    return eta, q_start


def ramp(q, q0):
    return np.clip((q - q0) / (1.0 - q0), 0.0, 1.0)


def main():
    raw = pd.read_csv(HERE / "efficiency_curves_raw.csv")

    eta_tab, power_tab, fits = {"q": Q_GRID}, {"q": Q_GRID}, []
    for name, g in raw.groupby("turbine"):
        eta, q_start = clean_curve(g)
        eta_design = eta[-1]                         # eta at q = 1
        power = Q_GRID * eta / eta_design
        eta_tab[name], power_tab[name] = eta, power

        # best linear ramp through (1, 1): scan q0, least squares on q in [0, 1]
        q0s = np.arange(0.0, 0.6, 0.005)
        rmse = [np.sqrt(np.mean((ramp(Q_GRID, q0) - power) ** 2)) for q0 in q0s]
        best = q0s[int(np.argmin(rmse))]
        rmse_paper = np.sqrt(np.mean((ramp(Q_GRID, PAPER_QMIN) - power) ** 2))
        fits.append({"turbine": name, "q_start": round(q_start, 3),
                     "eta_design": round(eta_design, 3),
                     "eta_peak": round(eta.max(), 3),
                     "best_q0": round(best, 3), "rmse_best": round(min(rmse), 4),
                     "rmse_paper_ramp": round(rmse_paper, 4),
                     "max_abs_err_paper_ramp": round(
                         np.max(np.abs(ramp(Q_GRID, PAPER_QMIN) - power)), 4)})

    pd.DataFrame(eta_tab).to_csv(HERE / "efficiency_curves_clean.csv", index=False)
    pd.DataFrame(power_tab).to_csv(HERE / "power_curves.csv", index=False)
    fit_df = pd.DataFrame(fits).set_index("turbine")
    fit_df.to_csv(HERE / "linear_fit.csv")
    print(fit_df.to_string())

    print("\nP/Pmax at selected q (paper ramp in last column):")
    sel = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    table = pd.DataFrame({LABELS[k]: np.interp(sel, Q_GRID, v)
                          for k, v in power_tab.items() if k != "q"}, index=sel)
    table["paper ramp"] = ramp(np.array(sel), PAPER_QMIN)
    print(table.round(2).to_string())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for name in LABELS:
        a1.plot(Q_GRID, eta_tab[name], color=COLORS[name], lw=2, label=LABELS[name])
        a2.plot(Q_GRID, power_tab[name], color=COLORS[name], lw=2, label=LABELS[name])
    a2.plot(Q_GRID, ramp(Q_GRID, PAPER_QMIN), "k--", lw=2,
            label="Paper ramp (q-0.2)/0.8")
    a2.plot(Q_GRID, Q_GRID, ":", color="grey", lw=1, label="q (constant efficiency)")
    a1.set_title("a) Efficiency  η(Q/Qmax)   [digitized, Sinagra et al.]")
    a2.set_title("b) Relative power  P/Pmax = (Q/Qmax)·η/η(Qmax)")
    for a in (a1, a2):
        a.set_xlim(0, 1); a.set_ylim(0, 1.02); a.grid(alpha=.3)
        a.set_xlabel("Q/Qmax")
    a1.set_ylabel("η"); a2.set_ylabel("P/Pmax")
    a1.legend(fontsize=8, loc="lower right"); a2.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "fig_efficiency_power.png", dpi=160)


if __name__ == "__main__":
    main()
