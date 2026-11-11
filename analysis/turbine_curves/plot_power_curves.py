"""Single-panel figure of the relative power curves P/Pmax(Q/Qmax) by turbine
type, with the paper's linear ramp, from power_curves.csv
(see build_power_curves.py). Writes fig_power_curves.png."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
COLORS = {"crossflow": "#7b3294", "pelton": "#1a9850", "kaplan": "#d73027",
          "francis": "#fc8d59", "propeller": "#2166ac"}
LABELS = {"crossflow": "Cross-flow", "pelton": "Pelton", "kaplan": "Kaplan",
          "francis": "Francis", "propeller": "Propeller"}


def main():
    curves = pd.read_csv(HERE / "power_curves.csv")
    q = curves["q"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 5.2))
    for name in LABELS:
        ax.plot(q, curves[name], color=COLORS[name], lw=2, label=LABELS[name])
    ax.plot(q, np.clip((q - 0.2) / 0.8, 0.0, 1.0), "k--", lw=2,
            label=r"$(Q/Q_{max} - 0.2)\,/\,0.8$")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=.3)
    ax.set_xlabel(r"$Q/Q_{max}$")
    ax.set_ylabel(r"$P/P_{max} = (Q/Q_{max})\cdot\eta(Q/Q_{max})\,/\,\eta(Q_{max})$")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "fig_power_curves.png", dpi=200)


if __name__ == "__main__":
    main()
