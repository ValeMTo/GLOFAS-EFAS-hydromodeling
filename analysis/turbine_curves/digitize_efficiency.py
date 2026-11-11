"""Digitize the efficiency-vs-discharge curves of Sinagra et al. Fig. 2b
(Kaplan, Pelton, Francis, Cross-flow, Propeller) from the figure image, by
colour hue, and write them to efficiency_curves.csv (q = Q/Qmax, eta)."""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.colors import rgb_to_hsv

HERE = Path(__file__).parent
IMG = HERE / "sinagra_fig2.png"

# axis calibration, in pixels of the original image (found from the frame
# lines and the 0.6/0.9 tick marks)
X0_PX, X1_PX = 1317.5, 2185.5      # q = 0, q = 1
Y0_PX, Y1_PX = 694.0, 103.5        # eta = 0, eta = 1

# hue ranges in degrees; Kaplan (red) and Francis (orange) are close, so
# their split is the delicate one -- check it on the overlay figure
HUES = {
    "kaplan": (2, 20),
    "francis": (20, 40),
    "pelton": (120, 165),
    "propeller": (205, 235),
    "crossflow": (265, 300),
}

# text labels drawn in the same colours as the curves, (x0, x1, y0, y1) px
LABEL_BOXES = [
    (1555, 1665, 122, 160),    # Pelton
    (1360, 1465, 222, 300),    # Cross-flow
    (1495, 1610, 345, 385),    # Kaplan
    (1605, 1720, 405, 450),    # Francis
    (1805, 1950, 420, 462),    # Propeller
]


def hue_mask(hsv, lo, hi):
    h, s, v = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    return (h >= lo) & (h < hi) & (s > 0.45) & (v > 0.35)


def main():
    rgb = np.array(Image.open(IMG).convert("RGB"))
    y_lo, y_hi = int(Y1_PX) - 5, int(Y0_PX) + 2
    x_lo, x_hi = int(X0_PX), int(X1_PX) + 1
    panel = rgb[y_lo:y_hi, x_lo:x_hi]
    hsv = rgb_to_hsv(panel / 255.0)

    blank = np.ones(panel.shape[:2], dtype=bool)
    for bx0, bx1, by0, by1 in LABEL_BOXES:
        blank[max(by0 - y_lo, 0):max(by1 - y_lo, 0),
              max(bx0 - x_lo, 0):max(bx1 - x_lo, 0)] = False

    rows = []
    for name, (lo, hi) in HUES.items():
        mask = hue_mask(hsv, lo, hi) & blank
        for col in range(mask.shape[1]):
            ys = np.where(mask[:, col])[0]
            if ys.size == 0:
                continue
            # a steep (near-vertical) segment spans many rows in one column:
            # keep the top of the run, which is where the curve leaves it
            row_px = ys.min() + y_lo if np.ptp(ys) > 6 else np.median(ys) + y_lo
            q = (col + x_lo - X0_PX) / (X1_PX - X0_PX)
            eta = (Y0_PX - row_px) / (Y0_PX - Y1_PX)
            rows.append((name, round(q, 4), round(eta, 4)))

    df = pd.DataFrame(rows, columns=["turbine", "q", "eta"])
    df.to_csv(HERE / "efficiency_curves_raw.csv", index=False)
    print(df.groupby("turbine").agg(n=("q", "size"), q_min=("q", "min"),
                                    q_max=("q", "max"), eta_max=("eta", "max")))


if __name__ == "__main__":
    main()
