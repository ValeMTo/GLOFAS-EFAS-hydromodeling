"""Annual RoR energy with turbine type by head AND variable net head.

    P/Pmax = clip( p_turb(q) * h(q), 0, 1 )

p_turb   digitized power curve of the turbine assigned from the plant's head
         (Kaplan < 30 m <= Francis < 300 m <= Pelton), see build_power_curves.py
h(q)     net head relative to design head Hd:

    h(q) = 1 - (1 - alpha) * [dy_down(q) - spill(q)] / Hd

dy_down  tailwater (valle) excursion y(q*Qmax) - y(Qmax): Manning normal depth
         of the FULL river flow (turbined + spilled both return downstream) on
         the LISFLOOD compound section of the GloFAS cell matched to the plant
         -- the same cell its discharge series comes from
spill    headwater (monte) rise in flood. Default 0: gated barrage, the pond
         is held at its regulation level and the gates open as the river grows
         (standard operation of low-head RoR). The fixed-crest alternative,
         ((q-1)*Qmax / (C_W * L))^(2/3) with L = bankfull top width, is kept
         behind SPILL_OVER_FIXED_CREST but rejected: on GloFAS geometry the
         overflow is ~10x the tailwater rise, so head would INCREASE in flood,
         contrary to observed low-head plant behaviour
alpha    how much of the river's rise reaches the headwater:
         0 = pond regulated at a fixed level (standard RoR operation)
         1 = headwater follows the river as much as the tailwater -> net head
             constant (the paper's assumption)
         in between, a linear blend of the two end-members
mode     "full"          head varies at part load (gain) and in flood (loss)
         "no_flood_loss" head varies only at part load; frozen at h = 1 for
                         q >= 1, i.e. the tailwater rise in flood is ignored

Outputs in out_head_alpha/:
    scenarios.csv   annual energy per (mode, alpha), vs constant-head turbine
                    curve (head effect alone) and vs the paper ramp (total)
    summary.txt     the digest, including the low/high-head split
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import ror_diagnose as rd  # noqa: E402
from turbine_type_energy import assign_turbine  # noqa: E402

OUTDIR = HERE / "out_head_alpha"
YEARS = list(range(2015, 2024))   # all years with inflow and ENTSO-E data
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
HEAD_MODES = ["full", "no_flood_loss"]
SPILL_OVER_FIXED_CREST = False   # False = gated barrage (pond held in flood)
C_W = 1.7                # broad-crested weir coefficient, SI: Q = C_W * L * h^1.5
HEAD_SPLIT = 15.0        # low-head vs the rest [m], same split as ror_diagnose.py


def load_inputs(years=rd.YEARS):
    """RoR plants, hourly inflow (time x plant), timestep [h], years used and
    the hourly time index."""
    domain = None
    for y in years:
        da, _ = rd.open_inflow(y)
        if da is not None:
            domain = set(da.plant.values)
            da.close()
            break
    if domain is None:
        sys.exit("[FATAL] no inflow files found")

    plants = rd.load_plants(domain)
    ror = plants[plants.technology == rd.ROR_TECH].copy()
    ids = list(ror.index)

    blocks, times, dt_ref, used = [], [], None, []
    for y in years:
        da, dt_h = rd.open_inflow(y)
        if da is None:
            print(f"[warn] missing inflow {y}")
            continue
        blocks.append(np.nan_to_num(da.sel(plant=ids).load().values.astype("float32"),
                                    nan=0.0, posinf=0.0, neginf=0.0))
        times.append(pd.to_datetime(da.time.values))
        da.close()
        dt_ref = dt_h if dt_ref is None else dt_ref
        used.append(y)
    return ror, np.concatenate(blocks, axis=0), dt_ref, used, times[0].append(times[1:])


def turbine_power(q, turbine, curves):
    """Constant-head P/Pmax from the digitized curve of each plant's turbine."""
    q_grid = curves["q"].to_numpy()
    p = np.empty(q.shape, dtype="float32")
    for t in ("kaplan", "francis", "pelton"):
        cols = np.where(turbine == t)[0]
        if cols.size:
            p[:, cols] = np.interp(np.clip(q[:, cols], 0.0, 1.0), q_grid,
                                   curves[t].to_numpy())
    return p


def head_ingredients(ror, qmax):
    """Tailwater excursion and headwater spill tables on MANNING_Q_GRID, plus
    the mask of plants with valid channel geometry."""
    lon, lat, _ = rd.load_matched_cells(ror.index.values)
    geom_raw = rd.sample_channel_geometry(lon, lat)
    geom_ok = np.isfinite(lon) & np.isfinite(lat) & rd.geometry_is_valid(geom_raw)
    geom = rd.prepare_geometry(geom_raw)
    dy_down = rd.build_manning_delta_y_table(qmax, **geom)

    weir_len = geom["bw"] + 2.0 * geom["side_slope"] * geom["bnkf"]
    grid = rd.MANNING_Q_GRID[:, None]
    spill_flow = np.maximum(grid - 1.0, 0.0) * qmax[None, :]
    spill = (spill_flow / (C_W * weir_len[None, :])) ** (2.0 / 3.0)
    if not SPILL_OVER_FIXED_CREST:
        spill = np.zeros_like(spill)
    return dy_down, spill, geom_ok


def head_table(dy_down, spill, head, alpha, mode):
    h = 1.0 - (1.0 - alpha) * (dy_down - spill) / np.maximum(head[None, :], 0.1)
    h = np.maximum(h, rd.H_FLOOR)
    if mode == "no_flood_loss":
        h = rd.freeze_above_design(h)
    return h


def main():
    OUTDIR.mkdir(exist_ok=True)
    curves = pd.read_csv(HERE / "power_curves.csv")

    ror, Q, dt_ref, used, _ = load_inputs(YEARS)
    qmax = ror.qmax.values.astype("float64")
    pnom = ror.p_nom.values.astype("float64")
    head = ror.head_m.values.astype("float64")
    q = Q / qmax[None, :].astype("float32")
    del Q

    def energy(p):
        return p.sum(axis=0, dtype="float64") * pnom * dt_ref / 1e6 / len(used)

    p_turb = turbine_power(q, assign_turbine(head), curves)
    E_paper = energy(rd.p_V0(q))
    E_turb = energy(p_turb)

    dy_down, spill, geom_ok = head_ingredients(ror, qmax)
    print(f"[info] geometry matched: {int(geom_ok.sum())}/{len(ror)} plants")

    low = head < HEAD_SPLIT
    rows = []
    for mode in HEAD_MODES:
        for alpha in ALPHAS:
            h = rd.h_manning_series(q, head_table(dy_down, spill, head, alpha, mode))
            E = energy(np.clip(p_turb * h, 0.0, 1.0))
            E = np.where(geom_ok, E, E_turb)
            rows.append({
                "mode": mode,
                "alpha": alpha,
                "E_TWh_per_year": E.sum(),
                "dE_head_only_pct": 100 * (E.sum() / E_turb.sum() - 1),
                "dE_head_only_lowhead_pct": 100 * (E[low].sum() / E_turb[low].sum() - 1),
                "dE_head_only_highhead_pct": 100 * (E[~low].sum() / E_turb[~low].sum() - 1),
                "dE_total_vs_paper_pct": 100 * (E.sum() / E_paper.sum() - 1),
            })
            del h

    S = pd.DataFrame(rows)
    S.round(3).to_csv(OUTDIR / "scenarios.csv", index=False)

    with open(OUTDIR / "summary.txt", "w") as f:
        w = f.write
        w("RoR ENERGY: TURBINE BY HEAD + VARIABLE NET HEAD (alpha sweep)\n")
        w("=" * 62 + "\n\n")
        w(f"years: {used}   RoR plants: {len(ror)} ({pnom.sum()/1e3:.2f} GW), "
          f"geometry matched: {int(geom_ok.sum())}\n")
        w(f"low-head (< {HEAD_SPLIT:g} m): {int(low.sum())} plants, "
          f"{pnom[low].sum()/1e3:.2f} GW\n\n")
        w(f"paper ramp, head constant          : {E_paper.sum():8.2f} TWh/yr\n")
        w(f"turbine by head, head constant     : {E_turb.sum():8.2f} TWh/yr "
          f"({100*(E_turb.sum()/E_paper.sum()-1):+.2f} % vs paper)\n\n")
        w(f"headwater in flood: {'fixed-crest overflow' if SPILL_OVER_FIXED_CREST else 'gated barrage, pond held at regulation level'}\n")
        w("alpha = 0: headwater regulated at a fixed level\n")
        w("alpha = 1: headwater follows the river -> net head constant\n")
        w("mode full: gain at part load + loss in flood; "
          "no_flood_loss: gain at part load only\n\n")
        w(S.round(2).to_string(index=False) + "\n")
    print((OUTDIR / "summary.txt").read_text())


if __name__ == "__main__":
    main()
