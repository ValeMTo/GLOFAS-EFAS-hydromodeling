"""Annual RoR energy under different turbine power curves, head constant.

Same plants, same GloFAS-SABER inflow, same Qmax as ror_diagnose.py; only the
P/Pmax(Q/Qmax) curve changes:

    paper_ramp   (q - 0.2)/0.8, the formulation in the paper
    all_<type>   every plant gets the digitized curve of that turbine type
                 (Sinagra et al. Fig. 2b, see build_power_curves.py)
    by_head      turbine type assigned from the plant's head:
                 Kaplan < 30 m <= Francis < 300 m <= Pelton

Above design flow (q > 1) every curve saturates at P/Pmax = 1.

Outputs in out_energy/:
    scenarios.csv     annual energy per scenario, and difference vs paper ramp
    by_class.csv      the same, split by the head-assigned turbine class
    by_country.csv    annual energy per country, paper ramp vs by_head
    summary.txt       the digest
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import ror_diagnose as rd  # noqa: E402

OUTDIR = HERE / "out_energy"
YEARS = list(range(2015, 2024))   # all years with inflow and ENTSO-E data
HEAD_KAPLAN_MAX = 30.0
HEAD_FRANCIS_MAX = 300.0
TURBINES = ["kaplan", "francis", "pelton", "crossflow", "propeller"]


def assign_turbine(head):
    return np.where(head < HEAD_KAPLAN_MAX, "kaplan",
                    np.where(head < HEAD_FRANCIS_MAX, "francis", "pelton"))


def main():
    OUTDIR.mkdir(exist_ok=True)
    curves = pd.read_csv(HERE / "power_curves.csv")
    q_grid = curves["q"].to_numpy()

    domain = None
    for y in YEARS:
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

    blocks, dt_ref, used = [], None, []
    for y in YEARS:
        da, dt_h = rd.open_inflow(y)
        if da is None:
            continue
        blocks.append(np.nan_to_num(da.sel(plant=ids).load().values.astype("float32"),
                                    nan=0.0, posinf=0.0, neginf=0.0))
        da.close()
        dt_ref = dt_h if dt_ref is None else dt_ref
        used.append(y)
    Q = np.concatenate(blocks, axis=0)
    del blocks

    qmax = ror.qmax.values.astype("float64")
    pnom = ror.p_nom.values.astype("float64")
    head = ror.head_m.values
    turbine = assign_turbine(head)
    q = Q / qmax[None, :].astype("float32")
    del Q

    def energy(p):
        return p.sum(axis=0, dtype="float64") * pnom * dt_ref / 1e6 / len(used)

    def p_curve(name, cols=slice(None)):
        qt = np.clip(q[:, cols], 0.0, 1.0)
        return np.interp(qt, q_grid, curves[name].to_numpy())

    E = {"paper_ramp": energy(rd.p_V0(q))}
    for t in TURBINES:
        E[f"all_{t}"] = energy(p_curve(t))
    e_head = np.zeros(len(ids))
    for t in ("kaplan", "francis", "pelton"):
        cols = np.where(turbine == t)[0]
        if cols.size:
            e_head[cols] = (p_curve(t, cols).sum(axis=0, dtype="float64")
                            * pnom[cols] * dt_ref / 1e6 / len(used))
    E["by_head"] = e_head

    E_obs_ror, _ = rd.observed_denominators(used)
    base = E["paper_ramp"].sum()
    scen = pd.DataFrame({
        "scenario": list(E),
        "E_TWh_per_year": [v.sum() for v in E.values()],
    })
    scen["dE_vs_paper_TWh"] = scen["E_TWh_per_year"] - base
    scen["dE_vs_paper_pct"] = 100 * scen["dE_vs_paper_TWh"] / base
    scen["pct_of_observed_RoR"] = 100 * scen["E_TWh_per_year"] / E_obs_ror
    scen.round(3).to_csv(OUTDIR / "scenarios.csv", index=False)

    df = pd.DataFrame({"turbine_by_head": turbine, "p_nom_MW": pnom,
                       "head_reported": ror.head_reported.values,
                       "country": ror.country.values,
                       **{k: v for k, v in E.items()}})
    by_class = df.groupby("turbine_by_head").agg(
        n_plants=("p_nom_MW", "size"), GW=("p_nom_MW", lambda s: s.sum() / 1e3),
        head_reported_share=("head_reported", "mean"),
        E_paper_TWh=("paper_ramp", "sum"), E_by_head_TWh=("by_head", "sum"))
    by_class["dE_pct"] = 100 * (by_class.E_by_head_TWh / by_class.E_paper_TWh - 1)
    by_class.round(3).to_csv(OUTDIR / "by_class.csv")

    by_country = df.groupby("country")[["paper_ramp", "by_head"]].sum()
    by_country["dE_pct"] = 100 * (by_country.by_head / by_country.paper_ramp - 1)
    by_country.round(3).to_csv(OUTDIR / "by_country.csv")

    with open(OUTDIR / "summary.txt", "w") as f:
        w = f.write
        w("RoR ANNUAL ENERGY UNDER DIFFERENT TURBINE CURVES (head constant)\n")
        w("=" * 66 + "\n\n")
        w(f"years: {used}   RoR plants: {len(ror)} ({pnom.sum()/1e3:.2f} GW)\n")
        w(f"observed RoR (ENTSO-E, usable countries): {E_obs_ror:.2f} TWh/yr\n")
        w(f"turbine by head: Kaplan < {HEAD_KAPLAN_MAX:g} m <= Francis < "
          f"{HEAD_FRANCIS_MAX:g} m <= Pelton\n\n")
        w(scen.round(2).to_string(index=False) + "\n\n")
        w("BY HEAD-ASSIGNED CLASS\n")
        w(by_class.round(2).to_string() + "\n")
    print((OUTDIR / "summary.txt").read_text())


if __name__ == "__main__":
    main()
