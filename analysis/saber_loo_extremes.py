#!/usr/bin/env python3
"""SABER leave-one-out (LOO) validation focused on hydrological extremes."""

from pathlib import Path
import sys
import os
import argparse
import numpy as np
import pandas as pd

SABER_ROOT = "/work/cmcc/ds07226/GLOFAS-EFAS-hydromodeling/external/saber-hbc"
sys.path.insert(0, SABER_ROOT)

from saber.saber import map_saber, _read_modelid_to_rivid, sanitize_sim, _read_sim_series
from saber.assign import _assign_nearest_cluster_gauge
import xarray as xr

BASE = Path("/work/cmcc/ds07226/GLOFAS-EFAS-hydromodeling/data/hydro_workflow/saber_global/glofas/output_grdc_to_drain")
ASSIGN_TABLE = BASE / "workdir" / "tables" / "assign_table_full.parquet"
GAUGE_DATA   = BASE / "gauge_data_grdc"
TARGET_MAP   = BASE / "workdir" / "tables" / "target_model_map.csv"
HINDCAST_ZARR = "/work/cmcc/ds07226/GLOFAS-EFAS-hydromodeling/data/hydro_workflow/saber_global/glofas/output_grdc_to_drain/hindcast_target.zarr"
OUTDIR = Path(__file__).resolve().parent / "saber_loo_out"

N_PER_CLUSTER = 100000   # >max cluster size -> take ALL gauges
MIN_PER_CLUSTER = 1
SEED = 123
FIT_GUMBEL = False
CAL_END = None
DIFFERENT_DAM_PENALTY_M = 10_000.0
CROSS_REGIME_PENALTY_M = 100_000.0

COL_MID, COL_GID = "model_id", "gauge_id"
COL_X, COL_Y = "x", "y"
COL_CID = "clstr_id"
COL_RPROP = "rprop"

LOW_Q  = [q/100 for q in range(1, 21)]    # 0.01 ... 0.20, percentile-by-percentile
HIGH_Q = [q/100 for q in range(80, 101)]  # 0.80 ... 1.00, percentile-by-percentile


def load_gauges():
    a = pd.read_parquet(ASSIGN_TABLE)
    g = a[a[COL_GID].notna()].copy()
    g[COL_MID] = g[COL_MID].astype(str)
    g[COL_GID] = g[COL_GID].astype(str)
    return a, g


def stratified_sample(gauges, n_per, min_per, seed):
    rng = np.random.default_rng(seed)
    picks = []
    for cid, sub in gauges.groupby(COL_CID):
        k = min(len(sub), max(min_per, n_per))
        k = min(k, len(sub))
        idx = rng.choice(sub.index.values, size=k, replace=False)
        picks.append(sub.loc[idx])
    return pd.concat(picks)


def find_loo_donor(gauge_row, gauges_all):
    cid = gauge_row[COL_CID]
    g_mid = str(gauge_row[COL_MID])
    pool = gauges_all[(gauges_all[COL_CID] == cid) &
                      (gauges_all[COL_MID].astype(str) != g_mid)].copy()
    if pool.empty:
        return None
    new_row = gauge_row.to_frame().T.copy()
    target_rprop = new_row[COL_RPROP].iloc[0] if COL_RPROP in new_row else ""
    assigned = _assign_nearest_cluster_gauge(
        new_row=new_row,
        gauges_df=pool,
        target_rprop=target_rprop,
        different_dam_penalty_m=DIFFERENT_DAM_PENALTY_M,
        cross_regime_penalty_m=CROSS_REGIME_PENALTY_M,
    )
    return assigned


def kge_prime(obs, sim):
    m = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[m], sim[m]
    if len(o) < 30 or o.mean() == 0 or o.std() == 0 or s.std() == 0:
        return np.nan
    r = np.corrcoef(o, s)[0, 1]
    beta = s.mean() / o.mean()
    gamma = (s.std()/s.mean()) / (o.std()/o.mean())
    return 1 - np.sqrt((r-1)**2 + (beta-1)**2 + (gamma-1)**2)


def tail_metrics(obs, sim_raw, sim_loo):
    out = {}
    df = pd.DataFrame({"obs": obs, "raw": sim_raw, "loo": sim_loo}).dropna()
    if len(df) < 100:
        return None
    for q in LOW_Q + HIGH_Q:
        oq = df["obs"].quantile(q)
        qtag = f"{round(q*100):02d}"
        for name in ("raw", "loo"):
            mq = df[name].quantile(q)
            out[f"{name}_relbias_q{qtag}"] = (mq - oq) / oq if oq != 0 else np.nan
            out[f"{name}_ratio_q{qtag}"] = mq / oq if oq != 0 else np.nan
    df["year"] = df.index.year
    amax = df.groupby("year").max()
    amin = df.groupby("year").min()
    for name in ("raw", "loo"):
        out[f"{name}_relbias_amax"] = ((amax[name]-amax["obs"])/amax["obs"]).replace([np.inf,-np.inf],np.nan).mean()
        out[f"{name}_relbias_amin"] = ((amin[name]-amin["obs"])/amin["obs"].replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).mean()
    out["kge_raw"] = kge_prime(df["obs"].values, df["raw"].values)
    out["kge_loo"] = kge_prime(df["obs"].values, df["loo"].values)
    out["n"] = len(df)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assign-only", action="store_true")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    a, gauges = load_gauges()
    print(f"Total gauges: {len(gauges)} | clusters: {gauges[COL_CID].value_counts().to_dict()}")

    sample = stratified_sample(gauges, N_PER_CLUSTER, MIN_PER_CLUSTER, SEED)
    print(f"Sampled: {len(sample)} | per cluster: {sample[COL_CID].value_counts().to_dict()}")

    donor_rows = []
    for _, grow in sample.iterrows():
        assigned = find_loo_donor(grow, gauges)
        if assigned is None:
            continue
        donor_rows.append({
            "gauge_mid": str(grow[COL_MID]),
            "gauge_gid": str(grow[COL_GID]),
            "cluster": grow[COL_CID],
            "donor_mid": str(assigned["asgn_mid"].iloc[0]),
            "donor_gid": str(assigned["asgn_gid"].iloc[0]),
        })
    donors = pd.DataFrame(donor_rows)
    donors.to_csv(OUTDIR / "loo_donors.csv", index=False)
    print(f"Donors found: {len(donors)} -> {OUTDIR/'loo_donors.csv'}")
    same = (donors["donor_gid"] == donors["gauge_gid"]).sum()
    print(f"WARNING donor==gauge cases: {same}" if same else "All donors differ from gauge: OK")

    if args.assign_only:
        print("assign-only mode: stopping before correction.")
        return

    modelid_to_rivid = _read_modelid_to_rivid(str(TARGET_MAP))
    ds = xr.open_zarr(HINDCAST_ZARR)
    if "Qout" not in ds.data_vars and "Qsim" in ds.data_vars:
        ds = ds.rename({"Qsim": "Qout"})   # SABER expects 'Qout'
    time_index = pd.to_datetime(ds["time"].values)

    results = []
    for i, r in donors.iterrows():
        g_mid, g_gid = r["gauge_mid"], r["gauge_gid"]
        d_mid, d_gid = r["donor_mid"], r["donor_gid"]
        try:
            corrected = map_saber(
                mid=g_mid, asgn_mid=d_mid, asgn_gid=d_gid,
                ds=ds, time_index=time_index,
                modelid_to_rivid=modelid_to_rivid,
                gauge_data_dir=str(GAUGE_DATA),
                calibration_end_year=CAL_END,
            )
            if corrected is None:
                results.append({"gauge_gid": g_gid, "status": "no_correction"}); continue

            obs_true = pd.read_csv(GAUGE_DATA / f"{g_gid}.csv", index_col=0)
            obs_true.index = pd.to_datetime(obs_true.index)
            obs_s = obs_true.iloc[:, 0]

            if g_mid not in modelid_to_rivid:
                results.append({"gauge_gid": g_gid, "status": "mid_not_in_map"}); continue
            sim_raw = _read_sim_series(ds, time_index, modelid_to_rivid[g_mid])
            sim_raw = sanitize_sim(sim_raw, drop_leap_day=True).iloc[:, 0]

            loo_s = corrected["Qmod"]
            loo_s.index = pd.to_datetime(loo_s.index)

            common = obs_s.index.intersection(loo_s.index).intersection(sim_raw.index)
            m = tail_metrics(obs_s.reindex(common), sim_raw.reindex(common), loo_s.reindex(common))
            if m is None:
                results.append({"gauge_gid": g_gid, "status": "too_few"}); continue
            m.update({"gauge_gid": g_gid, "donor_gid": d_gid,
                      "cluster": r["cluster"], "status": "ok"})
            results.append(m)
        except Exception as exc:
            results.append({"gauge_gid": g_gid, "status": "fail", "error": str(exc)})

        if (i+1) % 20 == 0:
            print(f"  processed {i+1}/{len(donors)}")

    ds.close()
    res = pd.DataFrame(results)
    res.to_csv(OUTDIR / "loo_extremes_metrics.csv", index=False)
    ok = res[res["status"] == "ok"]
    print(f"\nDone. OK: {len(ok)}/{len(res)} -> {OUTDIR/'loo_extremes_metrics.csv'}")
    print(f"Status: {res['status'].value_counts().to_dict()}")

    if len(ok):
        print("\n=== Median relative bias on tails (raw vs LOO) ===")
        cols = [c for c in ok.columns if "relbias" in c]
        summ = ok[cols].median().round(3)
        for c in sorted(summ.index):
            print(f"  {c:28s} {summ[c]:+.3f}")
        print(f"\nKGE' median  raw={ok['kge_raw'].median():.3f}  LOO={ok['kge_loo'].median():.3f}")


if __name__ == "__main__":
    main()
