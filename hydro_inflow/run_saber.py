#!/usr/bin/env python3
# This workflow runs the core SABER pipeline from the prepared hydrological drainage
# network and gauge/regulation inputs. It clusters the targets, assigns target
# nodes to suitable donor gauges, and runs SABER to correct targets.

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
import logging
import os

import numpy as np
import pandas as pd

from saber.io import read_config, init_workdir, get_state
from saber.cluster import cluster, predict_labels
from saber.table import init, mp_prop_gauges, mp_prop_regulated
from saber.assign import mp_assign
from saber.saber import mp_saber

from hydro_inflow.hydro_config import get_config, log_config_summary
from hydro_inflow.utils import setup_logging


logger = logging.getLogger(__name__)

# None = read automatically from workdir/tables/cluster_metrics.csv column 'knee'.
# If the file/column/value is missing, DEFAULT_N_CLUSTERS is used.
N_CLUSTERS = None
DEFAULT_N_CLUSTERS = 5


# ============================================================
# ASSIGNMENT TABLE HELPERS
# ============================================================

def normalize_assignment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize SABER assignment columns after initialization or propagation.

    This avoids treating string placeholders such as 'nan', 'None', or ''
    as valid gauge/regulation identifiers.
    """
    df = df.copy()

    for col in ["gauge_id", "reg_id", "asgn_gid", "asgn_mid"]:
        if col in df.columns:
            df[col] = df[col].replace(
                {
                    "": np.nan,
                    "nan": np.nan,
                    "NaN": np.nan,
                    "None": np.nan,
                    "NONE": np.nan,
                    "none": np.nan,
                }
            )

    for col in ["rprop", "gprop"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    return df


def prepare_assignment_table() -> pd.DataFrame:
    """
    Initialize SABER assignment table and propagate gauge/regulation information.
    """
    assign_df_full = init()
    assign_df_full = normalize_assignment_columns(assign_df_full)

    assign_df_full = mp_prop_gauges(assign_df_full)
    assign_df_full = normalize_assignment_columns(assign_df_full)

    assign_df_full = mp_prop_regulated(assign_df_full)
    assign_df_full = normalize_assignment_columns(assign_df_full)

    return assign_df_full


# ============================================================
# CLUSTER SELECTION
# ============================================================

def read_knee_n_clusters(
    cluster_metrics_csv: Path,
    default_n_clusters: int = DEFAULT_N_CLUSTERS,
) -> int:
    """
    Read the suggested number of clusters from cluster_metrics.csv.

    SABER clustering writes a 'knee' column. Any valid non-null value in that
    column is interpreted as the selected number of clusters. If the file,
    column, or value is missing, default_n_clusters is returned.
    """
    if not cluster_metrics_csv.exists():
        logger.warning(
            "Cluster metrics file not found: %s. Using default n_clusters=%s.",
            cluster_metrics_csv,
            default_n_clusters,
        )
        return default_n_clusters

    try:
        metrics = pd.read_csv(cluster_metrics_csv)
    except Exception as exc:
        logger.warning(
            "Could not read cluster metrics file: %s. Using default n_clusters=%s. Error: %s",
            cluster_metrics_csv,
            default_n_clusters,
            exc,
        )
        return default_n_clusters

    if "knee" not in metrics.columns:
        logger.warning(
            "Column 'knee' not found in %s. Using default n_clusters=%s.",
            cluster_metrics_csv,
            default_n_clusters,
        )
        return default_n_clusters

    knee_values = pd.to_numeric(metrics["knee"], errors="coerce").dropna()

    if knee_values.empty:
        logger.warning(
            "No valid values found in 'knee' column. Using default n_clusters=%s.",
            default_n_clusters,
        )
        return default_n_clusters

    n_clusters = int(knee_values.iloc[0])

    if n_clusters <= 0:
        logger.warning(
            "Invalid knee value %s. Using default n_clusters=%s.",
            n_clusters,
            default_n_clusters,
        )
        return default_n_clusters

    logger.info("Using n_clusters=%s from %s", n_clusters, cluster_metrics_csv)

    return n_clusters


def resolve_n_clusters(
    workdir: Path,
    requested_n_clusters: int | None,
    default_n_clusters: int = DEFAULT_N_CLUSTERS,
) -> int:
    """
    Resolve n_clusters from configured value or from cluster_metrics.csv.

    Priority:
    1. Explicit N_CLUSTERS value.
    2. workdir/tables/cluster_metrics.csv column 'knee'.
    3. default_n_clusters.
    """
    if requested_n_clusters is not None:
        logger.info("Using n_clusters=%s from script configuration.", requested_n_clusters)
        return int(requested_n_clusters)

    cluster_metrics_csv = workdir / "tables" / "cluster_metrics.csv"

    return read_knee_n_clusters(
        cluster_metrics_csv=cluster_metrics_csv,
        default_n_clusters=default_n_clusters,
    )


# ============================================================
# TARGET FILTERING
# ============================================================

def load_target_model_ids(target_model_map_csv: Path) -> set[str]:
    """
    Load target model ids used to restrict SABER assignment to target nodes.
    """
    targets = pd.read_csv(target_model_map_csv, dtype=str)

    if "model_id" not in targets.columns:
        raise KeyError(f"'model_id' column not found in {target_model_map_csv}")

    return set(targets["model_id"].astype(str))


def filter_targets(
    assign_df_full: pd.DataFrame,
    target_model_map_csv: Path,
) -> pd.DataFrame:
    """
    Keep only model nodes listed in target_model_map.csv.
    """
    target_model_ids = load_target_model_ids(target_model_map_csv)

    assign_df = assign_df_full[
        assign_df_full["model_id"].astype(str).isin(target_model_ids)
    ].copy()

    return assign_df.reset_index(drop=True)


# ============================================================
# PLOT HELPERS
# ============================================================

def dam_mid_from_rprop(rprop) -> str:
    """
    Extract regulation dam model id from SABER rprop string.
    """
    if rprop is None:
        return ""

    value = str(rprop).strip()

    if value == "" or value.lower() in {"nan", "none"}:
        return ""

    parts = value.split("-")
    return parts[-1] if len(parts) >= 3 else ""


def plot_assignments(
    assign_df: pd.DataFrame,
    drain_csv: Path,
    target_model_map_csv: Path,
    regulate_csv: Path,
    output_html: Path | None = None,
    show: bool = False,
) -> None:
    """
    Plot SABER assignments, targets, gauges, regulation nodes, and assignment links.

    This is a diagnostic HTML plot. It is not used as a paper figure.
    """
    import plotly.graph_objects as go
    import plotly.express as px

    drain = pd.read_csv(drain_csv, dtype=str)
    drain["x"] = pd.to_numeric(drain["x"], errors="coerce")
    drain["y"] = pd.to_numeric(drain["y"], errors="coerce")
    drain = drain.dropna(subset=["x", "y"]).copy()

    targets = pd.read_csv(target_model_map_csv, dtype=str)

    if "model_id" not in targets.columns:
        raise KeyError(f"'model_id' column not found in {target_model_map_csv}")

    target_ids = set(targets["model_id"].astype(str))

    reg = pd.read_csv(regulate_csv, dtype=str)
    reg["model_id"] = reg["model_id"].astype(str)

    reg["reg_id"] = (
        reg["reg_id"]
        .astype(str)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )

    reg_min = reg[["model_id", "reg_id"]].copy()
    reg_min = reg_min.rename(columns={"reg_id": "regtab_reg_id"})
    reg_min["regtab_reg_id"] = reg_min["regtab_reg_id"].fillna("")

    df = assign_df.copy()
    df["model_id"] = df["model_id"].astype(str)
    df["asgn_mid"] = df["asgn_mid"].astype(str)

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"]).copy()

    df = df.merge(reg_min, on="model_id", how="left")

    if "regtab_reg_id" not in df.columns:
        df["regtab_reg_id"] = ""

    df["regtab_reg_id"] = df["regtab_reg_id"].fillna("")

    df_targets = df[df["model_id"].isin(target_ids)].copy()

    gauge_mask = (
        df["gauge_id"]
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .notna()
    )
    df_gauges = df[gauge_mask].copy()

    df_targets["is_regulated"] = (
        df_targets["rprop"].fillna("").astype(str).str.strip() != ""
    )
    df_targets["reg_dam_mid"] = df_targets["rprop"].apply(dam_mid_from_rprop)

    df_gauges["is_regulated"] = (
        df_gauges["rprop"].fillna("").astype(str).str.strip() != ""
    )
    df_gauges["reg_dam_mid"] = df_gauges["rprop"].apply(dam_mid_from_rprop)

    donor_xy = df[["model_id", "x", "y", "gauge_id"]].copy()
    donor_xy.columns = ["asgn_mid", "don_x", "don_y", "don_gauge_id"]

    links = df_targets[
        ["model_id", "x", "y", "asgn_mid", "asgn_gid", "reason"]
    ].copy()

    links = links[~links["asgn_mid"].isin(["", "nan", "None", "unassigned"])]
    links = links.merge(donor_xy, on="asgn_mid", how="left")
    links = links.dropna(subset=["don_x", "don_y"]).copy()

    reasons = sorted(links["reason"].astype(str).unique())

    palette = (
        px.colors.qualitative.Dark24
        if len(reasons) > 10
        else px.colors.qualitative.Set1
    )
    reason_to_color = {
        reason: palette[i % len(palette)]
        for i, reason in enumerate(reasons)
    }

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=drain["x"],
            y=drain["y"],
            mode="markers",
            name="drain_table river points",
            marker=dict(size=6, color="deepskyblue", opacity=0.5),
            hoverinfo="skip",
        )
    )

    for reason in reasons:
        sub = links[links["reason"] == reason]

        if sub.empty:
            continue

        xs = []
        ys = []

        for _, row in sub.iterrows():
            xs.extend([row["x"], row["don_x"], None])
            ys.extend([row["y"], row["don_y"], None])

        fig.add_trace(
            go.Scattergl(
                x=xs,
                y=ys,
                mode="lines",
                name=reason,
                line=dict(color=reason_to_color[reason], width=3.5),
                opacity=0.7,
                hoverinfo="skip",
            )
        )

    reg_nodes = (
        reg_min[reg_min["regtab_reg_id"] != ""]
        .merge(
            drain[["model_id", "x", "y"]].drop_duplicates("model_id"),
            on="model_id",
            how="left",
        )
        .dropna(subset=["x", "y"])
    )

    fig.add_trace(
        go.Scattergl(
            x=reg_nodes["x"],
            y=reg_nodes["y"],
            mode="markers",
            name="regulation dams",
            marker=dict(size=12, color="purple", symbol="square"),
            text=(
                "dam mid: "
                + reg_nodes["model_id"].astype(str)
                + "<br>reg_id: "
                + reg_nodes["regtab_reg_id"].astype(str)
            ),
            hovertemplate="%{text}<br>x=%{x}<br>y=%{y}<extra></extra>",
        )
    )

    target_text = (
        "target mid: "
        + df_targets["model_id"].astype(str)
        + "<br>reason: "
        + df_targets["reason"].astype(str)
        + "<br>asgn_gid: "
        + df_targets["asgn_gid"].astype(str)
        + "<br>asgn_mid: "
        + df_targets["asgn_mid"].astype(str)
        + "<br>regulated: "
        + df_targets["is_regulated"].map({True: "YES", False: "NO"})
        + "<br>regulation dam mid: "
        + df_targets["reg_dam_mid"].replace("", "—")
    )

    fig.add_trace(
        go.Scattergl(
            x=df_targets["x"],
            y=df_targets["y"],
            mode="markers",
            name="targets",
            marker=dict(size=9, color="red"),
            text=target_text,
            hovertemplate="%{text}<br>x=%{x}<br>y=%{y}<extra></extra>",
        )
    )

    gauge_text = (
        "gauge: "
        + df_gauges["gauge_id"].astype(str)
        + "<br>mid: "
        + df_gauges["model_id"].astype(str)
        + "<br>regulated: "
        + df_gauges["is_regulated"].map({True: "YES", False: "NO"})
        + "<br>regulation dam mid: "
        + df_gauges["reg_dam_mid"].replace("", "—")
    )

    fig.add_trace(
        go.Scattergl(
            x=df_gauges["x"],
            y=df_gauges["y"],
            mode="markers",
            name="gauges",
            marker=dict(size=9, color="black"),
            text=gauge_text,
            hovertemplate="%{text}<br>x=%{x}<br>y=%{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title="SABER assignments: drain + targets + gauges + regulation",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
        width=1050,
        height=1050,
        legend=dict(itemsizing="constant"),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    if output_html is not None:
        output_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_html)
        logger.info("Diagnostic SABER assignment plot saved: %s", output_html)

    if show:
        fig.show()

    logger.debug(
        "Assignment reason counts, targets only:\n%s",
        df_targets["reason"].value_counts(dropna=False).to_string(),
    )


# ============================================================
# OUTPUTS
# ============================================================

def save_assignment_outputs(
    assign_df_full: pd.DataFrame,
    assign_df: pd.DataFrame,
    tables_dir: Path,
) -> None:
    """
    Save full and target-only assignment tables.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)

    assign_full_path = tables_dir / "assign_table_full.parquet"
    assign_target_path = tables_dir / "assign_table_targets.parquet"
    assign_target_csv_path = tables_dir / "assign_table_targets.csv"

    assign_df_full.to_parquet(assign_full_path, index=False)
    assign_df.to_parquet(assign_target_path, index=False)
    assign_df.to_csv(assign_target_csv_path, index=False)

    logger.debug("Saved full assignment table: %s", assign_full_path)
    logger.debug("Saved target assignment table: %s", assign_target_path)
    logger.debug("Saved target assignment CSV: %s", assign_target_csv_path)


# ============================================================
# SABER WORKFLOW
# ============================================================

def run_saber_workflow(
    config_path: Path,
    n_clusters: int | None,
    make_plot: bool,
    show_plot: bool,
    assignment_plot_html: Path,
) -> None:
    """
    Run the full SABER workflow from clustering to validation.
    """
    logger.info("Reading SABER config: %s", config_path)
    read_config(config_path)
    init_workdir()

    workdir = Path(get_state("workdir"))
    drain_table = Path(get_state("drain_table"))
    regulate_table = Path(get_state("regulate_table"))
    hindcast_zarr = get_state("hindcast_zarr")
    gauge_data = get_state("gauge_data")

    tables_dir = workdir / "tables"
    target_model_map_csv = tables_dir / "target_model_map.csv"

    if not target_model_map_csv.exists():
        raise FileNotFoundError(
            f"Target model map not found: {target_model_map_csv}"
        )

    logger.info("Step 1: running SABER clustering.")
    cluster(plot=False)

    n_clusters_resolved = resolve_n_clusters(
        workdir=workdir,
        requested_n_clusters=n_clusters,
        default_n_clusters=DEFAULT_N_CLUSTERS,
    )

    logger.info("Step 2: predicting labels with n_clusters=%s.", n_clusters_resolved)
    cluster_table = predict_labels(n_clusters=n_clusters_resolved)

    if "clstr_id" in cluster_table.columns:
        logger.debug(
            "Cluster label counts:\n%s",
            cluster_table["clstr_id"].value_counts().sort_index().to_string(),
        )
    else:
        logger.warning("'clstr_id' column not found in cluster_table.")

    logger.info("Step 3: initializing and propagating assignment table.")
    assign_df_full = prepare_assignment_table()

    logger.debug("Full assignment table shape: %s", assign_df_full.shape)

    if "gauge_id" in assign_df_full.columns:
        logger.info(
            "Gauges after propagation: %s",
            f"{assign_df_full['gauge_id'].notna().sum():,}",
        )

    if "reg_id" in assign_df_full.columns:
        logger.info(
            "Regulated nodes after propagation: %s",
            f"{assign_df_full['reg_id'].notna().sum():,}",
        )

    logger.info("Step 4: filtering assignment table to target model ids.")
    assign_df = filter_targets(
        assign_df_full=assign_df_full,
        target_model_map_csv=target_model_map_csv,
    )
    logger.debug("Target assignment table shape: %s", assign_df.shape)

    logger.info("Step 5: running SABER assignment.")
    assign_df = mp_assign(assign_df.reset_index(drop=True))
    assign_df = normalize_assignment_columns(assign_df)

    logger.debug(
        "Assignment reason counts:\n%s",
        assign_df["reason"].value_counts(dropna=False).to_string(),
    )

    save_assignment_outputs(
        assign_df_full=assign_df_full,
        assign_df=assign_df,
        tables_dir=tables_dir,
    )

    if make_plot:
        logger.info("Writing diagnostic SABER assignment plot.")
        plot_assignments(
            assign_df=assign_df,
            drain_csv=drain_table,
            target_model_map_csv=target_model_map_csv,
            regulate_csv=regulate_table,
            output_html=assignment_plot_html,
            show=show_plot,
        )
    else:
        logger.debug(
            "Diagnostic SABER assignment plot disabled. "
            "Set HYDRO_MAKE_DIAGNOSTIC_PLOTS=1 to enable it."
        )

    logger.info("Running SABER correction.")
    stats, _ = mp_saber(
        assign_df=assign_df,
        hindcast_zarr=hindcast_zarr,
        gauge_data_dir=gauge_data,
        target_model_map_csv=target_model_map_csv,
    )

    stats_df = pd.DataFrame([asdict(item) for item in stats])

    stats_path = tables_dir / "saber_stats.parquet"
    stats_csv_path = tables_dir / "saber_stats.csv"

    stats_df.to_parquet(stats_path, index=False)
    stats_df.to_csv(stats_csv_path, index=False)

    logger.info("Saved SABER stats: %s", stats_path)
    logger.info("Saved SABER stats CSV: %s", stats_csv_path)

    logger.info("SABER workflow completed.")


# ============================================================
# WORKFLOW
# ============================================================

def run_saber_pipeline(cfg: dict | None = None) -> None:
    cfg = cfg or get_config()

    log_config_summary(cfg)

    config_path = Path(cfg["saber_config_path"])

    make_plot = os.environ.get("HYDRO_MAKE_DIAGNOSTIC_PLOTS", "0") == "1"
    show_plot = os.environ.get("HYDRO_SHOW_PLOTS", "0") == "1"

    diagnostic_plot_dir = cfg["base_output_dir"] / "diagnostic_plots"
    assignment_plot_html = diagnostic_plot_dir / "saber_assignments.html"

    run_saber_workflow(
        config_path=config_path,
        n_clusters=N_CLUSTERS,
        make_plot=make_plot,
        show_plot=show_plot,
        assignment_plot_html=assignment_plot_html,
    )


def main() -> None:
    setup_logging()
    run_saber_pipeline()


if __name__ == "__main__":
    main()