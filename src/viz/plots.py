"""
All paper figures. Each function reads from all_runs.csv and saves PNG + PDF.
Never hardcode numbers — every figure is regenerated from recorded results.

Functions:
  fig_failure_repro       — Fig 1: failure mode bar chart
  fig_exp1_sasnet         — Fig 2: atmospheric correction ablation
  fig_exp2_ablation       — Fig 3: central socioeconomic+language fusion ablation (money figure)
  fig_exp3_loro           — Fig 4: leave-one-region-out heatmap
  fig_master_table        — Fig 5: master comparison table
  fig_pr_curves           — Fig 7: precision-recall curves
  fig_confidence_strata   — Fig 8: label agreement stacked bars
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from .palette import (
    CONFIG_COLORS, REGION_COLORS, CHANNEL_COLORS, DPI,
    NAVY, TEAL, STEEL, SLATE, CORAL, SAND, apply_style
)


def _load_runs(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Results CSV not found: {csv_path}. "
            "Run experiments first (Phase 4 of BanglaSlumNet_Colab.ipynb)."
        )
    return pd.read_csv(str(p))


def _save_fig(fig: plt.Figure, output_dir: str, name: str, wide: bool = False):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(str(out / f"{name}.{ext}"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── Figure 1: failure mode reproduction ──────────────────────────────────────
def fig_failure_repro(csv_path: str, output_dir: str, wide: bool = False) -> plt.Figure:
    apply_style(wide)
    df = _load_runs(csv_path)

    configs = ["baseline_cnn", "full"]
    regions = ["korail", "old_dhaka", "gulshan_baridhara"]
    metric = "metric_fpr_control"

    fig, ax = plt.subplots(figsize=(5 if not wide else 7, 3.5))
    x = np.arange(len(regions))
    width = 0.25

    # Also include gram if available
    all_configs = ["gram", "baseline_cnn", "full"]
    offsets = np.linspace(-width, width, len(all_configs))

    for i, cfg in enumerate(all_configs):
        vals = []
        for reg in regions:
            row = df[(df["backbone_config"] == cfg) | (df["run_id"].str.startswith("gram"))]
            if cfg == "gram":
                row = df[df["run_id"].str.startswith("gram")]
            else:
                row = df[df["backbone_config"] == cfg]
            val = row.get(metric, pd.Series([float("nan")])).mean()
            vals.append(val if pd.notna(val) else 0.0)

        color = CONFIG_COLORS.get(cfg, SLATE)
        ax.bar(x + offsets[i], vals, width, label=cfg, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", "\n") for r in regions])
    ax.set_ylabel("False Positive Rate on control")
    ax.set_title("Failure mode: FPR on formal-dense control regions")
    ax.legend(frameon=False)
    ax.set_ylim(0, 1)
    _save_fig(fig, output_dir, "fig1_failure_repro", wide)
    return fig


# ── Figure 2: SAS-Net ablation ────────────────────────────────────────────────
def fig_exp1_sasnet(csv_path: str, output_dir: str, wide: bool = False) -> plt.Figure:
    apply_style(wide)
    df = _load_runs(csv_path)
    exp_df = df[df["experiment"] == "exp1_sasnet_ablation"]

    conditions = ["none", "classical", "sasnet"]
    labels = ["Raw", "Classical", "SAS-Net (ours)"]
    colors = [SLATE, STEEL, NAVY]

    hc_ious = [exp_df[exp_df["run_id"].str.contains(c)]["metric_hc_iou"].mean() for c in conditions]
    all_ious = [exp_df[exp_df["run_id"].str.contains(c)]["metric_all_iou"].mean() for c in conditions]

    fig, ax = plt.subplots(figsize=(4 if not wide else 6, 3.5))
    x = np.arange(len(conditions))
    w = 0.35
    ax.bar(x - w/2, hc_ious, w, label="HC-IoU", color=[NAVY]*3, alpha=0.85)
    ax.bar(x + w/2, all_ious, w, label="All-IoU", color=[TEAL]*3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("IoU")
    ax.set_title("Experiment 1: Atmospheric correction ablation")
    ax.legend(frameon=False)
    ax.set_ylim(0, 1)
    _save_fig(fig, output_dir, "fig2_exp1_sasnet", wide)
    return fig


# ── Figure 3: Central fusion ablation (money figure) ─────────────────────────
def fig_exp2_ablation(csv_path: str, output_dir: str, wide: bool = False) -> plt.Figure:
    apply_style(wide)
    df = _load_runs(csv_path)
    exp_df = df[df["experiment"] == "exp2_fusion_ablation"]

    # Ordered sequence of ablation steps
    step_ids = [
        "exp2_visual_only",
        "exp2_vlm_lang",
        "exp2_viirs_only",
        "exp2_viirs_pop",
        "exp2_viirs_pop_roads",
        "exp2_viirs_pop_roads_poverty",
        "exp2_full",
    ]
    step_labels = [
        "Visual\nonly",
        "+Lang\nconcept",
        "+VIIRS\n(NTL)",
        "+Pop",
        "+Roads",
        "+Poverty",
        "Full\nfusion",
    ]

    hc_ious = []
    fpr_vals = []
    for sid in step_ids:
        row = exp_df[exp_df["run_id"] == sid]
        hc_ious.append(row["metric_hc_iou"].values[0] if len(row) else float("nan"))
        fpr_vals.append(row["metric_fpr_control_old_dhaka"].values[0] if len(row) else float("nan"))

    x = np.arange(len(step_ids))
    fig, ax1 = plt.subplots(figsize=(6 if not wide else 8, 4))
    ax2 = ax1.twinx()

    ax1.plot(x, hc_ious, "o-", color=NAVY, label="HC-IoU (↑)", linewidth=2)
    ax2.plot(x, fpr_vals, "s--", color=CORAL, label="FPR Old Dhaka (↓)", linewidth=2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(step_labels, fontsize=8)
    ax1.set_ylabel("HC-IoU", color=NAVY)
    ax2.set_ylabel("FPR on Old Dhaka", color=CORAL)
    ax1.set_ylim(0, 1); ax2.set_ylim(0, 1)
    ax1.set_title("Experiment 2: Socioeconomic + language fusion ablation")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="lower right")

    _save_fig(fig, output_dir, "fig3_exp2_ablation", wide)
    return fig


# ── Figure 4: LORO generalization ────────────────────────────────────────────
def fig_exp3_loro(csv_path: str, output_dir: str, wide: bool = False) -> plt.Figure:
    apply_style(wide)
    df = _load_runs(csv_path)
    exp_df = df[df["experiment"] == "exp3_loro"]

    regions = ["korail", "bhashantek", "karail_extension", "old_dhaka", "gulshan_baridhara"]
    hc_ious = [exp_df[exp_df["run_id"] == f"exp3_loro_{r}"]["metric_hc_iou"].values
               for r in regions]
    hc_ious = [v[0] if len(v) else float("nan") for v in hc_ious]

    fig, ax = plt.subplots(figsize=(5 if not wide else 7, 3.5))
    colors = [REGION_COLORS.get(r, SLATE) for r in regions]
    ax.bar(range(len(regions)), hc_ious, color=colors, alpha=0.85)
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels([r.replace("_", "\n") for r in regions], fontsize=8)
    ax.set_ylabel("HC-IoU (held-out region)")
    ax.set_title("Experiment 3: Leave-one-region-out generalization")
    ax.set_ylim(0, 1)
    _save_fig(fig, output_dir, "fig4_exp3_loro", wide)
    return fig


# ── Figure 5: Master comparison table ────────────────────────────────────────
def fig_master_table(csv_path: str, output_dir: str, wide: bool = False) -> plt.Figure:
    apply_style(wide)
    df = _load_runs(csv_path)

    cols = ["run_id", "backbone_config", "metric_hc_iou", "metric_all_iou",
            "metric_precision", "metric_recall", "metric_f1",
            "metric_fpr_control_old_dhaka", "metric_korail_recall"]
    avail = [c for c in cols if c in df.columns]
    table_df = df[avail].copy()
    table_df = table_df.round(3)

    fig, ax = plt.subplots(figsize=(10 if wide else 8, 0.4 * max(len(table_df), 3) + 1))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.3)
    ax.set_title("Master comparison table", fontsize=9, y=0.98)
    _save_fig(fig, output_dir, "fig5_master_table", wide)
    return fig


# ── Figure 7: PR curves ───────────────────────────────────────────────────────
def fig_pr_curves(csv_path: str, output_dir: str, wide: bool = False) -> plt.Figure:
    apply_style(wide)
    df = _load_runs(csv_path)

    fig, ax = plt.subplots(figsize=(4 if not wide else 6, 4))
    for cfg, color in CONFIG_COLORS.items():
        row = df[df["backbone_config"] == cfg]
        if len(row) == 0:
            continue
        p = row["metric_precision"].mean()
        r = row["metric_recall"].mean()
        ax.scatter([r], [p], color=color, s=60, label=cfg, zorder=5)
        # Placeholder iso-F1 curves would go here with per-threshold data
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Precision-Recall (HC test set)")
    ax.legend(frameon=False)
    _save_fig(fig, output_dir, "fig7_pr_curves", wide)
    return fig


# ── Figure 8: Confidence strata ───────────────────────────────────────────────
def fig_confidence_strata(confidence_json: str, output_dir: str, wide: bool = False) -> plt.Figure:
    import json
    apply_style(wide)
    with open(confidence_json) as f:
        data = json.load(f)

    by_region = data.get("by_region", {})
    if not by_region:
        # Use tiles list to build summary
        tiles = data.get("tiles", [])
        for t in tiles:
            r = t["tile_id"].split("_")[0]
            if r not in by_region:
                by_region[r] = {"n0": 0, "n1": 0, "n2": 0, "n3": 0}
            for k in range(4):
                by_region[r][f"n{k}"] = by_region[r].get(f"n{k}", 0) + t.get(f"agreement_{k}", 0)

    regions = list(by_region.keys())
    agree_keys = ["agreement_0", "agreement_1", "agreement_2", "agreement_3"]
    colors = [SLATE, STEEL, TEAL, NAVY]
    labels = ["0-signal", "1-signal", "2-signal", "3-signal (HC)"]

    fig, ax = plt.subplots(figsize=(5 if not wide else 7, 3.5))
    bottoms = np.zeros(len(regions))
    for ki, (key, color, label) in enumerate(zip(agree_keys, colors, labels)):
        vals = np.array([by_region[r].get(key, 0) for r in regions], dtype=float)
        ax.bar(range(len(regions)), vals, bottom=bottoms, color=color, label=label, alpha=0.85)
        bottoms += vals

    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels([r.replace("_", "\n") for r in regions], fontsize=8)
    ax.set_ylabel("Pixel count")
    ax.set_title("Label agreement strata by region")
    ax.legend(frameon=False, loc="upper right")
    _save_fig(fig, output_dir, "fig8_confidence_strata", wide)
    return fig


# ── Synthetic smoke tests ──────────────────────────────────────────────────────
def _smoke_test():
    """Verify every figure function runs on synthetic data without errors."""
    import tempfile, json, csv

    with tempfile.TemporaryDirectory() as tmp:
        # Write minimal synthetic all_runs.csv
        csv_path = f"{tmp}/all_runs.csv"
        rows = []
        for cfg in ["baseline_cnn", "vlm_visual", "vlm_lang", "full"]:
            for exp_id, exp in [("exp2_visual_only", "exp2_fusion_ablation"),
                                  ("exp2_full", "exp2_fusion_ablation"),
                                  ("exp1_raw_vlmvisual", "exp1_sasnet_ablation"),
                                  ("exp3_loro_korail", "exp3_loro")]:
                rows.append({
                    "run_id": exp_id, "experiment": exp, "backbone_config": cfg,
                    "metric_hc_iou": 0.65, "metric_all_iou": 0.55,
                    "metric_precision": 0.70, "metric_recall": 0.60,
                    "metric_f1": 0.65, "metric_fpr_control_old_dhaka": 0.20,
                    "metric_fpr_control_gulshan": 0.15, "metric_korail_recall": 0.80,
                })
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        # Write minimal confidence.json
        conf_path = f"{tmp}/confidence.json"
        with open(conf_path, "w") as f:
            json.dump({
                "tiles": [{"tile_id": "korail_0001", "agreement_0": 100, "agreement_1": 200,
                            "agreement_2": 300, "agreement_3": 400}],
                "by_region": {}
            }, f)

        for fn, kwargs in [
            (fig_failure_repro, {"csv_path": csv_path}),
            (fig_exp1_sasnet,   {"csv_path": csv_path}),
            (fig_exp2_ablation, {"csv_path": csv_path}),
            (fig_exp3_loro,     {"csv_path": csv_path}),
            (fig_master_table,  {"csv_path": csv_path}),
            (fig_pr_curves,     {"csv_path": csv_path}),
            (fig_confidence_strata, {"confidence_json": conf_path}),
        ]:
            fn(output_dir=tmp, **kwargs)

    print("plots.py smoke test passed (all figures generated on synthetic data).")


if __name__ == "__main__":
    _smoke_test()
