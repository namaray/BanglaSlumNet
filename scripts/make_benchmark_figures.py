"""
Paper figures for the open-data benchmark, drawn from the result tables and maps written by
run_cell_benchmark.py, run_weak_label_cost.py, analyze_strata.py, make_slum_maps.py and
score_gram.py.

Colours follow the dataviz reference palette, validated with its checker: categorical slots
blue / orange / aqua (all-pairs safe for maps), blue vs orange for two-way comparisons, and a
light-to-dark blue ramp for the cumulative feature sets.

CLI:
    python scripts/make_benchmark_figures.py
Writes results/figures/benchmark/fig_*.png and .pdf.
"""

import argparse
import os
import sys

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from shapely.geometry import box  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.grid import load_benchmark_config, load_grid  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = {"s2": "#a8c8ef", "s2_ob": "#2a78d6", "all": "#12407a"}     # cumulative feature sets
FSET_NAME = {"s2": "Sentinel-2", "s2_ob": "+ buildings", "all": "+ land cover & context"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2de"
NEUTRAL_LIGHT, NEUTRAL_MID = "#ecebe7", "#b9b8b2"


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def save(fig, out_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  [done] {name}")


def top_legend(fig, ax, title="Inputs"):
    """One legend above all panels, so it never covers data."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(labels),
               title=title, title_fontsize=7)


def grouped_bars(ax, groups, series, values, errors, colors, labels):
    """Bars grouped by metric; 2 px-equivalent gaps between adjacent bars."""
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, s in enumerate(series):
        ax.bar(x + (i - (n - 1) / 2) * width, values[s], width * 0.92, yerr=errors[s], color=colors[s],
               label=labels[s], error_kw={"elinewidth": 0.6, "capsize": 1.5, "ecolor": MUTED})
    ax.set_xticks(x, groups)
    ax.grid(axis="x", visible=False)


def fig_label_audit(cfg, out):
    """Share of each v4 region box that ESA maps as slum, grouped by the box's label."""
    esa = gpd.read_file(cfg["paths"]["esa_shp"]).to_crs(cfg["grid"]["crs"])
    slum = esa.geometry.buffer(0).union_all()
    with open(cfg["weak_label_cost"]["regions_yaml"]) as f:
        regions = yaml.safe_load(f)["regions"]
    rows = []
    for name, r in regions.items():
        b = r["bbox"]
        bx = gpd.GeoSeries([box(b["lon_min"], b["lat_min"], b["lon_max"], b["lat_max"])], crs=4326) \
            .to_crs(cfg["grid"]["crs"]).iloc[0]
        rows.append({"region": name.replace("_", " "), "informal": r["type"] == "informal",
                     "share": slum.intersection(bx).area / bx.area * 100})
    df = pd.DataFrame(rows).sort_values(["informal", "share"])
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    colors = [BLUE if inf else ORANGE for inf in df.informal]
    ax.barh(df.region, df.share, color=colors, height=0.72)
    for y, v in enumerate(df.share):
        ax.text(v + 0.6, y, f"{v:.0f}%", va="center", fontsize=6.5, color=MUTED)
    ax.set_xlabel("Share of region box mapped as slum by ESA (%)")
    ax.set_xlim(0, max(df.share) * 1.2)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=BLUE, label="labelled slum (informal box)"),
                       Patch(color=ORANGE, label="labelled non-slum (formal box)")],
              loc="lower right")
    save(fig, out, "fig_label_audit")


def fig_ablation(cfg, out):
    s = pd.read_csv(os.path.join(cfg["benchmark"]["tables_dir"], "benchmark_summary.csv"))
    ref = cfg["reference_year"]
    s = s[(s.train_year == ref) & (s.test_year == ref)].set_index("features")
    fsets = list(cfg["benchmark"]["feature_sets"])
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.4), gridspec_kw={"width_ratios": [2, 3]})
    for ax, metrics, title in [
        (axes[0], {"auc": "AUC", "recall": "Recall"}, "Higher is better"),
        (axes[1], {"fpr": "All non-slum", "fpr_dense": "Dense formal", "fpr_lookalike": "Dense low-rise\nlook-alikes"},
         "False-positive rate (lower is better)")]:
        vals = {f: [s.loc[f, f"{m}_mean"] for m in metrics] for f in fsets}
        errs = {f: [s.loc[f, f"{m}_std"] for m in metrics] for f in fsets}
        grouped_bars(ax, list(metrics.values()), fsets, vals, errs, RAMP, FSET_NAME)
        ax.set_title(title, loc="left", color=INK)
        ax.set_ylim(0, 1)
    top_legend(fig, axes[0])
    save(fig, out, "fig_ablation")


def fig_weak_labels(cfg, out, features="s2_ob"):
    s = pd.read_csv(os.path.join(cfg["benchmark"]["tables_dir"], "weak_label_cost_summary.csv"))
    s = s[s.features == features].set_index("arm")
    arms = ["weak_boxes", "esa_boxes", "esa_all"]
    names = ["Region-box\nlabels", "ESA labels,\nsame cells", "ESA labels,\nall cells"]
    colors = [ORANGE, "#a8c8ef", BLUE]
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.3))
    for ax, m, title in [(axes[0], "auc", "AUC"), (axes[1], "fpr", "False-positive rate")]:
        mean, sd = s.loc[arms, f"{m}_mean"], s.loc[arms, f"{m}_std"]
        ax.bar(names, mean, yerr=sd, color=colors, width=0.68,
               error_kw={"elinewidth": 0.6, "capsize": 1.5, "ecolor": MUTED})
        for i, (v, e) in enumerate(zip(mean, sd)):
            ax.text(i, v + e + 0.03, f"{v:.2f}", ha="center", fontsize=6.5, color=MUTED)
        ax.set_title(title, loc="left", color=INK)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="x", visible=False)
    save(fig, out, "fig_weak_labels")


def fig_gram_roc(cfg, out, tag="z16_zeroshot", features="all"):
    t = cfg["benchmark"]["tables_dir"]
    cell = pd.read_csv(os.path.join(t, f"gram_{tag}_cell_probs.csv"))
    cell = cell[cell.label != -1]
    y = cfg["reference_year"]
    ours = pd.read_csv(os.path.join(cfg["benchmark"]["runs_dir"], f"oof_{y}_{y}_{features}.csv"))
    d = cell.merge(ours[["row", "col", "prob"]], on=["row", "col"])
    auc = pd.read_csv(os.path.join(t, f"gram_{tag}_cells.csv")).set_index("model").auc
    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    ax.plot([0, 1], [0, 1], color=NEUTRAL_MID, lw=1, ls="--")
    for p, color, name, a in [(d.prob, BLUE, "Ours", auc.iloc[1]), (d.gram_prob, ORANGE, "GRAM, zero-shot", auc.iloc[0])]:
        fpr, tpr, _ = roc_curve(d.label, p)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC {a:.2f})")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)  # below the plot: no free corner
    ax.set_title(f"Same {len(d)} cells ({int(d.label.sum())} slum)", loc="left", color=INK)
    save(fig, out, "fig_gram_roc")


def _map_axes(ax, grid, mapped, esa=None):
    left, bottom, right, top = grid.bounds
    mapped.boundary.plot(ax=ax, color=INK, lw=0.5)
    if esa is not None:
        esa.boundary.plot(ax=ax, color=INK, lw=0.15)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_maps(cfg, out):
    grid, _ = load_grid(os.path.join(cfg["paths"]["derived_dir"], "grid"))
    mp = cfg["maps"]["maps_dir"]
    years = sorted(cfg["years"])
    first, last = years[0], years[-1]
    mapped = gpd.read_file(cfg["paths"]["mapped_area"]).to_crs(grid.crs)
    esa = gpd.read_file(cfg["paths"]["esa_shp"]).to_crs(grid.crs)
    left, bottom, right, top = grid.bounds
    ext = (left, right, bottom, top)
    with rasterio.open(os.path.join(mp, f"slum_prob_{last}.tif")) as src:
        prob = src.read(1)
    with rasterio.open(os.path.join(mp, f"slum_change_{first}_{last}.tif")) as src:
        change = src.read(1)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 4.4))
    im = axes[0].imshow(np.ma.masked_invalid(prob), extent=ext, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    _map_axes(axes[0], grid, mapped, esa)
    axes[0].set_title(f"(a) Slum probability {last}, ESA {cfg['reference_year']} outlines", loc="left", color=INK)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.035, pad=0.01)
    cb.outline.set_visible(False)
    cmap = ListedColormap([NEUTRAL_LIGHT, BLUE, ORANGE, AQUA, NEUTRAL_MID])
    axes[1].imshow(np.ma.masked_equal(change, 0), extent=ext, cmap=cmap, vmin=0.5, vmax=5.5, interpolation="nearest")
    _map_axes(axes[1], grid, mapped)
    axes[1].set_title(f"(b) Change {first}→{last} (confident switches only)", loc="left", color=INK)
    names = ["stable non-slum", "stable slum", "new slum", "no longer slum", "uncertain"]
    axes[1].legend(handles=[Patch(color=c, label=n) for c, n in zip(cmap.colors, names)],
                   loc="upper left", bbox_to_anchor=(1.0, 0.98), fontsize=6.5)
    save(fig, out, "fig_maps")


def fig_recall_breakdown(cfg, out):
    t = cfg["benchmark"]["tables_dir"]
    by_type = pd.read_csv(os.path.join(t, "strata_slum_type.csv"))
    by_size = pd.read_csv(os.path.join(t, "strata_slum_size.csv"))
    fsets = list(cfg["benchmark"]["feature_sets"])
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.3))
    for ax, df, key, order, title in [
        (axes[0], by_type, "slum_type", ["High density", "Low density"], "Recall by ESA slum type"),
        (axes[1], by_size, "polygon_size", ["< 1 ha", "1-10 ha", "> 10 ha"], "Recall by slum size")]:
        piv = df.pivot(index=key, columns="features", values="recall").reindex(order)
        n = df.groupby(key).cells.first().reindex(order)
        grouped_bars(ax, [f"{o}\n(n={int(c)})" for o, c in zip(order, n)], fsets,
                     {f: piv[f].values for f in fsets}, {f: None for f in fsets}, RAMP, FSET_NAME)
        ax.set_ylim(0, 1)
        ax.set_title(title, loc="left", color=INK)
    top_legend(fig, axes[0])
    save(fig, out, "fig_recall_breakdown")


def run(config):
    cfg = load_benchmark_config(config)
    out = os.path.join(os.path.dirname(cfg["benchmark"]["tables_dir"]), "figures", "benchmark")
    os.makedirs(out, exist_ok=True)
    style()
    for f in (fig_label_audit, fig_ablation, fig_weak_labels, fig_gram_roc, fig_maps, fig_recall_breakdown):
        f(cfg, out)
    print(f"  figures in {out}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="config/benchmark.yaml")
    run(p.parse_args().config)
