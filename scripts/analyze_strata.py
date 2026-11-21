"""
Breakdowns and sensitivity checks for the cell benchmark (docs/MASTER_PLAN.md §5.4, §11):
  1. recall by ESA slum type (NB_LOCTYP) and by the size of the polygon covering each slum cell
  2. false-positive rate on the dense-formal stratum for different density cuts
  3. how results move when the slum-share threshold for a slum cell changes

1 and 2 reuse the reference-year out-of-fold predictions from scripts/run_cell_benchmark.py;
3 re-labels the reference year and reruns the spatial CV with the same folds.

CLI:
    python scripts/analyze_strata.py
Writes results/tables/strata_slum_type.csv, strata_slum_size.csv,
strata_dense_sensitivity.csv and label_threshold_sensitivity.csv.
"""

import argparse
import copy
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.grid import (LABEL_IGNORE, LABEL_SLUM, assign_labels, load_benchmark_config,  # noqa: E402
                           load_grid)
from src.eval.cell_metrics import binary_metrics, stratum_fpr  # noqa: E402
from run_cell_benchmark import feature_columns, load_cells  # noqa: E402

SIZE_BINS = [0, 100, 1000, np.inf]  # polygon area in 10 m pixels: < 1 ha, 1-10 ha, > 10 ha
SIZE_NAMES = ["< 1 ha", "1-10 ha", "> 10 ha"]


def dominant_polygon(cfg, grid):
    """Per cell: ESA polygon index covering most of the cell (-1 if none)."""
    res = cfg["grid"]["pixel_m"]
    k = int(round(grid.cell_m / res))
    esa = gpd.read_file(cfg["paths"]["esa_shp"]).to_crs(grid.crs)
    ids = rasterize([(g, i + 1) for i, g in enumerate(esa.geometry.buffer(0))],
                    out_shape=(grid.nrows * k, grid.ncols * k), transform=grid.transform(res),
                    fill=0, dtype="int32")
    blocks = ids.reshape(grid.nrows, k, grid.ncols, k).swapaxes(1, 2).reshape(grid.nrows * grid.ncols, k * k)
    dom = np.full(len(blocks), -1)
    for i in np.flatnonzero(blocks.max(axis=1) > 0):
        v = blocks[i][blocks[i] > 0]
        dom[i] = np.bincount(v).argmax() - 1
    return esa, dom


def oof(cfg, fset):
    y = cfg["reference_year"]
    return pd.read_csv(os.path.join(cfg["benchmark"]["runs_dir"], f"oof_{y}_{y}_{fset}.csv"))


def slum_breakdowns(cfg, grid, thr):
    esa, dom = dominant_polygon(cfg, grid)
    size_px = (esa.geometry.area / cfg["grid"]["pixel_m"] ** 2).values
    type_rows, size_rows = [], []
    for fset in cfg["benchmark"]["feature_sets"]:
        o = oof(cfg, fset)
        s = o[o.label == LABEL_SLUM].copy()
        idx = s.row.values * grid.ncols + s.col.values
        poly = dom[idx]
        s["slum_type"] = np.where(poly >= 0, esa.NB_LOCTYP.values[np.maximum(poly, 0)], "unknown")
        s["size"] = pd.cut(np.where(poly >= 0, size_px[np.maximum(poly, 0)], np.nan), SIZE_BINS, labels=SIZE_NAMES)
        s["found"] = s.prob >= thr
        for t, g in s.groupby("slum_type"):
            type_rows.append({"features": fset, "slum_type": t, "cells": len(g), "recall": g.found.mean()})
        for t, g in s.groupby("size", observed=True):
            size_rows.append({"features": fset, "polygon_size": t, "cells": len(g), "recall": g.found.mean()})
    return pd.DataFrame(type_rows), pd.DataFrame(size_rows)


def dense_sensitivity(cfg, thr, quantiles=(0.25, 0.5, 0.75)):
    ref = load_cells(cfg, cfg["reference_year"])
    rows = []
    for fset in cfg["benchmark"]["feature_sets"]:
        o = oof(cfg, fset).merge(ref[["row", "col", "ob_presence"]], on=["row", "col"])
        for q in quantiles:
            cut = ref.loc[ref.label == LABEL_SLUM, "ob_presence"].quantile(q)
            st = stratum_fpr(o.label, o.prob, o.ob_presence >= cut, thr)
            rows.append({"features": fset, "dense_slum_quantile": q, "ob_presence_min": round(float(cut), 3),
                         "nonslum_cells_in_stratum": st["n"], "fpr": st["fpr"]})
    return pd.DataFrame(rows)


def label_threshold_sensitivity(cfg, grid, frac, thr, values=(0.3, 0.5, 0.7)):
    b, y = cfg["benchmark"], cfg["reference_year"]
    cells = load_cells(cfg, y)
    with rasterio.open(os.path.join(cfg["paths"]["derived_dir"], f"features_{y}.tif")) as src:
        built = dict(zip(src.descriptions, src.read()))[cfg["labels"]["built_band"]]
    rows = []
    for v in values:
        c = copy.deepcopy(cfg)
        c["labels"]["slum_min_frac"] = v
        df = cells.assign(label=assign_labels(frac, built, c).ravel())
        lab = df[df.label != LABEL_IGNORE]
        dense_cut = lab.loc[lab.label == LABEL_SLUM, "ob_presence"].quantile(cfg["strata"]["dense_slum_quantile"])
        for fset in ["s2", "all"]:
            cols = feature_columns(df, b["feature_sets"][fset])
            ms = []
            for f in sorted(lab.fold.unique()):
                tr, te = lab[lab.fold != f], lab[lab.fold == f]
                model = HistGradientBoostingClassifier(
                    max_iter=b["gbm"]["max_iter"], learning_rate=b["gbm"]["learning_rate"],
                    class_weight="balanced", random_state=b["gbm"]["seed"]).fit(tr[cols], tr.label)
                p = model.predict_proba(te[cols])[:, 1]
                ms.append({**binary_metrics(te.label, p, thr),
                           "fpr_dense": stratum_fpr(te.label, p, te.ob_presence >= dense_cut, thr)["fpr"]})
            m = pd.DataFrame(ms)
            rows.append({"slum_min_frac": v, "features": fset, "n_slum": int((lab.label == LABEL_SLUM).sum()),
                         "n_nonslum": int((lab.label != LABEL_SLUM).sum()),
                         **{f"{k}_mean": m[k].mean() for k in ["auc", "balanced_acc", "recall", "fpr", "fpr_dense"]}})
    return pd.DataFrame(rows)


def run(config):
    cfg = load_benchmark_config(config)
    thr, tables = cfg["benchmark"]["threshold"], cfg["benchmark"]["tables_dir"]
    grid, frac = load_grid(os.path.join(cfg["paths"]["derived_dir"], "grid"))

    by_type, by_size = slum_breakdowns(cfg, grid, thr)
    dense = dense_sensitivity(cfg, thr)
    label_sens = label_threshold_sensitivity(cfg, grid, frac, thr)
    for name, df in [("strata_slum_type", by_type), ("strata_slum_size", by_size),
                     ("strata_dense_sensitivity", dense), ("label_threshold_sensitivity", label_sens)]:
        df.to_csv(os.path.join(tables, f"{name}.csv"), index=False)

    pd.set_option("display.width", 160)
    print("\n  Recall by ESA slum type:\n" + by_type.pivot(index="slum_type", columns="features", values="recall")
          .join(by_type.groupby("slum_type").cells.first()).round(3).to_string())
    print("\n  Recall by size of the covering polygon:\n" + by_size.pivot(index="polygon_size", columns="features", values="recall")
          .join(by_size.groupby("polygon_size").cells.first()).round(3).to_string())
    print("\n  Dense-formal FPR by density cut:\n" + dense.pivot_table(index=["dense_slum_quantile", "ob_presence_min", "nonslum_cells_in_stratum"],
          columns="features", values="fpr").round(3).to_string())
    print("\n  Slum-share threshold for a slum cell:\n" + label_sens.round(3).to_string(index=False))
    print(f"\n  [done] {tables}/strata_*.csv, label_threshold_sensitivity.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="config/benchmark.yaml")
    run(p.parse_args().config)
