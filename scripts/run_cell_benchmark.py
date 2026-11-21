"""
E2/E3 cell benchmark: gradient boosting on per-cell features under spatial cross-validation,
with the feature-set ablation and hard-negative strata (docs/MASTER_PLAN.md §5.4–§6).

For each [train_year, test_year] run and feature set, fold f is predicted by a model trained
on the train year's cells in every other fold, so test cells are always in unseen blocks
(and, across years, never in a block seen in training).

Needs data/derived/cells_<year>.csv (scripts/build_cell_table.py).

CLI:
    python scripts/run_cell_benchmark.py
Writes results/tables/benchmark_folds.csv (per fold), results/tables/benchmark_summary.csv
(mean ± std over folds) and out-of-fold predictions under results/runs/benchmark/.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.grid import LABEL_IGNORE, LABEL_SLUM, load_benchmark_config  # noqa: E402
from src.eval.cell_metrics import binary_metrics, stratum_fpr  # noqa: E402


def load_cells(cfg, year):
    return pd.read_csv(os.path.join(cfg["paths"]["derived_dir"], f"cells_{year}.csv"))


def formal_district_mask(df, cfg):
    """Cells whose centre lies in one of the named formal-district boxes."""
    s = cfg["strata"]
    with open(s["regions_yaml"]) as f:
        regions = yaml.safe_load(f)["regions"]
    to_utm = Transformer.from_crs("EPSG:4326", cfg["grid"]["crs"], always_xy=True)
    inside = np.zeros(len(df), dtype=bool)
    for name in s["formal_districts"]:
        b = regions[name]["bbox"]
        (x0, x1), (y0, y1) = to_utm.transform([b["lon_min"], b["lon_max"]], [b["lat_min"], b["lat_max"]])
        inside |= df.x.between(min(x0, x1), max(x0, x1)).values & df.y.between(min(y0, y1), max(y0, y1)).values
    return inside


def dense_threshold(cfg):
    """Building presence of a typical (low-quantile) slum cell in the reference year."""
    ref = load_cells(cfg, cfg["reference_year"])
    q = cfg["strata"]["dense_slum_quantile"]
    return float(ref.loc[ref.label == LABEL_SLUM, "ob_presence"].quantile(q))


def feature_columns(df, prefixes):
    return [c for c in df.columns if any(c.startswith(p) for p in prefixes)]


def run(config):
    cfg = load_benchmark_config(config)
    b = cfg["benchmark"]
    dense_min = dense_threshold(cfg)
    print(f"  dense-formal stratum: non-slum cells with ob_presence >= {dense_min:.3f} "
          f"(the {cfg['strata']['dense_slum_quantile']:.0%} quantile of {cfg['reference_year']} slum cells)")
    cells = {y: load_cells(cfg, y) for y in {y for run in b["runs"] for y in run}}
    os.makedirs(b["tables_dir"], exist_ok=True)
    os.makedirs(b["runs_dir"], exist_ok=True)

    rows = []
    for train_year, test_year in b["runs"]:
        train_all = cells[train_year][cells[train_year].label != LABEL_IGNORE]
        test_all = cells[test_year][cells[test_year].label != LABEL_IGNORE].copy()
        test_all["dense"] = test_all.ob_presence >= dense_min
        test_all["formal_district"] = formal_district_mask(test_all, cfg)
        for fset, prefixes in b["feature_sets"].items():
            cols = feature_columns(train_all, prefixes)
            oof = pd.Series(np.nan, index=test_all.index)
            for f in sorted(test_all.fold.unique()):
                tr, te = train_all[train_all.fold != f], test_all[test_all.fold == f]
                model = HistGradientBoostingClassifier(
                    max_iter=b["gbm"]["max_iter"], learning_rate=b["gbm"]["learning_rate"],
                    class_weight="balanced", random_state=b["gbm"]["seed"])
                model.fit(tr[cols], tr.label)
                p = model.predict_proba(te[cols])[:, 1]
                oof[te.index] = p
                m = binary_metrics(te.label, p, b["threshold"])
                dense = stratum_fpr(te.label, p, te.dense, b["threshold"])
                formal = stratum_fpr(te.label, p, te.formal_district, b["threshold"])
                rows.append({"train_year": train_year, "test_year": test_year, "features": fset, "fold": int(f),
                             **m, "fpr_dense": dense["fpr"], "n_dense": dense["n"],
                             "fpr_formal_districts": formal["fpr"], "n_formal_districts": formal["n"]})
            out = test_all[["row", "col", "fold", "label", "slum_frac", "dense", "formal_district"]].assign(prob=oof)
            out.to_csv(os.path.join(b["runs_dir"], f"oof_{train_year}_{test_year}_{fset}.csv"), index=False)

    folds = pd.DataFrame(rows)
    folds.to_csv(os.path.join(b["tables_dir"], "benchmark_folds.csv"), index=False)
    keys = ["train_year", "test_year", "features"]
    metric_cols = ["auc", "balanced_acc", "precision", "recall", "f1", "iou", "fpr", "fpr_dense",
                   "fpr_formal_districts", "pred_pos_rate"]
    summary = folds.groupby(keys, sort=False)[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    counts = folds.groupby(keys, sort=False)[["n_slum", "n_nonslum", "n_dense", "n_formal_districts"]].sum()
    summary = summary.join(counts).reset_index()
    summary.to_csv(os.path.join(b["tables_dir"], "benchmark_summary.csv"), index=False)

    show = ["auc", "balanced_acc", "recall", "fpr", "fpr_dense", "fpr_formal_districts"]
    print(f"\n  {'train→test':10s} {'features':8s} " + " ".join(f"{c:>17s}" for c in show))
    for r in summary.itertuples():
        vals = " ".join(f"{getattr(r, c + '_mean'):8.3f} ± {getattr(r, c + '_std'):5.3f}" for c in show)
        print(f"  {r.train_year}→{r.test_year}  {r.features:8s} {vals}")
    print(f"\n  cells tested per run: slum {summary.n_slum.iloc[0]:,}, non-slum {summary.n_nonslum.iloc[0]:,}, "
          f"dense-formal {summary.n_dense.iloc[0]:,}, formal-district {summary.n_formal_districts.iloc[0]:,} "
          f"(first run)")
    print(f"  [done] {b['tables_dir']}/benchmark_summary.csv, benchmark_folds.csv; predictions in {b['runs_dir']}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="config/benchmark.yaml")
    run(p.parse_args().config)
