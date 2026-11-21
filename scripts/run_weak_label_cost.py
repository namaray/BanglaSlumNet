"""
E4: what do region-box weak labels cost? (docs/MASTER_PLAN.md §6)

Three training arms, each tested on ESA labels in held-out spatial folds:
  a  weak_boxes  built-up cells in the v4 region boxes, labelled by box type
                 (informal -> slum, formal_dense -> non-slum); a cell in overlapping boxes
                 appears once per box, as in the v4 per-region exports, so conflicting
                 boxes give it both labels
  b  esa_boxes   the same box cells, labelled by ESA (cells ESA ignores are dropped)
  c  esa_all     every ESA-labelled cell in the mapped area (the E2 setting)
a vs b isolates label quality; b vs c isolates training coverage. All arms share the
built-up mask (labels.built_band >= labels.built_min) and the features.

Needs data/derived/cells_<year>.csv (scripts/build_cell_table.py).

CLI:
    python scripts/run_weak_label_cost.py
Writes results/tables/weak_label_cost_folds.csv and weak_label_cost_summary.csv.
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
from src.data.grid import LABEL_IGNORE, LABEL_NONSLUM, LABEL_SLUM, load_benchmark_config  # noqa: E402
from src.eval.cell_metrics import binary_metrics, stratum_fpr  # noqa: E402
from run_cell_benchmark import dense_threshold, feature_columns, formal_district_mask, load_cells  # noqa: E402


def box_memberships(df, cfg):
    """One row per (cell, region box) the cell centre falls in, with the box's weak label."""
    w = cfg["weak_label_cost"]
    with open(w["regions_yaml"]) as f:
        regions = yaml.safe_load(f)["regions"]
    to_utm = Transformer.from_crs("EPSG:4326", cfg["grid"]["crs"], always_xy=True)
    built = df[cfg["labels"]["built_band"]] >= cfg["labels"]["built_min"]
    parts = []
    for name, r in regions.items():
        b = r["bbox"]
        (x0, x1), (y0, y1) = to_utm.transform([b["lon_min"], b["lon_max"]], [b["lat_min"], b["lat_max"]])
        inside = df.x.between(min(x0, x1), max(x0, x1)) & df.y.between(min(y0, y1), max(y0, y1)) & built
        weak = LABEL_SLUM if r["type"] == "informal" else LABEL_NONSLUM
        parts.append(df[inside].assign(region=name, weak_label=weak))
    return pd.concat(parts)


def audit(memb):
    """How the weak labels compare with ESA on the box cells (one row per cell)."""
    per_cell = memb.groupby(level=0).weak_label.agg(["min", "max"])
    conflict = per_cell["min"] != per_cell["max"]
    cells = memb[~memb.index.duplicated()]
    esa = cells[cells.label != LABEL_IGNORE]
    weak_slum, weak_non = esa[esa.weak_label == LABEL_SLUM], esa[esa.weak_label == LABEL_NONSLUM]
    print(f"  box cells (built-up): {len(cells):,} distinct, {len(memb):,} training rows; "
          f"{int(conflict.sum()):,} cells carry both labels")
    print(f"  of weak 'slum' cells ESA labels: slum {(weak_slum.label == LABEL_SLUM).mean():.0%}, "
          f"non-slum {(weak_slum.label == LABEL_NONSLUM).mean():.0%} (n={len(weak_slum):,})")
    print(f"  of weak 'non-slum' cells ESA labels: slum {(weak_non.label == LABEL_SLUM).mean():.0%}, "
          f"non-slum {(weak_non.label == LABEL_NONSLUM).mean():.0%} (n={len(weak_non):,})")


def run(config):
    cfg = load_benchmark_config(config)
    b, year = cfg["benchmark"], cfg["weak_label_cost"]["year"]
    cells = load_cells(cfg, year)
    memb = box_memberships(cells, cfg)
    audit(memb)

    box_idx = memb.index.unique()
    arms = {
        "weak_boxes": memb.assign(target=memb.weak_label),
        "esa_boxes": cells.loc[box_idx][cells.loc[box_idx].label != LABEL_IGNORE].assign(target=lambda d: d.label),
        "esa_all": cells[cells.label != LABEL_IGNORE].assign(target=lambda d: d.label),
    }
    test_all = cells[cells.label != LABEL_IGNORE].copy()
    test_all["dense"] = test_all.ob_presence >= dense_threshold(cfg)
    test_all["formal_district"] = formal_district_mask(test_all, cfg)
    test_all["in_boxes"] = test_all.index.isin(box_idx)

    rows = []
    for fset, prefixes in b["feature_sets"].items():
        cols = feature_columns(cells, prefixes)
        for arm, train_all in arms.items():
            for f in sorted(test_all.fold.unique()):
                tr, te = train_all[train_all.fold != f], test_all[test_all.fold == f]
                if tr.target.nunique() < 2:
                    continue
                model = HistGradientBoostingClassifier(
                    max_iter=b["gbm"]["max_iter"], learning_rate=b["gbm"]["learning_rate"],
                    class_weight="balanced", random_state=b["gbm"]["seed"])
                model.fit(tr[cols], tr.target)
                p = model.predict_proba(te[cols])[:, 1]
                m = binary_metrics(te.label, p, b["threshold"])
                inbox = te.in_boxes.values
                m_box = binary_metrics(te.label[inbox], p[inbox], b["threshold"]) if inbox.any() else {}
                rows.append({"features": fset, "arm": arm, "fold": int(f), "n_train": len(tr),
                             "train_slum_share": float((tr.target == LABEL_SLUM).mean()), **m,
                             "fpr_dense": stratum_fpr(te.label, p, te.dense, b["threshold"])["fpr"],
                             "fpr_formal_districts": stratum_fpr(te.label, p, te.formal_district, b["threshold"])["fpr"],
                             "auc_in_boxes": m_box.get("auc", np.nan)})

    folds = pd.DataFrame(rows)
    os.makedirs(b["tables_dir"], exist_ok=True)
    folds.to_csv(os.path.join(b["tables_dir"], "weak_label_cost_folds.csv"), index=False)
    metric_cols = ["auc", "balanced_acc", "precision", "recall", "fpr", "fpr_dense",
                   "fpr_formal_districts", "pred_pos_rate", "auc_in_boxes"]
    summary = folds.groupby(["features", "arm"], sort=False)[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.join(folds.groupby(["features", "arm"], sort=False)[["n_train", "train_slum_share"]].mean())
    summary = summary.reset_index()
    summary.to_csv(os.path.join(b["tables_dir"], "weak_label_cost_summary.csv"), index=False)

    show = ["auc", "balanced_acc", "recall", "fpr", "fpr_dense", "pred_pos_rate"]
    print(f"\n  {'features':8s} {'arm':11s} {'n_train':>7s} " + " ".join(f"{c:>16s}" for c in show))
    for r in summary.itertuples():
        vals = " ".join(f"{getattr(r, c + '_mean'):7.3f} ± {getattr(r, c + '_std'):5.3f}" for c in show)
        print(f"  {r.features:8s} {r.arm:11s} {r.n_train:7.0f} {vals}")
    print(f"  [done] {b['tables_dir']}/weak_label_cost_summary.csv, weak_label_cost_folds.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="config/benchmark.yaml")
    run(p.parse_args().config)
