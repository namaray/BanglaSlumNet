"""
Assemble the per-cell benchmark tables: labels, spatial folds and features for each year
(docs/MASTER_PLAN.md §5.2–§5.3).

Needs data/derived/grid.{json,npz} (scripts/build_benchmark_grid.py) and
data/derived/features_<year>.tif (gee/export_cell_features.py) for every year in
config/benchmark.yaml.

Folds are drawn once from the reference-year labels and reused for every year, so the
same place is never in training for one year and in testing for another. Later years
mask Open Buildings change since the reference year (grid.assign_labels).

CLI:
    python scripts/build_cell_table.py
Writes data/derived/cells_<year>.csv (one row per grid cell; label -1 = ignore).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.grid import (LABEL_IGNORE, LABEL_NONSLUM, LABEL_SLUM, assign_folds,  # noqa: E402
                           assign_labels, load_benchmark_config, load_grid)


def read_features(path, grid):
    with rasterio.open(path) as src:
        if src.transform != grid.transform() or (src.height, src.width) != (grid.nrows, grid.ncols):
            raise ValueError(f"{path} is not aligned to the benchmark grid; re-export it.")
        return dict(zip(src.descriptions, src.read()))


def main(config):
    cfg = load_benchmark_config(config)
    d = cfg["paths"]["derived_dir"]
    grid, frac = load_grid(os.path.join(d, "grid"))
    ref = cfg["reference_year"]
    feats = {y: read_features(os.path.join(d, f"features_{y}.tif"), grid) for y in cfg["years"]}

    def ob(y):
        return {"building_height": feats[y]["ob_height_mean"], "building_presence": feats[y]["ob_presence"]}

    labels = {y: assign_labels(frac, feats[y][cfg["labels"]["built_band"]], cfg,
                               ob=None if y == ref else ob(y), ob_ref=None if y == ref else ob(ref))
              for y in cfg["years"]}
    folds = assign_folds(grid, labels[ref], cfg)

    rows, cols = np.indices((grid.nrows, grid.ncols))
    x = grid.left + (cols + 0.5) * grid.cell_m
    y_ = grid.top - (rows + 0.5) * grid.cell_m
    for yr in cfg["years"]:
        df = pd.DataFrame({"row": rows.ravel(), "col": cols.ravel(), "x": x.ravel(), "y": y_.ravel(),
                           "fold": folds.ravel(), "label": labels[yr].ravel(),
                           "slum_frac": frac["slum_frac"].ravel(), "mapped": frac["mapped"].ravel()})
        for name, arr in feats[yr].items():
            df[name] = arr.ravel()
        out = os.path.join(d, f"cells_{yr}.csv")
        df.to_csv(out, index=False, float_format="%.5g")
        lab = df[df.label != LABEL_IGNORE]
        per_fold = lab.groupby("fold").label.agg(slum=lambda s: (s == LABEL_SLUM).sum(),
                                                 nonslum=lambda s: (s == LABEL_NONSLUM).sum())
        print(f"  {yr}: slum {int((lab.label == LABEL_SLUM).sum()):,} | non-slum "
              f"{int((lab.label == LABEL_NONSLUM).sum()):,} | per fold (slum/non-slum): "
              + ", ".join(f"{r.slum}/{r.nonslum}" for r in per_fold.itertuples()))
        print(f"  [done] {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="config/benchmark.yaml")
    main(p.parse_args().config)
