"""
Score GRAM slum-probability maps against the ESA reference, at pixel level and on the
100 m benchmark cells, side by side with our cell model on exactly the same cells
(docs/MASTER_PLAN.md §3.3, E1).

Input: a directory of GRAM outputs, one 2-D probability array per Web-Mercator XYZ tile,
named like <anything>_x<X>_y<Y>_prob.npy (the convention of the gram-zero-shot branch),
plus the tile zoom. Works for the old zoom-16 zero-shot run and for the planned zoom-18
run with test-time adaptation.

Needs the ESA shapefile, data/derived/grid.{json,npz}, data/derived/cells_<year>.csv and,
for the comparison, out-of-fold predictions from scripts/run_cell_benchmark.py.

CLI:
    python scripts/score_gram.py --gram_dir data/raw/gram_zero_shot_z16/outputs --zoom 16 --tag z16_zeroshot
Writes results/tables/gram_<tag>_pixels.csv, gram_<tag>_cells.csv (side by side) and
gram_<tag>_cell_probs.csv.
"""

import argparse
import glob
import os
import re
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.grid import LABEL_IGNORE, load_benchmark_config, load_grid  # noqa: E402
from src.eval.cell_metrics import binary_metrics, stratum_fpr  # noqa: E402

WEB_MERC_HALF = 20037508.342789244


def tile_pixel_centres(x, y, zoom, h, w):
    """Web-Mercator coordinates of the pixel centres of XYZ tile (x, y)."""
    size = 2 * WEB_MERC_HALF / 2 ** zoom
    x0, y0 = -WEB_MERC_HALF + x * size, WEB_MERC_HALF - y * size
    cx = x0 + (np.arange(w) + 0.5) * size / w
    cy = y0 - (np.arange(h) + 0.5) * size / h
    return np.meshgrid(cx, cy)


def load_gram_pixels(gram_dir, zoom, grid_crs):
    """All GRAM pixels as a table: location, tile, x/y in the grid CRS, probability."""
    to_grid = Transformer.from_crs("EPSG:3857", grid_crs, always_xy=True)
    parts = []
    for f in sorted(glob.glob(os.path.join(gram_dir, "*_prob.npy"))):
        m = re.search(r"(.*)_x(\d+)_y(\d+)_prob\.npy$", os.path.basename(f))
        if not m:
            continue
        loc, x, y = m.group(1), int(m.group(2)), int(m.group(3))
        p = np.load(f)
        p = p.astype("float32") / 255.0 if p.dtype == np.uint8 else p.astype("float32")  # Kaggle maps are uint8
        mx, my = tile_pixel_centres(x, y, zoom, *p.shape)
        gx, gy = to_grid.transform(mx.ravel(), my.ravel())
        parts.append(pd.DataFrame({"location": loc, "tile": f"{x}_{y}", "x": gx, "y": gy, "prob": p.ravel()}))
    if not parts:
        raise FileNotFoundError(f"No *_x<X>_y<Y>_prob.npy files in {gram_dir}")
    return pd.concat(parts, ignore_index=True)


def run(gram_dir, zoom, tag, config, features):
    cfg = load_benchmark_config(config)
    thr, tables = cfg["benchmark"]["threshold"], cfg["benchmark"]["tables_dir"]
    grid, frac = load_grid(os.path.join(cfg["paths"]["derived_dir"], "grid"))
    year = cfg["reference_year"]
    px = load_gram_pixels(gram_dir, zoom, grid.crs)

    # Pixel level: ESA polygon membership inside the mapped area (outside it absence means nothing)
    esa = gpd.read_file(cfg["paths"]["esa_shp"]).to_crs(grid.crs)
    slum = esa.geometry.buffer(0).union_all()
    mapped = gpd.read_file(cfg["paths"]["mapped_area"]).to_crs(grid.crs).union_all()
    shapely.prepare(slum)
    shapely.prepare(mapped)
    px["esa_slum"] = shapely.contains_xy(slum, px.x.values, px.y.values)
    px["in_mapped"] = shapely.contains_xy(mapped, px.x.values, px.y.values)
    pix_rows = []
    for loc, g in px[px.in_mapped].groupby("location"):
        pix_rows.append({"location": loc, "pixels": len(g), "esa_slum_share": g.esa_slum.mean(),
                         **binary_metrics(g.esa_slum, g.prob, thr)})
    g = px[px.in_mapped]
    pix_rows.append({"location": "ALL", "pixels": len(g), "esa_slum_share": g.esa_slum.mean(),
                     **binary_metrics(g.esa_slum, g.prob, thr)})
    pix = pd.DataFrame(pix_rows)

    # Cell level: mean GRAM probability per benchmark cell, keeping well-covered cells
    col = np.floor((px.x - grid.left) / grid.cell_m).astype(int)
    row = np.floor((grid.top - px.y) / grid.cell_m).astype(int)
    px["row"], px["col"] = row, col
    cell = px.groupby(["row", "col"]).agg(gram_prob=("prob", "mean"), n_px=("prob", "size")).reset_index()
    side = int(round(np.sqrt(px.groupby("tile").size().iloc[0])))  # pixels per tile side (256, or 64 for sweeps)
    pixel_m = 2 * WEB_MERC_HALF / 2 ** zoom / side * np.cos(np.radians(23.75))
    full = (grid.cell_m / pixel_m) ** 2
    cell = cell[cell.n_px >= 0.9 * full]
    cells = pd.read_csv(os.path.join(cfg["paths"]["derived_dir"], f"cells_{year}.csv"))
    cell = cell.merge(cells[["row", "col", "label", "slum_frac", "ob_presence", "ob_height_mean"]], on=["row", "col"])
    cell.to_csv(os.path.join(tables, f"gram_{tag}_cell_probs.csv"), index=False)

    lab = cell[cell.label != LABEL_IGNORE].copy()
    ours = pd.read_csv(os.path.join(cfg["benchmark"]["runs_dir"], f"oof_{year}_{year}_{features}.csv"))
    lab = lab.merge(ours[["row", "col", "prob", "lookalike"]].rename(columns={"prob": "ours_prob"}),
                    on=["row", "col"], how="left")
    cmp_rows = []
    for name, p in [("GRAM", lab.gram_prob), (f"ours ({features})", lab.ours_prob)]:
        m = binary_metrics(lab.label, p, thr)
        cmp_rows.append({"model": name, "cells": len(lab), **m,
                         "fpr_lookalike": stratum_fpr(lab.label, p, lab.lookalike.fillna(False), thr)["fpr"]})
    cmp = pd.DataFrame(cmp_rows)
    pix.to_csv(os.path.join(tables, f"gram_{tag}_pixels.csv"), index=False)
    cmp.to_csv(os.path.join(tables, f"gram_{tag}_cells.csv"), index=False)

    pd.set_option("display.width", 160)
    print(f"  GRAM pixels: {len(px):,} ({int(px.in_mapped.sum()):,} inside the ESA mapped area), pixel ~{pixel_m:.2f} m")
    print("\n  Pixel level vs ESA (inside mapped area):\n" + pix[["location", "pixels", "esa_slum_share", "auc", "precision",
          "recall", "iou", "fpr", "pred_pos_rate"]].round(3).to_string(index=False))
    print(f"\n  Cell level, same {len(lab)} labelled cells ({int(lab.label.sum())} slum):\n" + cmp[["model", "auc", "balanced_acc",
          "precision", "recall", "fpr", "fpr_lookalike", "pred_pos_rate"]].round(3).to_string(index=False))
    print(f"  [done] {tables}/gram_{tag}_pixels.csv, gram_{tag}_cells.csv, gram_{tag}_cell_probs.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--gram_dir", required=True)
    p.add_argument("--zoom", type=int, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--features", default="all", help="feature set of our model to compare against")
    p.add_argument("--config", default="config/benchmark.yaml")
    a = p.parse_args()
    run(a.gram_dir, a.zoom, a.tag, a.config, a.features)
