"""
E5: slum probability maps for each year and the 2017 -> 2023 change map, plus a sample of
cells for a visual spot-check (docs/MASTER_PLAN.md §4.2, §6).

Every built-up cell in the mapped area is predicted out of fold: the model for fold f is
trained on that year's labelled cells in the other folds, so no cell's own block is ever
seen in training. Non-built and unmapped cells are left empty.

Needs data/derived/grid.{json,npz} and data/derived/cells_<year>.csv.

CLI:
    python scripts/make_slum_maps.py
Writes results/maps/slum_prob_<year>.tif, results/maps/slum_change_<first>_<last>.tif
(1 stable non-slum, 2 stable slum, 3 new slum, 4 no longer slum, 5 uncertain; 0 = no data),
results/tables/slum_map_summary.csv, and a blind spot-check sheet
results/tables/spotcheck_<first>_<last>.csv (fill is_slum_now: yes / no / unsure for the
~100 m area around each point in current satellite imagery) with its answer key
spotcheck_<first>_<last>_key.csv.

Change requires a confident switch (maps.change_low / maps.change_high); cells that cross 0.5
less decisively are "uncertain", since yearly maps differ by noise far more than slums change.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.grid import LABEL_IGNORE, LABEL_SLUM, load_benchmark_config, load_grid  # noqa: E402
from run_cell_benchmark import feature_columns, load_cells  # noqa: E402

CHANGE = {1: "stable_nonslum", 2: "stable_slum", 3: "new_slum", 4: "no_longer_slum", 5: "uncertain"}


def predict_year(cfg, year):
    b, mp = cfg["benchmark"], cfg["maps"]
    df = load_cells(cfg, year)
    cols = feature_columns(df, b["feature_sets"][mp["features"]])
    target = (df.mapped == 1) & (df[cfg["labels"]["built_band"]] >= cfg["labels"]["built_min"])
    lab = df[df.label != LABEL_IGNORE]
    prob = np.full(len(df), np.nan, dtype="float32")
    for f in sorted(df.fold.unique()):
        tr = lab[lab.fold != f]
        model = HistGradientBoostingClassifier(
            max_iter=b["gbm"]["max_iter"], learning_rate=b["gbm"]["learning_rate"],
            class_weight="balanced", random_state=b["gbm"]["seed"]).fit(tr[cols], tr.label)
        sel = (target & (df.fold == f)).values
        if sel.any():
            prob[sel] = model.predict_proba(df.loc[sel, cols])[:, 1]
    return df, prob


def write_raster(path, arr, grid, dtype, nodata):
    with rasterio.open(path, "w", driver="GTiff", height=grid.nrows, width=grid.ncols, count=1,
                       dtype=dtype, crs=grid.crs, transform=grid.transform(), nodata=nodata,
                       compress="deflate") as dst:
        dst.write(arr.astype(dtype), 1)


def run(config):
    cfg = load_benchmark_config(config)
    mp, thr = cfg["maps"], cfg["benchmark"]["threshold"]
    grid, _ = load_grid(os.path.join(cfg["paths"]["derived_dir"], "grid"))
    os.makedirs(mp["maps_dir"], exist_ok=True)
    tables = cfg["benchmark"]["tables_dir"]
    os.makedirs(tables, exist_ok=True)
    cell_km2 = grid.cell_m ** 2 / 1e6

    years = sorted(cfg["years"])
    probs, summary, ref_df = {}, [], None
    for y in years:
        df, p = predict_year(cfg, y)
        ref_df = df if ref_df is None else ref_df
        probs[y] = p
        write_raster(os.path.join(mp["maps_dir"], f"slum_prob_{y}.tif"),
                     p.reshape(grid.nrows, grid.ncols), grid, "float32", np.nan)
        pred = p >= thr
        lab = df.label.values
        summary.append({"year": y, "cells_mapped_built": int(np.isfinite(p).sum()),
                        "pred_slum_cells": int(np.nansum(pred)), "pred_slum_km2": float(np.nansum(pred) * cell_km2),
                        "esa_slum_cells": int((lab == LABEL_SLUM).sum()),
                        "recall_on_esa_slum": float(pred[lab == LABEL_SLUM].mean()),
                        "pred_slum_outside_esa_labels": int((pred & (lab == LABEL_IGNORE)).sum())})

    first, last = years[0], years[-1]
    a, b = probs[first], probs[last]
    valid = np.isfinite(a) & np.isfinite(b)
    sa, sb = a >= thr, b >= thr
    lo, hi = mp["change_low"], mp["change_high"]
    change = np.zeros(a.shape, dtype="uint8")
    change[valid & ~sa & ~sb] = 1
    change[valid & sa & sb] = 2
    change[valid & (sa != sb)] = 5
    change[valid & (a < lo) & (b >= hi)] = 3
    change[valid & (a >= hi) & (b < lo)] = 4
    write_raster(os.path.join(mp["maps_dir"], f"slum_change_{first}_{last}.tif"),
                 change.reshape(grid.nrows, grid.ncols), grid, "uint8", 0)

    summ = pd.DataFrame(summary)
    for code, name in CHANGE.items():
        summ.loc[summ.year == last, f"{name}_km2"] = float((change == code).sum() * cell_km2)
    summ.to_csv(os.path.join(tables, "slum_map_summary.csv"), index=False)

    # Spot-check sample: cells per change class. The checker gets a blind sheet (location and
    # link only, shuffled); the class and probabilities go to a separate key.
    rng = np.random.default_rng(mp["spotcheck_seed"])
    to_ll = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    picks = []
    for code, name in CHANGE.items():
        idx = np.flatnonzero(change == code)
        for i in rng.choice(idx, size=min(mp["spotcheck_per_class"], len(idx)), replace=False):
            lon, lat = to_ll.transform(ref_df.x.iat[i], ref_df.y.iat[i])
            picks.append({"change": name, "row": int(ref_df.row.iat[i]), "col": int(ref_df.col.iat[i]),
                          "lat": round(lat, 6), "lon": round(lon, 6),
                          f"prob_{first}": round(float(a[i]), 3), f"prob_{last}": round(float(b[i]), 3),
                          "esa_slum_frac": round(float(ref_df.slum_frac.iat[i]), 2)})
    key = pd.DataFrame(picks).sample(frac=1, random_state=mp["spotcheck_seed"]).reset_index(drop=True)
    key.insert(0, "id", range(1, len(key) + 1))
    key.to_csv(os.path.join(tables, f"spotcheck_{first}_{last}_key.csv"), index=False)
    spot = key[["id", "lat", "lon"]].assign(
        google_maps=[f"https://www.google.com/maps/@{la:.6f},{lo:.6f},19z/data=!3m1!1e3"
                     for la, lo in zip(key.lat, key.lon)],
        is_slum_now="", notes="")
    spot.to_csv(os.path.join(tables, f"spotcheck_{first}_{last}.csv"), index=False)

    for r in summ.itertuples():
        print(f"  {r.year}: predicted slum {r.pred_slum_km2:.1f} km2 ({r.pred_slum_cells:,} cells) | "
              f"recall on ESA slum cells {r.recall_on_esa_slum:.0%} | predicted slum in cells ESA leaves "
              f"unlabelled {r.pred_slum_outside_esa_labels:,}")
    print("  change " + f"{first}->{last}: " + ", ".join(
        f"{name} {(change == code).sum() * cell_km2:.1f} km2" for code, name in CHANGE.items()))
    print(f"  [done] {mp['maps_dir']}/slum_prob_*.tif, slum_change_{first}_{last}.tif; "
          f"{tables}/slum_map_summary.csv, spotcheck_{first}_{last}.csv ({len(spot)} cells)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="config/benchmark.yaml")
    run(p.parse_args().config)
