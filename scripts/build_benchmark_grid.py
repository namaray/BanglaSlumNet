"""
Build the 100 m benchmark grid and the ESA reference fractions on it
(slum_frac, edge, mapped). Earth Engine layers and final labels come later
(gee/export_cell_features.py, scripts/build_cell_table.py).

Needs data/derived/esa_mapped_area.geojson from scripts/build_mapped_area.py.

CLI:
    python scripts/build_benchmark_grid.py
Writes data/derived/grid.{json,npz} and data/derived/esa_slum_frac.tif (for QGIS).
"""

import argparse
import os
import sys

import geopandas as gpd
import numpy as np
import rasterio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.grid import esa_fractions, load_benchmark_config, make_grid, save_grid  # noqa: E402


def main(config: str):
    cfg = load_benchmark_config(config)
    paths = cfg["paths"]
    mapped = gpd.read_file(paths["mapped_area"])
    esa = gpd.read_file(paths["esa_shp"])
    grid = make_grid(mapped, cfg)
    frac = esa_fractions(grid, esa, mapped, cfg)

    out = os.path.join(paths["derived_dir"], "grid")
    save_grid(grid, out, **frac)
    tif = os.path.join(paths["derived_dir"], "esa_slum_frac.tif")
    with rasterio.open(tif, "w", driver="GTiff", height=grid.nrows, width=grid.ncols, count=1,
                       dtype="float32", crs=grid.crs, transform=grid.transform()) as dst:
        dst.write(frac["slum_frac"], 1)

    sf, m, near = frac["slum_frac"], frac["mapped"], frac["near_slum"]
    lab = cfg["labels"]
    slum = m & (sf >= lab["slum_min_frac"])
    nonslum = m & ~near & (sf <= lab["nonslum_max_frac"])
    print(f"  grid {grid.nrows} x {grid.ncols} cells of {grid.cell_m:.0f} m ({grid.nrows*grid.ncols:,} cells)")
    print(f"  ESA slum area on grid: {sf.sum() * grid.cell_m**2 / 1e6:.2f} km2")
    print(f"  mapped (inner) cells: {m.sum():,} ({m.sum() * grid.cell_m**2 / 1e6:.1f} km2)")
    print(f"  before the built-up filter: slum (share >= {lab['slum_min_frac']}) {int(slum.sum()):,}"
          f" | non-slum {int(nonslum.sum()):,} | ignored (mixed or near a polygon) {int((m & ~slum & ~nonslum).sum()):,}")
    print(f"  slum area captured by slum cells: {(sf * slum).sum() / (sf * m).sum():.1%} of ESA slum area in mapped cells")
    print(f"  [done] {out}.npz/.json, {tif}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="config/benchmark.yaml")
    main(p.parse_args().config)
