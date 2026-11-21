"""
Per-cell features on the 100 m benchmark grid for one year, computed in Earth Engine and
downloaded as a multi-band GeoTIFF aligned to data/derived/grid.json
(docs/MASTER_PLAN.md §4.1, §5.5; parameters in config/benchmark.yaml).

Bands (groups match the E3 ablation):
  s2_*     Sentinel-2 L2A dry-season median composite (SCL cloud mask): per-cell mean and
           stdDev of each band and index over its 10 m pixels
  ob_*     Open Buildings 2.5D Temporal for the year: mean presence, mean and stdDev of
           height, building count per cell
  dw_*     Dynamic World over the same window: share of pixels whose modal class is built,
           mean built probability
  ctx_*    VIIRS night lights (median of monthly composites) and WorldPop population

CLI:
    python gee/export_cell_features.py --year 2017
Writes data/derived/features_<year>.tif and a .json sidecar with band names.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests

try:
    import ee
except ImportError:
    ee = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gee.ee_export_utils import init_ee, season_dates  # noqa: E402
from src.data.grid import load_benchmark_config, load_grid  # noqa: E402

SCL_INVALID = [0, 1, 3, 8, 9, 10, 11]  # no data, saturated, shadow, cloud med/high, cirrus, snow
DW_BUILT = 6


def _to_cells(img, grid, src_scale, reducer, max_pixels=1024):
    """Aggregate a fine image to the grid cells with an explicit reducer."""
    return (img.setDefaultProjection(crs=grid.crs, scale=src_scale)
               .reduceResolution(reducer=reducer, maxPixels=max_pixels)
               .reproject(crs=grid.crs, crsTransform=grid.ee_crs_transform()))


def s2_cells(year, region, grid, cfg):
    f = cfg["features"]
    start, end = season_dates(year, cfg["season"])
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
           .filterDate(start, end).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", f["s2_max_cloud_pct"])))
    n = col.size().getInfo()
    if n == 0:
        raise RuntimeError(f"No Sentinel-2 L2A scenes for {year} {cfg['season']} ({start}..{end}).")

    def prep(im):
        ok = im.select("SCL").remap(SCL_INVALID, [0] * len(SCL_INVALID), 1)
        return im.select(f["s2_bands"]).divide(10000).updateMask(ok)

    comp = col.map(prep).median()
    idx = {"NDVI": ("B8", "B4"), "NDBI": ("B11", "B8"), "NDWI": ("B3", "B8")}
    comp = comp.addBands([comp.normalizedDifference(list(idx[k])).rename(k) for k in f["s2_indices"]])
    reducer = ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)
    cells = _to_cells(comp, grid, 10, reducer)
    names = cells.bandNames().map(lambda b: ee.String("s2_").cat(b))
    print(f"    Sentinel-2: {n} scenes {start}..{end}")
    return cells.rename(names)


def ob_cells(year, region, grid, cfg):
    s = cfg["features"]["ob_scale_m"]
    ob = (ee.ImageCollection("GOOGLE/Research/open-buildings-temporal/v1").filterBounds(region)
          .filter(ee.Filter.calendarRange(year, year, "year")).mosaic())
    native = 0.5  # stored resolution; fractional counts are per native pixel
    fine = ob.select(["building_presence", "building_height", "building_fractional_count"])
    # Sample at the ~4 m effective resolution (averaging from 0.5 m exceeds EE's 2^31-pixel limit)
    mean = _to_cells(fine, grid, s, ee.Reducer.mean())
    h_sd = _to_cells(fine.select("building_height"), grid, s, ee.Reducer.stdDev())
    count = mean.select("building_fractional_count").multiply((grid.cell_m / native) ** 2)
    return (mean.select(["building_presence", "building_height"], ["ob_presence", "ob_height_mean"])
            .addBands(h_sd.rename("ob_height_std")).addBands(count.rename("ob_count")))


def dw_cells(year, region, grid, cfg):
    start, end = season_dates(year, cfg["season"])
    dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(region).filterDate(start, end)
    built = dw.select("label").mode().eq(DW_BUILT).rename("dw_built_frac")
    prob = dw.select("built").mean().rename("dw_built_prob")
    return _to_cells(built.toFloat().addBands(prob), grid, 10, ee.Reducer.mean())


def ctx_cells(year, region, grid):
    monthly = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterBounds(region)
               .filter(ee.Filter.calendarRange(year, year, "year")).select("avg_rad"))
    # Composites lose their projection; restore the native ~463 m grid before bilinear
    # resampling, otherwise EE interpolates on a 1-degree default grid.
    viirs = monthly.median().setDefaultProjection(monthly.first().projection())
    pop = (ee.ImageCollection("WorldPop/GP/100m/R2025A/v1/pop").filterBounds(region)
           .filter(ee.Filter.eq("year", year)).mosaic().unmask(0))  # no data (water) = no people
    img = viirs.rename("ctx_viirs").resample("bilinear").addBands(pop.rename("ctx_pop"))
    return img.reproject(crs=grid.crs, crsTransform=grid.ee_crs_transform())


def build_groups(year, grid, cfg):
    left, bottom, right, top = grid.bounds
    region = ee.Geometry.Rectangle([left, bottom, right, top], proj=grid.crs, geodesic=False)
    return {"s2": s2_cells(year, region, grid, cfg), "ob": ob_cells(year, region, grid, cfg),
            "dw": dw_cells(year, region, grid, cfg), "ctx": ctx_cells(year, region, grid)}


def _download(img, grid, row0, nrows):
    """One GeoTIFF request for grid rows [row0, row0 + nrows); returns (bands, rows, cols)."""
    from rasterio.io import MemoryFile
    row0, nrows = int(row0), int(nrows)  # numpy ints are not EE-serializable
    t = [float(v) for v in grid.ee_crs_transform()]
    t[5] = float(grid.top - row0 * grid.cell_m)
    url = img.toFloat().getDownloadURL({"crs": grid.crs, "crs_transform": t,
                                        "dimensions": f"{grid.ncols}x{nrows}", "format": "GEO_TIFF"})
    r = requests.get(url, timeout=900)
    if not r.ok:
        raise RuntimeError(f"Earth Engine download failed ({r.status_code}): {r.text[:500]}")
    with MemoryFile(r.content) as mem, mem.open() as src:
        return src.read()


def download_group(img, grid, name, strips=(1, 2, 4, 8)):
    """Download a group, splitting into more row strips whenever EE runs out of memory."""
    import numpy as np
    for n in strips:
        try:
            edges = np.linspace(0, grid.nrows, n + 1).astype(int)
            parts = [_download(img, grid, a, b - a) for a, b in zip(edges[:-1], edges[1:])]
            print(f"    {name}: ok in {n} strip(s)")
            return np.concatenate(parts, axis=1)
        except RuntimeError as e:
            if "memory" not in str(e).lower() and "too large" not in str(e).lower():
                raise
            print(f"    {name}: EE out of memory with {n} strip(s), retrying smaller")
    raise RuntimeError(f"{name}: still out of memory with {strips[-1]} strips")


def run(year, config="config/benchmark.yaml", overwrite=False):
    import numpy as np
    import rasterio
    cfg = load_benchmark_config(config)
    grid, _ = load_grid(os.path.join(cfg["paths"]["derived_dir"], "grid"))
    out = Path(cfg["paths"]["derived_dir"]) / f"features_{year}.tif"
    if out.exists() and not overwrite:
        print(f"  [skip] exists: {out}")
        return out
    arrays, bands = [], []
    for name, img in build_groups(year, grid, cfg).items():
        bands += img.bandNames().getInfo()
        arrays.append(download_group(img, grid, name))
    data = np.concatenate(arrays, axis=0).astype("float32")
    with rasterio.open(out, "w", driver="GTiff", height=grid.nrows, width=grid.ncols,
                       count=len(bands), dtype="float32", crs=grid.crs, transform=grid.transform(),
                       compress="deflate") as dst:
        dst.write(data)
        dst.descriptions = tuple(bands)
    out.with_suffix(".json").write_text(json.dumps({"year": year, "bands": bands}, indent=2))
    print(f"  [done] {out} ({len(bands)} bands)")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--config", default="config/benchmark.yaml")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    init_ee(load_benchmark_config(a.config)["ee_project"], authenticate=False)
    run(a.year, a.config, a.overwrite)
