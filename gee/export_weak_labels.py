"""
Python `ee` version of gee/03_weak_labels.js — run inside the Colab notebook.
Exports a 3-band weak-label raster per region: [noisy_label, hc_geo, agreement_score].

Rules (§6.1):
  slum         = OSM/builtup ∩ GHSL built-up ∩ VIIRS dark   (≤ city median)
  formal-dense = OSM/builtup ∩ GHSL built-up ∩ VIIRS bright (> city median)
The 4th signal (LocateAnything) is fused later in Python (refresh_hc_with_la).

TODO_VERIFY: OSM-residential proxy uses Dynamic World 'built' class; swap for a
real OSM-residential raster if available. Confirm GHSL/VIIRS asset IDs are current.

Notebook usage:
    from gee.ee_export_utils import init_ee
    from gee.export_weak_labels import run_all
    init_ee('banglaslumnet-research')
    run_all(output_dir=cfg.paths.labels_dir, regions_yaml='config/regions_dhaka.yaml',
            reference_year=2022, viirs_percentile=cfg.weak_labels.viirs_percentile_threshold)
"""

from pathlib import Path
from typing import List

try:
    import ee
except ImportError:
    ee = None

try:
    from .ee_export_utils import (init_ee, load_regions, region_geometry,
                                  city_geometry, export_image)
except ImportError:  # direct CLI execution
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(__file__))
    from ee_export_utils import (init_ee, load_regions, region_geometry,
                                 city_geometry, export_image)


def build_weak_labels(regions: dict, reference_year: int, viirs_percentile: float):
    """Construct the 3-band weak-label image (noisy_label, hc_geo, agreement_score)."""
    city = city_geometry(regions)

    ghsl = (ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S/2020")
            .select("built_surface").gt(0).rename("ghsl"))

    # OSM-residential proxy: Dynamic World 'built' class (label 6). TODO_VERIFY.
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
          .filterDate(f"{reference_year}-01-01", f"{reference_year}-12-31")
          .filterBounds(city).select("label").mode())
    osm = dw.eq(6).rename("osm")

    viirs = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
             .filterDate(f"{reference_year}-01-01", f"{reference_year}-12-31")
             .filterBounds(city).select("avg_rad").median().rename("viirs"))
    city_med = viirs.reduceRegion(
        reducer=ee.Reducer.percentile([int(viirs_percentile)]),
        geometry=city, scale=500, maxPixels=int(1e9)
    ).getNumber("viirs")
    dark = viirs.lt(city_med).rename("dark")
    bright = viirs.gte(city_med).rename("bright")

    slum_sig = ghsl.add(osm).add(dark)       # 0–3
    formal_sig = ghsl.add(osm).add(bright)   # 0–3

    noisy = (ee.Image(0)
             .where(slum_sig.gte(2), 1)
             .where(formal_sig.gte(2), 2)
             .rename("noisy_label").toByte())
    hc_slum = slum_sig.eq(3)
    hc_formal = formal_sig.eq(3)
    hc_geo = hc_slum.Or(hc_formal).rename("hc_geo").toByte()
    agreement = slum_sig.max(formal_sig).rename("agreement").toByte()

    return noisy.addBands(hc_geo).addBands(agreement)


def run_all(output_dir: str, regions_yaml: str, reference_year: int = 2022,
            viirs_percentile: float = 50.0, scale: int = 10,
            skip_existing: bool = True) -> List[str]:
    regions = load_regions(regions_yaml)
    img = build_weak_labels(regions, reference_year, viirs_percentile)
    written = []
    for region, meta in regions.items():
        geom = region_geometry(meta["bbox"])
        out = Path(output_dir) / f"weeklabels_{region}.tif"
        written.append(export_image(img, geom, str(out), scale=scale,
                                    skip_existing=skip_existing))
    print(f"Weak-label export complete: {len(written)} regions.")
    return written


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--output_dir", default="data/labels")
    p.add_argument("--regions_yaml", default="config/regions_dhaka.yaml")
    p.add_argument("--reference_year", type=int, default=2022)
    p.add_argument("--viirs_percentile", type=float, default=50.0)
    a = p.parse_args()
    init_ee(a.project)
    run_all(a.output_dir, a.regions_yaml, a.reference_year, a.viirs_percentile)


if __name__ == "__main__":
    main()
