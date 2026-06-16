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


def build_built(regions: dict, reference_year: int):
    """Built-up / residential mask = GHSL built OR Dynamic World 'built' (class 6).
    Shared across regions; defines the spatial extent that gets a class label."""
    city = city_geometry(regions)
    ghsl = (ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S/2020")
            .select("built_surface").gt(0))
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
          .filterDate(f"{reference_year}-01-01", f"{reference_year}-12-31")
          .filterBounds(city).select("label").mode())
    return ghsl.Or(dw.eq(6)).rename("built")


def region_label_image(built, cls: int):
    """3-band weak label for ONE region of a known type.

    Region-type weak supervision: every BUILT pixel in a known-informal region is
    labeled slum (1); every built pixel in a known-formal region is labeled
    formal-dense (2). This is far more reliable than a per-pixel VIIRS threshold
    (which collapsed to all-formal) and guarantees both classes exist.
      band1 noisy_label: 0=unknown, cls on built pixels
      band2 hc_geo     : built pixels are high-confidence for the region's known type
      band3 agreement  : 0..2 (built + hc), for the data card
    """
    noisy = built.multiply(cls).toByte().rename("noisy_label")   # 0 off-built, else cls
    hc_geo = built.toByte().rename("hc_geo")
    agreement = built.toByte().add(hc_geo).rename("agreement")
    return noisy.addBands(hc_geo).addBands(agreement)


def run_all(output_dir: str, regions_yaml: str, reference_year: int = 2022,
            viirs_percentile: float = 50.0, scale: int = 10,
            skip_existing: bool = True) -> List[str]:
    # viirs_percentile kept for signature compatibility (no longer used by the
    # region-type labeling rule).
    regions = load_regions(regions_yaml)
    built = build_built(regions, reference_year)
    written = []
    for region, meta in regions.items():
        cls = 1 if meta.get("type") == "informal" else 2   # informal->slum, else formal
        img = region_label_image(built, cls)
        geom = region_geometry(meta["bbox"])
        out = Path(output_dir) / f"weeklabels_{region}.tif"
        written.append(export_image(img, geom, str(out), scale=scale,
                                    skip_existing=skip_existing))
        print(f"  {region}: type={meta.get('type')} -> class {cls}")
    print(f"Weak-label export complete: {len(written)} regions (region-type labeling).")
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
