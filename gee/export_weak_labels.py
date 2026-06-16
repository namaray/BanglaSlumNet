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
    """Construct the 3-band weak-label image (noisy_label, hc_geo, agreement_score).

    Rule: among BUILT residential pixels, classify by VIIRS nighttime brightness —
    darker = informal (slum), brighter = formal. The dark/bright threshold is computed
    over BUILT pixels only (not the whole bbox, which rural/water dilutes), so it
    actually separates the urban core. Slum and formal are assigned mutually
    exclusively (no overwrite). HC = built pixels in the VIIRS tails (clear cases).
    """
    city = city_geometry(regions)

    ghsl = (ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S/2020")
            .select("built_surface").gt(0).rename("ghsl"))

    # Built-up / residential: use GHSL OR Dynamic World 'built' (more recall than the
    # strict intersection, which rarely co-registered at 10 m and gave 0 HC).
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
          .filterDate(f"{reference_year}-01-01", f"{reference_year}-12-31")
          .filterBounds(city).select("label").mode())
    dw_built = dw.eq(6).rename("dw")
    built = ghsl.Or(dw_built).rename("built")

    viirs = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
             .filterDate(f"{reference_year}-01-01", f"{reference_year}-12-31")
             .filterBounds(city).select("avg_rad").median().rename("viirs"))

    # Thresholds computed over BUILT pixels only (mask out rural/water dilution),
    # so dark/bright actually splits the urban fabric. Tails define high confidence.
    viirs_built = viirs.updateMask(built)
    pct = viirs_built.reduceRegion(
        reducer=ee.Reducer.percentile([25, int(viirs_percentile), 75]),
        geometry=city, scale=500, maxPixels=int(1e9))
    med = pct.getNumber(f"viirs_p{int(viirs_percentile)}")
    p25 = pct.getNumber("viirs_p25")
    p75 = pct.getNumber("viirs_p75")

    dark = viirs.lt(med)
    bright = viirs.gte(med)

    # Mutually exclusive noisy labels: 0=unknown, 1=slum, 2=formal-dense.
    noisy = (ee.Image(0)
             .where(built.And(bright), 2)   # formal first
             .where(built.And(dark), 1)     # slum last (dark wins on built pixels)
             .rename("noisy_label").toByte())

    # High-confidence = built AND in a VIIRS tail (unambiguously dark or bright).
    hc_slum = built.And(viirs.lt(p25))
    hc_formal = built.And(viirs.gte(p75))
    hc_geo = hc_slum.Or(hc_formal).rename("hc_geo").toByte()

    # Agreement score (0-3): built + (dark|bright) + (in a tail) — for the data card.
    agreement = (built.toByte()
                 .add(built.And(dark.Or(bright)).toByte())
                 .add(hc_geo)
                 .rename("agreement").toByte())

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
