"""
Python `ee` version of gee/01_export_s2_composites.js — run inside the Colab notebook.
Exports Sentinel-2 seasonal best-pixel composites per region/year/season to Drive.

Notebook usage:
    from gee.ee_export_utils import init_ee
    from gee.export_s2_composites import run_all
    init_ee('banglaslumnet-research')
    run_all(output_dir=cfg.paths.tiles_dir, regions_yaml='config/regions_dhaka.yaml',
            years=cfg.data.training_years, seasons=cfg.data.seasons, bands=cfg.data.s2_bands)

CLI usage:
    python gee/export_s2_composites.py --project banglaslumnet-research --output_dir data/tiles
"""

from pathlib import Path
from typing import List

try:
    import ee
except ImportError:
    ee = None

try:
    from .ee_export_utils import (init_ee, load_regions, region_geometry,
                                  season_dates, export_image)
except ImportError:  # direct CLI execution (python gee/export_s2_composites.py)
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(__file__))
    from ee_export_utils import (init_ee, load_regions, region_geometry,
                                 season_dates, export_image)


def s2_composite(region_geom, year: int, season: str, bands: List[str]):
    """Cloud-masked Sentinel-2 SR median composite for one region/year/season."""
    start, end = season_dates(year, season)
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(region_geom)
           .filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)))

    def mask_clouds(img):
        scl = img.select("SCL")
        # exclude defective(1)/shadow(3)/cloud-med(8)/cloud-high(9)/cirrus(10)
        mask = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)))
        return img.updateMask(mask).divide(10000).select(bands)

    # unmask(0) gives the composite a full footprint over the whole box, so the export
    # is NOT cropped to the valid-data rectangle (cloud/water-masked edges otherwise
    # shrink the GeoTIFF and starve water-adjacent regions of tiles).
    return col.map(mask_clouds).median().clip(region_geom).unmask(0).toFloat()


def run_all(output_dir: str, regions_yaml: str,
            years: List[int], seasons: List[str], bands: List[str],
            scale: int = 10, skip_existing: bool = True) -> List[str]:
    """Export every region × year × season composite. Returns list of written paths."""
    regions = load_regions(regions_yaml)
    written = []
    for region, meta in regions.items():
        geom = region_geometry(meta["bbox"])
        for year in years:
            for season in seasons:
                img = s2_composite(geom, year, season, list(bands))
                out = Path(output_dir) / f"s2_{region}_{year}_{season}.tif"
                written.append(export_image(img, geom, str(out), scale=scale,
                                            skip_existing=skip_existing))
    print(f"S2 export complete: {len(written)} composites.")
    return written


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--output_dir", default="data/tiles")
    p.add_argument("--regions_yaml", default="config/regions_dhaka.yaml")
    p.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2022, 2023])
    p.add_argument("--seasons", nargs="+", default=["dry", "wet"])
    p.add_argument("--bands", nargs="+", default=["B2", "B3", "B4", "B8"])
    a = p.parse_args()
    init_ee(a.project)
    run_all(a.output_dir, a.regions_yaml, a.years, a.seasons, a.bands)


if __name__ == "__main__":
    main()
