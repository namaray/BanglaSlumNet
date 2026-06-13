"""
Python `ee` version of gee/04_export_socioeconomic.js — run inside the Colab notebook.
Exports the socioeconomic E-tensor per region as a multi-band GeoTIFF.

Bands are written IN THE CONFIG CHANNEL ORDER so the tiling step can resolve them
even if the GeoTIFF loses band descriptions (positional fallback in tiling.py).

Default channel order (config.fusion.socioeconomic_channels):
  [viirs, worldpop, ghspop, osm_roads, wb_poverty, ghsl_builtup]

TODO_VERIFY: osm_roads uses an accessibility proxy; wb_poverty is a zero placeholder.
Swap both for real GEE assets (GRIP roads / WB GriddedPoverty or RWI) before use.

Notebook usage:
    from gee.ee_export_utils import init_ee
    from gee.export_socioeconomic import run_all
    init_ee('banglaslumnet-research')
    run_all(output_dir=cfg.paths.socioeconomic_dir, regions_yaml='config/regions_dhaka.yaml',
            channels=cfg.fusion.socioeconomic_channels, reference_year=2022)
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


def _channel_image(channel: str, reference_year: int, city):
    """Return a single-band ee.Image for the named channel, renamed to `channel`."""
    y0, y1 = f"{reference_year}-01-01", f"{reference_year}-12-31"
    if channel == "viirs":
        return (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
                .filterDate(y0, y1).filterBounds(city).select("avg_rad").median())
    if channel == "worldpop":
        return (ee.ImageCollection("WorldPop/GP/100m/pop")
                .filterDate(y0, y1).filterBounds(city).select("population").mosaic())
    if channel == "ghspop":
        return ee.Image("JRC/GHSL/P2023A/GHS_POP/2020").select("population_count")
    if channel == "ghsl_builtup":
        return ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S/2020").select("built_surface")
    if channel == "osm_roads":
        # TODO_VERIFY: replace with a proper road-density raster (e.g. GRIP4).
        return ee.Image("Oxford/MAP/accessibility_to_cities_2015_v1_0").select("accessibility")
    if channel == "wb_poverty":
        # TODO_VERIFY: replace with WB GriddedPoverty or Meta/Chi RWI asset.
        return ee.Image.constant(0)
    # Unknown channel -> zeros so band positions stay aligned with the config order.
    return ee.Image.constant(0)


def build_socioeconomic(regions: dict, channels: List[str], reference_year: int):
    """Stack the requested channels into one multi-band image, in config order."""
    city = city_geometry(regions)
    imgs = [_channel_image(ch, reference_year, city).rename(ch).toFloat() for ch in channels]
    stack = imgs[0]
    for im in imgs[1:]:
        stack = stack.addBands(im)
    return stack


def run_all(output_dir: str, regions_yaml: str, channels: List[str],
            reference_year: int = 2022, scale: int = 10,
            skip_existing: bool = True) -> List[str]:
    regions = load_regions(regions_yaml)
    img = build_socioeconomic(regions, list(channels), reference_year)
    written = []
    for region, meta in regions.items():
        geom = region_geometry(meta["bbox"])
        out = Path(output_dir) / f"socioeconomic_{region}.tif"
        written.append(export_image(img, geom, str(out), scale=scale,
                                    skip_existing=skip_existing))
    print(f"Socioeconomic export complete: {len(written)} regions "
          f"({len(channels)} bands in order {list(channels)}).")
    return written


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--output_dir", default="data/socioeconomic")
    p.add_argument("--regions_yaml", default="config/regions_dhaka.yaml")
    p.add_argument("--channels", nargs="+",
                   default=["viirs", "worldpop", "ghspop", "osm_roads", "wb_poverty", "ghsl_builtup"])
    p.add_argument("--reference_year", type=int, default=2022)
    a = p.parse_args()
    init_ee(a.project)
    run_all(a.output_dir, a.regions_yaml, a.channels, a.reference_year)


if __name__ == "__main__":
    main()
