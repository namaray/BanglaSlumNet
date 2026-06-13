"""
Shared helpers for the Python `ee` export path (importable from the Colab notebook).

This is the in-notebook alternative to pasting the .js scripts into the GEE Code
Editor. It mirrors download_s2.py: ee.Initialize(project=...) then geemap direct
download straight into the Drive-mounted output dir — no Task queue to monitor.

All region boxes are read from config/regions_dhaka.yaml (config-driven, never
hardcoded). VERIFY those TODO_VERIFY boxes before exporting or you waste the pull.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import ee
except ImportError:
    ee = None  # importable without ee; init_ee() raises a clear error if missing

try:
    import yaml
except ImportError:
    yaml = None


def init_ee(project_id: str, authenticate: bool = True):
    """Initialize Earth Engine. In Colab, authenticate=True triggers the auth flow once."""
    if ee is None:
        raise ImportError("earthengine-api not installed. pip install earthengine-api geemap")
    try:
        ee.Initialize(project=project_id)
    except Exception:
        if authenticate:
            ee.Authenticate()
            ee.Initialize(project=project_id)
        else:
            raise
    print(f"Earth Engine ready (project={project_id})")


def load_regions(regions_yaml: str) -> Dict[str, Dict]:
    """Return {region_name: {bbox, type, ...}} from the regions YAML."""
    if yaml is None:
        raise ImportError("pyyaml required")
    with open(regions_yaml) as f:
        cfg = yaml.safe_load(f)
    return cfg["regions"]


def region_geometry(bbox: Dict):
    """Build an ee.Geometry.Rectangle from a bbox dict (lon_min, lat_min, lon_max, lat_max)."""
    return ee.Geometry.Rectangle(
        [bbox["lon_min"], bbox["lat_min"], bbox["lon_max"], bbox["lat_max"]]
    )


def city_geometry(regions: Dict[str, Dict], pad: float = 0.02):
    """Bounding box covering all regions (for city-median VIIRS threshold etc.)."""
    lons, lats = [], []
    for r in regions.values():
        bb = r["bbox"]
        lons += [bb["lon_min"], bb["lon_max"]]
        lats += [bb["lat_min"], bb["lat_max"]]
    return ee.Geometry.Rectangle([min(lons) - pad, min(lats) - pad,
                                  max(lons) + pad, max(lats) + pad])


def season_dates(year: int, season: str) -> Tuple[str, str]:
    """Return (start, end) ISO dates for a Dhaka season.
    dry = Dec(prev year)–Feb; wet = Jun–Aug."""
    if season == "dry":
        return f"{year - 1}-12-01", f"{year}-02-28"
    elif season == "wet":
        return f"{year}-06-01", f"{year}-08-31"
    else:
        # default to full year
        return f"{year}-01-01", f"{year}-12-31"


def export_image(image, region_geom, out_path: str, scale: int = 10,
                 crs: str = "EPSG:4326", skip_existing: bool = True):
    """
    Direct download an ee.Image to a local/Drive GeoTIFF via geemap.
    Skips if the file already exists (cache-safe, re-run friendly).
    """
    import geemap
    out_path = str(out_path)
    if skip_existing and os.path.exists(out_path):
        print(f"  [skip] exists: {os.path.basename(out_path)}")
        return out_path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    geemap.ee_export_image(
        image.clip(region_geom),
        filename=out_path,
        scale=scale,
        region=region_geom,
        crs=crs,
        file_per_band=False,
    )
    print(f"  [done] {os.path.basename(out_path)}")
    return out_path
