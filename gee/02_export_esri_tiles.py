"""
Export ESRI World Imagery z16 tiles for GRAM baseline comparison only.
This is NOT used for BanglaSlumNet training — only for the head-to-head
failure-mode reproduction at GRAM's native 1.2 m/px resolution.

Usage (run from Colab after mounting Drive):
    python gee/02_export_esri_tiles.py --region korail --output_dir /gdrive/MyDrive/BanglaSlumNet/data/tiles

TODO_VERIFY: ESRI tile access requires a valid ArcGIS/ESRI account or public tile endpoint.
TODO_VERIFY: Confirm zoom level 16 tile bounds align with region bounding boxes.
"""

import argparse
import math
import os
import urllib.request
from pathlib import Path

import numpy as np
try:
    from PIL import Image
except ImportError:
    raise ImportError("Install Pillow: pip install Pillow")

# ESRI World Imagery tile URL template (public endpoint)
ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ZOOM = 16  # ~1.2 m/px

# TODO_VERIFY: all bounding boxes
REGIONS = {
    "korail":             (90.4110, 23.7830, 90.4220, 23.7930),
    "bhashantek":         (90.3790, 23.8330, 90.3920, 23.8450),
    "karail_extension":   (90.4180, 23.7900, 90.4290, 23.7990),
    "old_dhaka":          (90.3940, 23.7090, 90.4160, 23.7260),
    "gulshan_baridhara":  (90.4100, 23.7760, 90.4340, 23.8000),
}


def lon_lat_to_tile(lon: float, lat: float, zoom: int):
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def download_tile(z: int, x: int, y: int, output_path: Path):
    if output_path.exists():
        return
    url = ESRI_URL.format(z=z, x=x, y=y)
    headers = {"User-Agent": "BanglaSlumNet-research/1.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    output_path.write_bytes(data)


def export_region(region_name: str, bbox: tuple, output_dir: Path):
    lon_min, lat_min, lon_max, lat_max = bbox
    x_min, y_max_tile = lon_lat_to_tile(lon_min, lat_min, ZOOM)
    x_max, y_min_tile = lon_lat_to_tile(lon_max, lat_max, ZOOM)

    region_dir = output_dir / f"esri_z{ZOOM}_{region_name}"
    region_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {region_name}: x={x_min}..{x_max}, y={y_min_tile}..{y_max_tile}")
    for x in range(x_min, x_max + 1):
        for y in range(y_min_tile, y_max_tile + 1):
            tile_path = region_dir / f"{ZOOM}_{x}_{y}.png"
            try:
                download_tile(ZOOM, x, y, tile_path)
            except Exception as e:
                print(f"  Warning: failed tile {ZOOM}/{x}/{y}: {e}")
    print(f"  Done → {region_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download ESRI z16 tiles for GRAM baseline")
    parser.add_argument("--region", choices=list(REGIONS.keys()) + ["all"], default="all")
    parser.add_argument("--output_dir", required=True, help="Output directory for tiles")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    regions_to_run = REGIONS if args.region == "all" else {args.region: REGIONS[args.region]}
    for name, bbox in regions_to_run.items():
        export_region(name, bbox, output_dir)


if __name__ == "__main__":
    main()
