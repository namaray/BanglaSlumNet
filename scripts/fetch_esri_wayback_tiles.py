"""
Download Esri World Imagery Wayback tiles at zoom 16 (GRAM's training zoom, per the
GRAM-Dataset README) covering the ESA mapped area, for the GRAM baseline only (E1).

Licence: Esri World Imagery is used here for non-commercial research, as the GRAM authors
did. The tiles stay in git-ignored data/raw/ and private storage; never redistribute them,
and never use them as benchmark input.

The release is chosen with --release (the Wayback release number from the Wayback config).
Tiles are fetched politely (few threads, retries, skip existing).

CLI:
    python scripts/fetch_esri_wayback_tiles.py --release 10768
Writes data/raw/esri_wayback_z16/<release>/dhaka_x<X>_y<Y>.jpg and manifest.csv.
"""

import argparse
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.grid import load_benchmark_config  # noqa: E402

TILE_URL = ("https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/"
            "default028mm/MapServer/tile/{release}/{z}/{y}/{x}")
HEADERS = {"User-Agent": "BanglaSlumNet-research/1.0 (non-commercial research; GRAM baseline)"}


def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def tile_bounds(x, y, z):
    n = 2 ** z
    lon0, lon1 = x / n * 360 - 180, (x + 1) / n * 360 - 180
    lat0 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon0, lat0, lon1, lat1


def tiles_covering(area, z):
    lon0, lat0, lon1, lat1 = area.bounds
    x0, y1 = lonlat_to_tile(lon0, lat0, z)
    x1, y0 = lonlat_to_tile(lon1, lat1, z)
    return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)
            if box(*tile_bounds(x, y, z)).intersects(area)]


def fetch(args):
    x, y, z, release, path = args
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path, os.path.getsize(path), "skip"
    url = TILE_URL.format(release=release, z=z, y=y, x=x)
    for attempt in range(4):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.ok and r.headers.get("Content-Type", "").startswith("image"):
                with open(path, "wb") as f:
                    f.write(r.content)
                return path, len(r.content), "ok"
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return path, 0, "failed"


def run(release, zoom, config, threads):
    cfg = load_benchmark_config(config)
    area = gpd.read_file(cfg["paths"]["mapped_area"]).to_crs(4326).union_all()
    tiles = tiles_covering(area, zoom)
    out = os.path.join("data", "raw", f"esri_wayback_z{zoom}", str(release))
    os.makedirs(out, exist_ok=True)
    jobs = [(x, y, zoom, release, os.path.join(out, f"dhaka_x{x}_y{y}.jpg")) for x, y in tiles]
    print(f"  {len(jobs)} zoom-{zoom} tiles cover the mapped area; release {release} -> {out}")
    with ThreadPoolExecutor(max_workers=threads) as pool:
        results = list(pool.map(fetch, jobs))
    man = pd.DataFrame([{"x": j[0], "y": j[1], "path": os.path.basename(r[0]), "bytes": r[1], "status": r[2]}
                        for j, r in zip(jobs, results)])
    man.to_csv(os.path.join(out, "manifest.csv"), index=False)
    print(f"  ok {int((man.status == 'ok').sum())}, skipped (already there) {int((man.status == 'skip').sum())}, "
          f"failed {int((man.status == 'failed').sum())}; {man.bytes.sum()/1e6:.1f} MB")
    print(f"  [done] {out}/manifest.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--release", type=int, required=True, help="Wayback release number")
    p.add_argument("--zoom", type=int, default=16)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--config", default="config/benchmark.yaml")
    a = p.parse_args()
    run(a.release, a.zoom, a.config, a.threads)
