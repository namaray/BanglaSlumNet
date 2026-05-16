"""Fetch ESRI World Imagery tiles over Dhaka slums.

GRAM was trained on ESRI World Imagery at zoom 16 (256x256, ~1.2m/px), so we feed
it the same data type to isolate distribution shift from format mismatch.
"""
import os
import math
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "dhaka_tiles")
os.makedirs(OUT, exist_ok=True)

LOCATIONS = {
    "korail":    (23.7806, 90.4040),  # Korail slum
    "mirpur":    (23.8100, 90.3600),  # Mirpur
    "old_dhaka": (23.7100, 90.3900),  # Old Dhaka
}

ZOOM = 16
GRID = 3  # 3x3 → 9 tiles per location, 768x768 effective coverage


def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon = xtile / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ytile / n))))
    return lat, lon


def fetch_tile(z, x, y, out_path):
    url = f"https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    req = urllib.request.Request(url, headers={"User-Agent": "BanglaSlumNet/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        with open(out_path, "wb") as f:
            f.write(r.read())


def main():
    half = GRID // 2
    manifest = []
    for loc, (lat, lon) in LOCATIONS.items():
        cx, cy = deg2num(lat, lon, ZOOM)
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                x, y = cx + dx, cy + dy
                out = os.path.join(OUT, f"{loc}_z{ZOOM}_x{x}_y{y}.jpg")
                if os.path.exists(out) and os.path.getsize(out) > 1000:
                    print(f"skip {out}")
                else:
                    print(f"fetch {loc} dx={dx} dy={dy} → {out}")
                    fetch_tile(ZOOM, x, y, out)
                    time.sleep(0.1)
                tlat, tlon = num2deg(x, y, ZOOM)
                manifest.append((loc, x, y, tlat, tlon, out))
    with open(os.path.join(OUT, "manifest.csv"), "w") as f:
        f.write("location,x,y,lat,lon,path\n")
        for row in manifest:
            f.write(",".join(str(v) for v in row) + "\n")
    print(f"\nfetched {len(manifest)} tiles into {OUT}")


if __name__ == "__main__":
    main()
