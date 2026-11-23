"""
Build a local, blind spot-check page for E5: one recent satellite image per sampled cell with
the 100 m benchmark cell drawn on it, yes / no / unsure buttons, notes, autosave in the browser
and a "Download CSV" button.

Imagery: the latest Esri World Imagery Wayback release at zoom 18 (~0.55 m/px), fetched with
scripts/fetch_esri_wayback_tiles.fetch. Research use only: the page and tiles stay local
(git-ignored); never publish or redistribute them.

Needs results/tables/spotcheck_<first>_<last>.csv from scripts/make_slum_maps.py.

CLI:
    python scripts/make_spotcheck_page.py --release 3692
Open results/figures/spotcheck/index.html in a browser.
"""

import argparse
import base64
import html
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from fetch_esri_wayback_tiles import fetch  # noqa: E402

ZOOM, TILE, CROP_PX, CELL_M = 18, 256, 500, 100
WAYBACK_CONFIG = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"


def global_px(lon, lat, z):
    n = TILE * 2 ** z
    return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n


def capture_date(meta_url, lon, lat):
    p = {"geometry": f"{lon},{lat}", "geometryType": "esriGeometryPoint", "sr": 4326, "layers": "all",
         "tolerance": 1, "mapExtent": f"{lon-0.01},{lat-0.01},{lon+0.01},{lat+0.01}",
         "imageDisplay": "400,400,96", "returnGeometry": "false", "f": "json"}
    for _ in range(3):
        try:
            for r in requests.get(meta_url + "/identify", params=p, timeout=90).json().get("results", []):
                a = r["attributes"]
                if a.get("MinMapLevel", "").isdigit() and int(a["MinMapLevel"]) <= ZOOM <= int(a["MaxMapLevel"]):
                    d = a.get("SRC_DATE", "")
                    return f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 else d
            return "unknown"
        except Exception:
            pass
    return "unknown"


def make_image(row, release, tile_dir, out_path):
    px, py = global_px(row.lon, row.lat, ZOOM)
    tx, ty = int(px // TILE), int(py // TILE)
    jobs = [(x, y, ZOOM, release, os.path.join(tile_dir, f"dhaka_x{x}_y{y}.jpg"))
            for y in range(ty - 1, ty + 2) for x in range(tx - 1, tx + 2)]
    for j in jobs:
        if fetch(j)[2] == "failed":
            raise RuntimeError(f"tile fetch failed: {j}")
    mosaic = Image.new("RGB", (3 * TILE, 3 * TILE))
    for i, j in enumerate(jobs):
        mosaic.paste(Image.open(j[4]).convert("RGB"), ((i % 3) * TILE, (i // 3) * TILE))
    cx, cy = px - (tx - 1) * TILE, py - (ty - 1) * TILE
    half = CROP_PX / 2
    img = mosaic.crop((round(cx - half), round(cy - half), round(cx + half), round(cy + half)))
    m_per_px = 156543.03392 * math.cos(math.radians(row.lat)) / 2 ** ZOOM
    s = CELL_M / m_per_px / 2
    c = CROP_PX / 2
    d = ImageDraw.Draw(img)
    d.rectangle([c - s - 2, c - s - 2, c + s + 2, c + s + 2], outline=(0, 0, 0), width=2)  # dark halo
    d.rectangle([c - s, c - s, c + s, c + s], outline=(255, 214, 0), width=3)
    img.save(out_path, quality=92)


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Slum spot-check</title>
<style>
:root {{ --bg:#fcfcfb; --card:#ffffff; --ink:#0b0b0b; --muted:#52514e; --line:#e3e2de; --accent:#2a78d6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1a1a19; --card:#242423; --ink:#f0efec; --muted:#c3c2b7; --line:#3a3a38; --accent:#3987e5; }} }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 system-ui, sans-serif; }}
header {{ position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line); padding:12px 16px; z-index:1; }}
main {{ max-width:620px; margin:0 auto; padding:16px; }}
h1 {{ font-size:18px; margin:0 0 4px; }}
.help {{ color:var(--muted); font-size:13px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; margin:0 0 18px; }}
.card img {{ width:100%; height:auto; display:block; border-radius:4px; }}
.meta {{ display:flex; justify-content:space-between; gap:8px; font-size:13px; color:var(--muted); margin:8px 0; flex-wrap:wrap; }}
.choices {{ display:flex; gap:8px; }}
.choices label {{ flex:1; text-align:center; border:1px solid var(--line); border-radius:6px; padding:8px; cursor:pointer; }}
.choices input {{ display:none; }}
.choices input:checked + span {{ font-weight:600; color:var(--accent); }}
.choices label:has(input:checked) {{ border-color:var(--accent); }}
textarea {{ width:100%; box-sizing:border-box; margin-top:8px; background:var(--bg); color:var(--ink); border:1px solid var(--line); border-radius:6px; padding:6px; font:inherit; }}
button {{ background:var(--accent); color:#fff; border:0; border-radius:6px; padding:8px 14px; font:inherit; cursor:pointer; }}
a {{ color:var(--accent); }}
</style></head><body>
<header><h1>Slum spot-check &middot; <span id="done">0</span> / {n} answered</h1>
<div class="help">Judge only the <b>yellow square (100 m &times; 100 m)</b>. <b>Yes</b> if more than half of it is slum:
small tightly packed (often tin) roofs, narrow lanes, no road grid. <b>No</b> if it is mostly formal buildings
(even dense ones like Old Dhaka), or water, parks, factories, construction. <b>Unsure</b> if about half and half or unclear.
Answers save in this browser automatically. Imagery: Esri World Imagery Wayback {release_title}.</div>
<p><button onclick="download()">Download CSV</button></p></header>
<main>{cards}</main>
<script>
const KEY = "banglaslumnet_spotcheck_{release}";
const ids = {ids};
let state = {{}};
try {{ state = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{}}
function save() {{ try {{ localStorage.setItem(KEY, JSON.stringify(state)); }} catch (e) {{}} count(); }}
function count() {{ document.getElementById("done").textContent = ids.filter(i => state[i] && state[i].v).length; }}
for (const i of ids) {{
  const s = state[i] || {{}};
  if (s.v) {{ const r = document.querySelector(`input[name="v${{i}}"][value="${{s.v}}"]`); if (r) r.checked = true; }}
  const t = document.getElementById(`n${{i}}`); t.value = s.n || "";
  document.querySelectorAll(`input[name="v${{i}}"]`).forEach(r => r.addEventListener("change", () => {{ state[i] = {{...(state[i]||{{}}), v: r.value}}; save(); }}));
  t.addEventListener("input", () => {{ state[i] = {{...(state[i]||{{}}), n: t.value}}; save(); }});
}}
count();
const rows = {rows};
function download() {{
  const esc = s => '"' + String(s || "").replace(/"/g, '""') + '"';
  const lines = ["id,lat,lon,is_slum_now,notes"].concat(rows.map(r => [r.id, r.lat, r.lon, (state[r.id]||{{}}).v || "", esc((state[r.id]||{{}}).n)].join(",")));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\\n")], {{type: "text/csv"}}));
  a.download = "spotcheck_filled.csv"; a.click();
}}
</script></body></html>"""

CARD = """<section class="card" id="c{id}"><img src="data:image/jpeg;base64,{img}" alt="Satellite image for point {id}" loading="lazy">
<div class="meta"><span><b>#{id}</b> &middot; imagery {date}</span>
<a href="https://www.google.com/maps/search/?api=1&query={lat},{lon}" target="_blank" rel="noopener">open in Google Maps</a></div>
<div class="choices"><label><input type="radio" name="v{id}" value="yes"><span>Yes, slum</span></label>
<label><input type="radio" name="v{id}" value="no"><span>No</span></label>
<label><input type="radio" name="v{id}" value="unsure"><span>Unsure</span></label></div>
<textarea id="n{id}" rows="1" placeholder="notes (optional)"></textarea></section>"""


def run(release, sheet, out_dir):
    cfg = requests.get(WAYBACK_CONFIG, timeout=60).json()[str(release)]
    pts = pd.read_csv(sheet)
    tile_dir = os.path.join("data", "raw", f"esri_wayback_z{ZOOM}", str(release))
    os.makedirs(tile_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    rows = list(pts.itertuples())
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda r: make_image(r, release, tile_dir, os.path.join(out_dir, "images", f"id_{r.id:03d}.jpg")), rows))
        dates = list(pool.map(lambda r: capture_date(cfg["metadataLayerUrl"], r.lon, r.lat), rows))
    def b64(r):  # images are embedded so the page is one self-contained file
        with open(os.path.join(out_dir, "images", f"id_{r.id:03d}.jpg"), "rb") as f:
            return base64.b64encode(f.read()).decode()
    cards = "\n".join(CARD.format(id=r.id, lat=r.lat, lon=r.lon, date=html.escape(d), img=b64(r))
                      for r, d in zip(rows, dates))
    page = PAGE.format(n=len(rows), cards=cards, release=release, release_title=html.escape(cfg["itemTitle"].split("(")[-1].rstrip(")")),
                       ids=json.dumps([int(r.id) for r in rows]),
                       rows=json.dumps([{"id": int(r.id), "lat": r.lat, "lon": r.lon} for r in rows]))
    path = os.path.join(out_dir, "index.html")
    with open(path, "w") as f:
        f.write(page)
    print(f"  {len(rows)} points; imagery dates: {pd.Series(dates).value_counts().to_dict()}")
    print(f"  [done] {os.path.abspath(path)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--release", type=int, required=True)
    p.add_argument("--sheet", default="results/tables/spotcheck_2017_2023.csv")
    p.add_argument("--out", default="results/figures/spotcheck")
    a = p.parse_args()
    run(a.release, a.sheet, a.out)
