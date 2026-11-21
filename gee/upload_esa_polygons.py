"""
Upload the ESA EO4SD-Urban Dhaka informal-settlement polygons (2017) to an Earth Engine
table asset, so every export and the Colab notebook can use the reference labels
without shipping the shapefile around.

Source (CC-BY 4.0, cite ESA EO4SD-Urban / World Bank):
  https://datacatalogfiles.worldbank.org/ddh-published/0041703/1/DR0052090/eo4sd_dhaka_informal_2017.zip
Expected local copy (git-ignored):
  data/raw/external_boundaries/esa_eo4sd_urban/EO4SD_DHAKA_INFORMAL_2017.shp

Geometries are simplified by --simplify_m metres (default 0.5 m, far below the 10 m
Sentinel-2 pixel) only to keep the upload request small.

CLI:
    python gee/upload_esa_polygons.py --project banglaslumnet-research
"""

import argparse
import json

import geopandas as gpd

try:
    import ee
except ImportError:
    ee = None

try:
    from .ee_export_utils import init_ee
except ImportError:  # direct CLI execution
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(__file__))
    from ee_export_utils import init_ee

DEFAULT_SHP = "data/raw/external_boundaries/esa_eo4sd_urban/EO4SD_DHAKA_INFORMAL_2017.shp"
DEFAULT_ASSET = "projects/banglaslumnet-research/assets/esa_eo4sd_dhaka_informal_2017"
KEEP_FIELDS = {"ID": "esa_id", "NB_LOCTYP": "loc_type", "AL3_NAMEF": "ward", "AREA": "area_m2"}
MAX_REQUEST_BYTES = 8_000_000  # EE rejects client-side payloads around 10 MB


def load_polygons(shp_path: str, simplify_m: float) -> gpd.GeoDataFrame:
    """Read, repair, simplify (in the native metric CRS) and reproject to EPSG:4326."""
    src = gpd.read_file(shp_path)
    gdf = gpd.GeoDataFrame(src[list(KEEP_FIELDS)].rename(columns=KEEP_FIELDS),
                           geometry=src.geometry.buffer(0).simplify(simplify_m), crs=src.crs)
    gdf["esa_id"] = gdf["esa_id"].astype(int)
    return gdf.to_crs("EPSG:4326")


def to_feature_collection(gdf: gpd.GeoDataFrame):
    geojson = json.loads(gdf.to_json())
    size = len(json.dumps(geojson))
    if size > MAX_REQUEST_BYTES:
        raise ValueError(f"GeoJSON is {size/1e6:.1f} MB; raise --simplify_m to shrink it.")
    print(f"  {len(gdf)} polygons, {size/1e6:.2f} MB GeoJSON")
    return ee.FeatureCollection(geojson)


def asset_exists(asset_id: str) -> bool:
    try:
        ee.data.getAsset(asset_id)
        return True
    except ee.EEException:
        return False


def run(shp_path: str = DEFAULT_SHP, asset_id: str = DEFAULT_ASSET,
        simplify_m: float = 0.5, overwrite: bool = False):
    """Start an Export.table.toAsset task; returns the task (or None if skipped)."""
    if asset_exists(asset_id):
        if not overwrite:
            print(f"  [skip] asset exists: {asset_id} (use --overwrite to replace)")
            return None
        ee.data.deleteAsset(asset_id)
    fc = to_feature_collection(load_polygons(shp_path, simplify_m))
    task = ee.batch.Export.table.toAsset(collection=fc, description="esa_eo4sd_dhaka_2017",
                                         assetId=asset_id)
    task.start()
    print(f"  [started] {asset_id} (task {task.id}); watch it at code.earthengine.google.com/tasks")
    return task


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--project", default="banglaslumnet-research")
    p.add_argument("--shp", default=DEFAULT_SHP)
    p.add_argument("--asset_id", default=DEFAULT_ASSET)
    p.add_argument("--simplify_m", type=float, default=0.5)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    init_ee(a.project, authenticate=False)
    run(a.shp, a.asset_id, a.simplify_m, a.overwrite)
