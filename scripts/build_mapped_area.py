"""
Reconstruct the area ESA EO4SD-Urban actually mapped ("Dhaka Core City", 305.8 km2 =
expanded Dhaka North + South City Corporations), so that absence of an ESA slum polygon
can be read as non-slum only inside it.

ESA does not publish its ward polygons, only each slum polygon's ward name (AL3_NAMEF)
and ward area (AREA3). We locate every slum polygon in the geoBoundaries ADM4 units
(BBS / OCHA 2020, CC BY 3.0 IGO) and keep a unit only if it is no larger than
--max_ratio times the combined ESA area of the wards it holds. That drops units that
extend far beyond the city (e.g. Tongi, Siddirganj), erring on the side of mapping too
little rather than too much. With the defaults the result is ~303 km2 and contains
99.4% of ESA polygons; the rest fall in excluded units and are left as ignore.

Inputs (git-ignored, see docs/MASTER_PLAN.md §4):
  data/raw/external_boundaries/esa_eo4sd_urban/EO4SD_DHAKA_INFORMAL_2017.shp
  data/raw/external_boundaries/geoboundaries_bgd/geoBoundaries-BGD-ADM4_simplified.geojson

CLI:
    python scripts/build_mapped_area.py
"""

import argparse
from pathlib import Path

import geopandas as gpd
from shapely.ops import unary_union

ESA_SHP = "data/raw/external_boundaries/esa_eo4sd_urban/EO4SD_DHAKA_INFORMAL_2017.shp"
ADM4 = "data/raw/external_boundaries/geoboundaries_bgd/geoBoundaries-BGD-ADM4_simplified.geojson"
OUT = "data/derived/esa_mapped_area.geojson"
UTM = "EPSG:32646"  # ESA's native CRS; metric, so areas are in m2


def build(esa_shp: str = ESA_SHP, adm4_path: str = ADM4, max_ratio: float = 1.25) -> gpd.GeoDataFrame:
    esa = gpd.read_file(esa_shp).to_crs(UTM)
    adm = gpd.read_file(adm4_path).to_crs(UTM)

    pts = gpd.GeoDataFrame(esa[["ID", "AL3_NAMEF"]], geometry=esa.representative_point(), crs=UTM)
    hits = gpd.sjoin(pts, adm[["shapeID", "geometry"]], predicate="within", how="inner")

    ward_km2 = esa.groupby("AL3_NAMEF").AREA3.first().astype(float) / 1e6
    units = hits.groupby("shapeID").AL3_NAMEF.agg(lambda s: sorted(set(s))).to_frame("wards")
    units["ward_km2"] = units.wards.map(lambda ws: ward_km2[ws].sum())
    units["unit_km2"] = adm.set_index("shapeID").area.reindex(units.index) / 1e6
    keep = units.index[units.unit_km2 / units.ward_km2 <= max_ratio]

    area = unary_union(adm.set_index("shapeID").loc[keep].geometry)
    inside = esa.representative_point().within(area).mean()
    esa_core_km2 = float(esa.AREA2.iloc[0]) / 1e6
    print(f"  {len(keep)} of {len(units)} ADM4 units kept; mapped area {area.area/1e6:.1f} km2 "
          f"(ESA core city {esa_core_km2:.1f} km2); ESA polygons inside {inside:.1%}")
    return gpd.GeoDataFrame({"n_units": [len(keep)], "area_km2": [round(area.area / 1e6, 2)],
                             "max_ratio": [max_ratio]}, geometry=[area], crs=UTM)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--esa", default=ESA_SHP)
    p.add_argument("--adm4", default=ADM4)
    p.add_argument("--max_ratio", type=float, default=1.25)
    p.add_argument("--out", default=OUT)
    a = p.parse_args()
    gdf = build(a.esa, a.adm4, a.max_ratio)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_crs("EPSG:4326").to_file(a.out, driver="GeoJSON")
    print(f"  [done] {a.out}")
