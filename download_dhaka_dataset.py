import os
import ee
import geemap

project_id = "banglaslumnet-research"
ee.Initialize(project=project_id)
print("✅ Connected to Earth Engine!")

# ==========================================
# 1. GLOBAL SETTINGS
# ==========================================
N_TILES = 50
SEED = 42
TILE_RADIUS_M = 2560
SCALE = 10
OUTPUT_DIR = os.path.join(os.getcwd(), "dhaka_dataset")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. DHAKA BOUNDARY
# ==========================================
dhaka_boundary = (
    ee.FeatureCollection("FAO/GAUL/2015/level2")
    .filter(ee.Filter.eq("ADM2_NAME", "Dhaka"))
    .geometry()
)

print(f"Generating {N_TILES} random sample locations inside Dhaka...")
random_points = ee.FeatureCollection.randomPoints(
    region=dhaka_boundary,
    points=N_TILES,
    seed=SEED
)
points_list = random_points.toList(N_TILES)

# ==========================================
# 3. SATELLITE HELPERS
# ==========================================
def get_hazy(region):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate("2023-11-01", "2024-01-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
        .select(["B4", "B3", "B2", "B8"])
        .toFloat()
    )

def get_clear(region):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate("2024-03-01", "2024-05-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 5))
        .median()
        .select(["B4", "B3", "B2", "B8"])
        .toFloat()
    )

# ==========================================
# 4. AUXILIARY DATA HELPERS
# ==========================================
def get_viirs_city():
    return (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterDate("2023-01-01", "2023-12-31")
        .median()
        .select("avg_rad")
        .rename("ntl")
        .unmask(0)
        .toFloat()
    )

def get_worldpop():
    return (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filter(ee.Filter.eq("country", "BGD"))
        .median()
        .select(0)
        .rename("pop")
        .unmask(0)
        .toFloat()
    )

def get_open_buildings(region):
    gob_fc = (
        ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons")
        .filterBounds(region)
        .map(lambda f: f.set("built", 1))
    )
    return (
        gob_fc.reduceToImage(properties=["built"], reducer=ee.Reducer.max())
        .unmask(0)
        .rename("gob")
        .toFloat()
    )

def export_image(img, out_path, region, scale=SCALE):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"  ↪ Skipping existing file: {os.path.basename(out_path)}")
        return

    geemap.ee_export_image(
        img,
        filename=out_path,
        scale=scale,
        region=region,
        file_per_band=False
    )

# ==========================================
# 5. PRECOMPUTE CITY-WIDE LAYERS
# ==========================================
print("Preparing city-wide auxiliary rasters...")
viirs_city = get_viirs_city().clip(dhaka_boundary)
worldpop = get_worldpop()

print("✅ Auxiliary rasters ready.")

# ==========================================
# 6. EXPORT LOOP
# ==========================================
print(f"Downloading/exporting {N_TILES} Dhaka tiles...")
for i in range(N_TILES):
    tile_id = f"dhaka_{str(i + 1).zfill(3)}"
    print(f"Processing Tile {i+1}/{N_TILES}  ->  {tile_id}")

    point = ee.Feature(points_list.get(i)).geometry()
    region = point.buffer(TILE_RADIUS_M).bounds()

    hazy = get_hazy(region)
    clear = get_clear(region)
    ntl = viirs_city.clip(region).rename("ntl")
    pop = worldpop.clip(region).rename("pop")
    gob = get_open_buildings(region).clip(region).rename("gob")

    export_image(hazy,  os.path.join(OUTPUT_DIR, f"{tile_id}_hazy.tif"),  region)
    export_image(clear, os.path.join(OUTPUT_DIR, f"{tile_id}_clear.tif"), region)
    export_image(ntl,   os.path.join(OUTPUT_DIR, f"{tile_id}_ntl.tif"),   region)
    export_image(pop,   os.path.join(OUTPUT_DIR, f"{tile_id}_pop.tif"),   region)
    export_image(gob,   os.path.join(OUTPUT_DIR, f"{tile_id}_gob.tif"),   region)

print("✅ Dhaka raw dataset export complete!")
print("Next: run generate_weak_labels.py")