import ee
import geemap
import os

# 1. Connect to Earth Engine
project_id = 'banglaslumnet-research' # <--- PASTE YOUR PROJECT ID HERE
ee.Initialize(project=project_id)
print("✅ Connected to Earth Engine!")

# 2. Define the exact same 3 locations
locations = {
    "mirpur": ee.Geometry.Point([90.36, 23.81]),
    "korail": ee.Geometry.Point([90.40, 23.79]),
    "old_dhaka": ee.Geometry.Point([90.39, 23.71])
}

# 3. Create the output folder (we will save them in the same folder as the pairs)
output_dir = os.path.join(os.getcwd(), 'paired_dataset')
os.makedirs(output_dir, exist_ok=True)

print("Preparing global socioeconomic datasets...")

for name, point in locations.items():
    print(f"\nProcessing Socioeconomic Data for: {name.upper()}...")
    
    # EXACT same 512x512 pixel bounding box
    region = point.buffer(2560).bounds()
    
    # --- 1. NIGHTTIME LIGHTS (VIIRS) ---
    # Take the median brightness for the whole year to avoid cloud/moonlight noise
    ntl = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
           .filterDate('2023-01-01', '2023-12-31')
           .median()
           .select('avg_rad'))
    
    # --- 2. POPULATION DENSITY (WorldPop) ---
    # Get the latest population estimates for Bangladesh
    pop = (ee.ImageCollection("WorldPop/GP/100m/pop")
           .filter(ee.Filter.eq('country', 'BGD'))
           .median()
           .unmask(0)) # Fill empty areas (like water) with 0 people
    
    # --- 3. GOOGLE OPEN BUILDINGS (Density/Footprints) ---
    # This dataset is polygons. We must "rasterize" it into an image 
    # where 1 = Building and 0 = No Building
    gob_polygons = ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons").filterBounds(region)
    # Assign a value of 1 to every building
    gob_polygons = gob_polygons.map(lambda f: f.set('built', 1))
    # Convert the shapes into a raster image
    gob = gob_polygons.reduceToImage(properties=['built'], reducer=ee.Reducer.max()).unmask(0)

    # --- EXPORT ---
    print(" -> Downloading Nighttime Lights...")
    geemap.ee_export_image(ntl, filename=os.path.join(output_dir, f"{name}_ntl.tif"), scale=10, region=region, file_per_band=False)
    
    print(" -> Downloading Population...")
    geemap.ee_export_image(pop, filename=os.path.join(output_dir, f"{name}_pop.tif"), scale=10, region=region, file_per_band=False)
    
    print(" -> Downloading Google Open Buildings...")
    geemap.ee_export_image(gob, filename=os.path.join(output_dir, f"{name}_gob.tif"), scale=10, region=region, file_per_band=False)

print("\n✅ All Socioeconomic features downloaded successfully!")