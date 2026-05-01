import ee
import geemap
import os

# 1. Connect to Earth Engine
project_id = 'banglaslumnet-research' # <--- PASTE YOUR PROJECT ID HERE
ee.Initialize(project=project_id)
print("✅ Connected to Earth Engine!")

# 2. Define 3 different locations in Dhaka to get a mix of textures
locations = {
    "mirpur": ee.Geometry.Point([90.36, 23.81]),
    "korail": ee.Geometry.Point([90.40, 23.79]),
    "old_dhaka": ee.Geometry.Point([90.39, 23.71])
}

# 3. Create the output folder
output_dir = os.path.join(os.getcwd(), 'paired_dataset')
os.makedirs(output_dir, exist_ok=True)

# 4. Loop through our locations and download the Hazy and Clear image for each
for name, point in locations.items():
    print(f"\nProcessing location: {name.upper()}...")
    
    # 512x512 pixel bounding box
    region = point.buffer(2560).bounds()
    
    # --- GET HAZY IMAGE (Nov 2023 - Jan 2024) ---
    hazy_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                       .filterBounds(region)
                       .filterDate('2023-11-01', '2024-01-31')
                       .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))) # Allow some clouds/haze
    hazy_image = hazy_collection.median().select(['B4', 'B3', 'B2', 'B8'])
    
    # --- GET CLEAR IMAGE (Mar 2024 - May 2024) ---
    clear_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .filterBounds(region)
                        .filterDate('2024-03-01', '2024-05-31')
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))) # Strictly clear
    clear_image = clear_collection.median().select(['B4', 'B3', 'B2', 'B8'])
    
    # File paths
    hazy_path = os.path.join(output_dir, f"{name}_hazy.tif")
    clear_path = os.path.join(output_dir, f"{name}_clear.tif")
    
    # Download Hazy
    print(f" -> Downloading Hazy image...")
    geemap.ee_export_image(hazy_image, filename=hazy_path, scale=10, region=region, file_per_band=False)
    
    # Download Clear
    print(f" -> Downloading Clear image...")
    geemap.ee_export_image(clear_image, filename=clear_path, scale=10, region=region, file_per_band=False)

print("\n✅ All Hazy/Clear pairs downloaded successfully!")