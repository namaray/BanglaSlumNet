import ee
import geemap
import os

# 1. Connect to Google Earth Engine using your project
project_id = 'banglaslumnet-research' # <--- PASTE YOUR PROJECT ID HERE (e.g., 'banglaslumnet-research-123456')
ee.Initialize(project=project_id)
print("✅ Connected to Earth Engine!")

# 2. Set our target: A point in Dhaka (Mirpur/Korail area)
dhaka_poi = ee.Geometry.Point([90.40, 23.79])

# Create a 5.12 km x 5.12 km box around the point.
# At 10 meters per pixel, this perfectly creates a 512x512 pixel image.
region = dhaka_poi.buffer(2560).bounds()

# 3. Find the Satellite Images
print("Searching for Sentinel-2 images...")
dataset = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterBounds(region)
           .filterDate('2023-01-01', '2023-03-31') # Clear season (Jan-Mar)
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))) # Less than 10% clouds

# Take the median of the clear images to get one perfect, cloud-free image
clean_image = dataset.median()

# Select the colors: Red (B4), Green (B3), Blue (B2), and Near-Infrared (B8)
# We need 13 bands eventually, but these 4 are enough to test the pipeline!
clean_image = clean_image.select(['B4', 'B3', 'B2', 'B8'])

# 4. Download it!
output_filename = 'dhaka_sentinel2_tile.tif'
output_path = os.path.join(os.getcwd(), output_filename)
print(f"Downloading image to {output_path} ... this might take a minute.")

geemap.ee_export_image(
    clean_image, 
    filename=output_path, 
    scale=10, 
    region=region, 
    file_per_band=False
)

print("✅ Download Complete! You have your Sentinel-2 data.")