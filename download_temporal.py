import ee
import geemap
import os

# 1. Connect to Earth Engine
project_id = 'banglaslumnet-research' # <--- PASTE YOUR PROJECT ID HERE
ee.Initialize(project=project_id)
print("✅ Connected to Earth Engine!")

# 2. Define our 3 testing locations
locations = {
    "mirpur": ee.Geometry.Point([90.36, 23.81]),
    "korail": ee.Geometry.Point([90.40, 23.79]),
    "old_dhaka": ee.Geometry.Point([90.39, 23.71])
}

# We will grab a 3-year sequence to test the ConvLSTM
years = [2021, 2022, 2023]

output_dir = os.path.join(os.getcwd(), 'temporal_dataset')
os.makedirs(output_dir, exist_ok=True)

# --- HELPER FUNCTIONS ---
def get_s2_clear(year, region):
    """Gets a clean Sentinel-2 optical composite for a given year."""
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(region)
                  .filterDate(f'{year}-01-01', f'{year}-05-31')
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
    return collection.median().select(['B4', 'B3', 'B2', 'B8'])

def get_sar_nochange_mask(year1, year2, region):
    """
    Calculates the radar difference between two years.
    Returns 1 if no structural change, 0 if a building was built/destroyed.
    """
    # Sentinel-1 Radar penetrates clouds and bounces off hard structures like roofs
    sar1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterBounds(region).filterDate(f'{year1}-01-01', f'{year1}-12-31')
            .filter(ee.Filter.eq('instrumentMode', 'IW')).select('VV').median())
    
    sar2 = (ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterBounds(region).filterDate(f'{year2}-01-01', f'{year2}-12-31')
            .filter(ee.Filter.eq('instrumentMode', 'IW')).select('VV').median())
    
    # Calculate absolute difference in radar backscatter
    diff = sar1.subtract(sar2).abs()
    
    # If the difference is less than 3.0 decibels, assume NO CHANGE (Return 1)
    # If difference > 3.0, a new roof was likely built or demolished (Return 0)
    no_change_mask = diff.lt(3.0)
    return no_change_mask

# --- MAIN DOWNLOAD LOOP ---
print("Warming up the time machine and satellite radar...")

for name, point in locations.items():
    print(f"\nProcessing Time-Series for: {name.upper()}...")
    region = point.buffer(2560).bounds()
    
    # 1. Download Optical Images for 2021, 2022, and 2023
    for yr in years:
        print(f" -> Downloading S2 Optical for {yr}...")
        img = get_s2_clear(yr, region)
        out_path = os.path.join(output_dir, f"{name}_s2_{yr}.tif")
        geemap.ee_export_image(img, filename=out_path, scale=10, region=region, file_per_band=False)
        
    # 2. Download Radar Change Masks (2021->2022, and 2022->2023)
    for i in range(len(years)-1):
        y1, y2 = years[i], years[i+1]
        print(f" -> Downloading SAR Radar Mask for {y1}-{y2}...")
        sar_mask = get_sar_nochange_mask(y1, y2, region)
        out_path = os.path.join(output_dir, f"{name}_sar_{y1}_{y2}.tif")
        geemap.ee_export_image(sar_mask, filename=out_path, scale=10, region=region, file_per_band=False)

print("\n✅ Temporal & Radar Dataset Downloaded Successfully!")