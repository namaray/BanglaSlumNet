
G:\Uni Work\BanglaSlumNet>(echo # decoder.py   & echo ```python   & type "decoder.py"   & echo ```   & echo.) 
# decoder.py 
```python 
import torch
import torch.nn as nn

class SegmentationDecoder(nn.Module):
    def __init__(self, in_channels=256):
        super(SegmentationDecoder, self).__init__()
        
        # Block 1: 64x64 -> 128x128
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Block 2: 128x128 -> 256x256
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Block 3: 256x256 -> 512x512
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Final Output Layer: Compress 32 channels into 1 channel (Slum Probability)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()  # Squashes values between 0.0 (Not Slum) and 1.0 (Slum)
        )

    def forward(self, F):
        # F is your fused feature map from Stage 2[Batch, 256, 64, 64]
        d1 = self.up1(F)    #[Batch, 128, 128, 128]
        d2 = self.up2(d1)   #[Batch, 64, 256, 256]
        d3 = self.up3(d2)   # [Batch, 32, 512, 512]
        out = self.final_conv(d3) #[Batch, 1, 512, 512]
        return out

# --- THE GRAND FINALE: END-TO-END TEST ---
if __name__ == "__main__":
    from sasnet import StructureEncoder
    from stage2_fusion import CrossAttentionFusion
    
    print("🚀 Initializing the FULL BanglaSlumNet Pipeline...")
    
    # 1. Initialize all models
    encoder = StructureEncoder()
    fusion = CrossAttentionFusion()
    decoder = SegmentationDecoder()
    
    # 2. Create Dummy Data (Simulating a real dataloader batch)
    print("\nLoading dummy satellite and socioeconomic data...")
    dummy_sentinel = torch.rand(1, 4, 512, 512)
    dummy_SE =[torch.rand(1, 1, 512, 512) for _ in range(5)]
    
    # 3. THE FORWARD PASS
    print("\nRunning Forward Pass...")
    
    # Stage 1: Disentanglement
    print(" -> Running Stage 1 (SAS-Net Encoder)...")
    s_m = encoder(dummy_sentinel)
    
    # Stage 2: Fusion
    print(" -> Running Stage 2 (Cross-Attention)...")
    F = fusion(s_m, dummy_SE)
    
    # Decoder: Final Mask
    print(" -> Running Decoder...")
    prediction_mask = decoder(F)
    
    print("\n==============================================")
    print(f"🎉 FINAL PREDICTION SHAPE: {prediction_mask.shape}")
    print(f"Min probability: {prediction_mask.min().item():.4f}")
    print(f"Max probability: {prediction_mask.max().item():.4f}")
    print("==============================================")``` 


G:\Uni Work\BanglaSlumNet>(echo # download_dhaka_dataset.py   & echo ```python   & type "download_dhaka_dataset.py"   & echo ```   & echo.) 
# download_dhaka_dataset.py 
```python 
import ee
import geemap
import os

project_id = 'banglaslumnet-research' # <-- PASTE YOUR PROJECT ID
ee.Initialize(project=project_id)
print("✅ Connected to Earth Engine!")

# 1. Get the official boundary of Dhaka
dhaka_boundary = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq('ADM2_NAME', 'Dhaka')).geometry()

# 2. Generate 50 random center points inside Dhaka
print("Generating 50 random sample locations inside Dhaka...")
random_points = ee.FeatureCollection.randomPoints(region=dhaka_boundary, points=50, seed=42)
points_list = random_points.toList(50)

# 3. Setup output directory
output_dir = os.path.join(os.getcwd(), 'dhaka_dataset')
os.makedirs(output_dir, exist_ok=True)

# --- HELPER FUNCTIONS (Same as before, but batched) ---
def get_hazy(region):
    return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(region).filterDate('2023-11-01', '2024-01-31').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).median().select(['B4', 'B3', 'B2', 'B8'])

def get_clear(region):
    return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(region).filterDate('2024-03-01', '2024-05-31').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5)).median().select(['B4', 'B3', 'B2', 'B8'])

def get_socio(region):
    ntl = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate('2023-01-01', '2023-12-31').median().select('avg_rad')
    pop = ee.ImageCollection("WorldPop/GP/100m/pop").filter(ee.Filter.eq('country', 'BGD')).median().unmask(0)
    gob_poly = ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons").filterBounds(region).map(lambda f: f.set('built', 1))
    gob = gob_poly.reduceToImage(properties=['built'], reducer=ee.Reducer.max()).unmask(0)
    return ntl, pop, gob

# 4. Download Loop
print("Downloading data for 50 tiles. This will take roughly 10-15 minutes...")
for i in range(50):
    print(f"Processing Tile {i+1}/50...")
    point = ee.Feature(points_list.get(i)).geometry()
    region = point.buffer(2560).bounds() # 512x512 pixels
    
    # Get Images
    hazy, clear = get_hazy(region), get_clear(region)
    ntl, pop, gob = get_socio(region)
    
    # Export (Saving with a simple index name: dhaka_01_hazy.tif, etc.)
    base_name = f"dhaka_{str(i+1).zfill(2)}"
    geemap.ee_export_image(hazy, filename=os.path.join(output_dir, f"{base_name}_hazy.tif"), scale=10, region=region, file_per_band=False)
    geemap.ee_export_image(clear, filename=os.path.join(output_dir, f"{base_name}_clear.tif"), scale=10, region=region, file_per_band=False)
    geemap.ee_export_image(ntl, filename=os.path.join(output_dir, f"{base_name}_ntl.tif"), scale=10, region=region, file_per_band=False)
    geemap.ee_export_image(pop, filename=os.path.join(output_dir, f"{base_name}_pop.tif"), scale=10, region=region, file_per_band=False)
    geemap.ee_export_image(gob, filename=os.path.join(output_dir, f"{base_name}_gob.tif"), scale=10, region=region, file_per_band=False)

print("✅ Dhaka Dataset Download Complete!")
``` 


G:\Uni Work\BanglaSlumNet>(echo # download_pairs.py   & echo ```python   & type "download_pairs.py"   & echo ```   & echo.) 
# download_pairs.py 
```python 
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

print("\n✅ All Hazy/Clear pairs downloaded successfully!")``` 


G:\Uni Work\BanglaSlumNet>(echo # download_s2.py   & echo ```python   & type "download_s2.py"   & echo ```   & echo.) 
# download_s2.py 
```python 
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

print("✅ Download Complete! You have your Sentinel-2 data.")``` 


G:\Uni Work\BanglaSlumNet>(echo # download_socioeconomic.py   & echo ```python   & type "download_socioeconomic.py"   & echo ```   & echo.) 
# download_socioeconomic.py 
```python 
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

print("\n✅ All Socioeconomic features downloaded successfully!")``` 


G:\Uni Work\BanglaSlumNet>(echo # download_temporal.py   & echo ```python   & type "download_temporal.py"   & echo ```   & echo.) 
# download_temporal.py 
```python 
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

print("\n✅ Temporal & Radar Dataset Downloaded Successfully!")``` 


G:\Uni Work\BanglaSlumNet>(echo # inference_export.py   & echo ```python   & type "inference_export.py"   & echo ```   & echo.) 
# inference_export.py 
```python 
import os
import torch
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_origin
import numpy as np
import cv2
import geopandas as gpd
from shapely.geometry import shape

# Import our models
from sasnet import StructureEncoder
from stage2_fusion import CrossAttentionFusion
from decoder import SegmentationDecoder

def run_inference_and_export():
    device = torch.device('cpu') # Forcing CPU for this test
    print("🚀 Initializing Inference Pipeline...")

    # 1. Load the Models (In reality, you would load your trained .pth weights here!)
    # e.g., struct_enc.load_state_dict(torch.load('best_sasnet.pth'))
    struct_enc = StructureEncoder().to(device)
    struct_enc.eval() # Set to evaluation mode!
    
    fusion = CrossAttentionFusion(num_se_channels=3).to(device)
    fusion.eval()
    
    decoder = SegmentationDecoder().to(device)
    decoder.eval()

    # 2. Load the input data (Let's use Korail from our dataset)
    print("Loading satellite and socioeconomic data for Korail...")
    def load_tif(path):
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)
            meta = src.meta # WE NEED THIS META TO SAVE THE MAP EXACTLY WHERE IT BELONGS ON EARTH!
        return torch.from_numpy(img)[:, :512, :512], meta

    # We use the Clear image + SE data
    s2_img, s2_meta = load_tif('paired_dataset/korail_clear.tif')
    s2_img = (s2_img / 3000.0).clamp(0, 1).unsqueeze(0) # Add batch dimension

    ntl, _ = load_tif('paired_dataset/korail_ntl.tif')
    pop, _ = load_tif('paired_dataset/korail_pop.tif')
    gob, _ = load_tif('paired_dataset/korail_gob.tif')
    
    se_stack =[
        (ntl / (ntl.max() + 1e-5)).unsqueeze(0),
        (pop / (pop.max() + 1e-5)).unsqueeze(0),
        (gob / (gob.max() + 1e-5)).unsqueeze(0)
    ]

    # 3. RUN THE FORWARD PASS (No gradients needed)
    print("Running AI Prediction...")
    with torch.no_grad():
        s_m = struct_enc(s2_img)
        f_m = fusion(s_m, se_stack)
        # Note: In inference, we use the final Sigmoid layer!
        prob_mask = decoder(f_m) 

    # 4. POST-PROCESSING (Blueprint Section 3.6)
    print("Applying Morphological Closing & Thresholding...")
    prob_numpy = prob_mask.squeeze().cpu().numpy()
    
    # Threshold at 0.5
    binary_mask = (prob_numpy > 0.5).astype(np.uint8)
    
    # Morphological Closing (5x5 kernel) to remove isolated noise pixels
    kernel = np.ones((5, 5), np.uint8)
    smoothed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # 5. EXPORT TO GEOTIFF
    output_dir = os.path.join(os.getcwd(), 'final_outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    tif_out_path = os.path.join(output_dir, 'korail_slum_prediction.tif')
    print(f"Exporting GeoTIFF to: {tif_out_path}")
    
    out_meta = s2_meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": 512,
        "width": 512,
        "count": 1,
        "dtype": 'uint8'
    })
    
    with rasterio.open(tif_out_path, 'w', **out_meta) as dest:
        dest.write(smoothed_mask, 1)

    # 6. EXPORT TO SHAPEFILE (.shp)
    shp_out_path = os.path.join(output_dir, 'korail_slum_polygons.shp')
    print(f"Exporting Vector Shapefile to: {shp_out_path}")
    
    # Convert pixels to vector polygons
    mask_generator = shapes(smoothed_mask, mask=(smoothed_mask == 1), transform=s2_meta['transform'])
    polygons =[{"geometry": shape(geom), "properties": {"class": "slum"}} for geom, val in mask_generator]
    
    if len(polygons) > 0:
        gdf = gpd.GeoDataFrame.from_features(polygons, crs=s2_meta['crs'])
        gdf.to_file(shp_out_path)
        print(f"✅ Generated {len(polygons)} slum polygons!")
    else:
        print("✅ No slum pixels detected in this tile.")

    print("🎉 INFERENCE COMPLETE!")

if __name__ == "__main__":
    run_inference_and_export()``` 


G:\Uni Work\BanglaSlumNet>(echo # sasnet.py   & echo ```python   & type "sasnet.py"   & echo ```   & echo.) 
# sasnet.py 
```python 
import torch
import torch.nn as nn
import torchvision.models as models

# ==========================================
# 1. STRUCTURE ENCODER (The Physical Buildings)
# ==========================================
class StructureEncoder(nn.Module):
    def __init__(self, in_channels=4):
        super(StructureEncoder, self).__init__()
        
        effnet = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        
        # Replace the first layer for 4-channel input (RGB + NIR)
        original_first_conv = effnet.features[0][0]
        self.first_conv = nn.Conv2d(in_channels=in_channels, 
                                    out_channels=original_first_conv.out_channels, 
                                    kernel_size=original_first_conv.kernel_size, 
                                    stride=original_first_conv.stride, 
                                    padding=original_first_conv.padding, 
                                    bias=False)
        
        with torch.no_grad():
            self.first_conv.weight[:, :3] = original_first_conv.weight
            self.first_conv.weight[:, 3] = original_first_conv.weight.mean(dim=1)
            
        # FIX: We only take the first 4 blocks of EfficientNet!
        # This stops the compression at Stride 8 (64x64 resolution) instead of Stride 32 (16x16)
        self.backbone = nn.Sequential(*list(effnet.features.children())[1:4])
        
        # At block 4, EffNet-B2 has 48 channels. We project it to 256 as per the blueprint.
        self.adapter = nn.Conv2d(48, 256, kernel_size=1)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.backbone(x)
        s_m = self.adapter(x)
        return s_m

# ==========================================
# 2. APPEARANCE ENCODER (The Smog / Haze)
# ==========================================
class AppearanceEncoder(nn.Module):
    def __init__(self, in_channels=4):
        super(AppearanceEncoder, self).__init__()
        # Blueprint: 4x strided Conv3x3 + Global AvgPool + FC(32)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU()
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 32) # Outputs a 32-dim vector

    def forward(self, x):
        x = self.convs(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        a_m = self.fc(x)
        return a_m

# ==========================================
# 3. AdaIN (Adaptive Instance Normalization)
# ==========================================
class AdaIN(nn.Module):
    def __init__(self, style_dim=32, num_features=256):
        super(AdaIN, self).__init__()
        # Learns to map the 32-dim appearance vector to the exact mean/std needed to tint the image
        self.fc_gamma = nn.Linear(style_dim, num_features)
        self.fc_beta = nn.Linear(style_dim, num_features)

    def forward(self, structure, appearance):
        # 1. Normalize the structure (strip away any existing weather/style)
        b, c, h, w = structure.size()
        structure_view = structure.view(b, c, -1)
        mean = structure_view.mean(dim=2, keepdim=True).unsqueeze(3)
        std = structure_view.std(dim=2, keepdim=True).unsqueeze(3) + 1e-5
        normalized_structure = (structure - mean) / std
        
        # 2. Generate the new weather/style multipliers from the appearance code
        gamma = self.fc_gamma(appearance).view(b, c, 1, 1)
        beta = self.fc_beta(appearance).view(b, c, 1, 1)
        
        # 3. Apply the new style!
        styled_output = normalized_structure * gamma + beta
        return styled_output

# --- Let's test the whole SAS-Net Encoder combo! ---
if __name__ == "__main__":
    dummy_image = torch.rand(1, 4, 512, 512)
    
    struct_enc = StructureEncoder()
    appear_enc = AppearanceEncoder()
    adain = AdaIN(style_dim=32, num_features=256)
    
    # 1. Extract Structure (Buildings)
    s_m = struct_enc(dummy_image)
    print(f"✅ Structure Code Shape: {s_m.shape} (Blueprint says H/8 x W/8 x 256 -> 64x64)")
    
    # 2. Extract Appearance (Smog)
    a_m = appear_enc(dummy_image)
    print(f"✅ Appearance Code Shape: {a_m.shape} (Blueprint says 32-dim vector)")
    
    # 3. Mix them together!
    recombined = adain(s_m, a_m)
    print(f"✅ Recombined AdaIN Shape: {recombined.shape}")``` 


G:\Uni Work\BanglaSlumNet>(echo # slum_dataloader.py   & echo ```python   & type "slum_dataloader.py"   & echo ```   & echo.) 
# slum_dataloader.py 
```python 
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
import os

class BanglaSlumDataset(Dataset):
    def __init__(self, image_paths):
        """
        image_paths: A list of file paths to our .tif images
        """
        self.image_paths = image_paths

    def __len__(self):
        # Tells PyTorch how many images we have in total
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Tells PyTorch how to open ONE specific image
        img_path = self.image_paths[idx]
        
        with rasterio.open(img_path) as src:
            # Read all 4 bands
            img_array = src.read() 
        
        # 1. Convert to float32 (Neural Networks hate integers)
        img_array = img_array.astype(np.float32)
        
        # 2. Normalize the data to be between 0 and 1
        # Sentinel-2 raw values usually max out around 3000 for normal land
        img_array = img_array / 3000.0
        img_array = np.clip(img_array, 0.0, 1.0)
        
        # 3. Convert the numpy array into a PyTorch Tensor
        tensor_image = torch.from_numpy(img_array)
        
        # --- NEW LINE: Force exact 512x512 crop ---
        tensor_image = tensor_image[:, :512, :512]
        
        return tensor_image
# --- Let's test it! ---
if __name__ == "__main__":
    # We only have 1 image right now, so we make a list of just one file
    my_files =['dhaka_sentinel2_tile.tif']
    
    # 1. Create the Dataset
    my_dataset = BanglaSlumDataset(image_paths=my_files)
    
    # 2. Create the DataLoader (We tell it to grab 1 image per batch)
    my_dataloader = DataLoader(my_dataset, batch_size=1, shuffle=False)
    
    print("✅ DataLoader created successfully!")
    
    # 3. Fetch the first batch of data
    for batch in my_dataloader:
        print(f"Data type: {type(batch)}")
        print(f"Batch Shape: {batch.shape} -> [Batch_Size, Channels, Height, Width]")
        print(f"Max pixel value: {batch.max():.4f}")
        print(f"Min pixel value: {batch.min():.4f}")
        break  # We only want to test the first batch``` 


G:\Uni Work\BanglaSlumNet>(echo # stage1_gan.py   & echo ```python   & type "stage1_gan.py"   & echo ```   & echo.) 
# stage1_gan.py 
```python 
import torch
import torch.nn as nn

# ==========================================
# 1. THE PATCHGAN DISCRIMINATOR
# ==========================================
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=4):
        super(PatchGANDiscriminator, self).__init__()
        
        # A 4-layer Convolutional Network that outputs a "grid" of True/False scores 
        # instead of just one single score for the whole image.
        self.model = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4 (Blueprint specifies 4 Conv layers)
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output Layer (1 channel logit map)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# ==========================================
# 2. THE BLUEPRINT LOSS FUNCTIONS (Section 2.5)
# ==========================================
class SASNetLoss(nn.Module):
    def __init__(self, lambda_adv=0.1, lambda_scene=1.0):
        super(SASNetLoss, self).__init__()
        self.lambda_adv = lambda_adv
        self.lambda_scene = lambda_scene
        
        # Loss Metrics
        self.mse_loss = nn.MSELoss()              # For L_recon
        self.l1_loss = nn.L1Loss()                # For L_scene
        self.bce_logits = nn.BCEWithLogitsLoss()  # For L_adv

    def forward(self, fake_clear_img, real_clear_img, s_hazy, s_clear, disc_pred_fake):
        # 1. L_recon (Pixel MSE Reconstruction)
        # Does the generated image look EXACTLY like the real clear image pixel-by-pixel?
        L_recon = self.mse_loss(fake_clear_img, real_clear_img)
        
        # 2. L_scene (Structure Invariance L1)
        # The physical buildings (s_m) must remain identical whether it's hazy or clear!
        L_scene = self.l1_loss(s_hazy, s_clear)
        
        # 3. L_adv (PatchGAN Adversarial Loss)
        # Trick the discriminator into thinking the fake image is real (Target = 1.0)
        target_real = torch.ones_like(disc_pred_fake)
        L_adv = self.bce_logits(disc_pred_fake, target_real)
        
        # 4. Total Stage 1 Loss (Blueprint Eq: L_recon + λ1*L_adv + λ2*L_scene)
        L_stage1 = L_recon + (self.lambda_adv * L_adv) + (self.lambda_scene * L_scene)
        
        return L_stage1, L_recon, L_adv, L_scene

# --- Let's test the Math! ---
if __name__ == "__main__":
    print("Initializing PatchGAN Discriminator and Loss Functions...")
    discriminator = PatchGANDiscriminator(in_channels=4)
    loss_calculator = SASNetLoss()
    
    # Fake tensors simulating the training loop
    fake_clear = torch.rand(1, 4, 512, 512)
    real_clear = torch.rand(1, 4, 512, 512)
    
    s_hazy = torch.rand(1, 256, 64, 64) # Structure extracted from Hazy image
    s_clear = torch.rand(1, 256, 64, 64) # Structure extracted from Clear image
    
    print("\nRunning Discriminator...")
    disc_output = discriminator(fake_clear)
    print(f"✅ Discriminator Output Shape: {disc_output.shape} (Should be a downscaled grid map)")
    
    print("\nCalculating Loss...")
    total_loss, l_rec, l_adv, l_sce = loss_calculator(fake_clear, real_clear, s_hazy, s_clear, disc_output)
    
    print(f"✅ L_recon Loss: {l_rec.item():.4f}")
    print(f"✅ L_scene Loss: {l_sce.item():.4f}")
    print(f"✅ L_adv Loss:   {l_adv.item():.4f}")
    print(f"🔥 TOTAL LOSS:   {total_loss.item():.4f}")``` 


G:\Uni Work\BanglaSlumNet>(echo # stage2_fusion.py   & echo ```python   & type "stage2_fusion.py"   & echo ```   & echo.) 
# stage2_fusion.py 
```python 
import torch
import torch.nn as nn

# ==========================================
# 1. SOCIOECONOMIC CHANNEL ENCODER
# ==========================================
class SEChannelEncoder(nn.Module):
    """
    Takes ONE socioeconomic raster (e.g., Nighttime Lights) of size 512x512
    and shrinks it down to 64x64 with 64 feature channels.
    """
    def __init__(self):
        super(SEChannelEncoder, self).__init__()
        # 3 layers of Conv->BN->ReLU with Stride 2 to shrink 512 -> 256 -> 128 -> 64
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. CROSS-ATTENTION FUSION
# ==========================================
class CrossAttentionFusion(nn.Module):
    def __init__(self, num_se_channels=5, d_model=256, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        
        # We need 5 independent encoders (NTL, Pop, GOB/OSM, Poverty, Kilns)
        self.se_encoders = nn.ModuleList([SEChannelEncoder() for _ in range(num_se_channels)])
        
        # After concatenating the 5 encoders (5 * 64 = 320), we project it to 256 to match s_m
        self.se_project = nn.Conv2d(num_se_channels * 64, d_model, kernel_size=1)
        
        # The Multi-Head Attention Mechanism
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

    def forward(self, S, se_inputs):
        """
        S: The clean structure code from SAS-Net [Batch, 256, 64, 64]
        se_inputs: List of 5 tensors, each [Batch, 1, 512, 512]
        """
        b, c, h, w = S.shape
        
        # 1. Encode all 5 socioeconomic channels independently
        encoded_se =[]
        for i in range(len(self.se_encoders)):
            encoded_se.append(self.se_encoders[i](se_inputs[i]))
            
        # 2. Concatenate them all together and project to d_model (256)
        E = torch.cat(encoded_se, dim=1) # Shape: [Batch, 320, 64, 64]
        E = self.se_project(E)           # Shape:[Batch, 256, 64, 64]
        
        # 3. Reshape for PyTorch's Attention layer
        # Attention expects sequences, so we flatten the 64x64 grid into 4096 "pixels"
        # Shape becomes [Batch, 4096, 256]
        S_flat = S.view(b, c, -1).permute(0, 2, 1) 
        E_flat = E.view(b, c, -1).permute(0, 2, 1)
        
        # 4. CROSS ATTENTION! 
        # Query = Structure (What am I looking at?)
        # Key/Value = Socioeconomic (Is this a slum?)
        attn_output, _ = self.multihead_attn(query=S_flat, key=E_flat, value=E_flat)
        
        # 5. Reshape back to image format[Batch, 256, 64, 64]
        attn_output = attn_output.permute(0, 2, 1).view(b, c, h, w)
        
        # 6. RESIDUAL ADDITION (As strictly specified in your blueprint equation)
        # F = CrossAttention(S, E) + S
        F = attn_output + S
        
        return F

# --- Let's test the Fusion! ---
if __name__ == "__main__":
    print("Initializing Cross-Attention Fusion Module...")
    fusion_module = CrossAttentionFusion(num_se_channels=5)
    
    # Fake structure code (coming from your SAS-Net)
    dummy_S = torch.rand(1, 256, 64, 64)
    
    # Fake Socioeconomic Rasters (5 separate maps of 512x512)
    dummy_NTL = torch.rand(1, 1, 512, 512)
    dummy_Pop = torch.rand(1, 1, 512, 512)
    dummy_GOB = torch.rand(1, 1, 512, 512)
    dummy_Poverty = torch.rand(1, 1, 512, 512)
    dummy_Kiln = torch.rand(1, 1, 512, 512)
    
    se_inputs =[dummy_NTL, dummy_Pop, dummy_GOB, dummy_Poverty, dummy_Kiln]
    
    print("\nFusing Visual Structure with Socioeconomic Context...")
    F = fusion_module(dummy_S, se_inputs)
    
    print(f"✅ Fused Feature Map (F) Shape: {F.shape}")``` 


G:\Uni Work\BanglaSlumNet>(echo # stage3_temporal.py   & echo ```python   & type "stage3_temporal.py"   & echo ```   & echo.) 
# stage3_temporal.py 
```python 
import torch
import torch.nn as nn

# ==========================================
# 1. THE CONVLSTM CELL (Blueprint Sec 2.4)
# ==========================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        # We combine the 4 math gates (Input, Forget, Cell, Output) into ONE convolution 
        # for extreme efficiency. It outputs 4 * hidden_dim channels.
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding)

    def forward(self, x_t, cur_state):
        h_cur, c_cur = cur_state
        
        # Concatenate the current input with the past memory
        combined = torch.cat([x_t, h_cur], dim=1) 
        combined_conv = self.conv(combined)
        
        # Split the convolution back into the 4 LSTM gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i) # Input gate
        f = torch.sigmoid(cc_f) # Forget gate
        o = torch.sigmoid(cc_o) # Output gate
        g = torch.tanh(cc_g)    # Cell state update
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

# ==========================================
# 2. THE 10-YEAR SEQUENCE LOOP
# ==========================================
class TemporalModule(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super(TemporalModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell = ConvLSTMCell(input_dim, hidden_dim)

    def forward(self, x_seq):
        # x_seq shape: [Batch, Time, Channels, Height, Width]
        b, seq_len, c, h, w = x_seq.size()
        
        # Start with a blank memory for Year 1
        h_t = torch.zeros(b, self.hidden_dim, h, w, device=x_seq.device)
        c_t = torch.zeros(b, self.hidden_dim, h, w, device=x_seq.device)
        
        outputs =[]
        
        # Loop through Time (Year 1 to Year T)
        for t in range(seq_len):
            h_t, c_t = self.cell(x_seq[:, t, :, :, :], (h_t, c_t))
            outputs.append(h_t)
            
        # Stack the outputs back together along the Time dimension
        # Result shape: [Batch, Time, Hidden_Dim, Height, Width]
        outputs = torch.stack(outputs, dim=1)
        return outputs

# ==========================================
# 3. TEMPORAL SMOOTHNESS LOSS (Blueprint Sec 2.5)
# ==========================================
class TemporalLoss(nn.Module):
    def __init__(self):
        super(TemporalLoss, self).__init__()

    def forward(self, y_hat_seq, M_nochange):
        """
        y_hat_seq: Predictions across time[Batch, Time, 1, 512, 512]
        M_nochange: Binary SAR mask (1 = no construction, 0 = construction detected)
        """
        # We calculate the difference between Year 2 and Year 1, Year 3 and Year 2, etc.
        diff = y_hat_seq[:, 1:] - y_hat_seq[:, :-1]
        
        # Square the difference
        squared_diff = diff ** 2
        
        # Multiply by the SAR change mask.
        # If SAR says "No Change" (1), but the AI predicted a change, the AI is severely punished!
        # If SAR says "Construction" (0), the punishment is wiped out (0), allowing the prediction to change.
        l_temp = (squared_diff * M_nochange).sum(dim=1).mean()
        
        return l_temp

# --- LET'S TEST TIME TRAVEL! ---
if __name__ == "__main__":
    print("⏳ Initializing 2D ConvLSTM...")
    conv_lstm = TemporalModule(input_dim=256, hidden_dim=128)
    calc_temp_loss = TemporalLoss()
    
    # Simulate a "3-Year" sequence coming out of Stage 2 Cross-Attention
    #[Batch=1, Time=3, Channels=256, Height=64, Width=64]
    dummy_seq = torch.rand(1, 3, 256, 64, 64)
    
    print("\nPushing 3 years of data through ConvLSTM...")
    out_seq = conv_lstm(dummy_seq)
    
    print(f"✅ Output Sequence Shape: {out_seq.shape}")
    print("   -> (Expected: [1, 3, 128, 64, 64])")
    
    # ---------------------------------------------
    # Testing the SAR Temporal Loss
    # ---------------------------------------------
    # Simulate the final Decoded predictions[Batch=1, Time=3, Channels=1, Height=512, Width=512]
    dummy_preds = torch.rand(1, 3, 1, 512, 512)
    
    # Simulate the SAR "No Change" Mask for the transitions (Time-1 length)
    #[Batch=1, Time=2, Channels=1, Height=512, Width=512]
    dummy_sar_mask = torch.randint(0, 2, (1, 2, 1, 512, 512)).float()
    
    print("\nCalculating SAR-Masked Temporal Loss...")
    loss_val = calc_temp_loss(dummy_preds, dummy_sar_mask)
    print(f"✅ Temporal Loss value: {loss_val.item():.4f}")``` 


G:\Uni Work\BanglaSlumNet>(echo # train_stage1.py   & echo ```python   & type "train_stage1.py"   & echo ```   & echo.) 
# train_stage1.py 
```python 
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np

# Import the modules we already built!
from sasnet import StructureEncoder, AppearanceEncoder, AdaIN
from stage1_gan import PatchGANDiscriminator, SASNetLoss

# ==========================================
# 1. PAIRED DATASET HANDLER
# ==========================================
class PairedSlumDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Find all the 'hazy' files
        self.hazy_files =[f for f in os.listdir(data_dir) if 'hazy.tif' in f]

    def __len__(self):
        return len(self.hazy_files)

    def load_tif(self, path):
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)
        img = img / 3000.0
        img = np.clip(img, 0.0, 1.0)
        tensor = torch.from_numpy(img)[:, :512, :512] # Crop to 512x512
        return tensor

    def __getitem__(self, idx):
        hazy_name = self.hazy_files[idx]
        clear_name = hazy_name.replace('hazy', 'clear')
        
        hazy_tensor = self.load_tif(os.path.join(self.data_dir, hazy_name))
        clear_tensor = self.load_tif(os.path.join(self.data_dir, clear_name))
        return hazy_tensor, clear_tensor

# ==========================================
# 2. STAGE 1 GENERATOR (SAS-Net Wrapper)
# ==========================================
class ImageDecoder(nn.Module):
    """Upsamples the 64x64 feature map back to a 512x512, 4-channel satellite image."""
    def __init__(self):
        super(ImageDecoder, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 4, 1), nn.Sigmoid() # 4 channels for RGB+NIR
        )
    def forward(self, x):
        return self.up(x)

class Stage1Generator(nn.Module):
    def __init__(self):
        super(Stage1Generator, self).__init__()
        self.struct_enc = StructureEncoder()
        self.appear_enc = AppearanceEncoder()
        self.adain = AdaIN()
        self.decoder = ImageDecoder()

    def forward(self, hazy_img, clear_img):
        s_hazy = self.struct_enc(hazy_img)    # Extract buildings from smog
        s_clear = self.struct_enc(clear_img)  # Extract buildings from clear (for L_scene loss)
        a_clear = self.appear_enc(clear_img)  # Extract clean weather
        
        # Mix hazy buildings with clean weather!
        fused = self.adain(s_hazy, a_clear)
        fake_clear_img = self.decoder(fused)
        return fake_clear_img, s_hazy, s_clear

# ==========================================
# 3. THE TRAINING LOOP
# ==========================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on: {device}")

    # 1. Load Data
    dataset = PairedSlumDataset(data_dir=os.path.join(os.getcwd(), 'paired_dataset'))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"📦 Loaded {len(dataset)} paired images.")

    # 2. Initialize Models
    G = Stage1Generator().to(device)
    D = PatchGANDiscriminator().to(device)
    criterion = SASNetLoss().to(device)

    # 3. Optimizers (As specified in Blueprint Appendix A)
    opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.9, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.9, 0.999))

    epochs = 5 # Just a quick test to see if it works!

    print("\n🔥 Starting Training Loop...")
    for epoch in range(epochs):
        for i, (hazy, clear) in enumerate(dataloader):
            hazy, clear = hazy.to(device), clear.to(device)

            # ---------------------
            # Train Discriminator
            # ---------------------
            opt_D.zero_grad()
            fake_clear, _, _ = G(hazy, clear)
            
            # Predict on Real & Fake
            pred_real = D(clear)
            pred_fake = D(fake_clear.detach())
            
            # D Loss: 1.0 for Real, 0.0 for Fake
            loss_D_real = nn.BCEWithLogitsLoss()(pred_real, torch.ones_like(pred_real))
            loss_D_fake = nn.BCEWithLogitsLoss()(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            
            loss_D.backward()
            opt_D.step()

            # ---------------------
            # Train Generator (SAS-Net)
            # ---------------------
            opt_G.zero_grad()
            # We want D to think our fake image is real!
            pred_fake_for_G = D(fake_clear)
            _, s_hazy, s_clear = G(hazy, clear)
            
            # Use your custom SASNetLoss module
            loss_G, l_rec, l_adv, l_sce = criterion(fake_clear, clear, s_hazy, s_clear, pred_fake_for_G)
            
            loss_G.backward()
            opt_G.step()

        # Print stats at the end of each epoch
        print(f"Epoch[{epoch+1}/{epochs}] | D_Loss: {loss_D.item():.4f} | G_Loss: {loss_G.item():.4f} (Rec:{l_rec.item():.4f}, Adv:{l_adv.item():.4f}, Sce:{l_sce.item():.4f})")
        
    print("✅ Mission 2 Complete: SAS-Net trained successfully!")

if __name__ == "__main__":
    train()``` 


G:\Uni Work\BanglaSlumNet>(echo # train_stage2.py   & echo ```python   & type "train_stage2.py"   & echo ```   & echo.) 
# train_stage2.py 
```python 
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np

# Import our custom modules
from sasnet import StructureEncoder
from stage2_fusion import CrossAttentionFusion
from decoder import SegmentationDecoder

# ==========================================
# 1. THE LOSS FUNCTION (Blueprint Sec 2.5)
# ==========================================
class Stage2Loss(nn.Module):
    def __init__(self, w_slum=3.0, w_nonslum=1.0):
        super(Stage2Loss, self).__init__()
        # We use a trick in PyTorch: BCEWithLogitsLoss is more stable than normal BCE
        # We apply the class weights (3.0 for slum) to penalize missing a slum 3x more!
        pos_weight = torch.tensor([w_slum / w_nonslum])
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, pred_logits, target_mask):
        # 1. Binary Cross Entropy Loss
        l_bce = self.bce(pred_logits, target_mask)
        
        # 2. Soft Differentiable IoU Loss
        pred_probs = torch.sigmoid(pred_logits)
        intersection = (pred_probs * target_mask).sum(dim=(1,2,3))
        union = (pred_probs + target_mask - (pred_probs * target_mask)).sum(dim=(1,2,3))
        l_iou = 1.0 - (intersection / (union + 1e-6)).mean()
        
        # 3. Total Loss (Blueprint Eq: L_IoU + 0.5 * L_BCE)
        total_loss = l_iou + (0.5 * l_bce)
        return total_loss, l_iou, l_bce

# ==========================================
# 2. MULTI-MODAL DATALOADER
# ==========================================
class MultiModalSlumDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.locations = ['mirpur', 'korail', 'old_dhaka']

    def __len__(self):
        return len(self.locations)

    def load_tif(self, path, is_se=False):
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)
            
        if not is_se:
            # Satellite normalization
            img = img / 3000.0
            img = np.clip(img, 0.0, 1.0)
        else:
            # Socioeconomic normalization (Min-Max scaling trick for safety)
            img_max = img.max() if img.max() > 0 else 1.0
            img = img / img_max
            
        tensor = torch.from_numpy(img)[:, :512, :512]
        return tensor

    def __getitem__(self, idx):
        loc = self.locations[idx]
        
        # 1. Load Visuals
        clear_img = self.load_tif(os.path.join(self.data_dir, f"{loc}_clear.tif"), is_se=False)
        
        # 2. Load Socioeconomics
        ntl = self.load_tif(os.path.join(self.data_dir, f"{loc}_ntl.tif"), is_se=True)
        pop = self.load_tif(os.path.join(self.data_dir, f"{loc}_pop.tif"), is_se=True)
        gob = self.load_tif(os.path.join(self.data_dir, f"{loc}_gob.tif"), is_se=True)
        
        se_stack = [ntl, pop, gob]
        
        # 3. Dummy Ground Truth Label (Random 0s and 1s)
        # In a real run, this would be the GRAM dataset masks
        dummy_label = torch.randint(0, 2, (1, 512, 512)).float()
        
        return clear_img, se_stack, dummy_label

# ==========================================
# 3. THE MASTER STAGE 2 TRAINING LOOP
# ==========================================
def train_stage2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Stage 2 Training on: {device}")

    # Load Data
    dataset = MultiModalSlumDataset(data_dir=os.path.join(os.getcwd(), 'paired_dataset'))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize Models
    struct_enc = StructureEncoder().to(device)
    # We only have 3 SE channels downloaded right now (NTL, Pop, GOB)
    fusion = CrossAttentionFusion(num_se_channels=3).to(device)
    decoder = SegmentationDecoder().to(device)
    
    criterion = Stage2Loss().to(device)

    # Optimizer: We only train Fusion and Decoder! Structure Encoder is frozen (from Stage 1)
    opt = optim.Adam(list(fusion.parameters()) + list(decoder.parameters()), lr=1e-4)

    epochs = 5
    print("\n🔥 Starting Stage 2 Fusion Training Loop...")
    
    for epoch in range(epochs):
        for clear_img, se_stack, target_mask in dataloader:
            clear_img = clear_img.to(device)
            se_stack = [se.to(device) for se in se_stack]
            target_mask = target_mask.to(device)

            opt.zero_grad()
            
            # 1. Extract Visual Structure (No gradients needed here)
            with torch.no_grad():
                s_clear = struct_enc(clear_img)
            
            # 2. Cross-Attention Fusion
            F = fusion(s_clear, se_stack)
            
            # 3. Decode to Slum Mask (We remove the final Sigmoid in decoder to use BCEWithLogits)
            # A quick hack: pass through the first layer of the final conv, skipping sigmoid
            pred_logits = decoder.final_conv[0](decoder.up3(decoder.up2(decoder.up1(F))))
            
            # 4. Calculate Loss
            loss, l_iou, l_bce = criterion(pred_logits, target_mask)
            
            # 5. Backpropagate
            loss.backward()
            opt.step()

        print(f"Epoch[{epoch+1}/{epochs}] | Total Loss: {loss.item():.4f} (IoU:{l_iou.item():.4f}, BCE:{l_bce.item():.4f})")
        
    print("✅ Mission 3 Complete: Cross-Attention Fusion trained successfully!")

if __name__ == "__main__":
    train_stage2()``` 


G:\Uni Work\BanglaSlumNet>(echo # train_stage3.py   & echo ```python   & type "train_stage3.py"   & echo ```   & echo.) 
# train_stage3.py 
```python 
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np

# Import EVERYTHING we have built!
from sasnet import StructureEncoder
from stage2_fusion import CrossAttentionFusion
from decoder import SegmentationDecoder
from stage3_temporal import TemporalModule, TemporalLoss
from train_stage2 import Stage2Loss

# ==========================================
# 1. 5D TEMPORAL DATALOADER
# ==========================================
class TemporalSlumDataset(Dataset):
    def __init__(self, temp_dir, se_dir):
        self.temp_dir = temp_dir
        self.se_dir = se_dir
        self.locations =['mirpur', 'korail', 'old_dhaka']
        self.years =[2021, 2022, 2023]

    def __len__(self):
        return len(self.locations)

    def load_tif(self, path, is_mask=False):
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)
        
        if not is_mask:
            img = np.clip(img / 3000.0, 0.0, 1.0)
        else:
            img = np.clip(img, 0.0, 1.0)
            
        tensor = torch.from_numpy(img)[:, :512, :512]
        return tensor

    def __getitem__(self, idx):
        loc = self.locations[idx]
        
        # 1. Load 3 Years of Optical Data [Time=3, Channels=4, H=512, W=512]
        opt_seq =[]
        for yr in self.years:
            img = self.load_tif(os.path.join(self.temp_dir, f"{loc}_s2_{yr}.tif"))
            opt_seq.append(img)
        opt_seq = torch.stack(opt_seq) 
        
        # 2. Load 2 Years of SAR Radar Masks[Time=2, Channels=1, H=512, W=512]
        sar_seq =[]
        for i in range(len(self.years)-1):
            y1, y2 = self.years[i], self.years[i+1]
            sar = self.load_tif(os.path.join(self.temp_dir, f"{loc}_sar_{y1}_{y2}.tif"), is_mask=True)
            sar_seq.append(sar)
        sar_seq = torch.stack(sar_seq)
        
        # 3. Load Socioeconomic Data (Static for this test)
        ntl = self.load_tif(os.path.join(self.se_dir, f"{loc}_ntl.tif"), is_mask=True)
        pop = self.load_tif(os.path.join(self.se_dir, f"{loc}_pop.tif"), is_mask=True)
        gob = self.load_tif(os.path.join(self.se_dir, f"{loc}_gob.tif"), is_mask=True)
        se_stack =[ntl, pop, gob]
        
        # 4. Dummy Labels for 3 Years
        dummy_labels = torch.randint(0, 2, (3, 1, 512, 512)).float()
        
        return opt_seq, sar_seq, se_stack, dummy_labels

# ==========================================
# 2. THE MASTER TRAINING LOOP
# ==========================================
def train_stage3():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Stage 3 (Temporal) Training on: {device}")

    # Load Data
    dataset = TemporalSlumDataset(
        temp_dir=os.path.join(os.getcwd(), 'temporal_dataset'),
        se_dir=os.path.join(os.getcwd(), 'paired_dataset')
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize FULL PIPELINE
    print("Loading Models...")
    struct_enc = StructureEncoder().to(device)
    fusion = CrossAttentionFusion(num_se_channels=3).to(device)
    temporal = TemporalModule(input_dim=256, hidden_dim=128).to(device)
    
    # Notice: ConvLSTM outputs 128 channels, so we tell the Decoder to expect 128!
    decoder = SegmentationDecoder(in_channels=128).to(device) 
    
    criterion_seg = Stage2Loss().to(device)
    criterion_temp = TemporalLoss().to(device)

    # Optimizer: Fine-tuning Temporal Module and Decoder
    opt = optim.Adam(list(temporal.parameters()) + list(decoder.parameters()), lr=5e-5)

    epochs = 3
    print("\n🔥 Starting Stage 3 End-to-End Temporal Training...")
    
    for epoch in range(epochs):
        for opt_seq, sar_seq, se_stack, target_labels in dataloader:
            
            # Move to device
            opt_seq, sar_seq, target_labels = opt_seq.to(device), sar_seq.to(device), target_labels.to(device)
            se_stack =[se.to(device) for se in se_stack]

            # Remove batch dimension for iteration (since batch_size=1)
            opt_seq, sar_seq, target_labels = opt_seq[0], sar_seq[0], target_labels[0]
            
            opt.zero_grad()
            
            # --- STEP 1: Process each year independently through Stages 1 & 2 ---
            fused_features_seq =[]
            for t in range(3): # Loop over 2021, 2022, 2023
                with torch.no_grad(): # Freeze Stages 1 & 2
                    s_t = struct_enc(opt_seq[t].unsqueeze(0))
                    f_t = fusion(s_t, se_stack)
                    fused_features_seq.append(f_t.squeeze(0))
            
            # Stack into a Time Sequence: [Batch, Time, Channels, H, W]
            F_seq = torch.stack(fused_features_seq).unsqueeze(0) 
            
            # --- STEP 2: The Time Machine (ConvLSTM) ---
            H_seq = temporal(F_seq) #[Batch, Time, 128, 64, 64]
            
            # --- STEP 3: Decode each year into a Slum Mask ---
            pred_seq =[]
            for t in range(3):
                # Pass through decoder (skipping final sigmoid for BCEWithLogits)
                h_t = H_seq[:, t, :, :, :]
                logits_t = decoder.final_conv[0](decoder.up3(decoder.up2(decoder.up1(h_t))))
                pred_seq.append(logits_t)
            
            pred_seq = torch.stack(pred_seq, dim=1) #[Batch, Time, 1, 512, 512]
            
            # --- STEP 4: Calculate Master Loss ---
            # Segmentation Loss (Average across all 3 years)
            loss_seg = 0
            for t in range(3):
                l_total, _, _ = criterion_seg(pred_seq[:, t], target_labels[t].unsqueeze(0))
                loss_seg += l_total
            loss_seg = loss_seg / 3.0
            
            # Temporal Smoothness Loss (Checked against SAR Radar)
            loss_temp = criterion_temp(torch.sigmoid(pred_seq), sar_seq.unsqueeze(0))
            
            # Blueprint Eq: L_stage3 = L_stage2 + 0.3 * L_temp
            final_loss = loss_seg + (0.3 * loss_temp)
            
            # --- STEP 5: Backpropagate through Time ---
            final_loss.backward()
            opt.step()

        print(f"Epoch[{epoch+1}/{epochs}] | Final Loss: {final_loss.item():.4f} (Seg: {loss_seg.item():.4f}, Temp: {loss_temp.item():.4f})")
        
    print("✅ Mission 4 Complete: 10-Year ConvLSTM Engine Validated!")

if __name__ == "__main__":
    train_stage3()``` 


G:\Uni Work\BanglaSlumNet>(echo # view_tile.py   & echo ```python   & type "view_tile.py"   & echo ```   & echo.) 
# view_tile.py 
```python 
import rasterio
import matplotlib.pyplot as plt
import numpy as np

# 1. Open the satellite image
image_path = 'dhaka_sentinel2_tile.tif'
with rasterio.open(image_path) as src:
    # Read the 4 bands (Red, Green, Blue, NIR)
    # rasterio reads as (Channels, Height, Width)
    img_tensor = src.read()

print(f"✅ Successfully loaded image!")
print(f"Shape of the data: {img_tensor.shape} (Channels, Height, Width)")

# 2. Extract the RGB bands to show a normal photo
# Bands are 1-indexed in rasterio. 
# In our download: 1=Red, 2=Green, 3=Blue, 4=NIR
red = img_tensor[0]
green = img_tensor[1]
blue = img_tensor[2]

# Stack them into a normal image shape (Height, Width, Channels)
rgb_image = np.dstack((red, green, blue))

# 3. Satellite images are dark by default (raw physics data). 
# We normalize the brightness so our human eyes can see it.
rgb_image = rgb_image / 3000.0  # 3000 is a standard Sentinel-2 brightness cap
rgb_image = np.clip(rgb_image, 0, 1)

# 4. Show the image!
plt.figure(figsize=(8, 8))
plt.imshow(rgb_image)
plt.title("Dhaka Sentinel-2 Tile (512x512)")
plt.axis('off')
plt.show()``` 

