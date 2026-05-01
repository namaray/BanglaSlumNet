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
    run_inference_and_export()