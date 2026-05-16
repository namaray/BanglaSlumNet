import os
import glob
import torch
import rasterio
from rasterio.features import shapes
import numpy as np
import cv2
import geopandas as gpd
from shapely.geometry import shape

from sasnet import StructureEncoder
from stage2_fusion import CrossAttentionFusion
from decoder import SegmentationDecoder


# ==========================================
# CONFIG
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR   = os.path.join(os.getcwd(), "dhaka_dataset")
OUTPUT_DIR = os.path.join(os.getcwd(), "final_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FUSION_CKPT  = os.path.join(os.getcwd(), "checkpoints_stage2", "best_fusion.pth")
DECODER_CKPT = os.path.join(os.getcwd(), "checkpoints_stage2", "best_decoder.pth")

THRESHOLD    = 0.5
MORPH_KERNEL = 5


# ==========================================
# HELPERS
# ==========================================
def load_tif(path):
    with rasterio.open(path) as src:
        arr  = src.read().astype(np.float32)[:, :512, :512]
        meta = src.meta.copy()
    return arr, meta


def normalize_satellite(arr):
    return np.clip(arr / 3000.0, 0.0, 1.0).astype(np.float32)


def normalize_aux(arr):
    vmax = float(arr.max())
    return (arr / vmax if vmax > 0 else arr).astype(np.float32)


def load_models():
    struct_enc = StructureEncoder().to(DEVICE)
    fusion     = CrossAttentionFusion(num_se_channels=3).to(DEVICE)
    decoder    = SegmentationDecoder().to(DEVICE)

    for ckpt, model, name in [
        (None,        struct_enc, "StructureEncoder"),
        (FUSION_CKPT, fusion,     "Fusion"),
        (DECODER_CKPT, decoder,   "Decoder"),
    ]:
        if ckpt and os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)
            print(f"✅ Loaded {name}: {ckpt}")
        else:
            print(f"⚠️  No checkpoint for {name} — using random weights.")

    struct_enc.eval()
    fusion.eval()
    decoder.eval()
    return struct_enc, fusion, decoder


def export_geotiff(array, ref_meta, out_path, dtype="float32"):
    out_meta = ref_meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": array.shape[0],
        "width":  array.shape[1],
        "count":  1,
        "dtype":  dtype,
    })
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(array.astype(dtype), 1)


def export_polygons(binary_mask, ref_meta, out_path):
    gen = shapes(
        binary_mask.astype(np.uint8),
        mask=(binary_mask == 1),
        transform=ref_meta["transform"]
    )
    polys = [
        {"geometry": shape(geom), "properties": {"class": "slum"}}
        for geom, val in gen if val == 1
    ]
    if not polys:
        return 0
    gdf = gpd.GeoDataFrame.from_features(polys, crs=ref_meta["crs"])
    gdf.to_file(out_path)
    return len(polys)


# ==========================================
# SINGLE TILE
# ==========================================
def run_tile(tile_id, struct_enc, fusion, decoder):
    clear_path = os.path.join(DATA_DIR, f"{tile_id}_clear.tif")
    ntl_path   = os.path.join(DATA_DIR, f"{tile_id}_ntl.tif")
    pop_path   = os.path.join(DATA_DIR, f"{tile_id}_pop.tif")
    gob_path   = os.path.join(DATA_DIR, f"{tile_id}_gob.tif")

    required = [clear_path, ntl_path, pop_path, gob_path]
    missing  = [p for p in required if not os.path.exists(p)]
    if missing:
        print(f"  ⚠️  Skipping {tile_id} — missing: {[os.path.basename(p) for p in missing]}")
        return

    clear_arr, meta = load_tif(clear_path)
    ntl_arr, _      = load_tif(ntl_path)
    pop_arr, _      = load_tif(pop_path)
    gob_arr, _      = load_tif(gob_path)

    clear_t = torch.from_numpy(normalize_satellite(clear_arr)).unsqueeze(0).to(DEVICE)
    se_stack = [
        torch.from_numpy(normalize_aux(ntl_arr)).unsqueeze(0).to(DEVICE),
        torch.from_numpy(normalize_aux(pop_arr)).unsqueeze(0).to(DEVICE),
        torch.from_numpy(normalize_aux(gob_arr)).unsqueeze(0).to(DEVICE),
    ]

    with torch.no_grad():
        s_m    = struct_enc(clear_t)
        f_m    = fusion(s_m, se_stack)
        logits = decoder(f_m)
        probs  = torch.sigmoid(logits)

    prob_np   = probs.squeeze().cpu().numpy()
    binary    = (prob_np >= THRESHOLD).astype(np.uint8)
    kernel    = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    smoothed  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Export probability raster (float32)  — primary artifact
    prob_out = os.path.join(OUTPUT_DIR, f"{tile_id}_prob.tif")
    export_geotiff(prob_np, meta, prob_out, dtype="float32")

    # Export binary prediction raster (uint8)
    bin_out = os.path.join(OUTPUT_DIR, f"{tile_id}_pred.tif")
    export_geotiff(smoothed, meta, bin_out, dtype="uint8")

    # Export vector shapefile (optional — skip if no detections)
    shp_out   = os.path.join(OUTPUT_DIR, f"{tile_id}_polygons.shp")
    n_polys   = export_polygons(smoothed, meta, shp_out)

    print(f"  ✅ {tile_id} → prob.tif, pred.tif | {n_polys} polygons")


# ==========================================
# BATCH RUNNER
# ==========================================
def run_batch():
    print(f"🚀 Batch Inference  |  device={DEVICE}")
    print(f"   DATA_DIR   : {DATA_DIR}")
    print(f"   OUTPUT_DIR : {OUTPUT_DIR}\n")

    struct_enc, fusion, decoder = load_models()

    # Discover every tile that has a _clear.tif anchor file
    clear_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_clear.tif")))
    tile_ids    = [os.path.basename(f).replace("_clear.tif", "") for f in clear_files]

    if not tile_ids:
        print("❌ No tiles found. Run download_dhaka_dataset.py first.")
        return

    print(f"Found {len(tile_ids)} tiles. Running...\n")

    for i, tile_id in enumerate(tile_ids, 1):
        print(f"[{i}/{len(tile_ids)}] {tile_id}")
        run_tile(tile_id, struct_enc, fusion, decoder)

    print(f"\n🎉 Done. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_batch()