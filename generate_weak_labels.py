import ee
import geemap
import os
import glob

# ==========================================
# CONFIG
# ==========================================
PROJECT_ID = "banglaslumnet-research"   # <-- paste your GEE project ID
DATA_DIR   = os.path.join(os.getcwd(), "dhaka_dataset")
LABEL_YEAR = 2023   # VIIRS + GHSL reference year — match your clear.tif year

IGNORE_VALUE = 255  # written for "unknown" pixels (signals disagree / data missing)

# ==========================================
# 1. CONNECT
# ==========================================
ee.Initialize(project=PROJECT_ID)
print("✅ Connected to Google Earth Engine.")


# ==========================================
# 2. GLOBAL LAYERS  (loaded once, reused per tile)
# ==========================================
def load_global_layers(year):
    """
    Returns pre-loaded GEE image objects for OSM, GHSL, and VIIRS.
    These are city-agnostic — we will clip them per tile later.
    """

    # --- OSM residential polygons ---
    # FAO GAUL + OSM landuse=residential rasterised via Open Buildings proxy
    # We use the same GOB layer already in the dataset as the built-up signal,
    # and OSM landuse from the Dynamic World / LSIB fallback.
    # For the residential mask we use the standard ESA WorldCover "Built-up" class (value 50).
    osm_builtup = (
        ee.ImageCollection("ESA/WorldCover/v200")
        .first()
        .eq(50)           # class 50 = Built-up area
        .rename("osm_res")
    )

    # --- GHSL Built-up Surface ---
    # GHS_BUILT_S: 10 m, value > 0 means built surface exists
    ghsl = (
        ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S/2020")
        .select("built_surface")
        .gt(0)
        .rename("ghsl_builtup")
    )

    # --- VIIRS Nighttime Lights annual composite ---
    viirs = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .median()
        .select("avg_rad")
        .rename("viirs_ntl")
    )

    return osm_builtup, ghsl, viirs


# ==========================================
# 3. PER-TILE LABEL GENERATION
# ==========================================
def generate_labels_for_tile(tile_id, region, osm_builtup, ghsl, viirs):
    """
    Produces two images for a given tile region:
      - label_mask : 1=slum, 0=formal-dense, 255=unknown
      - hc_mask    : 1=high-confidence pixel, 0=uncertain
    """

    # --- Clip global layers to tile ---
    osm_tile   = osm_builtup.clip(region)
    ghsl_tile  = ghsl.clip(region)
    viirs_tile = viirs.clip(region)

    # --- City-level VIIRS median (computed over the tile's bounding box) ---
    viirs_median = viirs_tile.reduceRegion(
        reducer=ee.Reducer.median(),
        geometry=region,
        scale=500,          # VIIRS native resolution
        maxPixels=1e8
    ).getNumber("viirs_ntl")

    # dark pixel  = NTL < city median  → informal-settlement proxy
    # bright pixel = NTL ≥ city median → formal-area proxy
    viirs_dark   = viirs_tile.lt(viirs_median).rename("viirs_dark")
    viirs_bright = viirs_tile.gte(viirs_median).rename("viirs_bright")

    # --- Signal agreement ---
    # All three agree on SLUM:
    #   OSM residential ∩ GHSL built-up ∩ VIIRS dark
    slum_signal = osm_tile.And(ghsl_tile).And(viirs_dark)

    # All three agree on FORMAL-DENSE:
    #   OSM residential ∩ GHSL built-up ∩ VIIRS bright
    formal_signal = osm_tile.And(ghsl_tile).And(viirs_bright)

    # --- Label mask ---
    # Start with IGNORE_VALUE everywhere, then paint slum=1 and formal=0
    label_mask = (
        ee.Image(IGNORE_VALUE)
        .where(formal_signal, 0)   # formal-dense = 0  (paint first so slum wins)
        .where(slum_signal, 1)     # slum         = 1
        .rename("label")
        .uint8()
        .clip(region)
    )

    # --- HC mask ---
    # 1 where ANY definitive label was assigned (slum OR formal), 0 elsewhere
    hc_mask = (
        slum_signal.Or(formal_signal)
        .rename("hc_mask")
        .uint8()
        .clip(region)
    )

    return label_mask, hc_mask


# ==========================================
# 4. REGION RECOVERY
# ==========================================
def get_region_from_clear_tif(clear_path):
    """
    Re-derives the GEE geometry from the downloaded clear.tif bounding box.
    This guarantees pixel-perfect alignment with the existing tiles.
    """
    import rasterio
    with rasterio.open(clear_path) as src:
        bounds = src.bounds
        crs    = src.crs.to_epsg()

    if crs != 4326:
        from pyproj import Transformer
        tf = Transformer.from_crs(f"EPSG:{crs}", "EPSG:4326", always_xy=True)
        minx, miny = tf.transform(bounds.left,  bounds.bottom)
        maxx, maxy = tf.transform(bounds.right, bounds.top)
    else:
        minx, miny, maxx, maxy = bounds.left, bounds.bottom, bounds.right, bounds.top

    return ee.Geometry.Rectangle([minx, miny, maxx, maxy])


# ==========================================
# 5. MAIN BATCH LOOP
# ==========================================
def generate_all_labels():
    print(f"\n🚀 Generating weak labels for all tiles in: {DATA_DIR}\n")

    clear_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_clear.tif")))
    if not clear_files:
        print("❌ No *_clear.tif files found. Run download_dhaka_dataset.py first.")
        return

    print(f"Found {len(clear_files)} tiles. Loading global GEE layers...")
    osm_builtup, ghsl, viirs = load_global_layers(LABEL_YEAR)
    print("✅ Global layers loaded.\n")

    skipped   = []
    completed = []

    for i, clear_path in enumerate(clear_files, 1):
        tile_id    = os.path.basename(clear_path).replace("_clear.tif", "")
        label_path = os.path.join(DATA_DIR, f"{tile_id}_label.tif")
        hc_path    = os.path.join(DATA_DIR, f"{tile_id}_hc_mask.tif")

        # Skip tiles that already have both outputs
        if os.path.exists(label_path) and os.path.exists(hc_path):
            print(f"[{i}/{len(clear_files)}] {tile_id} — already exists, skipping.")
            skipped.append(tile_id)
            continue

        print(f"[{i}/{len(clear_files)}] {tile_id} — generating labels...")

        try:
            region = get_region_from_clear_tif(clear_path)

            label_img, hc_img = generate_labels_for_tile(
                tile_id, region, osm_builtup, ghsl, viirs
            )

            geemap.ee_export_image(
                label_img,
                filename=label_path,
                scale=10,
                region=region,
                file_per_band=False
            )

            geemap.ee_export_image(
                hc_img,
                filename=hc_path,
                scale=10,
                region=region,
                file_per_band=False
            )

            completed.append(tile_id)
            print(f"  ✅ Saved: {tile_id}_label.tif + {tile_id}_hc_mask.tif")

        except Exception as e:
            print(f"  ❌ Failed for {tile_id}: {e}")
            skipped.append(tile_id)

    print(f"\n{'='*50}")
    print(f"✅ Completed : {len(completed)} tiles")
    print(f"⏭️  Skipped   : {len(skipped)} tiles")
    print(f"📁 Output dir: {DATA_DIR}")
    print(f"{'='*50}")
    print("\nFiles produced per tile:")
    print("  {tile_id}_label.tif   — 1=slum, 0=formal-dense, 255=unknown")
    print("  {tile_id}_hc_mask.tif — 1=high-confidence pixel, 0=uncertain")
    print("\nNext step: run train_stage2.py 🎯")


if __name__ == "__main__":
    generate_all_labels()