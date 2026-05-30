"""
Weak-label fusion and confidence stratification.
Ingests the GEE-exported label rasters and the LocateAnything validation JSON,
applies the 4-signal HC promotion rule, and writes per-tile label PNGs + confidence.json.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import rasterio
    from PIL import Image
except ImportError:
    raise ImportError("rasterio and Pillow required")

LABEL_UNKNOWN = 0
LABEL_SLUM = 1
LABEL_FORMAL = 2

# 4-signal HC: geospatial 3-way agreement AND LocateAnything sign-agrees
HC_MIN_GEO_SIGNALS = 3  # all 3 geospatial signals must agree
LA_SIGN_THRESHOLD = 0.0  # la_slum_score > 0 for slum, < 0 for formal


def load_geo_labels(label_tif: str):
    """Load GEE-exported label raster. Returns (noisy_label, hc_geo, agreement_score)."""
    with rasterio.open(label_tif) as src:
        noisy_label = src.read(1).astype(np.uint8)
        hc_geo = src.read(2).astype(np.uint8)
        agreement_score = src.read(3).astype(np.uint8)
        transform = src.transform
        crs = src.crs
    return noisy_label, hc_geo, agreement_score, transform, crs


def apply_la_validation(
    noisy_label: np.ndarray,
    hc_geo: np.ndarray,
    tile_id: str,
    la_validation: Optional[Dict],
    use_la: bool,
) -> np.ndarray:
    """
    Promote geospatial HC pixels to 4-signal HC if LocateAnything agrees.
    Returns binary HC mask: 1 = high-confidence, 0 = not.
    """
    if not use_la or la_validation is None or tile_id not in la_validation:
        # Fall back to 3-signal geo HC
        return (hc_geo > 0).astype(np.uint8)

    la_scores = la_validation[tile_id]
    la_slum_score = np.array(la_scores.get("la_slum_score_map", 0), dtype=np.float32)
    # la_slum_score may be a scalar (tile-level) or 256×256 array
    if np.isscalar(la_slum_score) or la_slum_score.ndim == 0:
        la_slum_score = np.full_like(noisy_label, float(la_slum_score), dtype=np.float32)

    geo_hc = (hc_geo > 0)
    slum_pixels = noisy_label == LABEL_SLUM
    formal_pixels = noisy_label == LABEL_FORMAL

    la_agrees_slum = la_slum_score > LA_SIGN_THRESHOLD
    la_agrees_formal = la_slum_score < -LA_SIGN_THRESHOLD

    hc_4signal = np.zeros_like(noisy_label, dtype=np.uint8)
    hc_4signal[geo_hc & slum_pixels & la_agrees_slum] = 1
    hc_4signal[geo_hc & formal_pixels & la_agrees_formal] = 1

    return hc_4signal


def process_tile(
    tile_id: str,
    label_tif: str,
    output_labels_dir: str,
    la_validation: Optional[Dict],
    use_la: bool,
) -> Dict:
    """
    Process one tile: load geo labels, apply LA validation, write PNG outputs.
    Returns per-tile signal stats for confidence.json.
    """
    output_dir = Path(output_labels_dir)
    noisy_label, hc_geo, agreement_score, transform, crs = load_geo_labels(label_tif)

    hc_mask = apply_la_validation(noisy_label, hc_geo, tile_id, la_validation, use_la)

    # Write noisy label PNG
    noisy_path = output_dir / f"{tile_id}_noisy.png"
    Image.fromarray(noisy_label).save(str(noisy_path))

    # Write HC mask PNG
    hc_path = output_dir / f"{tile_id}_hc.png"
    Image.fromarray(hc_mask * 255).save(str(hc_path))

    # Also write aligned GeoTIFF versions for rasterio loading
    profile = {
        "driver": "GTiff", "dtype": "uint8", "count": 1,
        "height": noisy_label.shape[0], "width": noisy_label.shape[1],
        "crs": crs, "transform": transform
    }
    with rasterio.open(str(output_dir / f"{tile_id}_noisy.tif"), "w", **profile) as dst:
        dst.write(noisy_label[np.newaxis])
    with rasterio.open(str(output_dir / f"{tile_id}_hc.tif"), "w", **profile) as dst:
        dst.write(hc_mask[np.newaxis])

    n_pixels = noisy_label.size
    stats = {
        "tile_id": tile_id,
        "n_slum": int((noisy_label == LABEL_SLUM).sum()),
        "n_formal": int((noisy_label == LABEL_FORMAL).sum()),
        "n_unknown": int((noisy_label == LABEL_UNKNOWN).sum()),
        "n_hc": int(hc_mask.sum()),
        "agreement_0": int((agreement_score == 0).sum()),
        "agreement_1": int((agreement_score == 1).sum()),
        "agreement_2": int((agreement_score == 2).sum()),
        "agreement_3": int((agreement_score == 3).sum()),
        "la_applied": use_la and la_validation is not None and tile_id in (la_validation or {}),
    }
    return stats


def build_confidence_json(
    label_tifs: List[str],
    output_labels_dir: str,
    la_validation_path: Optional[str],
    use_la: bool,
    output_confidence_path: str,
):
    """Process all tiles and write confidence.json."""
    la_validation = None
    if use_la and la_validation_path and Path(la_validation_path).exists():
        with open(la_validation_path) as f:
            la_validation = json.load(f)

    all_stats = []
    for tif_path in sorted(label_tifs):
        tile_id = Path(tif_path).stem.replace("weeklabels_", "")
        stats = process_tile(tile_id, tif_path, output_labels_dir, la_validation, use_la)
        all_stats.append(stats)

    # Per-region summary
    region_stats = {}
    for s in all_stats:
        region = s["tile_id"].split("_")[0]
        if region not in region_stats:
            region_stats[region] = {"n_hc": 0, "n_slum_hc": 0, "n_formal_hc": 0, "n_tiles": 0}
        region_stats[region]["n_tiles"] += 1
        region_stats[region]["n_hc"] += s["n_hc"]

    confidence = {"tiles": all_stats, "by_region": region_stats}
    Path(output_confidence_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_confidence_path, "w") as f:
        json.dump(confidence, f, indent=2)

    print(f"confidence.json written with {len(all_stats)} tiles")
    return confidence


# ── Smoke test ─────────────────────────────────────────────────────────────────
def _smoke_test():
    import tempfile
    from rasterio.transform import from_bounds

    with tempfile.TemporaryDirectory() as tmp:
        # Synthetic label raster: 3 bands (noisy, hc_geo, agreement)
        transform = from_bounds(0, 0, 1, 1, 256, 256)
        label_tif = os.path.join(tmp, "weeklabels_test.tif")
        data = np.zeros((3, 256, 256), dtype=np.uint8)
        data[0, :128, :] = LABEL_SLUM
        data[0, 128:, :] = LABEL_FORMAL
        data[1, :128, :] = 1  # hc_geo slum
        data[1, 128:, :] = 1  # hc_geo formal
        data[2] = 3            # all 3 signals agree

        with rasterio.open(label_tif, "w", driver="GTiff", dtype="uint8",
                           count=3, width=256, height=256,
                           crs="EPSG:4326", transform=transform) as dst:
            dst.write(data)

        stats = process_tile("test", label_tif, tmp, None, use_la=False)
        assert stats["n_slum"] > 0 and stats["n_formal"] > 0
        assert stats["n_hc"] > 0
        print("weak_labels.py smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
