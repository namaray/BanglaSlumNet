"""
Load, resample, and normalize the socioeconomic E-tensor.
Produces per-tile [C_eco, 256, 256] float32 arrays aligned to the RGB grid.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.warp import reproject
except ImportError:
    raise ImportError("rasterio required")


SOCIOECONOMIC_CHANNELS = ["viirs", "worldpop", "ghspop", "osm_roads", "wb_poverty", "ghsl_builtup"]


def load_and_resample_socioeconomic(
    socioec_tif: str,
    target_transform: Affine,
    target_shape: Tuple[int, int],
    channels: List[str],
    target_crs: str = "EPSG:4326",
) -> np.ndarray:
    """
    Load the socioeconomic GeoTIFF and resample each requested channel to
    the target 256×256 grid. Returns float32 [C, 256, 256].
    """
    with rasterio.open(socioec_tif) as src:
        # Band descriptions tell us which band is which channel
        desc = list(src.descriptions) if src.descriptions else []
        band_map = {d.lower(): i + 1 for i, d in enumerate(desc) if d}

        out = np.zeros((len(channels), *target_shape), dtype=np.float32)
        for ci, ch in enumerate(channels):
            band_idx = band_map.get(ch.lower())
            if band_idx is None:
                # Channel not in this file — leave as zeros
                continue

            src_data = src.read(band_idx).astype(np.float32)
            dst_data = np.zeros(target_shape, dtype=np.float32)
            reproject(
                source=src_data,
                destination=dst_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
            )
            out[ci] = dst_data

    return out


def normalize_socioeconomic(tensor: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> np.ndarray:
    """Robust per-channel percentile normalization to [0, 1]."""
    out = tensor.copy()
    for i in range(tensor.shape[0]):
        lo = np.percentile(tensor[i], p_lo)
        hi = np.percentile(tensor[i], p_hi)
        if hi > lo:
            out[i] = np.clip((tensor[i] - lo) / (hi - lo), 0, 1)
        else:
            out[i] = 0.0
    return out


def build_socioeconomic_tile(
    tile_id: str,
    socioec_dir: str,
    target_transform: Affine,
    target_shape: Tuple[int, int],
    channels: List[str],
    output_dir: Optional[str] = None,
) -> np.ndarray:
    """
    For a given tile, load the regional socioeconomic raster, crop/resample to
    the tile's 256×256 grid, normalize, and optionally cache as .npy.
    """
    # Infer region from tile_id (e.g., "korail_0012" → "korail")
    region = "_".join(tile_id.split("_")[:-1]) if "_" in tile_id else tile_id
    socioec_tif = Path(socioec_dir) / f"socioeconomic_{region}.tif"

    if not socioec_tif.exists():
        raise FileNotFoundError(
            f"Socioeconomic raster not found: {socioec_tif}. "
            "Run gee/04_export_socioeconomic.js first."
        )

    tensor = load_and_resample_socioeconomic(
        str(socioec_tif), target_transform, target_shape, channels
    )
    tensor = normalize_socioeconomic(tensor)

    assert tensor.shape == (len(channels), 256, 256), (
        f"Socioeconomic tensor shape mismatch for {tile_id}: {tensor.shape}"
    )

    if output_dir:
        cache_path = Path(output_dir) / f"{tile_id}_socioec.npy"
        np.save(str(cache_path), tensor)

    return tensor


# ── Smoke test ─────────────────────────────────────────────────────────────────
def _smoke_test():
    import tempfile
    from rasterio.transform import from_bounds

    channels = ["viirs", "worldpop"]
    target_transform = from_bounds(0, 0, 1, 1, 256, 256)
    target_shape = (256, 256)

    with tempfile.TemporaryDirectory() as tmp:
        # Write a synthetic socioeconomic raster
        tif_path = os.path.join(tmp, "socioeconomic_test.tif")
        data = np.random.rand(2, 512, 512).astype(np.float32)
        src_transform = from_bounds(0, 0, 1, 1, 512, 512)
        with rasterio.open(tif_path, "w", driver="GTiff", dtype="float32",
                           count=2, width=512, height=512,
                           crs="EPSG:4326", transform=src_transform) as dst:
            dst.write(data)
            # Set descriptions
            dst.update_tags(ns="rio_description", band_1="viirs", band_2="worldpop")

        tensor = load_and_resample_socioeconomic(
            tif_path, target_transform, target_shape, channels
        )
        normalized = normalize_socioeconomic(tensor)
        assert normalized.shape == (2, 256, 256)
        assert normalized.min() >= 0.0 and normalized.max() <= 1.0 + 1e-6
        print("socioeconomic.py smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
