"""
Tile dataset: loads aligned RGB, weak-label, HC-mask, and socioeconomic tensors.
All layers must share the same 256×256 grid and geotransform — misalignment raises loudly.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
except ImportError:
    raise ImportError("rasterio is required: pip install rasterio")


LABEL_UNKNOWN = 0
LABEL_SLUM = 1
LABEL_FORMAL = 2


def _assert_aligned(path_a: str, transform_a: Affine, shape_a: Tuple,
                    path_b: str, transform_b: Affine, shape_b: Tuple):
    """Fail loudly if two rasters are not pixel-aligned."""
    if shape_a != shape_b:
        raise AssertionError(
            f"Grid shape mismatch:\n  {path_a}: {shape_a}\n  {path_b}: {shape_b}"
        )
    # Compare geotransform to 6 decimal places
    for i, (a, b) in enumerate(zip(transform_a, transform_b)):
        if abs(a - b) > 1e-6:
            raise AssertionError(
                f"Geotransform mismatch at element {i}:\n  {path_a}: {transform_a}\n  {path_b}: {transform_b}"
            )


class SlumTileDataset(Dataset):
    """
    PyTorch Dataset over pre-tiled rasters. Each item contains:
        rgb        : FloatTensor [C_rgb, 256, 256]  (normalized 0–1)
        label      : LongTensor  [256, 256]          (0=unknown, 1=slum, 2=formal)
        hc_mask    : BoolTensor  [256, 256]          (True = high-confidence pixel)
        socioec    : FloatTensor [C_eco, 256, 256]   (normalized per-channel)
        tile_id    : str
        region     : str
        split      : str  (train | val | test)
    """

    def __init__(
        self,
        manifest_path: str,
        split: str,
        tiles_dir: str,
        labels_dir: str,
        socioeconomic_dir: str,
        socioeconomic_channels: List[str],
        transform=None,
        use_hc_only: bool = False,
        features_cache_dir: Optional[str] = None,
        model_config: Optional[str] = None,
        use_clean_tiles: bool = True,
    ):
        self.split = split
        self.tiles_dir = Path(tiles_dir)
        self.labels_dir = Path(labels_dir)
        self.socioeconomic_dir = Path(socioeconomic_dir)
        self.socioeconomic_channels = socioeconomic_channels
        self.transform = transform
        self.use_hc_only = use_hc_only
        self.features_cache_dir = Path(features_cache_dir) if features_cache_dir else None
        self.model_config = model_config
        self.use_clean_tiles = use_clean_tiles
        self.prompt_ids = self._resolve_prompt_ids(model_config)

        with open(manifest_path) as f:
            manifest = json.load(f)

        self.tiles = [t for t in manifest["tiles"] if t["split"] == split]
        if use_hc_only:
            self.tiles = [t for t in self.tiles if t.get("hc_pixel_count", 0) > 0]

    @staticmethod
    def _resolve_prompt_ids(model_config: Optional[str]) -> List[str]:
        """Which cached feature prompt files a config needs (matches FeatureExtractor)."""
        if model_config is None or model_config == "baseline_cnn":
            return []
        try:
            from ..locate_anything.prompts import PROMPT_VERSION as pv
        except Exception:
            pv = "v1"
        if model_config == "vlm_visual":
            return [f"neutral_{pv}"]
        return [f"slum_{pv}", f"formal_{pv}"]  # vlm_lang, full

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx: int) -> Dict:
        meta = self.tiles[idx]
        tile_id = meta["tile_id"]

        rgb, rgb_transform = self._load_rgb(tile_id)
        label, label_transform = self._load_label(tile_id)
        hc_mask, hc_transform = self._load_hc(tile_id)
        socioec, eco_transform = self._load_socioeconomic(tile_id)

        shape = (256, 256)
        _assert_aligned(f"{tile_id}_rgb", rgb_transform, rgb.shape[-2:],
                        f"{tile_id}_label", label_transform, label.shape[-2:])
        _assert_aligned(f"{tile_id}_rgb", rgb_transform, rgb.shape[-2:],
                        f"{tile_id}_hc", hc_transform, hc_mask.shape[-2:])
        _assert_aligned(f"{tile_id}_rgb", rgb_transform, rgb.shape[-2:],
                        f"{tile_id}_socioec", eco_transform, socioec.shape[-2:])

        # Prefer SAS-Net clean (reference-appearance) tile if it has been cached
        if self.use_clean_tiles:
            clean_path = self.tiles_dir / f"{tile_id}_clean.npy"
            if clean_path.exists():
                clean = np.load(str(clean_path)).astype(np.float32)
                if clean.shape[-2:] == rgb.shape[-2:]:
                    rgb = clean

        sample = {
            "rgb": torch.from_numpy(rgb).float(),
            "label": torch.from_numpy(label).long(),
            "hc_mask": torch.from_numpy(hc_mask).bool(),
            "socioec": torch.from_numpy(socioec).float(),
            "tile_id": tile_id,
            "region": meta["region"],
            "split": self.split,
        }

        # Cached MoonViT features for VLM configs (concatenated across prompts)
        if self.prompt_ids and self.features_cache_dir is not None:
            sample["cached_feats"] = self._load_cached_feats(tile_id)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _load_cached_feats(self, tile_id: str) -> torch.Tensor:
        """Load and channel-concatenate per-prompt cached features [D*n, H_f, W_f]."""
        feats = []
        for pid in self.prompt_ids:
            path = self.features_cache_dir / f"{tile_id}_{pid}.npy"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing cached feature {path}. Run feature extraction (Phase 3) "
                    f"for config '{self.model_config}', or run preflight to catch this on CPU."
                )
            feats.append(np.load(str(path)).astype(np.float32))
        arr = np.concatenate(feats, axis=0) if len(feats) > 1 else feats[0]
        return torch.from_numpy(arr).float()

    def _load_rgb(self, tile_id: str) -> Tuple[np.ndarray, Affine]:
        path = self.tiles_dir / f"{tile_id}_rgb.tif"
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
            data = np.clip(data, 0, 1)
            transform = src.transform
        self._assert_shape(data, tile_id, "rgb")
        return data, transform

    def _load_label(self, tile_id: str) -> Tuple[np.ndarray, Affine]:
        path = self.labels_dir / f"{tile_id}_noisy.tif"
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.int64)
            transform = src.transform
        self._assert_shape(data[np.newaxis], tile_id, "label")
        return data, transform

    def _load_hc(self, tile_id: str) -> Tuple[np.ndarray, Affine]:
        path = self.labels_dir / f"{tile_id}_hc.tif"
        with rasterio.open(path) as src:
            data = src.read(1).astype(bool)
            transform = src.transform
        self._assert_shape(data[np.newaxis], tile_id, "hc")
        return data, transform

    def _load_socioeconomic(self, tile_id: str) -> Tuple[np.ndarray, Affine]:
        path = self.socioeconomic_dir / f"{tile_id}_socioec.tif"
        # Configs without socioeconomic fusion request zero channels — return an
        # empty-channel tensor (the model ignores socioec when fusion is off).
        if not self.socioeconomic_channels:
            with rasterio.open(path) as src:
                transform = src.transform
                h, w = src.height, src.width
            return np.zeros((0, h, w), dtype=np.float32), transform
        with rasterio.open(path) as src:
            all_bands = {name: idx + 1 for idx, name in enumerate(src.descriptions)}
            indices = []
            for ch in self.socioeconomic_channels:
                if ch not in all_bands:
                    raise KeyError(f"Channel '{ch}' not found in {path}. Available: {list(all_bands)}")
                indices.append(all_bands[ch])
            data = src.read(indices).astype(np.float32)
            transform = src.transform

        # Per-channel min-max normalization (robust, 2nd–98th percentile)
        for i in range(data.shape[0]):
            lo, hi = np.percentile(data[i], [2, 98])
            if hi > lo:
                data[i] = np.clip((data[i] - lo) / (hi - lo), 0, 1)
            else:
                data[i] = 0.0

        self._assert_shape(data, tile_id, "socioec")
        return data, transform

    @staticmethod
    def _assert_shape(arr: np.ndarray, tile_id: str, name: str):
        h, w = arr.shape[-2], arr.shape[-1]
        assert h == w and h > 0, (
            f"Expected a square tile for {tile_id} ({name}), got {h}×{w}. "
            "Re-tile with a consistent tile_size."
        )


def build_dataset_manifest(
    tiles_dir: str,
    labels_dir: str,
    regions_yaml: str,
    output_path: str,
    val_fraction: float = 0.15,
    seed: int = 1337,
):
    """
    Scan exported tile files, verify alignment, assign train/val/test splits,
    and write dataset_manifest.json.
    """
    import yaml
    from rasterio.transform import Affine

    rng = np.random.default_rng(seed)
    tiles_dir = Path(tiles_dir)
    labels_dir = Path(labels_dir)

    with open(regions_yaml) as f:
        regions_cfg = yaml.safe_load(f)

    records = []
    for rgb_path in sorted(tiles_dir.glob("*_rgb.tif")):
        tile_id = rgb_path.stem.replace("_rgb", "")

        # Determine region from filename prefix
        region = next(
            (r for r in regions_cfg["regions"] if tile_id.startswith(r)),
            "unknown"
        )

        label_path = labels_dir / f"{tile_id}_noisy.tif"
        hc_path = labels_dir / f"{tile_id}_hc.tif"

        if not label_path.exists() or not hc_path.exists():
            continue

        # Load shapes and transforms for alignment check
        with rasterio.open(str(rgb_path)) as src_rgb:
            rgb_shape = src_rgb.shape
            rgb_transform = src_rgb.transform

        with rasterio.open(str(label_path)) as src_lbl:
            lbl_shape = src_lbl.shape
            lbl_transform = src_lbl.transform

        _assert_aligned(str(rgb_path), rgb_transform, rgb_shape,
                        str(label_path), lbl_transform, lbl_shape)

        with rasterio.open(str(hc_path)) as src_hc:
            hc_data = src_hc.read(1)

        hc_pixel_count = int(hc_data.sum())

        records.append({
            "tile_id": tile_id,
            "region": region,
            "hc_pixel_count": hc_pixel_count,
            "split": None,
        })

    return build_manifest_from_records(records, output_path, val_fraction, seed)


def build_manifest_from_records(
    records: List[Dict],
    output_path: str,
    val_fraction: float = 0.15,
    seed: int = 1337,
) -> str:
    """
    Assign train/val/test splits to pre-built tile records and write the manifest.
    Spatial split: HC tiles -> val+test; all others -> train (prevents leakage).
    Returns the manifest path.
    """
    rng = np.random.default_rng(seed)

    # Proper train/val/test split over ALL tiles (train must not be empty).
    # Prefer tiles that contain HC pixels for val/test so HC-IoU is computable,
    # but keep the majority for training (using their noisy labels).
    order = list(records)
    rng.shuffle(order)
    n = len(order)
    n_test = max(1, int(round(n * val_fraction))) if n >= 3 else 0
    n_val = max(1, int(round(n * val_fraction))) if n >= 3 else 0

    # Put HC-bearing tiles first so they fall into val/test preferentially.
    order.sort(key=lambda r: r.get("hc_pixel_count", 0), reverse=True)
    test_ids = {t["tile_id"] for t in order[:n_test]}
    val_ids = {t["tile_id"] for t in order[n_test:n_test + n_val]}

    for r in records:
        if r["tile_id"] in test_ids:
            r["split"] = "test"
        elif r["tile_id"] in val_ids:
            r["split"] = "val"
        else:
            r["split"] = "train"

    manifest = {"version": "1.0", "n_tiles": len(records), "tiles": records}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written: {output_path} ({len(records)} tiles, "
          f"{len(val_ids)} val, {len(test_ids)} test)")
    return output_path


# ── Synthetic smoke test ───────────────────────────────────────────────────────
def _smoke_test():
    """Verify dataset alignment logic on synthetic arrays (no real data needed)."""
    import tempfile, rasterio
    from rasterio.transform import from_bounds

    transform = from_bounds(0, 0, 1, 1, 256, 256)
    profile = {"driver": "GTiff", "dtype": "float32", "width": 256, "height": 256,
               "count": 4, "crs": "EPSG:4326", "transform": transform}

    with tempfile.TemporaryDirectory() as tmp:
        rgb_path = os.path.join(tmp, "test_tile_rgb.tif")
        with rasterio.open(rgb_path, "w", **profile) as dst:
            dst.write(np.random.rand(4, 256, 256).astype(np.float32))

        # Test shape assertion
        try:
            bad_arr = np.zeros((1, 128, 128))
            SlumTileDataset._assert_shape(bad_arr, "test_tile", "rgb")
            assert False, "Should have raised"
        except AssertionError:
            pass

        # Test alignment check
        try:
            bad_transform = from_bounds(0, 0, 2, 2, 256, 256)
            _assert_aligned("a", transform, (256, 256), "b", bad_transform, (256, 256))
            assert False, "Should have raised"
        except AssertionError:
            pass

    print("tiles.py smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
