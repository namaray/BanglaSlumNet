"""
Frozen MoonViT feature extraction with caching (Direction A core).

Extracts [D, H_f, W_f] feature maps from LocateAnything's vision encoder,
caches to .npy files, and automatically falls back to grounding_map mode
if the hidden_state hook fails (per §7.2).

Cache key: (tile_id, prompt_id) → data/features_cache/<tile_id>_<prompt_id>.npy
"""

import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .worker import LocateAnythingWorker
from .prompts import get_prompt, PROMPT_VERSION, SLUM_DISCRIMINATIVE, FORMAL_DISCRIMINATIVE, NEUTRAL


FEATURE_GRID_HW = (16, 16)  # MoonViT-SO-400M default patch grid; TODO_VERIFY at first load


class FeatureExtractor:
    """
    Wraps LocateAnythingWorker to extract and cache visual features for all tiles.
    feature_mode: 'hidden_state' (primary) or 'grounding_map' (fallback / explicit).
    """

    def __init__(
        self,
        worker: LocateAnythingWorker,
        cache_dir: str,
        feature_mode: str = "hidden_state",
        generation_mode: str = "slow",
        max_new_tokens: int = 512,
        feature_grid_hw: Tuple[int, int] = FEATURE_GRID_HW,
    ):
        self.worker = worker
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.feature_mode = feature_mode
        self.generation_mode = generation_mode
        self.max_new_tokens = max_new_tokens
        self.feature_grid_hw = feature_grid_hw
        self._mode_confirmed = False  # resolved on first call

    def _cache_path(self, tile_id: str, prompt_id: str) -> Path:
        return self.cache_dir / f"{tile_id}_{prompt_id}.npy"

    def _is_cached(self, tile_id: str, prompt_id: str) -> bool:
        return self._cache_path(tile_id, prompt_id).exists()

    def _save_cache(self, tile_id: str, prompt_id: str, feats: np.ndarray):
        np.save(str(self._cache_path(tile_id, prompt_id)), feats)

    def _load_cache(self, tile_id: str, prompt_id: str) -> np.ndarray:
        return np.load(str(self._cache_path(tile_id, prompt_id)))

    def extract(
        self,
        tile_id: str,
        image: Image.Image,
        model_config: str,
    ) -> Dict[str, np.ndarray]:
        """
        Extract features for one tile under the given model config.
        Returns dict mapping prompt_id → [D, H_f, W_f] float32 numpy array.
        Uses cache if available — never re-runs the VLM on a cached tile.
        """
        prompts = self._prompts_for_config(model_config)
        results = {}

        for prompt_id, prompt_text in prompts.items():
            if self._is_cached(tile_id, prompt_id):
                results[prompt_id] = self._load_cache(tile_id, prompt_id)
                continue

            feats = self._extract_one(image, prompt_text)
            arr = feats.numpy() if isinstance(feats, torch.Tensor) else feats
            self._save_cache(tile_id, prompt_id, arr)
            results[prompt_id] = arr

        return results

    def _extract_one(self, image: Image.Image, prompt: str) -> torch.Tensor:
        """Extract features for one image+prompt, with automatic fallback."""
        mode = self.feature_mode

        if mode == "hidden_state" and not self._mode_confirmed:
            feats = self.worker.extract_visual_features(
                image, prompt, feature_grid_hw=self.feature_grid_hw
            )
            if feats is None:
                warnings.warn(
                    "[FeatureExtractor] hidden_state mode failed. "
                    "Switching to grounding_map fallback for all remaining tiles."
                )
                self.feature_mode = "grounding_map"
                mode = "grounding_map"
            else:
                self._mode_confirmed = True
                return feats
        elif mode == "hidden_state":
            feats = self.worker.extract_visual_features(
                image, prompt, feature_grid_hw=self.feature_grid_hw
            )
            if feats is not None:
                return feats
            warnings.warn("[FeatureExtractor] hidden_state hook returned None; using grounding_map.")
            mode = "grounding_map"

        # grounding_map mode
        maps = self.worker.extract_grounding_map_features(
            image, [prompt],
            generation_mode=self.generation_mode,
            max_new_tokens=self.max_new_tokens,
            out_size=(self.feature_grid_hw[0] * 16, self.feature_grid_hw[1] * 16),
        )
        return maps[0:1]  # [1, H, W] — single-channel grounding map

    def _prompts_for_config(self, model_config: str) -> Dict[str, str]:
        """Return {prompt_id: prompt_text} for the given config."""
        if model_config == "vlm_visual":
            return {f"neutral_{PROMPT_VERSION}": NEUTRAL}
        elif model_config in ("vlm_lang", "full"):
            return {
                f"slum_{PROMPT_VERSION}": SLUM_DISCRIMINATIVE,
                f"formal_{PROMPT_VERSION}": FORMAL_DISCRIMINATIVE,
            }
        else:
            return {f"neutral_{PROMPT_VERSION}": NEUTRAL}

    def extract_all_tiles(
        self,
        tile_ids: List[str],
        image_loader,
        model_config: str,
        verbose: bool = True,
    ):
        """
        Extract and cache features for all tiles.
        image_loader: callable(tile_id) → PIL.Image
        This is the one-time VLM feature caching pass; subsequent epochs read from .npy.
        """
        self.worker.load()

        # Assert vision encoder is frozen
        enc = self.worker._resolve_vision_encoder()
        if enc is not None:
            frozen = all(not p.requires_grad for p in enc.parameters())
            assert frozen, (
                "Vision encoder parameters are not frozen! "
                "Set requires_grad=False on the entire vision encoder before extraction."
            )
            n_params = sum(p.numel() for p in enc.parameters())
            print(f"[FeatureExtractor] Vision encoder: {n_params:,} params (all frozen)")

        try:
            from tqdm.auto import tqdm
        except ImportError:
            def tqdm(x, **k):
                return x

        skipped = 0
        iterator = tqdm(tile_ids, desc="MoonViT features", unit="tile") if verbose else tile_ids
        for tile_id in iterator:
            prompts = self._prompts_for_config(model_config)
            all_cached = all(self._is_cached(tile_id, pid) for pid in prompts)
            if all_cached:
                skipped += 1
                continue

            image = image_loader(tile_id)
            self.extract(tile_id, image, model_config)

        print(f"Feature extraction complete. {skipped} tiles skipped (cached).")

    def load_cached_features(
        self,
        tile_id: str,
        model_config: str,
    ) -> Dict[str, torch.Tensor]:
        """Load cached features as tensors. Raises if not cached."""
        prompts = self._prompts_for_config(model_config)
        out = {}
        for prompt_id in prompts:
            path = self._cache_path(tile_id, prompt_id)
            assert path.exists(), (
                f"Features not cached for {tile_id}/{prompt_id}. "
                "Run extract_all_tiles() first."
            )
            out[prompt_id] = torch.from_numpy(np.load(str(path))).float()
        return out
