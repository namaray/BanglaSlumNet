"""
Direction B: LocateAnything zero-shot validation of weak labels.
Runs two grounding prompts per tile (slum + formal) on ESRI z16 tiles,
computes la_slum_score per pixel, and promotes to 4-signal HC where
geospatial signals + VLM sign agree.

Results cached to data/labels/la_validation.json — run once, never during training.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .worker import LocateAnythingWorker
from .prompts import LV_SLUM, LV_FORMAL, PROMPT_VERSION


def validate_tile(
    worker: LocateAnythingWorker,
    tile_id: str,
    image: Image.Image,
    generation_mode: str = "slow",
    max_new_tokens: int = 512,
) -> Dict:
    """
    Run slum + formal grounding prompts on one tile.
    Returns per-tile validation result with la_slum_score (scalar, tile-level).
    """
    start = time.time()

    # Run slum prompt
    text_slum = worker.generate(image, LV_SLUM, generation_mode=generation_mode,
                                 max_new_tokens=max_new_tokens)
    boxes_slum = worker.parse_boxes(text_slum, image.size)
    slum_mask = worker.rasterize_boxes(boxes_slum, image.size)
    slum_coverage = float(slum_mask.mean())

    # Run formal prompt
    text_formal = worker.generate(image, LV_FORMAL, generation_mode=generation_mode,
                                   max_new_tokens=max_new_tokens)
    boxes_formal = worker.parse_boxes(text_formal, image.size)
    formal_mask = worker.rasterize_boxes(boxes_formal, image.size)
    formal_coverage = float(formal_mask.mean())

    # la_slum_score ∈ [-1, 1]: positive → VLM says slum; negative → VLM says formal
    la_slum_score = slum_coverage - formal_coverage

    elapsed = time.time() - start
    return {
        "tile_id": tile_id,
        "prompt_version": PROMPT_VERSION,
        "slum_coverage": slum_coverage,
        "formal_coverage": formal_coverage,
        "la_slum_score": la_slum_score,
        "n_slum_boxes": len(boxes_slum),
        "n_formal_boxes": len(boxes_formal),
        "elapsed_s": round(elapsed, 2),
    }


def run_label_validation(
    worker: LocateAnythingWorker,
    tile_ids: List[str],
    image_loader,
    output_path: str,
    generation_mode: str = "slow",
    max_new_tokens: int = 512,
    resume: bool = True,
) -> Dict:
    """
    Validate all tiles and write la_validation.json.
    resume=True: skip tiles already in the JSON (safe to restart after disconnect).
    image_loader: callable(tile_id) → PIL.Image (ESRI z16 tile, high-res preferred)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if resuming
    results = {}
    if resume and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        results = {r["tile_id"]: r for r in existing.get("tiles", [])}
        print(f"Resuming: {len(results)}/{len(tile_ids)} tiles already validated.")

    worker.load()

    for i, tile_id in enumerate(tile_ids):
        if tile_id in results:
            continue

        try:
            image = image_loader(tile_id)
            result = validate_tile(
                worker, tile_id, image,
                generation_mode=generation_mode,
                max_new_tokens=max_new_tokens,
            )
            results[tile_id] = result
        except Exception as e:
            print(f"  Warning: validation failed for {tile_id}: {e}")
            results[tile_id] = {"tile_id": tile_id, "error": str(e), "la_slum_score": 0.0}

        # Save incrementally so a disconnect doesn't lose work
        if (i + 1) % 10 == 0 or (i + 1) == len(tile_ids):
            _save_validation(results, output_path)
            print(f"  [{i+1}/{len(tile_ids)}] saved checkpoint")

    _save_validation(results, output_path)
    print(f"Label validation complete → {output_path}")
    return results


def _save_validation(results: Dict, output_path: Path):
    data = {"prompt_version": PROMPT_VERSION, "tiles": list(results.values())}
    with open(str(output_path), "w") as f:
        json.dump(data, f, indent=2)


def load_validation(path: str) -> Dict[str, Dict]:
    """Load la_validation.json as {tile_id: result_dict}."""
    with open(path) as f:
        data = json.load(f)
    return {r["tile_id"]: r for r in data.get("tiles", [])}
