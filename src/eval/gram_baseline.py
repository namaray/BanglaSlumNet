"""
Wrap the existing GRAM run for head-to-head comparison.
Loads GRAM's per-tile probability outputs and computes the same metrics
on the same HC eval tiles as BanglaSlumNet.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .metrics import compute_metrics


def load_gram_predictions(gram_output_dir: str, tile_ids) -> torch.Tensor:
    """
    Load GRAM's per-tile probability maps from the existing gram_baseline/ outputs.
    Expected format: one .npy or .png per tile at gram_output_dir/<tile_id>_prob.npy.
    Returns [N, H, W] float tensor.
    """
    preds = []
    for tid in tile_ids:
        npy_path = Path(gram_output_dir) / f"{tid}_prob.npy"
        if npy_path.exists():
            arr = np.load(str(npy_path))
        else:
            # Try PNG (0–255 prob map)
            png_path = Path(gram_output_dir) / f"{tid}_prob.png"
            if png_path.exists():
                from PIL import Image
                arr = np.array(Image.open(str(png_path))).astype(np.float32) / 255.0
            else:
                # GRAM didn't produce output for this tile — fill with neutral 0.5
                arr = np.full((256, 256), 0.5, dtype=np.float32)
        # Resize to 256×256 if needed
        if arr.shape != (256, 256):
            import torch.nn.functional as _F
            t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
            arr = _F.interpolate(t, (256, 256), mode="bilinear", align_corners=False).squeeze().numpy()
        preds.append(arr)
    return torch.from_numpy(np.stack(preds, axis=0)).float()


def evaluate_gram(
    gram_output_dir: str,
    tile_ids,
    labels: torch.Tensor,
    hc_mask: Optional[torch.Tensor],
    results_dir: str,
    run_id: str = "gram_zeroshot",
) -> Dict:
    """
    Evaluate GRAM predictions on HC eval tiles and write results JSON.
    """
    preds = load_gram_predictions(gram_output_dir, tile_ids)
    metrics = compute_metrics(preds, labels, hc_mask=hc_mask)

    out = {
        "run_id": run_id,
        "experiment": "gram_head_to_head",
        "metrics": metrics,
        "gram_output_dir": gram_output_dir,
    }
    out_path = Path(results_dir) / f"{run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump(out, f, indent=2)

    print(f"GRAM eval: HC-IoU={metrics.get('hc_iou', 'nan'):.4f} → {out_path}")
    return metrics
