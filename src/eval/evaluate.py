"""
Run a model config on a split and dump results JSON.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..models.banglaslumnet import build_model
from .metrics import compute_metrics, ssim, psnr
from ..tracking.recorder import ResultsRecorder


def evaluate(
    config: dict,
    run_id: str,
    checkpoint_path: str,
    data_loader: DataLoader,
    results_dir: str,
    device: str = "cuda",
    split: str = "test",
) -> Dict:
    """
    Load checkpoint, run on data_loader, compute all metrics, write JSON.
    Returns the metrics dict.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_preds, all_labels, all_hc = [], [], []
    all_regions = []

    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Eval {run_id}", unit="batch", leave=False):
            rgb = batch["rgb"].to(device)
            label = batch["label"].to(device)
            hc_mask = batch.get("hc_mask", None)
            socioec = batch.get("socioec", None)
            cached_feats = batch.get("cached_feats", None)
            if socioec is not None:
                socioec = socioec.to(device)
            if cached_feats is not None:
                cached_feats = cached_feats.to(device)

            pred = model(rgb, cached_feats=cached_feats, socioec=socioec)
            all_preds.append(pred.squeeze(1).cpu())
            all_labels.append(label.cpu())
            if hc_mask is not None:
                all_hc.append(hc_mask.cpu())
            all_regions.extend(batch.get("region", ["unknown"] * rgb.shape[0]))

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    hc = torch.cat(all_hc) if all_hc else None

    global_metrics = compute_metrics(all_preds, all_labels, hc_mask=hc)

    # Per-region breakdown
    regions = list(set(all_regions))
    per_region = {}
    for reg in regions:
        reg_mask = torch.tensor([r == reg for r in all_regions])
        p_reg = all_preds[reg_mask]
        l_reg = all_labels[reg_mask]
        hc_reg = hc[reg_mask] if hc is not None else None
        per_region[reg] = compute_metrics(p_reg, l_reg, hc_mask=hc_reg)

    recorder = ResultsRecorder(results_dir=str(results_dir))
    record = recorder.record(
        run_id=run_id,
        experiment=config.get("_experiment_id", "unknown"),
        config=config,
        metrics=global_metrics,
        per_region=per_region,
        checkpoint=checkpoint_path,
    )

    return global_metrics
