"""
Stage 2 + fusion + decoder trainer.
Trains only the small heads over cached features — one epoch is seconds to minutes.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.banglaslumnet import BanglaSlumNet, build_model
from .losses import SegmentationLoss
from ..eval.metrics import compute_metrics
from ..tracking.recorder import ResultsRecorder
from ..tracking.registry import RunRegistry


def train_segmenter(
    config: dict,
    run_id: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: str,
    device: str = "cuda",
    resume_checkpoint: Optional[str] = None,
    registry: Optional[RunRegistry] = None,
) -> str:
    """
    Train Stage 2 segmentation head and return path to best checkpoint.
    VLM features are consumed from cache; the VLM encoder is never called here.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config).to(device)

    # Assert VLM backbone is frozen (if applicable)
    backbone_config = config.get("model", {}).get("config", "full")
    if backbone_config != "baseline_cnn":
        # No VLM parameters in the model object — features are pre-cached
        # Just verify projector is trainable and decoder is trainable
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Segmenter] Trainable params: {n_trainable:,} (all are fusion+decoder, VLM frozen/external)")

    cfg_tr = config.get("train", {})
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg_tr.get("lr", 3e-4),
        weight_decay=cfg_tr.get("weight_decay", 0.01),
    )
    epochs = cfg_tr.get("epochs", 80)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = SegmentationLoss(
        dice_weight=cfg_tr.get("dice_weight", 0.5),
        bce_weight=cfg_tr.get("bce_weight", 0.5),
        slum_weight=cfg_tr.get("slum_class_weight", 2.0),
        label_smoothing=cfg_tr.get("label_smoothing", 0.0),
    )

    best_val_iou = -1.0
    best_ckpt = None
    last_ckpt = None
    patience = cfg_tr.get("early_stopping_patience", 10)
    patience_counter = 0

    start_epoch = 0
    if resume_checkpoint and Path(resume_checkpoint).exists():
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_iou = ckpt.get("best_val_iou", -1.0)
        print(f"Resumed segmenter from epoch {start_epoch}")

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    amp_dtype = torch.bfloat16

    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"[{run_id}] epoch {epoch+1}/{epochs}", unit="batch", leave=False)
        for batch in pbar:
            rgb = batch["rgb"].to(device)
            label = batch["label"].to(device)
            hc_mask = batch.get("hc_mask", None)
            socioec = batch.get("socioec", None)
            cached_feats = batch.get("cached_feats", None)

            if socioec is not None:
                socioec = socioec.to(device)
            if cached_feats is not None:
                cached_feats = cached_feats.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                pred = model(rgb, cached_feats=cached_feats, socioec=socioec)
                loss = loss_fn(pred, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= max(len(train_loader), 1)
        scheduler.step()

        # Validation
        val_metrics = _evaluate_epoch(model, val_loader, device, amp_dtype)
        val_iou = val_metrics.get("hc_iou", 0.0)
        import math
        if val_iou is None or math.isnan(val_iou):
            val_iou = 0.0

        print(f"[Segmenter] Epoch {epoch+1}/{epochs} | loss={train_loss:.4f} "
              f"val_hc_iou={val_iou:.4f}")

        ckpt_payload = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "best_val_iou": best_val_iou, "config": config,
        }
        # Always keep a 'last' checkpoint so eval has something even if val never improves.
        last_ckpt = str(output_dir / "last.pt")
        torch.save(ckpt_payload, last_ckpt)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            best_ckpt = str(output_dir / "best.pt")
            torch.save(ckpt_payload, best_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    final_ckpt = best_ckpt or last_ckpt
    print(f"Training done. Best val HC-IoU: {best_val_iou:.4f} → {final_ckpt}")
    return final_ckpt


def _evaluate_epoch(model, loader, device, amp_dtype) -> Dict:
    model.eval()
    all_preds, all_labels, all_hc = [], [], []
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            label = batch["label"].to(device)
            hc_mask = batch.get("hc_mask", None)
            socioec = batch.get("socioec", None)
            cached_feats = batch.get("cached_feats", None)
            if socioec is not None:
                socioec = socioec.to(device)
            if cached_feats is not None:
                cached_feats = cached_feats.to(device)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                pred = model(rgb, cached_feats=cached_feats, socioec=socioec)

            all_preds.append(pred.squeeze(1).cpu())
            all_labels.append(label.cpu())
            if hc_mask is not None:
                all_hc.append(hc_mask.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    hc = torch.cat(all_hc) if all_hc else None
    return compute_metrics(all_preds, all_labels, hc_mask=hc)
