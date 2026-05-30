"""
Stage 1 trainer: SAS-Net atmospheric disentanglement.
Trains on raw S2 tiles only (no VLM in the loop).
Target: ≤ 1 A100-hour. Outputs cached clean tiles for Stage 2.
"""

import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.sasnet import SASNet
from .losses import SASNetLoss
from ..tracking.recorder import ResultsRecorder


def train_sasnet(
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: str,
    device: str = "cuda",
    resume_checkpoint: Optional[str] = None,
) -> str:
    """
    Train SAS-Net and return path to best checkpoint.
    """
    cfg_sas = config.get("sasnet", {})
    cfg_tr = config.get("train_sasnet", {})

    model = SASNet(
        in_channels=len(config.get("data", {}).get("s2_bands", ["B2", "B3", "B4", "B8"])),
        encoder_dim=cfg_sas.get("encoder_dim", 256),
        style_dim=cfg_sas.get("style_dim", 128),
        n_res=cfg_sas.get("num_res_blocks", 4),
    ).to(device)

    loss_fn = SASNetLoss(
        lambda_rec=cfg_tr.get("lambda_rec", 1.0),
        lambda_consist=cfg_tr.get("lambda_consist", 1.0),
        lambda_swap=cfg_tr.get("lambda_swap", 0.5),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg_tr.get("lr", 1e-4),
        weight_decay=config.get("train", {}).get("weight_decay", 0.01),
    )

    epochs = cfg_tr.get("epochs", 50)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_ckpt = None

    start_epoch = 0
    if resume_checkpoint and Path(resume_checkpoint).exists():
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed SAS-Net from epoch {start_epoch}")

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    amp_dtype = torch.bfloat16

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            # batch should contain two time-steps of the same location
            I_t1 = batch["rgb"].to(device)
            I_t2 = batch.get("rgb_t2", I_t1).to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                recon_t1, scene_t1, (sm1, ss1) = model(I_t1)
                _, scene_t2, (sm2, ss2) = model(I_t2)

                # Appearance swap: render I_t1 with I_t2's style
                swap_recon = model.renderer(scene_t1, sm2, ss2)
                loss = loss_fn(recon_t1, I_t1, scene_t1, scene_t2,
                               swap_recon=swap_recon, swap_target=I_t2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                I = batch["rgb"].to(device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                    recon, scene, _ = model(I)
                    scene2 = model.scene_encoder(I)
                    val_loss += loss_fn(recon, I, scene, scene2).item()
        val_loss /= max(len(val_loader), 1)

        print(f"[SASNet] Epoch {epoch+1}/{epochs} | train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt = str(output_dir / "sasnet_best.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": config,
            }, best_ckpt)

    print(f"SAS-Net training done. Best val loss: {best_val_loss:.4f} → {best_ckpt}")
    return best_ckpt


def cache_clean_tiles(
    checkpoint_path: str,
    data_loader: DataLoader,
    output_dir: str,
    config: dict,
    device: str = "cuda",
):
    """
    Run trained SAS-Net on all tiles and cache clean (reference-appearance) renders.
    This is the Stage 1 output that Stage 2 consumes.
    """
    import numpy as np, rasterio
    from rasterio.transform import from_bounds

    cfg_sas = config.get("sasnet", {})
    model = SASNet(
        in_channels=len(config.get("data", {}).get("s2_bands", ["B2", "B3", "B4", "B8"])),
        encoder_dim=cfg_sas.get("encoder_dim", 256),
        style_dim=cfg_sas.get("style_dim", 128),
        n_res=cfg_sas.get("num_res_blocks", 4),
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in data_loader:
            tile_ids = batch["tile_id"]
            rgb = batch["rgb"].to(device)
            clean = model.clean_tile(rgb).cpu().numpy()  # [B, C, 256, 256]
            for i, tid in enumerate(tile_ids):
                out_path = output_dir / f"{tid}_clean.npy"
                if not out_path.exists():
                    np.save(str(out_path), clean[i])

    print(f"Clean tiles cached to {output_dir}")
