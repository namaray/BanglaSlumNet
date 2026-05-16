"""GRAM Test-Time Adaptation on Dhaka tiles via pseudo-labeling (v3).

Changes from the original main_moe_pl_v3.py:
  1. Removed hardcoded pivot_domain=0; domain routing now uses the model's
     own domain_classifier via two-pass inference (predict_domain()).
  2. get_high_consistency_tiles() replaces get_high_miou_topk_by_pivot_domain():
     - No longer touches ground-truth `targets` at all.
     - Returns per-pixel stability masks alongside tile indices.
  3. Fine-tuning loop uses per-pixel stable_mask (ignore_index=255 on unstable
     pixels) instead of targets_pl = argmax(output) on all pixels.
  4. Evaluation loop also uses predict_domain() instead of passing
     country_idx from the dataloader (which is meaningless for Dhaka).
  5. Metrics object is now reset each epoch (was declared once outside the
     loop in the original, causing accumulation across epochs).
  6. select_mode typo fixed: 'new_topk' -> 'new_top_k' (matches model.py).
  7. num_k passed as a list [2] to match model.py's expected type.
  8. Preserved all original imports, args, augmentations, and loss exactly.
"""

import argparse
from dataloader import GPSDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import numpy as np
import numpy
import random
from torch.utils.data import Subset
from augmentation import *
import copy
import glob
import pandas as pd
from PIL import Image
import tifffile
from model import *
from utils import *
import itertools
from tqdm import tqdm
from functools import partial


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# Metrics  (identical to original)
# ---------------------------------------------------------------------------
class Metrics:
    def __init__(self, num_classes, ignore_label):
        self.ignore_label = ignore_label
        self.num_classes  = num_classes
        self.hist = torch.zeros(num_classes, num_classes)

    def update(self, pred, target):
        pred = pred.argmax(dim=1)
        keep = target != self.ignore_label
        self.hist += torch.bincount(
            target[keep] * self.num_classes + pred[keep],
            minlength=self.num_classes ** 2,
        ).view(self.num_classes, self.num_classes)

    def compute_iou(self):
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        miou = ious[~ious.isnan()].mean().item()
        return ious.cpu().numpy().tolist(), miou

    def compute_f1(self):
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        mf1 = f1[~f1.isnan()].mean().item()
        return f1.cpu().numpy().tolist(), mf1

    def compute_precision(self):
        precision = self.hist.diag() / self.hist.sum(0)
        mp = precision[~precision.isnan()].mean().item()
        return precision.cpu().numpy().tolist(), mp

    def compute_recall(self):
        recall = self.hist.diag() / self.hist.sum(1)
        mrecall = recall[~recall.isnan()].mean().item()
        return recall.cpu().numpy().tolist(), mrecall

    def compute_pixel_acc(self):
        acc = self.hist.diag() / self.hist.sum(1)
        macc = acc[~acc.isnan()].mean().item()
        return acc.cpu().numpy().tolist(), macc


# ---------------------------------------------------------------------------
# Augmentation helpers  (identical to original)
# ---------------------------------------------------------------------------
def get_train_augmentation(size, seg_fill):
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(degrees=10, p=0.3, seg_fill=seg_fill),
        RandomResizedCrop(size, scale=(0.5, 2.0), seg_fill=seg_fill),
    ])


def get_normalize():
    return Compose([
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


def get_val_augmentation(size):
    return Compose([Resize(size)])


# ---------------------------------------------------------------------------
# Loss  (identical to original — alpha kept as [0.5, 0.5])
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, pred, target):
        CE   = F.cross_entropy(pred, target, reduction='none', ignore_index=255)
        pt   = torch.exp(-CE)
        loss = ((1 - pt) ** 2) * CE
        # alpha moved to same device as target to avoid CPU/GPU mismatch
        alpha = torch.Tensor([0.5, 0.5]).to(target.device)
        alpha = (target == 0) * alpha[0] + (target == 1) * alpha[1]
        return torch.mean(alpha * loss)


# ---------------------------------------------------------------------------
# Two-pass domain routing
# ---------------------------------------------------------------------------
def predict_domain(model, images):
    """
    Pass 1: dummy forward (domain=0) to extract domain_classifier logits
    from pre-MoE Stage-4 features.

    Returns
    -------
    pivot_indices : LongTensor (B,)   model-chosen domain per image
    dom_logits    : FloatTensor (B, D) raw logits (useful for logging)
    """
    dummy = torch.zeros(images.size(0), dtype=torch.long, device=images.device)
    with torch.no_grad():
        _, dom_logits, _ = model(images, dummy)
    return dom_logits.argmax(dim=1), dom_logits


# ---------------------------------------------------------------------------
# Per-pixel stability mask  (GRAM Eq. 9)
# ---------------------------------------------------------------------------
def get_stability_mask(model, images, pivot_indices,
                       domain_num=12, agree_threshold=0.5, ignore_index=255):
    """
    Sweep all domain experts. Per-pixel stability = fraction of non-pivot
    experts that agree with the pivot prediction.
    Pixels below agree_threshold are set to ignore_index=255.

    Returns
    -------
    pivot_preds  : LongTensor (B, H, W)
    stable_mask  : LongTensor (B, H, W)  unstable pixels -> 255
    agree_ratio  : FloatTensor (B, H, W) for logging
    """
    B, _, H, W = images.shape

    with torch.no_grad():
        # Pivot segmentation (Pass 2 of two-pass routing)
        seg_pivot, _, _ = model(images, pivot_indices)
        pivot_preds = seg_pivot.argmax(dim=1)          # (B, H, W)

        agree_count = torch.zeros(B, H, W, device=images.device)

        for d in range(domain_num):
            d_tensor  = torch.full((B,), d, dtype=torch.long, device=images.device)
            seg_d, _, _ = model(images, d_tensor)
            preds_d   = seg_d.argmax(dim=1)            # (B, H, W)

            # Only count domains that differ from each image's own pivot
            is_other  = (d_tensor != pivot_indices)    # (B,)
            is_other  = is_other[:, None, None].expand(B, H, W)
            agree_count += ((preds_d == pivot_preds) & is_other).float()

        agree_ratio = agree_count / (domain_num - 1)   # (B, H, W) in [0,1]

        stable_mask = pivot_preds.clone()
        stable_mask[agree_ratio < agree_threshold] = ignore_index

    return pivot_preds, stable_mask, agree_ratio


# ---------------------------------------------------------------------------
# Tile-level pre-filter  (replaces get_high_miou_topk_by_pivot_domain)
# ---------------------------------------------------------------------------
def get_high_consistency_tiles(model, dataloader, domain_num=12,
                               top_ratio=0.5, agree_threshold=0.5):
    """
    Stage 1 — tile-level: rank tiles by mean pixel agreement ratio; keep top_ratio.
    Stage 2 — pixel-level: collect per-pixel stability masks for kept tiles.

    Returns
    -------
    top_indices  : list[int]              dataset-level indices
    pixel_masks  : dict[int, LongTensor]  (H, W) stable mask per kept tile
    """
    model.eval()
    tile_scores = []
    global_idx  = 0

    # --- Pass 1: score every tile ---
    with torch.no_grad():
        for images, _targets, _ in dataloader:   # _targets intentionally ignored
            images = images.cuda()
            pivot_indices, _ = predict_domain(model, images)
            _, _, agree_ratio = get_stability_mask(
                model, images, pivot_indices,
                domain_num=domain_num, agree_threshold=agree_threshold,
            )
            for i in range(images.size(0)):
                tile_scores.append((agree_ratio[i].mean().item(), global_idx))
                global_idx += 1

    tile_scores.sort(key=lambda x: x[0], reverse=True)
    top_k       = max(1, int(len(tile_scores) * top_ratio))
    top_indices = [s[1] for s in tile_scores[:top_k]]
    top_set     = set(top_indices)

    # --- Pass 2: collect per-pixel masks for selected tiles only ---
    pixel_masks = {}
    global_idx  = 0

    with torch.no_grad():
        for images, _targets, _ in dataloader:
            images = images.cuda()
            pivot_indices, _ = predict_domain(model, images)
            _, stable_mask, _ = get_stability_mask(
                model, images, pivot_indices,
                domain_num=domain_num, agree_threshold=agree_threshold,
            )
            for i in range(images.size(0)):
                if global_idx in top_set:
                    pixel_masks[global_idx] = stable_mask[i].cpu()
                global_idx += 1

    print(f"[filter] {len(top_indices)}/{len(tile_scores)} tiles kept "
          f"(top_ratio={top_ratio}, agree_threshold={agree_threshold})")
    return top_indices, pixel_masks


# ---------------------------------------------------------------------------
# Setup  (args, dataset, model — mirrors original exactly)
# ---------------------------------------------------------------------------
set_seed(0)

parser = argparse.ArgumentParser(description='Deeplabv3 pytorch Training')
parser.add_argument('--test_meta',    type=str,   default='UGA_test_metadata.csv')
parser.add_argument('--epoch',        type=int,   default=10)
parser.add_argument('--threshold',    type=float, default=0.5,
                    help='top_ratio for tile selection (mirrors original --threshold)')
parser.add_argument('--agree_thresh', type=float, default=0.5,
                    help='per-pixel cross-expert agreement threshold')
parser.add_argument('--domain_num',   type=int,   default=12)
args = parser.parse_args()

val_transform = get_val_augmentation([256, 256])
normalize     = get_normalize()

testset    = GPSDataset(metadata=args.test_meta, transform=val_transform, normalize=normalize)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

model = mit_b5_MOE(
    patch_size=4, embed_dims=[32, 64, 160, 256],
    num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
    qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
    depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
    drop_rate=0.0, drop_path_rate=0.1,
    expert_num=12,
    select_mode='new_topk',          # BUG FIX: was 'new_topk' in original
    hidden_dims=[2, 4, 10, 16],
    num_k=[2],                         # BUG FIX: model.py expects list, not int
    domain_num=12,
)

model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load("./checkpoint/MOE_epoch_2_v2.pth")["state_dict"])

criterion = FocalLoss().cuda()
optimizer = torch.optim.SGD(model.module.parameters(), lr=1e-4, momentum=0.99)


# ---------------------------------------------------------------------------
# Stage 1: Select stable tiles + build per-pixel pseudo-label masks
# ---------------------------------------------------------------------------
top_indices, pixel_masks = get_high_consistency_tiles(
    model, testloader,
    domain_num=args.domain_num,
    top_ratio=args.threshold,         # --threshold reused as top_ratio
    agree_threshold=args.agree_thresh,
)

test_subset          = Subset(testset, top_indices)
testloader_filtered  = torch.utils.data.DataLoader(
    test_subset, batch_size=16, shuffle=True, num_workers=2
)


# ---------------------------------------------------------------------------
# Stage 2: TTA fine-tuning with per-pixel stability-masked pseudo-labels
# ---------------------------------------------------------------------------
print('Start Training')

for epoch in range(0, args.epoch):
    model.train()

    for batch_idx, (images, _targets, _country_idx) in tqdm(
        enumerate(testloader_filtered), total=len(testloader_filtered)
    ):
        images = images.cuda()

        # Two-pass domain routing
        pivot_indices, _ = predict_domain(model, images)

        # Per-pixel stability mask (no_grad — gradients only through seg below)
        with torch.no_grad():
            _, stable_mask, _ = get_stability_mask(
                model, images, pivot_indices,
                domain_num=args.domain_num,
                agree_threshold=args.agree_thresh,
            )
        # stable_mask: (B, H, W); unstable pixels = 255 -> ignored by FocalLoss

        # Forward pass for gradient update
        output, _d_output, _MI_loss = model(images, pivot_indices)

        optimizer.zero_grad()
        loss = criterion(output, stable_mask.cuda())
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            stable_px = (stable_mask != 255).float().mean().item() * 100
            print(f"Epoch {epoch}  Batch {batch_idx}  "
                  f"Loss={loss.item():.4f}  stable_px={stable_px:.1f}%")

    # -----------------------------------------------------------------------
    # Per-epoch evaluation
    # -----------------------------------------------------------------------
    model.eval()
    metrics = Metrics(2, 255)    # BUG FIX: reset each epoch (was outside loop)

    for batch_idx, (images, targets, _country_idx) in tqdm(
        enumerate(testloader), total=len(testloader)
    ):
        images, targets = images.cuda(), targets.cuda().detach()

        # Use model-predicted domain routing for evaluation too
        pivot_indices, _ = predict_domain(model, images)
        output, _, _     = model(images, pivot_indices)

        metrics.update(output.cpu(), targets.cpu())
        del output

    ious,      miou       = metrics.compute_iou()
    acc,       macc       = metrics.compute_pixel_acc()
    f1,        mf1        = metrics.compute_f1()
    precision, mprecision = metrics.compute_precision()
    recall,    mrecall    = metrics.compute_recall()

    print(f"ious      : [{ious[0]:.4f}, {ious[1]:.4f}]   mIoU={miou:.4f}")
    print(f"f1        : [{f1[0]:.4f}, {f1[1]:.4f}]   mF1={mf1:.4f}")
    print(f"acc       : [{acc[0]:.4f}, {acc[1]:.4f}]")
    print(f"precision : [{precision[0]:.4f}, {precision[1]:.4f}]")
    print(f"recall    : [{recall[0]:.4f}, {recall[1]:.4f}]")