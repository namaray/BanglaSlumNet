"""
Segmentation losses and SAS-Net consistency loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        pred:   [B, 1, H, W] sigmoid probabilities
        target: [B, H, W] binary labels (0 or 1)
        mask:   [B, H, W] bool — restrict loss to masked pixels
        """
        p = pred.squeeze(1)
        t = target.float()
        if mask is not None:
            p = p[mask]
            t = t[mask]
        inter = (p * t).sum()
        denom = p.sum() + t.sum() + self.smooth
        return 1.0 - (2.0 * inter + self.smooth) / denom


class WeightedBCELoss(nn.Module):
    def __init__(
        self,
        slum_weight: float = 2.0,
        label_smoothing: float = 0.0,
        auto_balance: bool = True,
    ):
        super().__init__()
        self.slum_weight = slum_weight
        self.label_smoothing = label_smoothing
        self.auto_balance = auto_balance

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # binary_cross_entropy is unsafe under autocast (BF16) — force float32 here.
        p = pred.squeeze(1).float()
        t = target.float()
        if self.label_smoothing > 0:
            t = t * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        if self.auto_balance:
            weight_source = t[mask] if mask is not None else t.reshape(-1)
            n_pos = weight_source.sum().float()
            n_total = torch.tensor(float(weight_source.numel()), device=t.device)
            n_neg = n_total - n_pos
            if n_pos > 0 and n_neg > 0:
                # Equal total contribution from slum/formal pixels. This prevents
                # the region-type labels from rewarding all-slum or all-formal masks.
                pos_w = 0.5 * n_total / n_pos
                neg_w = 0.5 * n_total / n_neg
                weights = torch.where(target == 1, pos_w.expand_as(t), neg_w.expand_as(t)).float()
            else:
                weights = torch.ones_like(t).float()
        else:
            weights = torch.where(target == 1,
                                  torch.full_like(t, self.slum_weight),
                                  torch.ones_like(t)).float()
        p = p.clamp(1e-6, 1 - 1e-6)
        with torch.autocast(device_type=p.device.type, enabled=False):
            loss = F.binary_cross_entropy(p, t, weight=weights, reduction="none")
        if mask is not None:
            loss = loss[mask]
            if loss.numel() == 0:
                return pred.sum() * 0.0
        return loss.mean()


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        slum_weight: float = 2.0,
        label_smoothing: float = 0.0,
        auto_class_balance: bool = True,
    ):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = WeightedBCELoss(
            slum_weight=slum_weight,
            label_smoothing=label_smoothing,
            auto_balance=auto_class_balance,
        )
        self.dice_w = dice_weight
        self.bce_w = bce_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Only train on known labels (not LABEL_UNKNOWN=0 when in noisy-label mode)
        # When mask is provided, use it; otherwise use all non-unknown pixels
        if mask is None:
            mask = target != 0  # exclude unknown
        elif not mask.any():
            mask = target != 0
        if not mask.any():
            return pred.sum() * 0.0
        slum_target = (target == 1).long()
        return self.dice_w * self.dice(pred, slum_target, mask) + \
               self.bce_w * self.bce(pred, slum_target, mask)


class SASNetLoss(nn.Module):
    """
    Composite SAS-Net training loss:
      L_rec     = ||I - R(E_s(I), E_a(I))||   (reconstruction)
      L_consist = ||E_s(I_t1) - E_s(I_t2)||   (scene consistency across dates)
      L_swap    = ||I_t2 - R(E_s(I_t1), E_a(I_t2))||  (appearance swap re-render)
    """
    def __init__(
        self,
        lambda_rec: float = 1.0,
        lambda_consist: float = 1.0,
        lambda_swap: float = 0.5,
    ):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_consist = lambda_consist
        self.lambda_swap = lambda_swap

    def forward(
        self,
        recon: torch.Tensor,
        original: torch.Tensor,
        scene_t1: torch.Tensor,
        scene_t2: torch.Tensor,
        swap_recon: Optional[torch.Tensor] = None,
        swap_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        l_rec = F.l1_loss(recon, original)
        l_consist = F.mse_loss(scene_t1, scene_t2)
        total = self.lambda_rec * l_rec + self.lambda_consist * l_consist
        if swap_recon is not None and swap_target is not None:
            l_swap = F.l1_loss(swap_recon, swap_target)
            total = total + self.lambda_swap * l_swap
        return total
