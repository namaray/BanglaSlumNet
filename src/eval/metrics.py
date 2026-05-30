"""
Evaluation metrics: HC-IoU, All-IoU, P/R/F1, FPR-on-control, SSIM, PSNR.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _binarize(pred: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (pred >= threshold).long()


def iou(pred_bin: torch.Tensor, target_bin: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    if mask is not None:
        pred_bin = pred_bin[mask]
        target_bin = target_bin[mask]
    inter = (pred_bin & target_bin).sum().float()
    union = (pred_bin | target_bin).sum().float()
    if union == 0:
        return float("nan")
    return (inter / union).item()


def precision_recall_f1(
    pred_bin: torch.Tensor,
    target_bin: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    if mask is not None:
        pred_bin = pred_bin[mask]
        target_bin = target_bin[mask]
    tp = (pred_bin & target_bin).sum().float()
    fp = (pred_bin & ~target_bin).sum().float()
    fn = (~pred_bin & target_bin).sum().float()
    precision = (tp / (tp + fp + 1e-8)).item()
    recall = (tp / (tp + fn + 1e-8)).item()
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1}


def fpr(pred_bin: torch.Tensor, target_bin: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """False positive rate on a control (non-slum) region."""
    if mask is not None:
        pred_bin = pred_bin[mask]
        target_bin = target_bin[mask]
    fp = (pred_bin & ~target_bin).sum().float()
    tn = (~pred_bin & ~target_bin).sum().float()
    denom = fp + tn
    if denom == 0:
        return float("nan")
    return (fp / denom).item()


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Structural Similarity Index for image reconstruction quality."""
    try:
        from skimage.metrics import structural_similarity
        import numpy as np
        p = pred.cpu().numpy()
        t = target.cpu().numpy()
        vals = []
        for i in range(p.shape[0]):
            for c in range(p.shape[1]):
                val = structural_similarity(p[i, c], t[i, c], data_range=1.0)
                vals.append(val)
        return float(sum(vals) / len(vals))
    except ImportError:
        return float("nan")


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(torch.tensor(1.0 / mse)).item()


def compute_metrics(
    pred: torch.Tensor,
    label: torch.Tensor,
    hc_mask: Optional[torch.Tensor] = None,
    control_mask: Optional[torch.Tensor] = None,
    korail_mask: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    pred:         [N, H, W] sigmoid probabilities
    label:        [N, H, W] int (0=unknown, 1=slum, 2=formal)
    hc_mask:      [N, H, W] bool — HC eval pixels
    control_mask: [N, H, W] bool — formal-dense control region pixels
    korail_mask:  [N, H, W] bool — Korail region pixels
    """
    pred_bin = _binarize(pred, threshold)
    slum_label = (label == 1)

    # All-IoU: all labeled pixels
    labeled_mask = label != 0
    all_iou = iou(pred_bin, slum_label, mask=labeled_mask)

    # HC-IoU: high-confidence pixels only
    hc_iou = iou(pred_bin, slum_label, mask=hc_mask) if hc_mask is not None else float("nan")

    # Precision, recall, F1 on HC pixels
    if hc_mask is not None:
        prf = precision_recall_f1(pred_bin, slum_label, mask=hc_mask)
    else:
        prf = precision_recall_f1(pred_bin, slum_label, mask=labeled_mask)

    # FPR on formal-dense control regions
    fpr_control = fpr(pred_bin, slum_label, mask=control_mask) if control_mask is not None else float("nan")

    # Korail recall
    korail_recall = float("nan")
    if korail_mask is not None:
        prf_k = precision_recall_f1(pred_bin, slum_label, mask=korail_mask)
        korail_recall = prf_k["recall"]

    return {
        "hc_iou": hc_iou,
        "all_iou": all_iou,
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "fpr_control": fpr_control,
        "korail_recall": korail_recall,
    }
