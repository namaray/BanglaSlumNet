"""
Qualitative visualization: side-by-side prediction overlays.
Fig 6: N×M grid — tile | GRAM pred | baseline pred | BanglaSlumNet pred | HC ground truth.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .palette import NAVY, TEAL, CORAL, SLATE, DPI, apply_style


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.4) -> np.ndarray:
    """Blend a binary mask over an RGB image."""
    out = rgb.copy().astype(np.float32) / 255.0 if rgb.max() > 1 else rgb.copy().astype(np.float32)
    for c, v in enumerate(color):
        out[:, :, c] = np.where(mask > 0.5, out[:, :, c] * (1 - alpha) + v * alpha, out[:, :, c])
    return np.clip(out, 0, 1)


def hex_to_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def fig_qualitative(
    tile_rgbs: List[np.ndarray],
    gram_preds: List[np.ndarray],
    baseline_preds: List[np.ndarray],
    bangla_preds: List[np.ndarray],
    hc_gts: List[np.ndarray],
    tile_ids: List[str],
    output_dir: str,
    wide: bool = False,
) -> plt.Figure:
    """
    tile_rgbs:      list of [H, W, 3] float32 [0,1] RGB images
    gram_preds:     list of [H, W] float32 probability maps
    baseline_preds: list of [H, W] float32 probability maps
    bangla_preds:   list of [H, W] float32 probability maps
    hc_gts:         list of [H, W] binary HC ground-truth masks
    """
    apply_style(wide)
    N = len(tile_rgbs)
    ncols = 5  # tile, GRAM, baseline, BanglaSlumNet, HC GT
    col_titles = ["Input tile", "GRAM", "Baseline CNN", "BanglaSlumNet (full)", "HC Ground Truth"]

    fig, axes = plt.subplots(N, ncols, figsize=(ncols * 2.5, N * 2.5))
    if N == 1:
        axes = axes[np.newaxis, :]

    slum_color = hex_to_rgb(TEAL)
    gt_color = hex_to_rgb(NAVY)

    for i in range(N):
        rgb = tile_rgbs[i]

        # Column 0: input tile
        axes[i, 0].imshow(rgb[:, :, :3] if rgb.shape[2] >= 3 else np.stack([rgb[:, :, 0]]*3, -1))
        axes[i, 0].set_ylabel(tile_ids[i], fontsize=7, rotation=0, labelpad=40, va="center")

        # Columns 1-3: predictions overlaid on tile
        for col_idx, (preds, color) in enumerate([
            (gram_preds[i], hex_to_rgb(CORAL)),
            (baseline_preds[i], hex_to_rgb(SLATE)),
            (bangla_preds[i], slum_color),
        ], start=1):
            base_rgb = rgb[:, :, :3] if rgb.shape[2] >= 3 else np.stack([rgb[:, :, 0]]*3, -1)
            viz = overlay_mask(base_rgb, (preds > 0.5).astype(float), color)
            axes[i, col_idx].imshow(viz)

        # Column 4: HC ground truth
        base_rgb = rgb[:, :, :3] if rgb.shape[2] >= 3 else np.stack([rgb[:, :, 0]]*3, -1)
        gt_viz = overlay_mask(base_rgb, hc_gts[i].astype(float), gt_color)
        axes[i, 4].imshow(gt_viz)

        for ax in axes[i]:
            ax.axis("off")

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=8)

    patches = [
        mpatches.Patch(color=CORAL, label="Predicted slum"),
        mpatches.Patch(color=NAVY, label="HC ground truth"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(str(out / f"fig6_qualitative.{ext}"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return fig


def _smoke_test():
    import tempfile
    H, W = 64, 64
    N = 2
    rgbs = [np.random.rand(H, W, 3).astype(np.float32) for _ in range(N)]
    preds = [np.random.rand(H, W).astype(np.float32) for _ in range(N)]
    gts = [(np.random.rand(H, W) > 0.5).astype(np.float32) for _ in range(N)]
    ids = [f"tile_{i}" for i in range(N)]
    with tempfile.TemporaryDirectory() as tmp:
        fig_qualitative(rgbs, preds, preds, preds, gts, ids, tmp)
    print("qualitative.py smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
