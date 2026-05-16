"""GRAM zero-shot baseline on Dhaka tiles.

Loads the pretrained MoE checkpoint from DS4H-GIS/GRAM (epoch 2, v2) and runs
inference on Korail / Mirpur / Old Dhaka ESRI World Imagery tiles (z=16).

Output:
  - per-tile probability map (.npy + colorized .png)
  - per-location 3x3 mosaic of original | prob_heatmap | binary_overlay
  - summary CSV: per-tile mean/max slum probability and percent-slum-pixels
"""

import os
import sys
import glob
import csv
import math
import re
import torch
import torch.nn.functional as TF
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from model import mit_b5_MOE  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TILES_DIR = os.path.join(HERE, "dhaka_tiles")
OUT_DIR = os.path.join(HERE, "outputs")
CKPT = os.path.join(HERE, "checkpoint", "MOE_epoch_2_v2.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalization (matches GRAM's training in main_moe.py)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Country indices in GRAM's training set (alphabetical by metadata convention).
# We will sweep a few South-Asia-adjacent domains for the pseudo_domain_label.
# The 12 training cities: Cairo, Cape Town, Nairobi, Ouagadougou, Colombo,
# Karachi, Mumbai, Caracas, Medellín, Rio, Port-au-Prince, Tegucigalpa.
# Index mapping is unknown without metadata; we'll try a couple of candidates.
DOMAIN_CANDIDATES = [0, 4, 5, 6]  # try first/some-middle indices


def build_model():
    """Construct the GRAM MoE model and load checkpoint weights."""
    # mit_b5_MOE hard-codes its architectural hyperparams; ignore kwargs.
    model = mit_b5_MOE()
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    # Strip 'module.' prefix (saved from DataParallel wrapper)
    new_state = {}
    for k, v in state.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        new_state[nk] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:5]:
        print(f"       missing sample: {missing[:5]}")
    if unexpected[:5]:
        print(f"       unexpected sample: {unexpected[:5]}")
    model.eval()
    model.to(DEVICE)
    return model


def load_image(path):
    """Load a 256x256 RGB tile as a float tensor in [0,1], CHW."""
    img = Image.open(path).convert("RGB").resize((256, 256), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC
    # ImageNet normalize
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)  # 1,3,256,256
    return tensor, np.asarray(img)


def infer_one(model, tensor, domain_idx):
    """Run one forward pass with a chosen pseudo_domain_label."""
    tensor = tensor.to(DEVICE)
    cidx = torch.tensor([domain_idx], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        seg, _dom, _mi = model(tensor, cidx)
    # seg: (1, 2, H, W) — class 1 is "slum"
    probs = TF.softmax(seg, dim=1)[0, 1].cpu().numpy()  # H x W
    return probs


def colorize_prob(prob):
    """Map a [0,1] prob heatmap to RGB using a simple red ramp."""
    p = np.clip(prob, 0, 1)
    r = (p * 255).astype(np.uint8)
    g = np.zeros_like(r)
    b = ((1 - p) * 80).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    return rgb


def overlay(rgb, prob, alpha=0.55, threshold=0.5):
    """Overlay binary slum mask (prob > threshold) on RGB in red."""
    mask = (prob > threshold).astype(np.float32)[..., None]
    red = np.zeros_like(rgb, dtype=np.float32)
    red[..., 0] = 255
    blended = rgb.astype(np.float32) * (1 - alpha * mask) + red * (alpha * mask)
    return np.clip(blended, 0, 255).astype(np.uint8)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = build_model()

    tile_paths = sorted(glob.glob(os.path.join(TILES_DIR, "*_z16_x*_y*.jpg")))
    print(f"[run] {len(tile_paths)} tiles on {DEVICE}")

    rows = [["location", "tile", "domain_idx", "mean_prob", "max_prob",
             "pct_slum_p50", "pct_slum_p70"]]

    # First: pick the best domain index by mean activation on a Korail tile.
    probe = [p for p in tile_paths if "korail" in os.path.basename(p)][:1]
    if probe:
        t, _ = load_image(probe[0])
        sweep = {}
        for d in DOMAIN_CANDIDATES:
            pr = infer_one(model, t, d)
            sweep[d] = pr.mean()
            print(f"  [probe] domain_idx={d} mean_prob={pr.mean():.4f} max={pr.max():.4f}")
        # Use the domain that gives the highest mean activation (most confident)
        best_domain = max(sweep, key=sweep.get)
        print(f"[probe] using domain_idx={best_domain}")
    else:
        best_domain = 0

    # Group tiles by location for mosaic output
    by_loc = {}
    for p in tile_paths:
        name = os.path.basename(p)
        m = re.match(r"([a-z_]+)_z16_x(\d+)_y(\d+)\.jpg", name)
        if not m:
            continue
        loc = m.group(1).rstrip("_")
        x, y = int(m.group(2)), int(m.group(3))
        by_loc.setdefault(loc, []).append((x, y, p))

    for loc, entries in by_loc.items():
        entries.sort(key=lambda e: (e[1], e[0]))  # by y then x
        xs = sorted({e[0] for e in entries})
        ys = sorted({e[1] for e in entries})
        cols, rows_n = len(xs), len(ys)
        mosaic_rgb = np.zeros((rows_n * 256, cols * 256, 3), dtype=np.uint8)
        mosaic_heat = np.zeros_like(mosaic_rgb)
        mosaic_overlay = np.zeros_like(mosaic_rgb)
        for x, y, p in entries:
            ix = xs.index(x)
            iy = ys.index(y)
            t, rgb = load_image(p)
            prob = infer_one(model, t, best_domain)
            mean_p = float(prob.mean())
            max_p = float(prob.max())
            pct50 = float((prob > 0.5).mean() * 100)
            pct70 = float((prob > 0.7).mean() * 100)
            rows.append([loc, os.path.basename(p), best_domain,
                         round(mean_p, 4), round(max_p, 4),
                         round(pct50, 2), round(pct70, 2)])
            print(f"  {loc} x={x} y={y}  mean={mean_p:.3f} max={max_p:.3f} "
                  f"pct>0.5={pct50:.1f}%  pct>0.7={pct70:.1f}%")
            # Save per-tile prob map
            np.save(os.path.join(OUT_DIR, f"{loc}_x{x}_y{y}_prob.npy"), prob)
            heat = colorize_prob(prob)
            ov = overlay(rgb, prob, alpha=0.55, threshold=0.5)
            mosaic_rgb[iy*256:(iy+1)*256, ix*256:(ix+1)*256] = rgb
            mosaic_heat[iy*256:(iy+1)*256, ix*256:(ix+1)*256] = heat
            mosaic_overlay[iy*256:(iy+1)*256, ix*256:(ix+1)*256] = ov

        # Save combined mosaic: input | heat | overlay
        combined = np.concatenate([mosaic_rgb, mosaic_heat, mosaic_overlay], axis=1)
        out_png = os.path.join(OUT_DIR, f"{loc}_gram_baseline.png")
        Image.fromarray(combined).save(out_png)
        print(f"[saved] {out_png}")

    # Summary CSV
    csv_path = os.path.join(OUT_DIR, "gram_baseline_summary.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[saved] {csv_path}")

    # Aggregate stats by location
    print("\n=== Aggregate stats by location ===")
    agg = {}
    for r in rows[1:]:
        loc = r[0]
        agg.setdefault(loc, []).append((r[3], r[4], r[5], r[6]))
    for loc, vals in agg.items():
        arr = np.array(vals, dtype=np.float32)
        print(f"  {loc}: mean={arr[:,0].mean():.3f}  "
              f"max-of-max={arr[:,1].max():.3f}  "
              f"avg_pct>0.5={arr[:,2].mean():.1f}%  "
              f"avg_pct>0.7={arr[:,3].mean():.1f}%")


if __name__ == "__main__":
    main()
