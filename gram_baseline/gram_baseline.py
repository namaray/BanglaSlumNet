"""GRAM zero-shot baseline on Dhaka tiles.

Loads the pretrained MoE checkpoint from DS4H-GIS/GRAM (epoch 2, v2) and runs
inference on Korail / Mirpur / Old Dhaka ESRI World Imagery tiles (z=16).

Key change from v2: domain routing now uses the model's own built-in
domain_classifier (pre-MoE features -> domain logits) via a two-pass forward
pass, instead of a manual DOMAIN_CANDIDATES sweep.

Output:
  - per-tile probability map (.npy + colorized .png)
  - per-location 3x3 mosaic of original | prob_heatmap | binary_overlay
  - summary CSV: per-tile mean/max slum probability and percent-slum-pixels
    (now also records the model-predicted domain index per tile)
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
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# GRAM's 12 training cities (alphabetical index order from the official repo):
#   0: Cairo, 1: Cape Town, 2: Caracas, 3: Colombo, 4: Karachi,
#   5: Medellín, 6: Mumbai, 7: Nairobi, 8: Ouagadougou,
#   9: Port-au-Prince, 10: Rio, 11: Tegucigalpa
# We no longer hardcode candidates — the model's domain_classifier picks for us.
DOMAIN_NUM = 12


def build_model():
    """Construct the GRAM MoE model and load checkpoint weights."""
    model = mit_b5_MOE()
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    # Strip 'module.' prefix saved from DataParallel wrapper
    new_state = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
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
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)  # 1,3,256,256
    return tensor, np.asarray(img)


def infer_one(model, tensor):
    """
    Two-pass inference using the model's own domain_classifier.

    Pass 1: Forward with dummy domain index 0 to obtain pre-MoE domain logits.
            The segmentation output from this pass is discarded.
    Pass 2: Forward again with the model-predicted domain index so that the
            correct MoE experts are activated. This segmentation is returned.

    Returns
    -------
    probs : np.ndarray, shape (H, W)
        Per-pixel slum probability (class 1 softmax score).
    predicted_domain : int
        The domain index chosen by the model's domain_classifier.
    dom_logits : np.ndarray, shape (DOMAIN_NUM,)
        Raw domain logits — useful for logging / ablation.
    """
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        # --- Pass 1: get domain logits from pre-MoE features ---
        dummy_cidx = torch.tensor([0], dtype=torch.long, device=DEVICE)
        _, dom_logits_t, _ = model(tensor, dummy_cidx)
        # dom_logits_t: (1, DOMAIN_NUM)
        predicted_domain = int(dom_logits_t.argmax(dim=1).item())

        # --- Pass 2: route through the correct experts ---
        real_cidx = torch.tensor([predicted_domain], dtype=torch.long, device=DEVICE)
        seg, _, _ = model(tensor, real_cidx)

    # seg: (1, 2, H, W) — class 1 is "slum"
    probs = TF.softmax(seg, dim=1)[0, 1].cpu().numpy()  # H x W
    dom_logits_np = dom_logits_t[0].cpu().numpy()       # (DOMAIN_NUM,)
    return probs, predicted_domain, dom_logits_np


# ---------------------------------------------------------------------------
# City name lookup for readable logging
# ---------------------------------------------------------------------------
CITY_NAMES = {
    0: "Cairo", 1: "Cape Town", 2: "Caracas", 3: "Colombo",
    4: "Karachi", 5: "Medellín", 6: "Mumbai", 7: "Nairobi",
    8: "Ouagadougou", 9: "Port-au-Prince", 10: "Rio", 11: "Tegucigalpa",
}


def colorize_prob(prob):
    """Map a [0,1] prob heatmap to RGB using a simple red ramp."""
    p = np.clip(prob, 0, 1)
    r = (p * 255).astype(np.uint8)
    g = np.zeros_like(r)
    b = ((1 - p) * 80).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


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

    rows = [["location", "tile", "domain_idx", "domain_name",
             "mean_prob", "max_prob", "pct_slum_p50", "pct_slum_p70"]]

    # Group tiles by location for mosaic output
    by_loc: dict = {}
    for p in tile_paths:
        name = os.path.basename(p)
        m = re.match(r"([a-z_]+)_z16_x(\d+)_y(\d+)\.jpg", name)
        if not m:
            continue
        loc = m.group(1).rstrip("_")
        x, y = int(m.group(2)), int(m.group(3))
        by_loc.setdefault(loc, []).append((x, y, p))

    # Per-location domain distribution tracker (for the summary)
    domain_votes: dict = {}  # loc -> list of predicted domain indices

    for loc, entries in by_loc.items():
        entries.sort(key=lambda e: (e[1], e[0]))  # sort by y then x
        xs = sorted({e[0] for e in entries})
        ys = sorted({e[1] for e in entries})
        cols_n, rows_n = len(xs), len(ys)

        mosaic_rgb     = np.zeros((rows_n * 256, cols_n * 256, 3), dtype=np.uint8)
        mosaic_heat    = np.zeros_like(mosaic_rgb)
        mosaic_overlay = np.zeros_like(mosaic_rgb)

        domain_votes[loc] = []

        for x, y, p in entries:
            ix = xs.index(x)
            iy = ys.index(y)
            t, rgb = load_image(p)

            prob, pred_domain, dom_logits = infer_one(model, t)

            mean_p = float(prob.mean())
            max_p  = float(prob.max())
            pct50  = float((prob > 0.5).mean() * 100)
            pct70  = float((prob > 0.7).mean() * 100)
            domain_name = CITY_NAMES.get(pred_domain, str(pred_domain))

            rows.append([loc, os.path.basename(p), pred_domain, domain_name,
                         round(mean_p, 4), round(max_p, 4),
                         round(pct50, 2), round(pct70, 2)])
            domain_votes[loc].append(pred_domain)

            print(f"  {loc} x={x} y={y}  "
                  f"domain={pred_domain}({domain_name})  "
                  f"mean={mean_p:.3f} max={max_p:.3f} "
                  f"pct>0.5={pct50:.1f}%  pct>0.7={pct70:.1f}%")

            # Save per-tile prob map
            np.save(os.path.join(OUT_DIR, f"{loc}_x{x}_y{y}_prob.npy"), prob)

            heat = colorize_prob(prob)
            ov   = overlay(rgb, prob, alpha=0.55, threshold=0.5)

            mosaic_rgb    [iy*256:(iy+1)*256, ix*256:(ix+1)*256] = rgb
            mosaic_heat   [iy*256:(iy+1)*256, ix*256:(ix+1)*256] = heat
            mosaic_overlay[iy*256:(iy+1)*256, ix*256:(ix+1)*256] = ov

        # Save combined mosaic: original | heatmap | overlay
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
    agg: dict = {}
    for r in rows[1:]:
        loc = r[0]
        agg.setdefault(loc, []).append((r[4], r[5], r[6], r[7]))

    for loc, vals in agg.items():
        arr = np.array(vals, dtype=np.float32)
        votes = domain_votes[loc]
        # Most-common predicted domain across the 9 tiles
        majority_domain = max(set(votes), key=votes.count)
        majority_name   = CITY_NAMES.get(majority_domain, str(majority_domain))
        domain_dist     = {CITY_NAMES.get(d, d): votes.count(d) for d in set(votes)}
        print(f"  {loc}:")
        print(f"    mean_prob    = {arr[:,0].mean():.3f}")
        print(f"    max-of-max   = {arr[:,1].max():.3f}")
        print(f"    avg_pct>0.5  = {arr[:,2].mean():.1f}%")
        print(f"    avg_pct>0.7  = {arr[:,3].mean():.1f}%")
        print(f"    majority_domain = {majority_domain} ({majority_name})")
        print(f"    domain_distribution = {domain_dist}")

    # Cross-location domain agreement report
    print("\n=== Domain routing report (model's own classifier) ===")
    all_tiles = rows[1:]
    for r in all_tiles:
        print(f"  {r[1]:45s}  -> domain {r[2]:2d} ({r[3]})")


if __name__ == "__main__":
    main()