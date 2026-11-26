"""
GRAM on Dhaka, run on a Kaggle GPU notebook (E1, docs/MASTER_PLAN.md §3.3).

Expects a private Kaggle Dataset (built by scripts/build_kaggle_gram_bundle.py) holding:
  model.py                      GRAM's mit_b5_MOE (DS4H-GIS/GRAM, as patched on the gram-zero-shot branch)
  checkpoint/MOE_epoch_2_v2.pth released source-trained weights
  tiles/dhaka_x<X>_y<Y>.jpg     Esri Wayback zoom-16 tiles over the ESA mapped area

Steps, all label-free:
  1. Zero-shot sweep: predict every tile with each of the 12 domain indices (saved at 64x64).
  2. Pick the domain index whose predictions agree most with the other 11 experts (mean IoU of
     slum masks), GRAM's own consistency criterion, instead of probing a known slum.
  3. Test-time adaptation as in GRAM's main_moe_pl_v3.py: rank tiles by agreement between the
     pivot expert and the others, keep the top 50%, fine-tune 10 epochs on argmax pseudo-labels
     (focal loss, SGD lr 1e-4, momentum 0.99, batch 16, no augmentation).
  4. Save full-resolution probability maps (uint8, prob x 255) before and after adaptation.

Output: /kaggle/working/gram_outputs.zip -> download it from the notebook's Output tab.

Kaggle notebook cell:
    !python /kaggle/input/<your-dataset-name>/gram_tta_kaggle.py

Memory: one GRAM forward needs ~256 MB per tile in the decoder, so predictions run 8 tiles per
GPU, and adaptation splits GRAM's batch of 16 into micro-batches with gradient accumulation
(same effective batch). If a 16 GB GPU still runs out, lower --pred_batch / --micro_batch.
"""

import glob
import json
import os
import shutil
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch import

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DOMAINS, EPOCHS, BATCH, TOP_RATIO, LR = 12, 10, 16, 0.5, 1e-4
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def find_bundle():
    hits = glob.glob("/kaggle/input/**/checkpoint/MOE_epoch_2_v2.pth", recursive=True) or \
        glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint", "MOE_epoch_2_v2.pth"))
    if not hits:
        sys.exit("Bundle not found: add the dataset with checkpoint/MOE_epoch_2_v2.pth to the notebook.")
    return os.path.dirname(os.path.dirname(hits[0]))


def load_tiles(tile_dir, max_tiles=0):
    """Tiles as uint8 (N, 3, 256, 256); normalised per batch in to_input() to keep memory low."""
    paths = sorted(glob.glob(os.path.join(tile_dir, "*_x*_y*.jpg")))
    if max_tiles:
        paths = paths[:max_tiles]
    arr = np.stack([np.asarray(Image.open(p).convert("RGB").resize((256, 256), Image.BILINEAR)) for p in paths])
    return paths, torch.from_numpy(arr.transpose(0, 3, 1, 2).copy())


def to_input(xb, device):
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return (xb.to(device).float() / 255.0 - mean) / std


def build_model(bundle, device):
    sys.path.insert(0, bundle)
    from model import mit_b5_MOE
    model = mit_b5_MOE()
    state = torch.load(os.path.join(bundle, "checkpoint", "MOE_epoch_2_v2.pth"), map_location="cpu",
                       weights_only=False)
    state = state.get("state_dict", state)
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device)


@torch.no_grad()
def predict(model, x, domain, device, batch=32):
    """Slum probability per pixel as uint8 (prob x 255), (N, 256, 256) on CPU."""
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        xb = to_input(x[i:i + batch], device)
        d = torch.full((len(xb),), domain, dtype=torch.long, device=device)
        seg, _, _ = model(xb, d)
        out.append((F.softmax(seg.float(), dim=1)[:, 1] * 255).round().to(torch.uint8).cpu())
    return torch.cat(out)


def mask_iou(a, b):
    inter = (a & b).flatten(1).sum(1).float()
    union = (a | b).flatten(1).sum(1).float()
    return torch.where(union > 0, inter / union, torch.zeros_like(union))


def focal_loss(pred, target):
    ce = F.cross_entropy(pred, target, reduction="none", ignore_index=255)
    loss = (1 - torch.exp(-ce)) ** 2 * ce
    alpha = (target == 0) * 0.5 + (target == 1) * 0.5
    return torch.mean(alpha * loss)


def save_maps(paths, probs, out_dir, size=None):
    """probs: uint8 (prob x 255) maps; optionally area-downsampled for the sweep."""
    os.makedirs(out_dir, exist_ok=True)
    q = probs
    if size:
        q = F.interpolate(probs.float()[:, None], size=(size, size), mode="area")[:, 0].round().to(torch.uint8)
    q = q.numpy()
    for path, m in zip(paths, q):
        np.save(os.path.join(out_dir, os.path.basename(path).replace(".jpg", "_prob.npy")), m)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_tiles", type=int, default=0, help="smoke test only: use the first N tiles")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", default="")
    ap.add_argument("--pred_batch", type=int, default=8, help="tiles per GPU per prediction step")
    ap.add_argument("--micro_batch", type=int, default=4, help="tiles per GPU per adaptation micro-step")
    args = ap.parse_args()
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = max(torch.cuda.device_count(), 1)
    bundle = find_bundle()
    out = args.out or ("/kaggle/working/gram_outputs" if os.path.isdir("/kaggle/working")
                       else os.path.join(bundle, "gram_outputs"))
    os.makedirs(out, exist_ok=True)
    paths, x = load_tiles(os.path.join(bundle, "tiles"), args.max_tiles)
    print(f"[data] {len(paths)} tiles, device {device} ({torch.cuda.device_count()} GPU)")
    model = build_model(bundle, device)
    net = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    pred_batch = args.pred_batch * n_gpu
    micro = min(args.micro_batch * n_gpu, BATCH)
    accum = max(BATCH // micro, 1)
    print(f"[memory] prediction batch {pred_batch}; adaptation {accum} x {micro} = effective batch {accum * micro}")

    # 1. Zero-shot sweep over all domain indices
    probs = {}
    for d in range(DOMAINS):
        probs[d] = predict(net, x, d, device, batch=pred_batch)
        save_maps(paths, probs[d], os.path.join(out, f"sweep_domain{d:02d}"), size=64)
        print(f"[sweep] domain {d:2d}: mean prob {probs[d].float().mean() / 255:.3f}, "
              f"slum share {(probs[d] >= 128).float().mean():.3f}")

    # 2. Label-free domain choice: agreement of each expert with the other 11
    masks = {d: probs[d] >= 128 for d in range(DOMAINS)}  # prob >= 0.5
    agree = {d: float(np.mean([mask_iou(masks[d], masks[o]).mean().item() for o in range(DOMAINS) if o != d]))
             for d in range(DOMAINS)}
    pivot = max(agree, key=agree.get)
    print(f"[domain] mean IoU with other experts: { {d: round(v, 3) for d, v in agree.items()} } -> using {pivot}")
    save_maps(paths, probs[pivot], os.path.join(out, "zeroshot"))

    # 3. Test-time adaptation (GRAM main_moe_pl_v3.py)
    tile_score = torch.stack([mask_iou(masks[pivot], masks[o]) for o in range(DOMAINS) if o != pivot]).mean(0)
    keep = torch.argsort(tile_score, descending=True)[: int(len(paths) * TOP_RATIO)]
    del probs, masks
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.99)
    losses = []
    g = torch.Generator().manual_seed(0)
    for epoch in range(args.epochs):
        net.train()
        order = keep[torch.randperm(len(keep), generator=g)]
        ep = []
        for i in range(0, len(order), BATCH):  # one optimizer step per GRAM batch of 16
            batch_idx = order[i:i + BATCH]
            opt.zero_grad()
            step_loss = 0.0
            for j in range(0, len(batch_idx), micro):
                xb = to_input(x[batch_idx[j:j + micro]], device)
                d = torch.full((len(xb),), pivot, dtype=torch.long, device=device)
                seg, _, _ = net(xb, d)
                loss = focal_loss(seg, seg.argmax(1).detach()) * len(xb) / len(batch_idx)
                loss.backward()
                step_loss += loss.item()
                del seg, loss, xb
            opt.step()
            ep.append(step_loss)
        losses.append(float(np.mean(ep)))
        print(f"[tta] epoch {epoch + 1}/{args.epochs} loss {losses[-1]:.4f}")

    # 4. Adapted predictions
    adapted = predict(net, x, pivot, device, batch=pred_batch)
    save_maps(paths, adapted, os.path.join(out, "tta"))
    log = {"tiles": len(paths), "device": str(device), "gpus": torch.cuda.device_count(),
           "domain_agreement": agree, "pivot_domain": pivot, "tta_tiles": len(keep),
           "epochs": args.epochs, "batch": BATCH, "micro_batch": micro, "accum_steps": accum, "lr": LR, "top_ratio": TOP_RATIO, "epoch_loss": losses,
           "adapted_slum_share": float((adapted >= 128).float().mean()), "minutes": (time.time() - t0) / 60}
    with open(os.path.join(out, "log.json"), "w") as f:
        json.dump(log, f, indent=2)
    shutil.make_archive(out, "zip", out)
    print(f"[done] {log['minutes']:.1f} min -> {out}.zip")


if __name__ == "__main__":
    main()
