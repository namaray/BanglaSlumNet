import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import rasterio
import numpy as np

from sasnet import StructureEncoder
from stage2_fusion import CrossAttentionFusion
from decoder import SegmentationDecoder


# ==========================================
# CONFIG
# ==========================================
DATA_DIR = os.path.join(os.getcwd(), "dhaka_dataset")
CKPT_DIR = os.path.join(os.getcwd(), "checkpoints_stage2")
os.makedirs(CKPT_DIR, exist_ok=True)

STRUCT_CKPT = os.path.join(os.getcwd(), "checkpoints_stage1", "best_sasnet.pth")

BATCH_SIZE = 1       # keep at 1 for CPU; raise to 2-4 if you have a GPU
EPOCHS = 2           # set to 30 for a real training run
LR = 1e-4
VAL_SPLIT = 0.2
IGNORE_INDEX = 255
W_SLUM = 3.0
SAVE_EVERY = 5
SEED = 42
ATTN_SIZE = 16       # attention grid size; lower = less memory (try 8 if OOM)


# ==========================================
# DATASET
# ==========================================
class WeakLabelDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        clear_files = sorted(glob.glob(os.path.join(data_dir, "*_clear.tif")))
        self.tile_ids = []

        for cf in clear_files:
            tid = os.path.basename(cf).replace("_clear.tif", "")
            required = [
                f"{tid}_clear.tif",
                f"{tid}_ntl.tif",
                f"{tid}_pop.tif",
                f"{tid}_gob.tif",
                f"{tid}_label.tif",
                f"{tid}_hc_mask.tif",
            ]
            if all(os.path.exists(os.path.join(data_dir, f)) for f in required):
                self.tile_ids.append(tid)

        if len(self.tile_ids) == 0:
            raise RuntimeError(
                f"No complete tiles found in {data_dir}. "
                "Run dataset download and weak-label generation first."
            )

        print(f"✅ Dataset: {len(self.tile_ids)} complete tiles found.")

    def __len__(self):
        return len(self.tile_ids)

    def _load(self, path):
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)
        return arr[:, :512, :512]

    def __getitem__(self, idx):
        tid = self.tile_ids[idx]
        p = lambda suffix: os.path.join(self.data_dir, f"{tid}_{suffix}.tif")

        # Satellite image — clamp -inf/inf/nan before normalising
        clear_raw = self._load(p("clear"))
        clear_raw = np.nan_to_num(clear_raw, nan=0.0, posinf=3000.0, neginf=0.0)
        clear = np.clip(clear_raw / 3000.0, 0.0, 1.0)

        # Socioeconomic rasters — kill bad pixels then min-max normalise
        def norm_aux(arr):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            vmax = float(arr.max())
            return arr / vmax if vmax > 0 else arr

        ntl = norm_aux(self._load(p("ntl")))
        pop = norm_aux(self._load(p("pop")))
        gob = norm_aux(self._load(p("gob")))

        label   = self._load(p("label")).astype(np.float32)
        hc_mask = self._load(p("hc_mask")).astype(np.float32)

        return (
            torch.from_numpy(clear),
            [torch.from_numpy(ntl), torch.from_numpy(pop), torch.from_numpy(gob)],
            torch.from_numpy(label),
            torch.from_numpy(hc_mask),
        )


def collate_fn(batch):
    clears, se_stacks, labels, hc_masks = zip(*batch)
    clear_t = torch.stack(clears)
    label_t = torch.stack(labels)
    hc_t    = torch.stack(hc_masks)
    se_t = [
        torch.stack([se_stacks[b][c] for b in range(len(se_stacks))])
        for c in range(3)
    ]
    return clear_t, se_t, label_t, hc_t


# ==========================================
# LOSS
# ==========================================
class Stage2Loss(nn.Module):
    def __init__(self, w_slum=W_SLUM, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.ignore_index = ignore_index
        self.register_buffer("pos_weight", torch.tensor([w_slum], dtype=torch.float32))
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight,
            reduction="none"
        )

    def forward(self, logits, target):
        valid_mask  = (target != self.ignore_index).float()

        target_clean = target.clone()
        target_clean[target == self.ignore_index] = 0.0

        bce_raw = self.bce(logits, target_clean)
        l_bce   = (bce_raw * valid_mask).sum() / (valid_mask.sum() + 1e-6)

        probs    = torch.sigmoid(logits)
        probs_v  = probs * valid_mask
        target_v = target_clean * valid_mask

        intersection = (probs_v * target_v).sum(dim=(1, 2, 3))
        union        = (probs_v + target_v - probs_v * target_v).sum(dim=(1, 2, 3))
        l_iou = (1.0 - (intersection + 1e-6) / (union + 1e-6)).mean()

        total = l_iou + 0.5 * l_bce
        return total, l_iou, l_bce


# ==========================================
# METRICS
# ==========================================
def _stats(pred_binary, label_binary, mask):
    p  = pred_binary[mask]
    t  = label_binary[mask]
    tp = int((p * t).sum().item())
    fp = int((p * (1 - t)).sum().item())
    fn = int(((1 - p) * t).sum().item())
    tn = int(((1 - p) * (1 - t)).sum().item())
    return tp, fp, fn, tn

def _iou(tp, fp, fn):
    return tp / (tp + fp + fn + 1e-6)

def _prec(tp, fp):
    return tp / (tp + fp + 1e-6)

def _rec(tp, fn):
    return tp / (tp + fn + 1e-6)


# ==========================================
# CHECKPOINT LOADING
# ==========================================
def load_structure_encoder(struct_enc, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        print("⚠️ No Stage 1 checkpoint found — StructureEncoder will be trainable.")
        return False

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    if not isinstance(ckpt, dict):
        print("⚠️ Stage 1 checkpoint format not understood — skipping freeze.")
        return False

    cleaned = {}
    for k, v in ckpt.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        cleaned[nk] = v

    if any(k.startswith("structenc.") for k in cleaned.keys()):
        cleaned = {
            k.replace("structenc.", "", 1): v
            for k, v in cleaned.items()
            if k.startswith("structenc.")
        }

    missing, unexpected = struct_enc.load_state_dict(cleaned, strict=False)
    print(f"✅ Loaded Stage 1 weights: {ckpt_path}")
    print(f"   missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")
    return True


# ==========================================
# TRAINING LOOP
# ==========================================
def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Stage 2 Training | device={device}\n")

    full_dataset = WeakLabelDataset(DATA_DIR)
    n_val   = max(1, int(len(full_dataset) * VAL_SPLIT))
    n_train = len(full_dataset) - n_val

    train_set, val_set = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )

    print(f"Train tiles: {n_train} | Val tiles: {n_val}\n")

    struct_enc = StructureEncoder().to(device)
    fusion     = CrossAttentionFusion(num_se_channels=3, attn_size=ATTN_SIZE).to(device)
    decoder    = SegmentationDecoder().to(device)

    loaded_stage1 = load_structure_encoder(struct_enc, STRUCT_CKPT, device)

    if loaded_stage1:
        for param in struct_enc.parameters():
            param.requires_grad = False
        struct_enc.eval()
        print("🔒 StructureEncoder frozen.\n")
        trainable_params = list(fusion.parameters()) + list(decoder.parameters())
    else:
        struct_enc.train()
        print("🛠️ StructureEncoder left trainable.\n")
        trainable_params = (
            list(struct_enc.parameters()) +
            list(fusion.parameters()) +
            list(decoder.parameters())
        )

    optimizer = optim.Adam(trainable_params, lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    criterion = Stage2Loss().to(device)

    best_hc_iou = -1.0

    print("🔥 Starting training...\n")

    for epoch in range(1, EPOCHS + 1):
        if loaded_stage1:
            struct_enc.eval()
        else:
            struct_enc.train()

        fusion.train()
        decoder.train()

        train_loss = train_iou = train_bce = 0.0

        for batch_idx, (clear, se_stack, label, _) in enumerate(train_loader):
            clear    = clear.to(device)
            se_stack = [s.to(device) for s in se_stack]
            label    = label.to(device)

            optimizer.zero_grad()

            if loaded_stage1:
                with torch.no_grad():
                    s_m = struct_enc(clear)
            else:
                s_m = struct_enc(clear)

            f_m    = fusion(s_m, se_stack)
            logits = decoder(f_m)

            loss, l_iou, l_bce = criterion(logits, label)

            # Guard: skip batch if loss is NaN (bad tile slipping through)
            if not torch.isfinite(loss):
                print(f"  ⚠️ NaN/Inf loss at batch {batch_idx}, skipping.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_iou  += l_iou.item()
            train_bce  += l_bce.item()

        n_batches   = len(train_loader)
        train_loss /= n_batches
        train_iou  /= n_batches
        train_bce  /= n_batches

        # ---- Validation ----
        fusion.eval()
        decoder.eval()
        struct_enc.eval()

        hc_tp = hc_fp = hc_fn = hc_tn = 0
        all_tp = all_fp = all_fn = all_tn = 0

        with torch.no_grad():
            for clear, se_stack, label, hc_mask in val_loader:
                clear    = clear.to(device)
                se_stack = [s.to(device) for s in se_stack]
                label    = label.to(device)
                hc_mask  = hc_mask.to(device)

                s_m    = struct_enc(clear)
                f_m    = fusion(s_m, se_stack)
                logits = decoder(f_m)
                pred   = (torch.sigmoid(logits) >= 0.5).float()

                label_bin  = (label == 1).float()
                valid_mask = (label != IGNORE_INDEX)
                hc_eval    = (hc_mask == 1) & valid_mask

                a, b, c, d = _stats(pred, label_bin, hc_eval)
                hc_tp += a; hc_fp += b; hc_fn += c; hc_tn += d

                a, b, c, d = _stats(pred, label_bin, valid_mask)
                all_tp += a; all_fp += b; all_fn += c; all_tn += d

        mean_hc_iou  = _iou(hc_tp, hc_fp, hc_fn)
        mean_all_iou = _iou(all_tp, all_fp, all_fn)
        mean_prec    = _prec(hc_tp, hc_fp)
        mean_rec     = _rec(hc_tp, hc_fn)

        scheduler.step()

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"Loss={train_loss:.4f}  IoU={train_iou:.4f}  BCE={train_bce:.4f}  |  "
            f"HC-IoU={mean_hc_iou:.4f}  All-IoU={mean_all_iou:.4f}  "
            f"P={mean_prec:.4f}  R={mean_rec:.4f}"
        )

        if mean_hc_iou > best_hc_iou:
            best_hc_iou = mean_hc_iou
            torch.save(struct_enc.state_dict(), os.path.join(CKPT_DIR, "best_struct_enc.pth"))
            torch.save(fusion.state_dict(),     os.path.join(CKPT_DIR, "best_fusion.pth"))
            torch.save(decoder.state_dict(),    os.path.join(CKPT_DIR, "best_decoder.pth"))
            print(f"  💾 Best checkpoint saved (HC-IoU={best_hc_iou:.4f})")

        if epoch % SAVE_EVERY == 0:
            if not loaded_stage1:
                torch.save(
                    struct_enc.state_dict(),
                    os.path.join(CKPT_DIR, f"struct_enc_epoch{epoch:03d}.pth")
                )
            torch.save(fusion.state_dict(),   os.path.join(CKPT_DIR, f"fusion_epoch{epoch:03d}.pth"))
            torch.save(decoder.state_dict(),  os.path.join(CKPT_DIR, f"decoder_epoch{epoch:03d}.pth"))

    print(f"\n🎉 Training complete. Best HC-IoU = {best_hc_iou:.4f}")
    print(f"📁 Checkpoints saved to: {CKPT_DIR}")


if __name__ == "__main__":
    train()