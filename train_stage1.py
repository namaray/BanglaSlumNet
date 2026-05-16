import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np

from sasnet import StructureEncoder, AppearanceEncoder, AdaIN
from stage1_gan import PatchGANDiscriminator, SASNetLoss


# ==========================================
# CONFIG
# ==========================================
DATA_DIR  = os.path.join(os.getcwd(), "dhaka_dataset")
CKPT_DIR  = os.path.join(os.getcwd(), "checkpoints_stage1")
os.makedirs(CKPT_DIR, exist_ok=True)

BATCH_SIZE   = 1
EPOCHS       = 50
LR           = 1e-4
SAVE_EVERY   = 5


# ==========================================
# 1. DATASET
# ==========================================
class PairedSlumDataset(Dataset):
    """
    Loads hazy/clear pairs. Expects per-tile files:
      {tile_id}_hazy.tif
      {tile_id}_clear.tif
    Both must exist for a tile to be included.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        hazy_files = sorted(glob.glob(os.path.join(data_dir, "*_hazy.tif")))

        self.tile_ids = []
        for hf in hazy_files:
            tid        = os.path.basename(hf).replace("_hazy.tif", "")
            clear_path = os.path.join(data_dir, f"{tid}_clear.tif")
            if os.path.exists(clear_path):
                self.tile_ids.append(tid)

        if len(self.tile_ids) == 0:
            raise ValueError(
                f"No complete hazy/clear pairs found in {data_dir}.\n"
                "Run download_dhaka_dataset.py first."
            )
        print(f"✅ PairedSlumDataset: {len(self.tile_ids)} pairs loaded.")

    def __len__(self):
        return len(self.tile_ids)

    def _load_tif(self, path):
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)[:, :512, :512]
        img = np.clip(img / 3000.0, 0.0, 1.0)
        return torch.from_numpy(img)

    def __getitem__(self, idx):
        tid   = self.tile_ids[idx]
        hazy  = self._load_tif(os.path.join(self.data_dir, f"{tid}_hazy.tif"))
        clear = self._load_tif(os.path.join(self.data_dir, f"{tid}_clear.tif"))
        return hazy, clear


# ==========================================
# 2. GENERATOR
# ==========================================
class ImageDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 4, 1),
            nn.Sigmoid()   # output range 0-1, same as normalised satellite input
        )

    def forward(self, x):
        return self.up(x)


class Stage1Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.struct_enc = StructureEncoder()
        self.appear_enc = AppearanceEncoder()
        self.adain      = AdaIN()
        self.decoder    = ImageDecoder()

    def forward(self, hazy_img, clear_img):
        s_hazy  = self.struct_enc(hazy_img)    # structure from smoggy image
        s_clear = self.struct_enc(clear_img)   # structure from clear image (for L_scene)
        a_clear = self.appear_enc(clear_img)   # appearance from clear image
        fused        = self.adain(s_hazy, a_clear)
        fake_clear   = self.decoder(fused)
        return fake_clear, s_hazy, s_clear


# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Stage 1 Training  |  device={device}\n")

    dataset    = PairedSlumDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    G         = Stage1Generator().to(device)
    D         = PatchGANDiscriminator().to(device)
    criterion = SASNetLoss().to(device)

    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.9, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.9, 0.999))

    # Cosine decay — G and D on same schedule
    sched_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=EPOCHS, eta_min=1e-6)
    sched_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=EPOCHS, eta_min=1e-6)

    bce_loss = nn.BCEWithLogitsLoss()
    best_g_loss = float("inf")

    print("🔥 Starting Stage 1 Training Loop...\n")

    for epoch in range(1, EPOCHS + 1):
        G.train()
        D.train()

        epoch_loss_G = epoch_loss_D = 0.0
        epoch_rec = epoch_adv = epoch_sce = 0.0
        n = 0

        for hazy, clear in dataloader:
            hazy  = hazy.to(device)
            clear = clear.to(device)

            # ---- Train Discriminator ----
            opt_D.zero_grad()

            with torch.no_grad():
                fake_clear, _, _ = G(hazy, clear)

            pred_real = D(clear)
            pred_fake = D(fake_clear)

            loss_D = 0.5 * (
                bce_loss(pred_real, torch.ones_like(pred_real)) +
                bce_loss(pred_fake, torch.zeros_like(pred_fake))
            )
            loss_D.backward()
            opt_D.step()

            # ---- Train Generator ----
            opt_G.zero_grad()

            fake_clear, s_hazy, s_clear = G(hazy, clear)
            pred_fake_for_G             = D(fake_clear)

            loss_G, l_rec, l_adv, l_sce = criterion(
                fake_clear, clear, s_hazy, s_clear, pred_fake_for_G
            )
            loss_G.backward()
            opt_G.step()

            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()
            epoch_rec    += l_rec.item()
            epoch_adv    += l_adv.item()
            epoch_sce    += l_sce.item()
            n += 1

        sched_G.step()
        sched_D.step()

        avg_G   = epoch_loss_G / n
        avg_D   = epoch_loss_D / n
        avg_rec = epoch_rec / n
        avg_adv = epoch_adv / n
        avg_sce = epoch_sce / n

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}]  "
            f"D={avg_D:.4f}  G={avg_G:.4f}  "
            f"Rec={avg_rec:.4f}  Adv={avg_adv:.4f}  Sce={avg_sce:.4f}"
        )

        # Save best checkpoint (tracked by G loss — D checkpoint not needed for Stage 2)
        if avg_G < best_g_loss:
            best_g_loss = avg_G
            torch.save(
                G.struct_enc.state_dict(),
                os.path.join(CKPT_DIR, "best_sasnet.pth")
            )
            print(f"  💾 Best StructureEncoder saved  (G_loss={best_g_loss:.4f})")

        if epoch % SAVE_EVERY == 0:
            torch.save(G.state_dict(), os.path.join(CKPT_DIR, f"G_epoch{epoch:03d}.pth"))
            torch.save(D.state_dict(), os.path.join(CKPT_DIR, f"D_epoch{epoch:03d}.pth"))

    print(f"\n🎉 Stage 1 training complete. Best G loss = {best_g_loss:.4f}")
    print(f"   StructureEncoder checkpoint: {CKPT_DIR}/best_sasnet.pth")
    print("   Next step: run train_stage2.py 🎯")


if __name__ == "__main__":
    train()