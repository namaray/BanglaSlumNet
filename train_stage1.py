import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np

# Import the modules we already built!
from sasnet import StructureEncoder, AppearanceEncoder, AdaIN
from stage1_gan import PatchGANDiscriminator, SASNetLoss

# ==========================================
# 1. PAIRED DATASET HANDLER
# ==========================================
class PairedSlumDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Find all the 'hazy' files
        self.hazy_files =[f for f in os.listdir(data_dir) if 'hazy.tif' in f]

    def __len__(self):
        return len(self.hazy_files)

    def load_tif(self, path):
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)
        img = img / 3000.0
        img = np.clip(img, 0.0, 1.0)
        tensor = torch.from_numpy(img)[:, :512, :512] # Crop to 512x512
        return tensor

    def __getitem__(self, idx):
        hazy_name = self.hazy_files[idx]
        clear_name = hazy_name.replace('hazy', 'clear')
        
        hazy_tensor = self.load_tif(os.path.join(self.data_dir, hazy_name))
        clear_tensor = self.load_tif(os.path.join(self.data_dir, clear_name))
        return hazy_tensor, clear_tensor

# ==========================================
# 2. STAGE 1 GENERATOR (SAS-Net Wrapper)
# ==========================================
class ImageDecoder(nn.Module):
    """Upsamples the 64x64 feature map back to a 512x512, 4-channel satellite image."""
    def __init__(self):
        super(ImageDecoder, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 4, 1), nn.Sigmoid() # 4 channels for RGB+NIR
        )
    def forward(self, x):
        return self.up(x)

class Stage1Generator(nn.Module):
    def __init__(self):
        super(Stage1Generator, self).__init__()
        self.struct_enc = StructureEncoder()
        self.appear_enc = AppearanceEncoder()
        self.adain = AdaIN()
        self.decoder = ImageDecoder()

    def forward(self, hazy_img, clear_img):
        s_hazy = self.struct_enc(hazy_img)    # Extract buildings from smog
        s_clear = self.struct_enc(clear_img)  # Extract buildings from clear (for L_scene loss)
        a_clear = self.appear_enc(clear_img)  # Extract clean weather
        
        # Mix hazy buildings with clean weather!
        fused = self.adain(s_hazy, a_clear)
        fake_clear_img = self.decoder(fused)
        return fake_clear_img, s_hazy, s_clear

# ==========================================
# 3. THE TRAINING LOOP
# ==========================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on: {device}")

    # 1. Load Data
    dataset = PairedSlumDataset(data_dir=os.path.join(os.getcwd(), 'paired_dataset'))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"📦 Loaded {len(dataset)} paired images.")

    # 2. Initialize Models
    G = Stage1Generator().to(device)
    D = PatchGANDiscriminator().to(device)
    criterion = SASNetLoss().to(device)

    # 3. Optimizers (As specified in Blueprint Appendix A)
    opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.9, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.9, 0.999))

    epochs = 5 # Just a quick test to see if it works!

    print("\n🔥 Starting Training Loop...")
    for epoch in range(epochs):
        for i, (hazy, clear) in enumerate(dataloader):
            hazy, clear = hazy.to(device), clear.to(device)

            # ---------------------
            # Train Discriminator
            # ---------------------
            opt_D.zero_grad()
            fake_clear, _, _ = G(hazy, clear)
            
            # Predict on Real & Fake
            pred_real = D(clear)
            pred_fake = D(fake_clear.detach())
            
            # D Loss: 1.0 for Real, 0.0 for Fake
            loss_D_real = nn.BCEWithLogitsLoss()(pred_real, torch.ones_like(pred_real))
            loss_D_fake = nn.BCEWithLogitsLoss()(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            
            loss_D.backward()
            opt_D.step()

            # ---------------------
            # Train Generator (SAS-Net)
            # ---------------------
            opt_G.zero_grad()
            # We want D to think our fake image is real!
            pred_fake_for_G = D(fake_clear)
            _, s_hazy, s_clear = G(hazy, clear)
            
            # Use your custom SASNetLoss module
            loss_G, l_rec, l_adv, l_sce = criterion(fake_clear, clear, s_hazy, s_clear, pred_fake_for_G)
            
            loss_G.backward()
            opt_G.step()

        # Print stats at the end of each epoch
        print(f"Epoch[{epoch+1}/{epochs}] | D_Loss: {loss_D.item():.4f} | G_Loss: {loss_G.item():.4f} (Rec:{l_rec.item():.4f}, Adv:{l_adv.item():.4f}, Sce:{l_sce.item():.4f})")
        
    print("✅ Mission 2 Complete: SAS-Net trained successfully!")

if __name__ == "__main__":
    train()