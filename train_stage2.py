import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np

# Import our custom modules
from sasnet import StructureEncoder
from stage2_fusion import CrossAttentionFusion
from decoder import SegmentationDecoder

# ==========================================
# 1. THE LOSS FUNCTION (Blueprint Sec 2.5)
# ==========================================
class Stage2Loss(nn.Module):
    def __init__(self, w_slum=3.0, w_nonslum=1.0):
        super(Stage2Loss, self).__init__()
        # We use a trick in PyTorch: BCEWithLogitsLoss is more stable than normal BCE
        # We apply the class weights (3.0 for slum) to penalize missing a slum 3x more!
        pos_weight = torch.tensor([w_slum / w_nonslum])
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, pred_logits, target_mask):
        # 1. Binary Cross Entropy Loss
        l_bce = self.bce(pred_logits, target_mask)
        
        # 2. Soft Differentiable IoU Loss
        pred_probs = torch.sigmoid(pred_logits)
        intersection = (pred_probs * target_mask).sum(dim=(1,2,3))
        union = (pred_probs + target_mask - (pred_probs * target_mask)).sum(dim=(1,2,3))
        l_iou = 1.0 - (intersection / (union + 1e-6)).mean()
        
        # 3. Total Loss (Blueprint Eq: L_IoU + 0.5 * L_BCE)
        total_loss = l_iou + (0.5 * l_bce)
        return total_loss, l_iou, l_bce

# ==========================================
# 2. MULTI-MODAL DATALOADER
# ==========================================
class MultiModalSlumDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.locations = ['mirpur', 'korail', 'old_dhaka']

    def __len__(self):
        return len(self.locations)

    def load_tif(self, path, is_se=False):
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)
            
        if not is_se:
            # Satellite normalization
            img = img / 3000.0
            img = np.clip(img, 0.0, 1.0)
        else:
            # Socioeconomic normalization (Min-Max scaling trick for safety)
            img_max = img.max() if img.max() > 0 else 1.0
            img = img / img_max
            
        tensor = torch.from_numpy(img)[:, :512, :512]
        return tensor

    def __getitem__(self, idx):
        loc = self.locations[idx]
        
        # 1. Load Visuals
        clear_img = self.load_tif(os.path.join(self.data_dir, f"{loc}_clear.tif"), is_se=False)
        
        # 2. Load Socioeconomics
        ntl = self.load_tif(os.path.join(self.data_dir, f"{loc}_ntl.tif"), is_se=True)
        pop = self.load_tif(os.path.join(self.data_dir, f"{loc}_pop.tif"), is_se=True)
        gob = self.load_tif(os.path.join(self.data_dir, f"{loc}_gob.tif"), is_se=True)
        
        se_stack = [ntl, pop, gob]
        
        # 3. Dummy Ground Truth Label (Random 0s and 1s)
        # In a real run, this would be the GRAM dataset masks
        dummy_label = torch.randint(0, 2, (1, 512, 512)).float()
        
        return clear_img, se_stack, dummy_label

# ==========================================
# 3. THE MASTER STAGE 2 TRAINING LOOP
# ==========================================
def train_stage2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Stage 2 Training on: {device}")

    # Load Data
    dataset = MultiModalSlumDataset(data_dir=os.path.join(os.getcwd(), 'paired_dataset'))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize Models
    struct_enc = StructureEncoder().to(device)
    # We only have 3 SE channels downloaded right now (NTL, Pop, GOB)
    fusion = CrossAttentionFusion(num_se_channels=3).to(device)
    decoder = SegmentationDecoder().to(device)
    
    criterion = Stage2Loss().to(device)

    # Optimizer: We only train Fusion and Decoder! Structure Encoder is frozen (from Stage 1)
    opt = optim.Adam(list(fusion.parameters()) + list(decoder.parameters()), lr=1e-4)

    epochs = 5
    print("\n🔥 Starting Stage 2 Fusion Training Loop...")
    
    for epoch in range(epochs):
        for clear_img, se_stack, target_mask in dataloader:
            clear_img = clear_img.to(device)
            se_stack = [se.to(device) for se in se_stack]
            target_mask = target_mask.to(device)

            opt.zero_grad()
            
            # 1. Extract Visual Structure (No gradients needed here)
            with torch.no_grad():
                s_clear = struct_enc(clear_img)
            
            # 2. Cross-Attention Fusion
            F = fusion(s_clear, se_stack)
            
            # 3. Decode to Slum Mask (We remove the final Sigmoid in decoder to use BCEWithLogits)
            # A quick hack: pass through the first layer of the final conv, skipping sigmoid
            pred_logits = decoder.final_conv[0](decoder.up3(decoder.up2(decoder.up1(F))))
            
            # 4. Calculate Loss
            loss, l_iou, l_bce = criterion(pred_logits, target_mask)
            
            # 5. Backpropagate
            loss.backward()
            opt.step()

        print(f"Epoch[{epoch+1}/{epochs}] | Total Loss: {loss.item():.4f} (IoU:{l_iou.item():.4f}, BCE:{l_bce.item():.4f})")
        
    print("✅ Mission 3 Complete: Cross-Attention Fusion trained successfully!")

if __name__ == "__main__":
    train_stage2()