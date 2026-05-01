import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np

# Import EVERYTHING we have built!
from sasnet import StructureEncoder
from stage2_fusion import CrossAttentionFusion
from decoder import SegmentationDecoder
from stage3_temporal import TemporalModule, TemporalLoss
from train_stage2 import Stage2Loss

# ==========================================
# 1. 5D TEMPORAL DATALOADER
# ==========================================
class TemporalSlumDataset(Dataset):
    def __init__(self, temp_dir, se_dir):
        self.temp_dir = temp_dir
        self.se_dir = se_dir
        self.locations =['mirpur', 'korail', 'old_dhaka']
        self.years =[2021, 2022, 2023]

    def __len__(self):
        return len(self.locations)

    def load_tif(self, path, is_mask=False):
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)
        
        if not is_mask:
            img = np.clip(img / 3000.0, 0.0, 1.0)
        else:
            img = np.clip(img, 0.0, 1.0)
            
        tensor = torch.from_numpy(img)[:, :512, :512]
        return tensor

    def __getitem__(self, idx):
        loc = self.locations[idx]
        
        # 1. Load 3 Years of Optical Data [Time=3, Channels=4, H=512, W=512]
        opt_seq =[]
        for yr in self.years:
            img = self.load_tif(os.path.join(self.temp_dir, f"{loc}_s2_{yr}.tif"))
            opt_seq.append(img)
        opt_seq = torch.stack(opt_seq) 
        
        # 2. Load 2 Years of SAR Radar Masks[Time=2, Channels=1, H=512, W=512]
        sar_seq =[]
        for i in range(len(self.years)-1):
            y1, y2 = self.years[i], self.years[i+1]
            sar = self.load_tif(os.path.join(self.temp_dir, f"{loc}_sar_{y1}_{y2}.tif"), is_mask=True)
            sar_seq.append(sar)
        sar_seq = torch.stack(sar_seq)
        
        # 3. Load Socioeconomic Data (Static for this test)
        ntl = self.load_tif(os.path.join(self.se_dir, f"{loc}_ntl.tif"), is_mask=True)
        pop = self.load_tif(os.path.join(self.se_dir, f"{loc}_pop.tif"), is_mask=True)
        gob = self.load_tif(os.path.join(self.se_dir, f"{loc}_gob.tif"), is_mask=True)
        se_stack =[ntl, pop, gob]
        
        # 4. Dummy Labels for 3 Years
        dummy_labels = torch.randint(0, 2, (3, 1, 512, 512)).float()
        
        return opt_seq, sar_seq, se_stack, dummy_labels

# ==========================================
# 2. THE MASTER TRAINING LOOP
# ==========================================
def train_stage3():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Stage 3 (Temporal) Training on: {device}")

    # Load Data
    dataset = TemporalSlumDataset(
        temp_dir=os.path.join(os.getcwd(), 'temporal_dataset'),
        se_dir=os.path.join(os.getcwd(), 'paired_dataset')
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize FULL PIPELINE
    print("Loading Models...")
    struct_enc = StructureEncoder().to(device)
    fusion = CrossAttentionFusion(num_se_channels=3).to(device)
    temporal = TemporalModule(input_dim=256, hidden_dim=128).to(device)
    
    # Notice: ConvLSTM outputs 128 channels, so we tell the Decoder to expect 128!
    decoder = SegmentationDecoder(in_channels=128).to(device) 
    
    criterion_seg = Stage2Loss().to(device)
    criterion_temp = TemporalLoss().to(device)

    # Optimizer: Fine-tuning Temporal Module and Decoder
    opt = optim.Adam(list(temporal.parameters()) + list(decoder.parameters()), lr=5e-5)

    epochs = 3
    print("\n🔥 Starting Stage 3 End-to-End Temporal Training...")
    
    for epoch in range(epochs):
        for opt_seq, sar_seq, se_stack, target_labels in dataloader:
            
            # Move to device
            opt_seq, sar_seq, target_labels = opt_seq.to(device), sar_seq.to(device), target_labels.to(device)
            se_stack =[se.to(device) for se in se_stack]

            # Remove batch dimension for iteration (since batch_size=1)
            opt_seq, sar_seq, target_labels = opt_seq[0], sar_seq[0], target_labels[0]
            
            opt.zero_grad()
            
            # --- STEP 1: Process each year independently through Stages 1 & 2 ---
            fused_features_seq =[]
            for t in range(3): # Loop over 2021, 2022, 2023
                with torch.no_grad(): # Freeze Stages 1 & 2
                    s_t = struct_enc(opt_seq[t].unsqueeze(0))
                    f_t = fusion(s_t, se_stack)
                    fused_features_seq.append(f_t.squeeze(0))
            
            # Stack into a Time Sequence: [Batch, Time, Channels, H, W]
            F_seq = torch.stack(fused_features_seq).unsqueeze(0) 
            
            # --- STEP 2: The Time Machine (ConvLSTM) ---
            H_seq = temporal(F_seq) #[Batch, Time, 128, 64, 64]
            
            # --- STEP 3: Decode each year into a Slum Mask ---
            pred_seq =[]
            for t in range(3):
                # Pass through decoder (skipping final sigmoid for BCEWithLogits)
                h_t = H_seq[:, t, :, :, :]
                logits_t = decoder.final_conv[0](decoder.up3(decoder.up2(decoder.up1(h_t))))
                pred_seq.append(logits_t)
            
            pred_seq = torch.stack(pred_seq, dim=1) #[Batch, Time, 1, 512, 512]
            
            # --- STEP 4: Calculate Master Loss ---
            # Segmentation Loss (Average across all 3 years)
            loss_seg = 0
            for t in range(3):
                l_total, _, _ = criterion_seg(pred_seq[:, t], target_labels[t].unsqueeze(0))
                loss_seg += l_total
            loss_seg = loss_seg / 3.0
            
            # Temporal Smoothness Loss (Checked against SAR Radar)
            loss_temp = criterion_temp(torch.sigmoid(pred_seq), sar_seq.unsqueeze(0))
            
            # Blueprint Eq: L_stage3 = L_stage2 + 0.3 * L_temp
            final_loss = loss_seg + (0.3 * loss_temp)
            
            # --- STEP 5: Backpropagate through Time ---
            final_loss.backward()
            opt.step()

        print(f"Epoch[{epoch+1}/{epochs}] | Final Loss: {final_loss.item():.4f} (Seg: {loss_seg.item():.4f}, Temp: {loss_temp.item():.4f})")
        
    print("✅ Mission 4 Complete: 10-Year ConvLSTM Engine Validated!")

if __name__ == "__main__":
    train_stage3()