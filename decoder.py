import torch
import torch.nn as nn

class SegmentationDecoder(nn.Module):
    def __init__(self, in_channels=256):
        super(SegmentationDecoder, self).__init__()
        
        # Block 1: 64x64 -> 128x128
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Block 2: 128x128 -> 256x256
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Block 3: 256x256 -> 512x512
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Final Output Layer: Compress 32 channels into 1 channel (Slum Probability)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()  # Squashes values between 0.0 (Not Slum) and 1.0 (Slum)
        )

    def forward(self, F):
        # F is your fused feature map from Stage 2[Batch, 256, 64, 64]
        d1 = self.up1(F)    #[Batch, 128, 128, 128]
        d2 = self.up2(d1)   #[Batch, 64, 256, 256]
        d3 = self.up3(d2)   # [Batch, 32, 512, 512]
        out = self.final_conv(d3) #[Batch, 1, 512, 512]
        return out

# --- THE GRAND FINALE: END-TO-END TEST ---
if __name__ == "__main__":
    from sasnet import StructureEncoder
    from stage2_fusion import CrossAttentionFusion
    
    print("🚀 Initializing the FULL BanglaSlumNet Pipeline...")
    
    # 1. Initialize all models
    encoder = StructureEncoder()
    fusion = CrossAttentionFusion()
    decoder = SegmentationDecoder()
    
    # 2. Create Dummy Data (Simulating a real dataloader batch)
    print("\nLoading dummy satellite and socioeconomic data...")
    dummy_sentinel = torch.rand(1, 4, 512, 512)
    dummy_SE =[torch.rand(1, 1, 512, 512) for _ in range(5)]
    
    # 3. THE FORWARD PASS
    print("\nRunning Forward Pass...")
    
    # Stage 1: Disentanglement
    print(" -> Running Stage 1 (SAS-Net Encoder)...")
    s_m = encoder(dummy_sentinel)
    
    # Stage 2: Fusion
    print(" -> Running Stage 2 (Cross-Attention)...")
    F = fusion(s_m, dummy_SE)
    
    # Decoder: Final Mask
    print(" -> Running Decoder...")
    prediction_mask = decoder(F)
    
    print("\n==============================================")
    print(f"🎉 FINAL PREDICTION SHAPE: {prediction_mask.shape}")
    print(f"Min probability: {prediction_mask.min().item():.4f}")
    print(f"Max probability: {prediction_mask.max().item():.4f}")
    print("==============================================")