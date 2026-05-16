import torch
import torch.nn as nn


class SegmentationDecoder(nn.Module):
    def __init__(self, inchannels=256):
        super(SegmentationDecoder, self).__init__()

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(inchannels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.finalconv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, F):
        d1 = self.up1(F)
        d2 = self.up2(d1)
        d3 = self.up3(d2)
        out = self.finalconv(d3)
        return out

    def predict_proba(self, fused_features):
        logits = self.forward(fused_features)
        probs = torch.sigmoid(logits)
        return probs

    def predict_mask(self, fused_features, threshold=0.5):
        probs = self.predict_proba(fused_features)
        return (probs >= threshold).float()


# --- END-TO-END TEST ---
if __name__ == "__main__":
    from sasnet import StructureEncoder
    from stage2_fusion import CrossAttentionFusion

    print("🚀 Initializing the FULL BanglaSlumNet Pipeline...")

    encoder = StructureEncoder()
    fusion = CrossAttentionFusion(num_se_channels=3)
    decoder = SegmentationDecoder()

    print("\nLoading dummy satellite and socioeconomic data...")
    dummy_sentinel = torch.rand(1, 4, 512, 512)
    dummy_se = [torch.rand(1, 1, 512, 512) for _ in range(3)]

    print("\nRunning Forward Pass...")

    print(" -> Running Stage 1 (Structure Encoder)...")
    s_m = encoder(dummy_sentinel)

    print(" -> Running Stage 2 (Cross-Attention Fusion)...")
    fused = fusion(s_m, dummy_se)

    print(" -> Running Decoder...")
    logits = decoder(fused)
    probs = torch.sigmoid(logits)
    mask = (probs >= 0.5).float()

    print("\n==============================================")
    print(f"Logits shape: {logits.shape}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Binary mask shape: {mask.shape}")
    print(f"Min probability: {probs.min().item():.4f}")
    print(f"Max probability: {probs.max().item():.4f}")
    print("==============================================")