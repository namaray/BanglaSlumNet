import torch
import torch.nn as nn
import torchvision.models as models

# ==========================================
# 1. STRUCTURE ENCODER (The Physical Buildings)
# ==========================================
class StructureEncoder(nn.Module):
    def __init__(self, in_channels=4):
        super(StructureEncoder, self).__init__()
        
        effnet = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        
        # Replace the first layer for 4-channel input (RGB + NIR)
        original_first_conv = effnet.features[0][0]
        self.first_conv = nn.Conv2d(in_channels=in_channels, 
                                    out_channels=original_first_conv.out_channels, 
                                    kernel_size=original_first_conv.kernel_size, 
                                    stride=original_first_conv.stride, 
                                    padding=original_first_conv.padding, 
                                    bias=False)
        
        with torch.no_grad():
            self.first_conv.weight[:, :3] = original_first_conv.weight
            self.first_conv.weight[:, 3] = original_first_conv.weight.mean(dim=1)
            
        # FIX: We only take the first 4 blocks of EfficientNet!
        # This stops the compression at Stride 8 (64x64 resolution) instead of Stride 32 (16x16)
        self.backbone = nn.Sequential(*list(effnet.features.children())[1:4])
        
        # At block 4, EffNet-B2 has 48 channels. We project it to 256 as per the blueprint.
        self.adapter = nn.Conv2d(48, 256, kernel_size=1)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.backbone(x)
        s_m = self.adapter(x)
        return s_m

# ==========================================
# 2. APPEARANCE ENCODER (The Smog / Haze)
# ==========================================
class AppearanceEncoder(nn.Module):
    def __init__(self, in_channels=4):
        super(AppearanceEncoder, self).__init__()
        # Blueprint: 4x strided Conv3x3 + Global AvgPool + FC(32)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU()
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 32) # Outputs a 32-dim vector

    def forward(self, x):
        x = self.convs(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        a_m = self.fc(x)
        return a_m

# ==========================================
# 3. AdaIN (Adaptive Instance Normalization)
# ==========================================
class AdaIN(nn.Module):
    def __init__(self, style_dim=32, num_features=256):
        super(AdaIN, self).__init__()
        # Learns to map the 32-dim appearance vector to the exact mean/std needed to tint the image
        self.fc_gamma = nn.Linear(style_dim, num_features)
        self.fc_beta = nn.Linear(style_dim, num_features)

    def forward(self, structure, appearance):
        # 1. Normalize the structure (strip away any existing weather/style)
        b, c, h, w = structure.size()
        structure_view = structure.view(b, c, -1)
        mean = structure_view.mean(dim=2, keepdim=True).unsqueeze(3)
        std = structure_view.std(dim=2, keepdim=True).unsqueeze(3) + 1e-5
        normalized_structure = (structure - mean) / std
        
        # 2. Generate the new weather/style multipliers from the appearance code
        gamma = self.fc_gamma(appearance).view(b, c, 1, 1)
        beta = self.fc_beta(appearance).view(b, c, 1, 1)
        
        # 3. Apply the new style!
        styled_output = normalized_structure * gamma + beta
        return styled_output

# --- Let's test the whole SAS-Net Encoder combo! ---
if __name__ == "__main__":
    dummy_image = torch.rand(1, 4, 512, 512)
    
    struct_enc = StructureEncoder()
    appear_enc = AppearanceEncoder()
    adain = AdaIN(style_dim=32, num_features=256)
    
    # 1. Extract Structure (Buildings)
    s_m = struct_enc(dummy_image)
    print(f"✅ Structure Code Shape: {s_m.shape} (Blueprint says H/8 x W/8 x 256 -> 64x64)")
    
    # 2. Extract Appearance (Smog)
    a_m = appear_enc(dummy_image)
    print(f"✅ Appearance Code Shape: {a_m.shape} (Blueprint says 32-dim vector)")
    
    # 3. Mix them together!
    recombined = adain(s_m, a_m)
    print(f"✅ Recombined AdaIN Shape: {recombined.shape}")