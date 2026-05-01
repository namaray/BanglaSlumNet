import torch
import torch.nn as nn

# ==========================================
# 1. SOCIOECONOMIC CHANNEL ENCODER
# ==========================================
class SEChannelEncoder(nn.Module):
    """
    Takes ONE socioeconomic raster (e.g., Nighttime Lights) of size 512x512
    and shrinks it down to 64x64 with 64 feature channels.
    """
    def __init__(self):
        super(SEChannelEncoder, self).__init__()
        # 3 layers of Conv->BN->ReLU with Stride 2 to shrink 512 -> 256 -> 128 -> 64
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. CROSS-ATTENTION FUSION
# ==========================================
class CrossAttentionFusion(nn.Module):
    def __init__(self, num_se_channels=5, d_model=256, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        
        # We need 5 independent encoders (NTL, Pop, GOB/OSM, Poverty, Kilns)
        self.se_encoders = nn.ModuleList([SEChannelEncoder() for _ in range(num_se_channels)])
        
        # After concatenating the 5 encoders (5 * 64 = 320), we project it to 256 to match s_m
        self.se_project = nn.Conv2d(num_se_channels * 64, d_model, kernel_size=1)
        
        # The Multi-Head Attention Mechanism
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

    def forward(self, S, se_inputs):
        """
        S: The clean structure code from SAS-Net [Batch, 256, 64, 64]
        se_inputs: List of 5 tensors, each [Batch, 1, 512, 512]
        """
        b, c, h, w = S.shape
        
        # 1. Encode all 5 socioeconomic channels independently
        encoded_se =[]
        for i in range(len(self.se_encoders)):
            encoded_se.append(self.se_encoders[i](se_inputs[i]))
            
        # 2. Concatenate them all together and project to d_model (256)
        E = torch.cat(encoded_se, dim=1) # Shape: [Batch, 320, 64, 64]
        E = self.se_project(E)           # Shape:[Batch, 256, 64, 64]
        
        # 3. Reshape for PyTorch's Attention layer
        # Attention expects sequences, so we flatten the 64x64 grid into 4096 "pixels"
        # Shape becomes [Batch, 4096, 256]
        S_flat = S.view(b, c, -1).permute(0, 2, 1) 
        E_flat = E.view(b, c, -1).permute(0, 2, 1)
        
        # 4. CROSS ATTENTION! 
        # Query = Structure (What am I looking at?)
        # Key/Value = Socioeconomic (Is this a slum?)
        attn_output, _ = self.multihead_attn(query=S_flat, key=E_flat, value=E_flat)
        
        # 5. Reshape back to image format[Batch, 256, 64, 64]
        attn_output = attn_output.permute(0, 2, 1).view(b, c, h, w)
        
        # 6. RESIDUAL ADDITION (As strictly specified in your blueprint equation)
        # F = CrossAttention(S, E) + S
        F = attn_output + S
        
        return F

# --- Let's test the Fusion! ---
if __name__ == "__main__":
    print("Initializing Cross-Attention Fusion Module...")
    fusion_module = CrossAttentionFusion(num_se_channels=5)
    
    # Fake structure code (coming from your SAS-Net)
    dummy_S = torch.rand(1, 256, 64, 64)
    
    # Fake Socioeconomic Rasters (5 separate maps of 512x512)
    dummy_NTL = torch.rand(1, 1, 512, 512)
    dummy_Pop = torch.rand(1, 1, 512, 512)
    dummy_GOB = torch.rand(1, 1, 512, 512)
    dummy_Poverty = torch.rand(1, 1, 512, 512)
    dummy_Kiln = torch.rand(1, 1, 512, 512)
    
    se_inputs =[dummy_NTL, dummy_Pop, dummy_GOB, dummy_Poverty, dummy_Kiln]
    
    print("\nFusing Visual Structure with Socioeconomic Context...")
    F = fusion_module(dummy_S, se_inputs)
    
    print(f"✅ Fused Feature Map (F) Shape: {F.shape}")