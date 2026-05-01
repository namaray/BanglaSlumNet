import torch
import torch.nn as nn

# ==========================================
# 1. THE PATCHGAN DISCRIMINATOR
# ==========================================
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=4):
        super(PatchGANDiscriminator, self).__init__()
        
        # A 4-layer Convolutional Network that outputs a "grid" of True/False scores 
        # instead of just one single score for the whole image.
        self.model = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4 (Blueprint specifies 4 Conv layers)
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output Layer (1 channel logit map)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# ==========================================
# 2. THE BLUEPRINT LOSS FUNCTIONS (Section 2.5)
# ==========================================
class SASNetLoss(nn.Module):
    def __init__(self, lambda_adv=0.1, lambda_scene=1.0):
        super(SASNetLoss, self).__init__()
        self.lambda_adv = lambda_adv
        self.lambda_scene = lambda_scene
        
        # Loss Metrics
        self.mse_loss = nn.MSELoss()              # For L_recon
        self.l1_loss = nn.L1Loss()                # For L_scene
        self.bce_logits = nn.BCEWithLogitsLoss()  # For L_adv

    def forward(self, fake_clear_img, real_clear_img, s_hazy, s_clear, disc_pred_fake):
        # 1. L_recon (Pixel MSE Reconstruction)
        # Does the generated image look EXACTLY like the real clear image pixel-by-pixel?
        L_recon = self.mse_loss(fake_clear_img, real_clear_img)
        
        # 2. L_scene (Structure Invariance L1)
        # The physical buildings (s_m) must remain identical whether it's hazy or clear!
        L_scene = self.l1_loss(s_hazy, s_clear)
        
        # 3. L_adv (PatchGAN Adversarial Loss)
        # Trick the discriminator into thinking the fake image is real (Target = 1.0)
        target_real = torch.ones_like(disc_pred_fake)
        L_adv = self.bce_logits(disc_pred_fake, target_real)
        
        # 4. Total Stage 1 Loss (Blueprint Eq: L_recon + λ1*L_adv + λ2*L_scene)
        L_stage1 = L_recon + (self.lambda_adv * L_adv) + (self.lambda_scene * L_scene)
        
        return L_stage1, L_recon, L_adv, L_scene

# --- Let's test the Math! ---
if __name__ == "__main__":
    print("Initializing PatchGAN Discriminator and Loss Functions...")
    discriminator = PatchGANDiscriminator(in_channels=4)
    loss_calculator = SASNetLoss()
    
    # Fake tensors simulating the training loop
    fake_clear = torch.rand(1, 4, 512, 512)
    real_clear = torch.rand(1, 4, 512, 512)
    
    s_hazy = torch.rand(1, 256, 64, 64) # Structure extracted from Hazy image
    s_clear = torch.rand(1, 256, 64, 64) # Structure extracted from Clear image
    
    print("\nRunning Discriminator...")
    disc_output = discriminator(fake_clear)
    print(f"✅ Discriminator Output Shape: {disc_output.shape} (Should be a downscaled grid map)")
    
    print("\nCalculating Loss...")
    total_loss, l_rec, l_adv, l_sce = loss_calculator(fake_clear, real_clear, s_hazy, s_clear, disc_output)
    
    print(f"✅ L_recon Loss: {l_rec.item():.4f}")
    print(f"✅ L_scene Loss: {l_sce.item():.4f}")
    print(f"✅ L_adv Loss:   {l_adv.item():.4f}")
    print(f"🔥 TOTAL LOSS:   {total_loss.item():.4f}")