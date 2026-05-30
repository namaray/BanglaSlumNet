"""
Stage 1 — SAS-Net: Scene-Appearance Separation.
I = R(E_s(I), E_a(I)) + n
Disentangles structural content (E_s) from atmospheric/seasonal appearance (E_a)
via AdaIN-based rendering. Scene consistency loss forces E_s to be date-invariant.

~3.35M parameters total.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1), nn.InstanceNorm2d(dim), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1), nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.net(x)


class AdaIN(nn.Module):
    """Adaptive Instance Normalization: normalizes content, then shifts by style."""
    def forward(self, content: torch.Tensor, style_mean: torch.Tensor, style_std: torch.Tensor):
        B, C, H, W = content.shape
        mean = content.mean(dim=[2, 3], keepdim=True)
        std = content.std(dim=[2, 3], keepdim=True) + 1e-5
        normalized = (content - mean) / std
        style_mean = style_mean.view(B, C, 1, 1)
        style_std = style_std.view(B, C, 1, 1)
        return normalized * style_std + style_mean


class SceneEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, dim: int = 256, n_res: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, padding=3), nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, dim, 3, stride=2, padding=1), nn.InstanceNorm2d(dim), nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(*[ResBlock(dim) for _ in range(n_res)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_blocks(self.stem(x))


class AppearanceEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, style_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(in_channels * 64, 256), nn.ReLU(inplace=True),
            nn.Linear(256, style_dim * 2),
        )
        self.style_dim = style_dim

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        mean, std = out[:, :self.style_dim], F.softplus(out[:, self.style_dim:]) + 1e-5
        return mean, std


class Renderer(nn.Module):
    def __init__(self, dim: int = 256, out_channels: int = 4, n_res: int = 2):
        super().__init__()
        self.adain = AdaIN()
        self.res_blocks = nn.Sequential(*[ResBlock(dim) for _ in range(n_res)])
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(dim, 128, 4, stride=2, padding=1), nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 7, padding=3), nn.Sigmoid(),
        )

    def forward(self, scene: torch.Tensor, style_mean: torch.Tensor, style_std: torch.Tensor):
        x = self.adain(scene, style_mean, style_std)
        x = self.res_blocks(x)
        return self.upsample(x)


class SASNet(nn.Module):
    """
    Scene-Appearance Separation Network.
    Input: S2 tile [B, C_in, 256, 256]
    Output: clean tile rendered at reference appearance, scene features
    """
    def __init__(
        self,
        in_channels: int = 4,
        encoder_dim: int = 256,
        style_dim: int = 128,
        n_res: int = 4,
    ):
        super().__init__()
        self.scene_encoder = SceneEncoder(in_channels, encoder_dim, n_res)
        self.appearance_encoder = AppearanceEncoder(in_channels, style_dim)
        self.renderer = Renderer(encoder_dim, in_channels)

        # Fixed reference appearance (zero mean, unit std → normalized appearance)
        self.register_buffer("ref_mean", torch.zeros(style_dim))
        self.register_buffer("ref_std", torch.ones(style_dim))

    def forward(self, x: torch.Tensor, ref_appearance: bool = False):
        """
        x: input tile [B, C, 256, 256]
        ref_appearance: if True, render with reference (clean) appearance
        Returns: reconstructed tile, scene features, (style_mean, style_std)
        """
        scene = self.scene_encoder(x)
        style_mean, style_std = self.appearance_encoder(x)

        if ref_appearance:
            B = x.shape[0]
            sm = self.ref_mean.unsqueeze(0).expand(B, -1)
            ss = self.ref_std.unsqueeze(0).expand(B, -1)
        else:
            sm, ss = style_mean, style_std

        recon = self.renderer(scene, sm, ss)
        return recon, scene, (style_mean, style_std)

    def clean_tile(self, x: torch.Tensor) -> torch.Tensor:
        """Render x at the fixed reference appearance — the Stage 1 output."""
        recon, _, _ = self.forward(x, ref_appearance=True)
        return recon


def _smoke_test():
    net = SASNet(in_channels=4, encoder_dim=128, style_dim=64, n_res=2)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"SASNet params: {n_params:,}")
    x = torch.rand(2, 4, 256, 256)
    recon, scene, (sm, ss) = net(x)
    assert recon.shape == (2, 4, 256, 256), f"Bad recon shape: {recon.shape}"
    assert scene.shape[0] == 2
    clean = net.clean_tile(x)
    assert clean.shape == (2, 4, 256, 256)
    print("sasnet.py smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
