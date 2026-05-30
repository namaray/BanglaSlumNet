"""
Lightweight UNet-style segmentation decoder.
Input: fused feature map [B, D, H_f, W_f]
Output: binary slum probability [B, 1, 256, 256]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(16, out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class SegmentationDecoder(nn.Module):
    """
    Upsamples from [B, d_model, H_f, W_f] to [B, 1, 256, 256].
    n_blocks controls depth (3 blocks: 16→32→64→128... may need extra if H_f=16 → 256 needs 4×).
    """

    def __init__(self, in_channels: int = 512, num_upsample_blocks: int = 3):
        super().__init__()
        dims = [in_channels]
        for _ in range(num_upsample_blocks):
            dims.append(max(dims[-1] // 2, 32))

        self.blocks = nn.ModuleList([
            UpBlock(dims[i], dims[i + 1]) for i in range(num_upsample_blocks)
        ])
        self.head = nn.Conv2d(dims[-1], 1, kernel_size=1)

    def forward(self, x: torch.Tensor, target_size: int = 256) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        if x.shape[-1] != target_size:
            x = F.interpolate(x, size=(target_size, target_size), mode="bilinear", align_corners=False)
        return torch.sigmoid(self.head(x))


def _smoke_test():
    dec = SegmentationDecoder(in_channels=128, num_upsample_blocks=3)
    n = sum(p.numel() for p in dec.parameters())
    print(f"Decoder params: {n:,}")
    x = torch.rand(2, 128, 16, 16)
    out = dec(x)
    assert out.shape == (2, 1, 256, 256), f"Bad decoder output: {out.shape}"
    assert (out >= 0).all() and (out <= 1).all()
    print("decoder.py smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
