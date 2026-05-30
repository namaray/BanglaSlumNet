"""
Optical-only baseline: SegFormer-B0 or ResNet-UNet on clean RGB only.
No language, no socioeconomics. This is our controlled analogue of the GRAM failure —
high recall on Korail, high false positives on Old Dhaka.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResNetUNet(nn.Module):
    """Lightweight ResNet encoder + UNet decoder for binary segmentation."""
    def __init__(self, in_channels: int = 4):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128, stride=2)
        self.enc3 = ConvBlock(128, 256, stride=2)
        self.enc4 = ConvBlock(256, 512, stride=2)
        self.bottleneck = ConvBlock(512, 512)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)

        d4 = self.dec4(torch.cat([self.up4(b), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        return torch.sigmoid(self.head(d2))

    def feature_dim(self) -> int:
        return 512


class SegFormerLite(nn.Module):
    """
    Minimal SegFormer-style model using timm if available, else falls back to ResNetUNet.
    Using B0 variant (3.7M params) for the optical baseline.
    """
    def __init__(self, in_channels: int = 4):
        super().__init__()
        self._use_timm = False
        try:
            import timm
            backbone = timm.create_model(
                "mit_b0", pretrained=True, in_chans=in_channels, features_only=True
            )
            self.backbone = backbone
            # Decode head: simple upsampling from multi-scale features
            self.fuse = nn.Conv2d(256 + 160 + 64 + 32, 256, 1)
            self.head = nn.Sequential(
                nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
            )
            self._use_timm = True
        except Exception:
            self.fallback = ResNetUNet(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_timm:
            return self.fallback(x)

        feats = self.backbone(x)  # list of feature maps at different scales
        # Upsample all to stride-4 resolution and fuse
        target = feats[0].shape[-2:]
        upsampled = [F.interpolate(f, size=target, mode="bilinear", align_corners=False)
                     for f in feats]
        fused = self.fuse(torch.cat(upsampled, dim=1))
        out = self.head(fused)
        # Upsample to 256×256
        out = F.interpolate(out, size=(256, 256), mode="bilinear", align_corners=False)
        return torch.sigmoid(out)


class BaselineCNN(nn.Module):
    """Wrapper: selects backbone by config name."""
    def __init__(self, backbone: str = "segformer_b0", in_channels: int = 4):
        super().__init__()
        if backbone == "segformer_b0":
            self.net = SegFormerLite(in_channels)
        else:
            self.net = ResNetUNet(in_channels)

    def forward(self, rgb: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.net(rgb)


def _smoke_test():
    for backbone in ["segformer_b0", "resnet_unet"]:
        model = BaselineCNN(backbone=backbone, in_channels=4)
        n = sum(p.numel() for p in model.parameters())
        x = torch.rand(2, 4, 256, 256)
        out = model(x)
        assert out.shape == (2, 1, 256, 256), f"Bad baseline output ({backbone}): {out.shape}"
        print(f"baseline_cnn.py smoke test passed ({backbone}, {n:,} params).")


if __name__ == "__main__":
    _smoke_test()
