"""
BanglaSlumNet assembly: build_model(config) returns the right model composition
for baseline_cnn | vlm_visual | vlm_lang | full.

All four configs share the same decoder and eval code for apples-to-apples comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .baseline_cnn import BaselineCNN
from .fusion import FusionModule
from .decoder import SegmentationDecoder


class CachedFeatureBackbone(nn.Module):
    """
    Placeholder backbone for VLM configs: takes pre-extracted feature tensors
    from the feature cache rather than running the VLM inline during training.
    All VLM weights are external (frozen); this module is just a projection head.
    """
    def __init__(self, feature_dim: int, d_model: int = 512):
        super().__init__()
        # Trainable projection from cached feature dim → d_model
        self.proj = nn.Sequential(
            nn.Conv2d(feature_dim, d_model, 1),
            nn.GroupNorm(min(32, d_model), d_model),
            nn.GELU(),
        )

    def forward(self, cached_feats: torch.Tensor) -> torch.Tensor:
        return self.proj(cached_feats)


class BanglaSlumNet(nn.Module):
    """
    Unified model supporting all four configs from §3.2.
    backbone_config: 'baseline_cnn' | 'vlm_visual' | 'vlm_lang' | 'full'

    For VLM configs, cached_feats is expected in the forward pass
    (pre-extracted MoonViT features as FloatTensor [B, D_feat, H_f, W_f]).
    For baseline_cnn, only rgb is used.
    """

    def __init__(
        self,
        backbone_config: str = "full",
        # Backbone dims
        in_channels: int = 4,
        feature_dim: int = 256,          # MoonViT output dim; TODO_VERIFY at first load
        d_model: int = 512,
        # Fusion
        socioeconomic_channels: Optional[List[str]] = None,
        fusion_num_heads: int = 8,
        fusion_num_layers: int = 2,
        fusion_dropout: float = 0.1,
        # Decoder
        num_upsample_blocks: int = 3,
        # Baseline
        baseline_backbone: str = "segformer_b0",
    ):
        super().__init__()
        self.backbone_config = backbone_config
        self.socioeconomic_channels = socioeconomic_channels or []

        if backbone_config == "baseline_cnn":
            self.backbone = BaselineCNN(backbone=baseline_backbone, in_channels=in_channels)
            # Decoder unused for baseline_cnn (it has its own head)
            self.fusion = None
            self.decoder = None
            self.visual_proj = None

        else:
            # VLM configs: backbone is the cached feature projector
            self.visual_proj = CachedFeatureBackbone(feature_dim, d_model)

            use_fusion = (backbone_config == "full") and len(self.socioeconomic_channels) > 0
            self.fusion = FusionModule(
                visual_dim=d_model,
                socioeconomic_channels=self.socioeconomic_channels if use_fusion else ["__placeholder__"],
                d_model=d_model,
                num_heads=fusion_num_heads,
                num_layers=fusion_num_layers,
                dropout=fusion_dropout,
            ) if use_fusion else None

            self.decoder = SegmentationDecoder(
                in_channels=d_model,
                num_upsample_blocks=num_upsample_blocks,
            )

    def forward(
        self,
        rgb: torch.Tensor,
        cached_feats: Optional[torch.Tensor] = None,
        socioec: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        rgb:          [B, C_in, 256, 256]
        cached_feats: [B, D_feat, H_f, W_f] — required for VLM configs
        socioec:      [B, C_eco, 256, 256] — required for full config
        channel_mask: [C_eco] bool — optional ablation mask
        Returns: [B, 1, 256, 256] sigmoid slum probability
        """
        if self.backbone_config == "baseline_cnn":
            return self.backbone(rgb)

        assert cached_feats is not None, "VLM configs require cached_feats"

        V = self.visual_proj(cached_feats)  # [B, d_model, H_f, W_f]

        if self.fusion is not None and socioec is not None:
            # Downsample socioec to visual feature resolution
            H_f, W_f = V.shape[-2:]
            if socioec.shape[-2:] != (H_f, W_f):
                socioec_r = F.interpolate(socioec.float(), size=(H_f, W_f),
                                          mode="bilinear", align_corners=False)
            else:
                socioec_r = socioec
            V = self.fusion(V, socioec_r, channel_mask=channel_mask)

        return self.decoder(V)

    def count_trainable_params(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        parts = {}
        for name, child in self.named_children():
            parts[name] = sum(p.numel() for p in child.parameters() if p.requires_grad)
        parts["total"] = total
        return parts


def build_model(config: dict) -> BanglaSlumNet:
    """
    Factory: build BanglaSlumNet from the resolved OmegaConf/dict config.
    config keys: model.config, fusion.*, decoder.*, baseline_cnn.*, data.s2_bands
    """
    backbone_config = config.get("model", {}).get("config", "full")
    n_bands = len(config.get("data", {}).get("s2_bands", ["B2", "B3", "B4", "B8"]))
    socioeconomic_channels = config.get("fusion", {}).get("socioeconomic_channels", [])
    d_model = config.get("fusion", {}).get("d_model", 512)

    model = BanglaSlumNet(
        backbone_config=backbone_config,
        in_channels=n_bands,
        feature_dim=config.get("locate_anything", {}).get("feature_dim", 256),  # TODO_VERIFY
        d_model=d_model,
        socioeconomic_channels=socioeconomic_channels,
        fusion_num_heads=config.get("fusion", {}).get("num_heads", 8),
        fusion_num_layers=config.get("fusion", {}).get("num_layers", 2),
        fusion_dropout=config.get("fusion", {}).get("dropout", 0.1),
        num_upsample_blocks=config.get("decoder", {}).get("num_upsample_blocks", 3),
        baseline_backbone=config.get("baseline_cnn", {}).get("backbone", "segformer_b0"),
    )
    return model


def _smoke_test():
    """Forward pass for all 4 configs."""
    B, C, H, W = 2, 4, 256, 256
    D_feat, H_f, W_f = 256, 16, 16
    C_eco = 6

    rgb = torch.rand(B, C, H, W)
    cached_feats = torch.rand(B, D_feat, H_f, W_f)
    socioec = torch.rand(B, C_eco, H, W)
    eco_channels = ["viirs", "worldpop", "ghspop", "osm_roads", "wb_poverty", "ghsl_builtup"]

    configs = [
        ("baseline_cnn", dict(backbone_config="baseline_cnn")),
        ("vlm_visual", dict(backbone_config="vlm_visual", feature_dim=D_feat, d_model=128)),
        ("vlm_lang",   dict(backbone_config="vlm_lang",   feature_dim=D_feat, d_model=128)),
        ("full",       dict(backbone_config="full", feature_dim=D_feat, d_model=128,
                            socioeconomic_channels=eco_channels)),
    ]

    for name, kwargs in configs:
        model = BanglaSlumNet(**kwargs)
        params = model.count_trainable_params()
        if name == "baseline_cnn":
            out = model(rgb)
        else:
            out = model(rgb, cached_feats=cached_feats, socioec=socioec)
        assert out.shape == (B, 1, H, W), f"{name}: bad output {out.shape}"
        print(f"banglaslumnet.py [{name}] smoke test passed. trainable params: {params['total']:,}")


if __name__ == "__main__":
    _smoke_test()
