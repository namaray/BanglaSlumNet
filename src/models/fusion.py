"""
Cross-attention fusion module.
F = CrossAttention(Q=V, K=E, V=E) + V  (residual)
V: projected VLM visual feature grid [B, D, H_f, W_f]
E: socioeconomic tensor projected to same shape

Supports per-channel ablation via channel_mask (zero out individual E channels
without changing the projector shape — §8.2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        attn_out, _ = self.attn(q, kv, kv)
        query = query + attn_out
        query = query + self.ffn(self.norm_out(query))
        return query


class SocioeconomicProjector(nn.Module):
    """Project C_eco socioeconomic channels to d_model, spatially."""
    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, d_model, 1)
        self.norm = nn.GroupNorm(min(32, d_model), d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class VisualProjector(nn.Module):
    """Project VLM feature channels to d_model, spatially."""
    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, d_model, 1)
        self.norm = nn.GroupNorm(min(32, d_model), d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class FusionModule(nn.Module):
    """
    Cross-attention fusion of VLM visual features (V) with socioeconomic tensor (E).
    F = CrossAttn(Q=V, K=E, V=E) + V  (residual)

    channel_mask: optional [C_eco] boolean tensor — set False to zero out a channel
    for per-channel ablation without retraining projector shapes.
    """

    def __init__(
        self,
        visual_dim: int,
        socioeconomic_channels: List[str],
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        c_eco = len(socioeconomic_channels)
        self.socioeconomic_channels = socioeconomic_channels
        self.d_model = d_model

        self.visual_proj = VisualProjector(visual_dim, d_model)
        self.eco_proj = SocioeconomicProjector(c_eco, d_model)

        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        visual_feats: torch.Tensor,
        socioec: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        visual_feats: [B, D_v, H_f, W_f]
        socioec:      [B, C_eco, H_f, W_f]  (resampled to visual grid)
        channel_mask: [C_eco] bool tensor — ablation hook
        Returns: [B, d_model, H_f, W_f]
        """
        if channel_mask is not None:
            mask = channel_mask.float().to(socioec.device).view(1, -1, 1, 1)
            socioec = socioec * mask

        # Upsample/downsample socioec to match visual spatial dims
        H_f, W_f = visual_feats.shape[-2:]
        if socioec.shape[-2:] != (H_f, W_f):
            socioec = F.interpolate(socioec, size=(H_f, W_f), mode="bilinear", align_corners=False)

        V = self.visual_proj(visual_feats)  # [B, d_model, H_f, W_f]
        E = self.eco_proj(socioec)           # [B, d_model, H_f, W_f]

        B, D, H, W = V.shape
        V_flat = V.view(B, D, H * W).permute(0, 2, 1)  # [B, N, D]
        E_flat = E.view(B, D, H * W).permute(0, 2, 1)  # [B, N, D]

        F_out = V_flat
        for layer in self.cross_attn_layers:
            F_out = layer(F_out, E_flat)

        F_out = F_out.permute(0, 2, 1).view(B, D, H, W)
        return F_out + V  # residual


def _smoke_test():
    module = FusionModule(
        visual_dim=256,
        socioeconomic_channels=["viirs", "worldpop", "ghspop", "osm_roads", "wb_poverty", "ghsl_builtup"],
        d_model=128,
        num_heads=4,
        num_layers=2,
    )
    visual = torch.rand(2, 256, 16, 16)
    socioec = torch.rand(2, 6, 256, 256)
    out = module(visual, socioec)
    assert out.shape == (2, 128, 16, 16), f"Bad fusion output: {out.shape}"

    # Test channel mask ablation
    mask = torch.ones(6, dtype=torch.bool)
    mask[0] = False  # zero out VIIRS
    out_masked = module(visual, socioec, channel_mask=mask)
    assert out_masked.shape == out.shape
    print("fusion.py smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
