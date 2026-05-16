import torch
import torch.nn as nn
import torch.nn.functional as F


class SEChannelEncoder(nn.Module):
    def __init__(self, out_channels=64):
        super(SEChannelEncoder, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        num_se_channels=3,
        d_model=256,
        num_heads=8,
        se_hidden=64,
        dropout=0.1,
        attn_size=16
    ):
        super(CrossAttentionFusion, self).__init__()

        self.num_se_channels = num_se_channels
        self.d_model = d_model
        self.se_hidden = se_hidden
        self.attn_size = attn_size

        self.se_encoders = nn.ModuleList(
            [SEChannelEncoder(out_channels=se_hidden) for _ in range(num_se_channels)]
        )

        self.se_project = nn.Sequential(
            nn.Conv2d(num_se_channels * se_hidden, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )

        self.query_norm = nn.LayerNorm(d_model)
        self.key_value_norm = nn.LayerNorm(d_model)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.out_norm = nn.LayerNorm(d_model)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, S, se_inputs):
        if not isinstance(se_inputs, (list, tuple)):
            raise TypeError("se_inputs must be a list or tuple of tensors.")

        if len(se_inputs) != self.num_se_channels:
            raise ValueError(
                f"Expected {self.num_se_channels} socioeconomic channels, "
                f"but got {len(se_inputs)}."
            )

        b, c, h, w = S.shape
        if c != self.d_model:
            raise ValueError(f"Expected S to have {self.d_model} channels, got {c}.")

        encoded_se = []
        for i in range(self.num_se_channels):
            x = se_inputs[i]
            if x.dim() != 4 or x.shape[1] != 1:
                raise ValueError(
                    f"Each SE input must have shape [B, 1, H, W]. Got {x.shape} at index {i}."
                )
            encoded_se.append(self.se_encoders[i](x))

        E = torch.cat(encoded_se, dim=1)
        E = self.se_project(E)

        S_small = F.adaptive_avg_pool2d(S, (self.attn_size, self.attn_size))
        E_small = F.adaptive_avg_pool2d(E, (self.attn_size, self.attn_size))

        S_flat = S_small.flatten(2).transpose(1, 2)
        E_flat = E_small.flatten(2).transpose(1, 2)

        S_q = self.query_norm(S_flat)
        E_kv = self.key_value_norm(E_flat)

        attn_output, _ = self.multihead_attn(
            query=S_q,
            key=E_kv,
            value=E_kv,
            need_weights=False
        )

        attn_output = self.out_dropout(attn_output)
        fused_small = self.out_norm(attn_output + S_flat)
        fused_small = fused_small.transpose(1, 2).reshape(
            b, c, self.attn_size, self.attn_size
        )

        fused_up = F.interpolate(
            fused_small,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )

        fused = S + fused_up
        return fused


if __name__ == "__main__":
    print("Initializing Cross-Attention Fusion Module...")

    fusion_module = CrossAttentionFusion(num_se_channels=3, attn_size=16)

    dummy_S = torch.rand(1, 256, 64, 64)
    dummy_NTL = torch.rand(1, 1, 512, 512)
    dummy_Pop = torch.rand(1, 1, 512, 512)
    dummy_GOB = torch.rand(1, 1, 512, 512)

    se_inputs = [dummy_NTL, dummy_Pop, dummy_GOB]

    print("\nFusing Visual Structure with Socioeconomic Context...")
    F_out = fusion_module(dummy_S, se_inputs)

    print(f"✅ Fused Feature Map shape: {F_out.shape}")