from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class FPNAdapter(nn.Module):
    """Lightweight FPN: single-scale ViT spatial features -> 4-level pyramid (P3..P6).

    Input: (B, T, in_channels) patch tokens (CLS already excluded by backbone wrapper).
    Output: dict {'p3': (B, out_channels, 2H, 2W),
                  'p4': (B, out_channels, H, W),
                  'p5': (B, out_channels, H//2, W//2),
                  'p6': (B, out_channels, H//4, W//4)}
    where (H, W) is the spatial_shape passed at forward time.

    Architecture:
        1. Reshape (B, T, C_in) -> (B, C_in, H, W)
        2. 1x1 conv: C_in -> out_channels (256), GroupNorm, GELU (lateral connection)
        3. P4 = identity of lateral output (1/16 scale)
        4. P3 = 2x bilinear upsample + 3x3 refine conv (1/8 scale)
        5. P5 = 3x3 stride-2 conv (1/32 scale)
        6. P6 = 3x3 stride-2 conv from P5 (1/64 scale)

    Notes:
        - in_channels must come from BackboneInfo.spatial_dim (1536 for C-RADIOv4, 768 for DINOv2).
        - GroupNorm (NOT BatchNorm) to handle batch-size-1 inference.
    """

    def __init__(self, in_channels: int, out_channels: int = 256, num_groups: int = 32) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Lateral: project to out_channels
        self.lateral = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.GELU(),
        )
        # P3 refine after upsample
        self.p3_refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.GELU(),
        )
        # P5: stride-2 downsample from P4
        self.p5_down = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.GELU(),
        )
        # P6: stride-2 downsample from P5
        self.p6_down = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.GELU(),
        )

    def forward(self, tokens: torch.Tensor, spatial_shape: tuple[int, int]) -> dict[str, torch.Tensor]:
        batch, num_tokens, channels = tokens.shape
        h, w = spatial_shape
        if h * w != num_tokens:
            raise ValueError(f"spatial_shape ({h},{w})={h * w} doesn't match T={num_tokens}")
        if self.in_channels != channels:
            raise ValueError(f"Token dim {channels} doesn't match in_channels {self.in_channels}")

        # (B, T, C) -> (B, C, H, W)
        x = tokens.transpose(1, 2).contiguous().view(batch, channels, h, w)

        # Lateral: -> (B, 256, H, W)
        lat = self.lateral(x)

        # P4: identity (1/16 scale)
        p4 = lat

        # P3: 2x upsample + refine (1/8 scale)
        p3_up = functional.interpolate(lat, scale_factor=2, mode='bilinear', align_corners=False)
        p3 = self.p3_refine(p3_up)

        # P5: stride-2 down from P4 (1/32 scale)
        p5 = self.p5_down(p4)

        # P6: stride-2 down from P5 (1/64 scale)
        p6 = self.p6_down(p5)

        return {'p3': p3, 'p4': p4, 'p5': p5, 'p6': p6}
