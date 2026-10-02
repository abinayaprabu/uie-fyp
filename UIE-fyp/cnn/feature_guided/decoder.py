"""Decoder for the feature-guided enhancer: upsample -> skip -> refine.

Each stage is the standard U-Net decoder pattern, already proven in this repo
by ``cnn/unet.py``:

    DecoderBlock(in_ch, skip_ch, out_ch):
        Upsample(2, bilinear)            doubles H and W
        concat with the encoder skip     adds back the fine detail
        DoubleConv(in_ch + skip_ch -> out_ch)   two 3x3 Conv-BN-ReLU units

The four stages mirror the four encoder blocks exactly:

    256x14x14  ->  28x28 (+ 128ch skip)  ->  128x28x28
               ->  56x56 (+  64ch skip)  ->   64x56x56
               -> 112x112 (+ 32ch skip)  ->   32x112x112
               -> 224x224 (no skip)      ->   16x224x224
    head: Conv1x1 -> 3 channels -> Sigmoid   [targets live in [0, 1]]

Why skips matter here: colour cast and haze are low-frequency (carried by the
deep path), while edges and texture are high-frequency (carried by the skips).
The objective needs both.  The model stays small on purpose: 890 training
pairs and, in this project, a CPU-only sandbox.
"""
from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    """Two 3x3 Conv-BN-ReLU units (the U-Net building block)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    """Upsample + optional skip concatenation + DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                raise ValueError(f"skip {tuple(skip.shape[-2:])} does not match "
                                 f"upsampled {tuple(x.shape[-2:])}")
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class EnhancementDecoder(nn.Module):
    """4-stage decoder with skips -> 3-channel image in [0, 1]."""

    def __init__(self, widths: tuple[int, int, int, int] = (128, 64, 32, 16),
                 bottleneck_ch: int = 256, skip_channels: tuple[int, int, int] = (32, 64, 128)):
        super().__init__()
        w1, w2, w3, w4 = widths
        # skip_channels arrives as (32-channel s1, 64-channel s2, 128-channel s3)
        c_s1, c_s2, c_s3 = skip_channels
        self.up1 = DecoderBlock(bottleneck_ch, c_s3, w1)  # 14 -> 28   (+128 ch)
        self.up2 = DecoderBlock(w1, c_s2, w2)             # 28 -> 56   (+64 ch)
        self.up3 = DecoderBlock(w2, c_s1, w3)             # 56 -> 112  (+32 ch)
        self.up4 = DecoderBlock(w3, 0, w4)                # 112 -> 224
        self.head = nn.Conv2d(w4, 3, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor,
                skips: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        s1, s2, s3 = skips
        x = self.up1(x, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)
        x = self.up4(x, None)
        return self.act(self.head(x))
