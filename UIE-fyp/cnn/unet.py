"""Enhancement / restoration U-Net — an IMAGE-TO-IMAGE network.

Why this file exists
--------------------
Every other model in this repository (``HybridCNN``, ``ImageOnlyCNN``,
``FeatMLP``) is a **quality predictor**: it maps an image to two scalars,
``[SSIM, PSNR]``. Their encoders terminate in ``nn.AdaptiveAvgPool2d(1)``, which
collapses the spatial dimensions into a single vector, so those architectures
are *structurally incapable* of emitting an image.

``EnhancementUNet`` is the missing component required by the project objective
("develop an underwater image enhancement and restoration framework"). It is an
encoder-decoder with skip connections: it maps a degraded underwater image to an
**enhanced image of the same height and width**, shape ``(B, 3, H, W)``.

    preprocessed image (B,3,H,W) -> encoder -> bottleneck -> decoder -> enhanced
                                                                image (B,3,H,W)

Design notes
------------
* **3 levels, base width 32** (32 -> 64 -> 128). Two max-pools means the input
  must be divisible by 4; ``cnn.dataset_pairs.pad_to_multiple`` handles that for
  the full-resolution validation/test pass, so *any* UIEB size works.
* **Bilinear upsample + skip concatenation**, the standard U-Net decoder. Skips
  matter here specifically: colour cast and haze are low-frequency (carried by
  the deep path) while edges and fine detail are high-frequency (carried by the
  skips), and the objective asks for both.
* **Sigmoid output**, because both the input and the target are images in
  [0, 1]. This bounds the prediction to the valid colour range.
* **472,259 trainable parameters** — deliberately the same order as the
  quality-prediction CNNs (422,530 image-only / 427,106 hybrid), so the two
  halves of the framework are comparably sized. The count is asserted against
  ``config.ENH_EXPECTED_PARAMS`` at build time in ``cnn/train_enhance.py``.
* **No pretrained weights**, matching the deliberate choice made for the
  quality-prediction CNNs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """The standard U-Net building block: two 3x3 Conv-BN-ReLU units."""

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


class EnhancementUNet(nn.Module):
    """3-level U-Net. ``forward`` returns an IMAGE, shape ``(B, 3, H, W)``."""

    #: number of max-pools; the input must be divisible by ``2 ** LEVELS``
    LEVELS = 2

    def __init__(self, base: int = 32):
        super().__init__()
        c0, c1, c2 = base, base * 2, base * 4
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # encoder
        self.d1 = DoubleConv(3, c0)              # H
        self.d2 = DoubleConv(c0, c1)             # H/2
        self.d3 = DoubleConv(c1, c2)             # H/4  (bottleneck)
        # decoder (upsample, concatenate the skip, then refine)
        self.u2 = DoubleConv(c2 + c1, c1)        # H/2
        self.u1 = DoubleConv(c1 + c0, c0)        # H
        self.head = nn.Conv2d(c0, 3, 1)
        self.act = nn.Sigmoid()                  # targets live in [0, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.d1(x)
        s2 = self.d2(self.pool(s1))
        b = self.d3(self.pool(s2))
        y2 = self.u2(torch.cat([self.up(b), s2], dim=1))
        y1 = self.u1(torch.cat([self.up(y2), s1], dim=1))
        return self.act(self.head(y1))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
