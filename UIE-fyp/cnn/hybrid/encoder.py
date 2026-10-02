"""CNN encoder for the feature-guided enhancer.

WHY THIS FILE EXISTS
--------------------
The quality-prediction CNNs in ``cnn/model.py`` end in ``AdaptiveAvgPool2d(1)``:
they throw away the spatial grid and can only emit two numbers. Enhancement
needs the opposite: the encoder must KEEP the spatial feature maps so the
decoder can reconstruct an image, and the deepest map (256x14x14) is the
tensor that the handcrafted features will condition.

WHAT EACH BLOCK DOES (input -> output)
--------------------------------------
EncoderBlock(in_ch, out_ch):
    Conv2d(3x3, pad 1) -> BatchNorm -> ReLU -> MaxPool2d(2)
    input  (B, in_ch,  H,   W  )
    output (B, out_ch, H/2, W/2)      <- the map AFTER pooling
    the PRE-pool map is returned too, because the decoder needs it as a skip

HybridEncoder:
    input   (B, 3, 224, 224)
    block1  -> skip s1 (B,  32, 112, 112)
    block2  -> skip s2 (B,  64,  56,  56)
    block3  -> skip s3 (B, 128,  28,  28)
    block4  -> bottleneck b (B, 256, 14, 14)   (no skip needed: it feeds the decoder)

The widths (32/64/128/256) and the input size (224) are the frozen values in
``src/config.py`` (HYBRID_*).  Nothing here is pretrained.
"""
from __future__ import annotations

import torch
from torch import nn


class EncoderBlock(nn.Module):
    """Conv 3x3 -> BN -> ReLU -> MaxPool. Returns (pre_pool, pooled)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.act(self.bn(self.conv(x)))   # full-resolution (pre-pool) map
        return feat, self.pool(feat)


class HybridEncoder(nn.Module):
    """4-block encoder producing three skip maps and a 256x14x14 bottleneck."""

    def __init__(self, widths: tuple[int, int, int, int] = (32, 64, 128, 256)):
        super().__init__()
        w1, w2, w3, w4 = widths
        self.block1 = EncoderBlock(3, w1)
        self.block2 = EncoderBlock(w1, w2)
        self.block3 = EncoderBlock(w2, w3)
        self.block4 = EncoderBlock(w3, w4)
        self.out_channels = w4

    def forward(self, x: torch.Tensor):
        # Each block returns (pre-pool features, pooled features); the SKIPS
        # handed to the decoder are the POOLED maps at 112/56/28 -- exactly the
        # 32/64/128-channel tensors drawn in the locked architecture diagram --
        # and the bottleneck is the pooled output of block 4 at 256x14x14.
        _, p1 = self.block1(x)       # skip s1 (B,  32, 112, 112)
        _, p2 = self.block2(p1)      # skip s2 (B,  64,  56,  56)
        _, p3 = self.block3(p2)      # skip s3 (B, 128,  28,  28)
        _, bottleneck = self.block4(p3)   # (B, 256, 14, 14)
        return bottleneck, (p1, p2, p3)
