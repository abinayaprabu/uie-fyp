"""Model definitions (PyTorch).

Naming is strict (per the methodology review):
- An MLP on handcrafted features is an MLP, never a "CNN".
- ImageOnlyCNN: convolutions over the preprocessed image only.
- HybridCNN: image branch + handcrafted-feature branch fused before a shared
  regression head with TWO outputs (SSIM, PSNR) in normalised space.
- FeatMLP: torch MLP on handcrafted features (used as the MLP baseline so all
  neural models share framework/optimiser/early-stopping protocol).

The image branch is a small custom CNN (4 conv blocks + global average
pooling, ~0.43 M params at default widths) — deliberately lightweight for
890 training images on CPU. No pretrained weights: keeps the comparison
between image-only and hybrid strictly attributable to the handcrafted
features rather than to ImageNet priors.
"""
from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class ImageBranch(nn.Module):
    """4 conv blocks + GAP -> 256-d deep visual vector (for 224px input)."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out_dim = out_dim

    def forward(self, x):
        x = self.stem(x)
        return self.gap(x).flatten(1)  # (B, 256)


class FeatBranch(nn.Module):
    """Handcrafted-feature encoder: Linear -> ReLU -> Dropout -> 32-d."""

    def __init__(self, n_features: int, out_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x)


class RegressionHead(nn.Module):
    """Shared fusion head: Linear(->128) -> ReLU -> Dropout -> Linear(->2)."""

    def __init__(self, in_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),  # [ssim_norm, psnr_norm]
        )

    def forward(self, x):
        return self.net(x)


class HybridCNN(nn.Module):
    """Proposed model: CNN image branch + handcrafted-feature branch fusion."""

    def __init__(self, n_features: int):
        super().__init__()
        self.image_branch = ImageBranch()
        self.feat_branch = FeatBranch(n_features)
        self.head = RegressionHead(
            self.image_branch.out_dim + self.feat_branch.out_dim)

    def forward(self, image, feats):
        z = torch.cat([self.image_branch(image), self.feat_branch(feats)], dim=1)
        return self.head(z)


class ImageOnlyCNN(nn.Module):
    """Ablation/baseline: image branch + head (no handcrafted features)."""

    def __init__(self):
        super().__init__()
        self.image_branch = ImageBranch()
        self.head = RegressionHead(self.image_branch.out_dim)

    def forward(self, image, feats=None):  # feats ignored; kept for API parity
        return self.head(self.image_branch(image))


class FeatMLP(nn.Module):
    """MLP baseline on handcrafted features (NOT a CNN)."""

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    def forward(self, image=None, feats=None):
        return self.net(feats)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
