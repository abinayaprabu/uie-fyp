"""Loss functions for enhancement training.

PRIMARY: L1 (pixel-wise absolute error). Chosen because the targets are paired
images and L1 is the standard, robust choice for image restoration -- it
penalises errors linearly instead of squaring them, which avoids over-penalising
the few hard pixels and produces less blurry output than MSE.

OPTIONAL: L1 + lambda * (1 - SSIM), configurable through
``HYBRID_LOSS`` / ``HYBRID_LAMBDA_SSIM`` in ``src/config.py``.  No number is
copied from any paper: if used, lambda is chosen on TRAIN/VALIDATION only and
reported as our own setting.  The default is pure L1.

The SSIM term below is a differentiable re-implementation of the same
structural-similarity formula the project reports with (uniform window,
K1=0.01, K2=0.03), computed on the fly inside the training graph.
"""
from __future__ import annotations

import torch
from torch import nn


class L1Loss(nn.Module):
    """Plain pixel-wise L1 loss."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(pred - target))


class SSIMLoss(nn.Module):
    """1 - mean SSIM, differentiable, computed with a uniform window."""

    def __init__(self, window: int = 11, data_range: float = 1.0,
                 k1: float = 0.01, k2: float = 0.03):
        super().__init__()
        self.window = window
        self.data_range = data_range
        self.c1 = (k1 * data_range) ** 2
        self.c2 = (k2 * data_range) ** 2
        self.pool = nn.AvgPool2d(window, stride=1, padding=window // 2)
        # covariance normalisation used by scikit-image's default
        # (use_sample_covariance=True) -- kept consistent with src/iqa.py
        self.cov_norm = (window ** 2) / (window ** 2 - 1)

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ux = self.pool(x)
        uy = self.pool(y)
        uxx = self.pool(x * x)
        uyy = self.pool(y * y)
        uxy = self.pool(x * y)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        num = (2 * ux * uy + self.c1) * (2 * vxy + self.c2)
        den = (ux ** 2 + uy ** 2 + self.c1) * (vx + vy + self.c2)
        return (num / den).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - self._ssim(pred, target)


class CombinedLoss(nn.Module):
    """L1 + lambda * (1 - SSIM), with lambda explicit and configurable."""

    def __init__(self, lam: float, window: int = 11):
        super().__init__()
        self.lam = float(lam)
        self.l1 = L1Loss()
        self.ssim = SSIMLoss(window=window)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1(pred, target) + self.lam * self.ssim(pred, target)


def build_loss(name: str, lam: float = 0.0, window: int = 11) -> nn.Module:
    """Factory used by the training loop ('l1' or 'l1_ssim')."""
    name = name.lower()
    if name == "l1":
        return L1Loss()
    if name == "l1_ssim":
        if lam <= 0:
            raise ValueError("HYBRID_LAMBDA_SSIM must be > 0 when loss='l1_ssim'")
        return CombinedLoss(lam, window=window)
    raise ValueError(f"unknown loss {name!r}")
