"""Feature conditioning: selected handcrafted features -> FiLM (scale/bias).

THE IDEA IN ONE PARAGRAPH
-------------------------
The selected features (red ratio, dynamic range, entropy, ...) summarise the
GLOBAL state of the image: how much red was lost, how much contrast and
texture the scene has. The encoder's bottleneck knows WHERE things are but not
these global numbers. FiLM ("Feature-wise Linear Modulation", Perez et al.,
2018) lets the features set a per-channel gain gamma and offset beta on the
bottleneck:

    conditioned = F * (1 + gamma(features)) + beta(features)

i.e. the features turn 256 "knobs" that scale and shift the encoder's
channels before decoding. Reading it in viva terms: the features act like
per-channel brightness/contrast/colour knobs whose positions are learned from
the statistics of the input image.

WHY THIS EXACT FORM
-------------------
1. It is ONE simple, explainable mechanism (no attention, no gates stack).
2. It is initialised to IDENTITY: gamma and beta come from a zero-initialised
   last layer, so at step 0 the model computes conditioned = F * (1+0) + 0 = F,
   i.e. EXACTLY the image-only baseline. Any improvement or harm afterwards is
   learned feature use, not an architectural side effect.
3. The ablation is therefore clean: with the feature branch removed the model
   IS the baseline; with features shuffled the modulation is nonsense, and the
   sanity check in ``scripts/verify_hybrid.py`` shows how much the output moves.

SHAPES
------
    input features x        (B, k)          k = number of selected features
    hidden                  (B, 32)
    gamma, beta             (B, 256) each
    spatial F               (B, 256, 14, 14)
    output                  (B, 256, 14, 14)   [broadcast over H, W]
"""
from __future__ import annotations

import torch
from torch import nn


class FeatureConditioning(nn.Module):
    """Small MLP producing FiLM parameters (gamma, beta) for ``out_channels``.

    ``n_features`` comes from the frozen selection CSV, so nothing is hard-coded
    to 14 -- whatever Stage A selected is what the model consumes.
    """

    def __init__(self, n_features: int, hidden: int = 32, out_channels: int = 256,
                 dropout: float = 0.2, identity_init: bool = True):
        super().__init__()
        self.n_features = n_features
        self.out_channels = out_channels
        self.body = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.to_film = nn.Linear(hidden, 2 * out_channels)
        if identity_init:
            # Zero weights + zero bias => gamma = beta = 0 at initialisation.
            nn.init.zeros_(self.to_film.weight)
            nn.init.zeros_(self.to_film.bias)
        self.identity_init = identity_init

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (gamma, beta), each (B, out_channels)."""
        h = self.body(feats)
        film = self.to_film(h)
        gamma, beta = torch.chunk(film, 2, dim=1)
        return gamma, beta

    def modulate(self, spatial: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """Apply FiLM to the spatial bottleneck with correct broadcasting.

        spatial: (B, C, H, W);  feats: (B, k);  returns (B, C, H, W).
        """
        gamma, beta = self.forward(feats)
        return spatial * (1.0 + gamma)[:, :, None, None] + beta[:, :, None, None]
