"""The single proposed enhancement model (and its image-only twin).

    FeatureGuidedEnhancer(n_features=k, use_features=True)
        image  (B,3,224,224) -> encoder -> bottleneck (B,256,14,14) + skips
        feats  (B,k)         -> conditioning -> gamma, beta
        bottleneck <- FiLM-modulated
        decoder(bottleneck, skips) -> (B,3,224,224) in [0,1]

    FeatureGuidedEnhancer(n_features=k, use_features=False)
        -> the SAME encoder and decoder with no feature branch at all.
        This is the image-only baseline required by the ablation; training it
        with identical settings is what makes "do the features help?" a
        measured question instead of a claim.

At initialisation the two variants compute the identical function (FiLM is
identity-init), so the comparison starts from a level playing field.
"""
from __future__ import annotations

import torch
from torch import nn

from cnn.feature_guided.conditioning import FeatureConditioning
from cnn.feature_guided.decoder import EnhancementDecoder
from cnn.feature_guided.encoder import HybridEncoder


class FeatureGuidedEnhancer(nn.Module):
    """Feature-guided underwater image enhancement (one model, two variants)."""

    def __init__(self, n_features: int | None = None,
                 use_features: bool = True,
                 encoder_widths: tuple[int, int, int, int] = (32, 64, 128, 256),
                 decoder_widths: tuple[int, int, int, int] = (128, 64, 32, 16),
                 feat_hidden: int = 32, feat_dropout: float = 0.2):
        super().__init__()
        if use_features and not n_features:
            raise ValueError("use_features=True requires n_features > 0")
        self.use_features = bool(use_features)
        self.n_features = int(n_features) if n_features else 0

        self.encoder = HybridEncoder(encoder_widths)
        skip_channels = (encoder_widths[0], encoder_widths[1], encoder_widths[2])
        self.decoder = EnhancementDecoder(
            decoder_widths, bottleneck_ch=encoder_widths[3],
            skip_channels=skip_channels)
        self.conditioning = (
            FeatureConditioning(self.n_features, hidden=feat_hidden,
                                out_channels=encoder_widths[3],
                                dropout=feat_dropout, identity_init=True)
            if self.use_features else None)

    def forward(self, image: torch.Tensor,
                feats: torch.Tensor | None = None) -> torch.Tensor:
        bottleneck, skips = self.encoder(image)
        if self.conditioning is not None:
            if feats is None:
                raise ValueError("this variant needs the feature vector")
            bottleneck = self.conditioning.modulate(bottleneck, feats)
        return self.decoder(bottleneck, skips)


def count_params(model: nn.Module) -> int:
    """Trainable parameter count (printed at build time, never estimated)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
