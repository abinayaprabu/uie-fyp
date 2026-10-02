"""The ONE proposed model: feature-guided underwater image enhancement.

Pipeline implemented by this package:

    image (224x224x3 letterboxed)
        -> ENCODER (4 conv blocks)         -> spatial maps 32x112x112,
                                              64x56x56, 128x28x28, 256x14x14
        -> FiLM CONDITIONING from the selected handcrafted features
           (Linear(k->32) -> ReLU -> Dropout -> Linear(32->2*256) -> gamma, beta)
        -> DECODER (4 up-stages with skip connections) -> 3x224x224
        -> un-letterbox -> resize to the preprocessed geometry -> enhanced image

The image-only baseline is the same classes with ``use_features=False``: the
encoder and decoder are identical, so the ablation measures exactly one thing
(the feature-conditioning branch).

Modules:
    encoder.py       EncoderBlock, HybridEncoder       (keeps spatial maps)
    conditioning.py  FeatureConditioning               (FiLM, identity at init)
    decoder.py       DecoderBlock, EnhancementDecoder  (skips, sigmoid output)
    model.py         FeatureGuidedEnhancer             (the single model class)
    dataset.py       HybridPairs                       (paired 224 dataset)
    losses.py        l1_loss / ssim_loss / combined    (configurable loss)
"""
