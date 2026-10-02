"""Stage A — statistical validation of the 25 handcrafted image-quality features.

This package answers ONE question with real data (n = 623 training images):

    "Which handcrafted image-quality characteristics have a statistically
     detectable and practically useful association with enhancement quality
     (SSIM / PSNR), and which of them are redundant?"

The pipeline (each step in its own module, each fit on TRAIN ROWS ONLY):

    descriptive.py        mean / median / sd / IQR / skew / kurtosis      (context)
    hypothesis_tests.py   univariate regression F-test per feature x target
    multiple_testing.py   Benjamini-Hochberg FDR control over the 25 p-values
    correlation.py        Pearson r and Spearman rho (association, effect size)
    redundancy.py         |Pearson r| >= 0.90 feature-feature redundancy removal
    selector.py           the assembled screening -> ranking -> redundancy funnel

Nothing here sees the validation or test splits, and nothing here trains the
enhancement model. The output is a documented, reproducible selected feature
set consumed by the feature-conditioning branch of the enhancement CNN.
"""
