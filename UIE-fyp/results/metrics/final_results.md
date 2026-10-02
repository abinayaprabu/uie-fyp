# Final results — sealed test split (n = 133)

| Method | PSNR (dB) | SSIM | UIQM | UCIQE |
|---|---|---|---|---|
| Existing U-Net | 19.3242 ± 2.5316 | 0.8003 ± 0.0792 | 7.6281 ± 0.5239 | 25.6052 ± 3.8626 |
| Feature-guided (proposed) | 18.9744 ± 2.2645 | 0.6402 ± 0.1163 | 5.9293 ± 0.3379 | 25.2816 ± 3.7642 |
| Classical | 17.0889 ± 3.1581 | 0.7636 ± 0.1114 | 7.7645 ± 0.8886 | 22.6355 ± 4.0216 |
| Raw | 17.0633 ± 4.0496 | 0.7603 ± 0.1382 | 7.3322 ± 0.5569 | 20.4920 ± 6.7891 |

Means ± standard deviations over the same 133 images; 95% percentile-bootstrap CIs on the mean are in `final_results.csv`.

PSNR/SSIM are full-reference (distance to the UIEB reference). UIQM/UCIQE are no-reference: they reward colour/sharpness/contrast in the image itself, are NOT a distance to the ground truth, and are never compared with values from other papers.
