# Final results — sealed test split (n = 4)

| Method | PSNR (dB) | SSIM | UIQM | UCIQE |
|---|---|---|---|---|
| Raw | 18.7204 ± 3.6604 | 0.8235 ± 0.0870 | 6.9628 ± 1.1166 | 20.6614 ± 6.1540 |
| Existing U-Net | 18.5656 ± 3.7194 | 0.8322 ± 0.0567 | 7.4053 ± 1.0154 | 26.0145 ± 3.8840 |
| _paneltest | 18.5656 ± 3.7194 | 0.8322 ± 0.0567 | 7.4053 ± 1.0154 | 26.0145 ± 3.8840 |
| Classical | 16.9544 ± 4.4583 | 0.8160 ± 0.0822 | 7.3600 ± 1.5555 | 23.3737 ± 3.8626 |

Means ± standard deviations over the same 133 images; 95% percentile-bootstrap CIs on the mean are in `final_results.csv`.

PSNR/SSIM are full-reference (distance to the UIEB reference). UIQM/UCIQE are no-reference: they reward colour/sharpness/contrast in the image itself, are NOT a distance to the ground truth, and are never compared with values from other papers.
