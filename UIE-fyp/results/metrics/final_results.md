# Final results — sealed test split (n=133)

| Method | PSNR (dB) | SSIM | UIQM | UCIQE |
|---|---|---|---|---|
| Raw | 17.0633 | 0.7603 | 7.3322 | 20.4920 |
| Classical | 17.0889 | 0.7636 | 7.7645 | 22.6355 |
| Existing U-Net | 19.3242 | 0.8003 | 7.6281 | 25.6052 |

Means over the same 133 images; 95% percentile-bootstrap CIs on the mean are in `final_results.csv`.

UIQM and UCIQE are NO-REFERENCE scores computed by `src/nr_metrics.py`: higher means more colour/sharpness/contrast in the image itself, NOT closer to the reference. They must never be compared against numbers from other papers (different conventions), only between the rows above.
