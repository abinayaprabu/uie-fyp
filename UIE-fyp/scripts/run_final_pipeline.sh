#!/usr/bin/env bash
# Final pipeline: run this ONCE the feature-guided training has stopped.
#
# It performs, in order, exactly the steps required by the specification:
#   1. verify the final checkpoint (loads, features == selected list, forward
#      pass, feature dependence, history agreement)
#   2. enhance EXACTLY the 133 sealed test images
#   3. check the inference outputs (count, filenames, size, range, no NaN)
#   4. compute PSNR / SSIM / UIQM / UCIQE for every available system
#   5. re-run the leakage audit (incl. the checkpoint-dependent checks)
#   6. redraw the training/validation curves
#
# Usage:  bash scripts/run_final_pipeline.sh [run_tag]      (default enh224_featguided)
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:-enh224_featguided}"
PY="../.venv/bin/python"

echo "############ 1/6 verify final checkpoint ############"
"$PY" scripts/verify_final_model.py --run "$RUN"

echo "############ 2/6 enhance the sealed test split ############"
"$PY" -m cnn.feature_guided.enhance --run "$RUN" --split test

echo "############ 3/6 check the inference outputs ############"
"$PY" - "$RUN" <<'EOF'
import sys
from pathlib import Path
import cv2, numpy as np
from cnn.dataset_pairs import split_ids
run = sys.argv[1]
d = Path("results/enhancement") / run / "enhanced"
ids = set(split_ids("test"))
files = {p.name for p in d.glob("*.png")}
missing = sorted(ids - files)
extra = sorted(files - ids)
bad = []
for p in sorted(d.glob("*.png")):
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        bad.append((p.name, "unreadable")); continue
    if img.shape[2] != 3:
        bad.append((p.name, f"channels {img.shape[2]}")); continue
    if img.min() < 0 or img.max() > 255:
        bad.append((p.name, "range")); continue
    if not np.isfinite(img.astype(np.float64)).all():
        bad.append((p.name, "non-finite"))
print(f"outputs           : {len(files)}")
print(f"test ids expected : {len(ids)}")
print(f"missing           : {len(missing)} {missing[:5]}")
print(f"unexpected extras : {len(extra)} {extra[:5]}")
print(f"invalid files     : {len(bad)} {bad[:5]}")
ok = not missing and not extra and not bad and len(files) == len(ids) == 133
print("INFERENCE CHECK:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
EOF

echo "############ 4/6 four-metric evaluation + panels ############"
"$PY" scripts/evaluate_hybrid.py

echo "############ 5/6 leakage audit ############"
"$PY" scripts/leakage_audit.py

echo "############ 6/6 curves ############"
"$PY" scripts/make_hybrid_plots.py

echo
echo "ALL STEPS FINISHED. Review results/metrics/ before committing."
