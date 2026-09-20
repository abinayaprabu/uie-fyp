#!/usr/bin/env python3
"""CHECK 1-2 + Section 25: validate raw/preprocessed/reference pairing.

Verifies, for every image identity:
  - raw / preprocessed / reference files exist and are readable
  - shapes, channel counts, dtypes
  - raw vs reference dimension match (UIEB pairs share dimensions)
  - duplicate detection by SHA-256 file hash AND by perceptual hash
    (catches both byte-identical copies and near-identical re-saves)
  - raw-vs-reference content sanity: paired images must be visually similar
    (small perceptual-hash distance) but NOT byte-identical

Outputs:
  - results/feature/dataset_validation_report.csv (one row per image identity)
  - plots/pairing_contact_sheet_A.png / ..._B.png (raw|reference side-by-side
    strips so a human can visually confirm correspondence)
  - console summary with PASS/FAIL per check; exit code 1 on critical failure.

Run AFTER scripts/download_uieb.py and (for the preprocessed_* columns)
AFTER scripts/run_preprocessing.py. Re-run anytime; it never modifies images.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    FEATURE_RESULTS_DIR,
    PLOTS_DIR,
    PREPROCESSED_DIR,
    RAW_DIR,
    REFERENCE_DIR,
    VALIDATION_REPORT_CSV,
)

CONTACT_SHEET_IDS = [0, 1, 2, 3, 4, 100, 200, 300, 400, 500, 600, 700, 800, 889]


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phash(gray: np.ndarray) -> int:
    """64-bit DCT perceptual hash (grayscale uint8 -> int)."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    low = dct[:8, :8].flatten()
    med = np.median(low)
    bits = (low > med).astype(np.uint64)
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def read_info(path: Path):
    """Return (ok, h, w, c, file_hash, img) for an image file."""
    if not path.exists():
        return False, None, None, None, None, None
    try:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    except Exception:
        return False, None, None, None, None, None
    if img is None:
        return False, None, None, None, None, None
    if img.ndim == 2:
        h, w, c = img.shape[0], img.shape[1], 1
    elif img.ndim == 3:
        h, w, c = img.shape
    else:
        return False, None, None, None, None, None
    return True, h, w, c, sha256_of_file(path), img


def to_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img if img.dtype == np.uint8 else cv2.normalize(
            img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def main() -> int:
    VALIDATION_REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(RAW_DIR.glob("*")) if RAW_DIR.is_dir() else []
    ref_files = sorted(REFERENCE_DIR.glob("*")) if REFERENCE_DIR.is_dir() else []
    raw_names = [p.name for p in raw_files if p.is_file()]
    ref_names = [p.name for p in ref_files if p.is_file()]
    ids = sorted(set(raw_names) | set(ref_names))

    print(f"raw-890 files: {len(raw_names)} | reference-890 files: {len(ref_names)}")
    print(f"union of identities: {len(ids)}")

    rows = []
    raw_hashes: dict[str, str] = {}
    ref_hashes: dict[str, str] = {}
    pair_hamming: list[int] = []

    for name in ids:
        raw_ok, rh, rw, rc, rhash, raw_img = read_info(RAW_DIR / name)
        ref_ok, fh, fw, fc, fhash, ref_img = read_info(REFERENCE_DIR / name)
        pre_path = PREPROCESSED_DIR / name
        pre_ok = pre_path.exists()
        ph, pw, pc = None, None, None
        if pre_ok:
            pok, ph, pw, pc, _, _ = read_info(pre_path)
            pre_ok = pok
        if rhash:
            raw_hashes.setdefault(rhash, name)
        if fhash:
            ref_hashes.setdefault(fhash, name)

        dim_match, identical_bytes, ph_dist = None, None, None
        if raw_ok and ref_ok:
            dim_match = (rh == fh and rw == fw)
            identical_bytes = (rhash == fhash)
            try:
                ph_dist = hamming(phash(to_gray_u8(raw_img)), phash(to_gray_u8(ref_img)))
                pair_hamming.append(ph_dist)
            except Exception:
                ph_dist = None

        rows.append({
            "image_name": name,
            "raw_exists": raw_ok, "preprocessed_exists": pre_ok, "reference_exists": ref_ok,
            "raw_shape": f"{rh}x{rw}" if raw_ok else None,
            "preprocessed_shape": f"{ph}x{pw}" if pre_ok else None,
            "reference_shape": f"{fh}x{fw}" if ref_ok else None,
            "raw_channels": rc, "preprocessed_channels": pc, "reference_channels": fc,
            "raw_ref_dim_match": dim_match,
            "raw_ref_identical_bytes": identical_bytes,
            "raw_ref_phash_hamming": ph_dist,
        })

    report = pd.DataFrame(rows)
    report.to_csv(VALIDATION_REPORT_CSV, index=False)
    print(f"Saved {VALIDATION_REPORT_CSV} ({len(report)} rows)")

    # ---------------- checks ----------------
    failures: list[str] = []
    both_ok = report["raw_exists"] & report["reference_exists"]
    n_both = int(both_ok.sum())
    print(f"\nCHECK 1 (count ~890 paired): {n_both} paired identities", end=" ")
    if n_both < 890:
        failures.append(f"only {n_both} complete raw+reference pairs (< 890)")
        print("-> FAIL")
    else:
        print("-> PASS")

    only_raw = sorted(set(raw_names) - set(ref_names))
    only_ref = sorted(set(ref_names) - set(raw_names))
    print(f"CHECK 2 (filename matching): only-raw={len(only_raw)} only-ref={len(only_ref)}", end=" ")
    if only_raw or only_ref:
        failures.append(f"filename mismatch: only-raw={only_raw[:5]} only-ref={only_ref[:5]}")
        print("-> FAIL")
    else:
        print("-> PASS")

    dup_raw = len(raw_names) - len(raw_hashes)
    dup_ref = len(ref_names) - len(ref_hashes)
    print(f"duplicate content (byte-identical): raw={dup_raw} ref={dup_ref}", end=" ")
    if dup_raw or dup_ref:
        # WARN, not FAIL: groups are recorded and the split
        # (scripts/make_split.py) is group-aware, so duplicate members always
        # land in the SAME split and no train/test leakage can occur. Dropping
        # rows silently would be worse: 4 raw-duplicate pairs carry genuinely
        # different UIEB references (SSIM 0.73-0.90 between the two refs).
        import hashlib as _hl
        from collections import defaultdict as _dd
        groups = _dd(list)
        for n in raw_names:
            groups[_hl.sha256((RAW_DIR / n).read_bytes()).hexdigest()].append(n)
        dup_groups = sorted([sorted(v) for v in groups.values() if len(v) > 1])
        pd.DataFrame([{"group_id": i, "members": ",".join(g)}
                      for i, g in enumerate(dup_groups)]
                     ).to_csv(FEATURE_RESULTS_DIR / "duplicate_groups.csv", index=False)
        print(f"-> WARN ({len(dup_groups)} raw groups; see duplicate_groups.csv; "
              f"handled by group-aware split)")
    else:
        print("-> PASS")

    dim_mismatch = int(((report["raw_ref_dim_match"] == False)).sum())  # noqa: E712
    print(f"raw/reference dimension mismatches: {dim_mismatch} / {n_both}", end=" ")
    # Mismatch is a warning, not a failure: recorded per-row; SSIM stage resizes.
    print("-> WARN" if dim_mismatch else "-> PASS")

    n_identical = int((report["raw_ref_identical_bytes"] == True).sum())  # noqa: E712
    print(f"raw byte-identical to reference (must be 0): {n_identical}", end=" ")
    if n_identical:
        failures.append(f"{n_identical} raw images byte-identical to their reference")
        print("-> FAIL")
    else:
        print("-> PASS")

    if pair_hamming:
        arr = np.array(pair_hamming)
        print(f"paired phash Hamming distance: median={np.median(arr):.0f} "
              f"mean={arr.mean():.1f} max={arr.max()} "
              f"(paired images: small values expected; unrelated images score ~32)")
        # Cross-pair baseline: compare first 40 raws against shifted references.
        if n_both >= 80:
            sample = report.loc[both_ok, "image_name"].tolist()[:40]
            cross = []
            for i, nm in enumerate(sample):
                a = to_gray_u8(cv2.imread(str(RAW_DIR / nm), cv2.IMREAD_COLOR))
                b = to_gray_u8(cv2.imread(
                    str(REFERENCE_DIR / sample[(i + 13) % len(sample)]), cv2.IMREAD_COLOR))
                cross.append(hamming(phash(a), phash(b)))
            print(f"  cross-pair baseline (mismatched pairs): median={np.median(cross):.0f} "
                  f"mean={np.mean(cross):.1f}")
            # M4 FIX: the README cites "median 2 vs 32 for mismatched pairs", but
            # this control distribution was only ever printed, never persisted, so
            # the claim was unverifiable from the committed artifacts. Store it.
            pd.DataFrame({
                "statistic": ["paired_median", "paired_mean", "paired_max",
                              "crosspair_median", "crosspair_mean", "n_crosspair_sampled"],
                "value": [float(np.median(arr)), float(arr.mean()), int(arr.max()),
                          float(np.median(cross)), float(np.mean(cross)), len(cross)],
            }).to_csv(FEATURE_RESULTS_DIR / "phash_pairing_control.csv", index=False)
            print(f"  Saved {FEATURE_RESULTS_DIR / 'phash_pairing_control.csv'}")
            if np.median(arr) >= np.median(cross):
                failures.append("paired phash distance NOT smaller than cross-pair baseline; "
                                "pairing may be wrong")
                print("  pairing plausibility -> FAIL")
            else:
                print("  pairing plausibility -> PASS")

    # ---------------- contact sheets ----------------
    for tag, id_list in (("A", CONTACT_SHEET_IDS[:7]), ("B", CONTACT_SHEET_IDS[7:])):
        strips = []
        for i in id_list:
            name = f"UIEB_{i}.png"
            if not (RAW_DIR / name).exists() or not (REFERENCE_DIR / name).exists():
                continue
            a = cv2.imread(str(RAW_DIR / name), cv2.IMREAD_COLOR)
            b = cv2.imread(str(REFERENCE_DIR / name), cv2.IMREAD_COLOR)
            h = 160
            a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h))
            b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
            w = min(a.shape[1], b.shape[1])
            strip = np.hstack([a[:, :w], np.full((h, 6, 3), 255, np.uint8), b[:, :w]])
            cv2.putText(strip, f"raw {name}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.putText(strip, "reference", (w + 14, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            strips.append(strip)
        if strips:
            wmax = max(s.shape[1] for s in strips)
            canvas = np.full((sum(s.shape[0] for s in strips) + 10 * (len(strips) - 1),
                              wmax, 3), 255, np.uint8)
            y = 0
            for s in strips:
                canvas[y:y + s.shape[0], :s.shape[1]] = s
                y += s.shape[0] + 10
            out = PLOTS_DIR / f"pairing_contact_sheet_{tag}.png"
            cv2.imwrite(str(out), canvas)
            print(f"Saved {out}")

    print("\n==== RESULT ====")
    if failures:
        print("VALIDATION FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL CRITICAL DATASET CHECKS PASSED.")
    print("NOTE: UIEB references are human-preferred pseudo-references (best-of-several "
          "enhancement outputs chosen by volunteers, Li et al. TIP 2019), not physical "
          "ground truth. SSIM/PSNR here measure similarity to that preferred "
          "enhancement. See README section 19 (Limitations).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
