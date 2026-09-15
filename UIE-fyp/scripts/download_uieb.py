#!/usr/bin/env python3
"""Download the real UIEB dataset and organise it into dataset/{raw-890,reference-890}.

Official UIEB source (Li et al., TIP 2019):
    https://li-chongyi.github.io/proj_benchmark.html
    - 890 raw underwater images (~630 MB) + 890 reference images (~786 MB)
    - Official links are Google Drive / Baidu Cloud (see README for IDs).

Because Google Drive is unreachable from some sandboxes, this script uses the
GitHub mirror JJsnowx/UIEB_Dataset (verified: 890 raw + 890 reference PNGs,
identical contiguous filenames UIEB_0.png .. UIEB_889.png, plus 60
challenging images), downloaded as a single tarball from codeload.github.com.

The script is resumable (curl -C -) and verifies file counts + name
correspondence after extraction. Pairing *content* validity is checked
separately by scripts/validate_dataset.py (dimensions, readability,
raw-vs-reference similarity sanity checks).

Usage:
    python scripts/download_uieb.py [--keep-tarball]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
RAW_DIR = DATASET_DIR / "raw-890"
REF_DIR = DATASET_DIR / "reference-890"
CHALLENGE_DIR = DATASET_DIR / "challenging-60"
TMP_DIR = DATASET_DIR / "_tmp_uieb"

TARBALL_URL = "https://codeload.github.com/JJsnowx/UIEB_Dataset/tar.gz/refs/heads/main"
TARBALL = DATASET_DIR / "_uieb_mirror.tar.gz"

# Paths inside the mirror tarball (top-level dir is UIEB_Dataset-main/).
MIRROR_RAW = "UIEB_Dataset-main/UIEB Dataset/raw/UIEB_raw_reName"
MIRROR_REF = "UIEB_Dataset-main/UIEB Dataset/reference/UIEB_reference_reName"
MIRROR_CH = "UIEB_Dataset-main/UIEB Dataset/challenge/challenging-60"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def download_tarball() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    if TARBALL.exists() and TARBALL.stat().st_size > 1_000_000_000:
        print(f"Tarball already present ({TARBALL.stat().st_size/1e9:.2f} GB), skipping download.")
        return
    print(f"Downloading UIEB mirror tarball (~1.6 GB) -> {TARBALL}")
    print("This may take a while. The download is resumable; re-run on failure.")
    run([
        "curl", "-L", "--retry", "5", "--retry-all-errors", "-C", "-",
        "--max-time", "3600", "-o", str(TARBALL), TARBALL_URL,
    ])


def extract_and_organise() -> None:
    for d in (RAW_DIR, REF_DIR, CHALLENGE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True)
    print(f"Extracting {TARBALL} ...")
    with tarfile.open(TARBALL, "r:gz") as tf:
        members = [m for m in tf.getmembers()
                   if m.name.startswith(("UIEB_Dataset-main/UIEB Dataset/raw/",
                                          "UIEB_Dataset-main/UIEB Dataset/reference/",
                                          "UIEB_Dataset-main/UIEB Dataset/challenge/"))]
        print(f"  {len(members)} members to extract")
        tf.extractall(TMP_DIR, members=members)  # noqa: S202 - trusted mirror tarball

    mapping = [(TMP_DIR / MIRROR_RAW, RAW_DIR),
               (TMP_DIR / MIRROR_REF, REF_DIR),
               (TMP_DIR / MIRROR_CH, CHALLENGE_DIR)]
    for src, dst in mapping:
        files = sorted(p for p in src.iterdir() if p.is_file()) if src.is_dir() else []
        print(f"  {src.name}: {len(files)} files -> {dst}")
        for f in files:
            shutil.move(str(f), str(dst / f.name))
    shutil.rmtree(TMP_DIR, ignore_errors=True)


def verify() -> bool:
    raw = sorted(p.name for p in RAW_DIR.glob("*") if p.is_file())
    ref = sorted(p.name for p in REF_DIR.glob("*") if p.is_file())
    ch = sorted(p.name for p in CHALLENGE_DIR.glob("*") if p.is_file())
    print(f"raw-890: {len(raw)} files | reference-890: {len(ref)} files | challenging-60: {len(ch)} files")
    ok = True
    if len(raw) != 890:
        print(f"ERROR: expected 890 raw images, found {len(raw)}"); ok = False
    if len(ref) != 890:
        print(f"ERROR: expected 890 reference images, found {len(ref)}"); ok = False
    if set(raw) != set(ref):
        print(f"ERROR: raw/reference filename mismatch "
              f"(only-raw={len(set(raw)-set(ref))}, only-ref={len(set(ref)-set(raw))})"); ok = False
    else:
        print("Filename correspondence raw <-> reference: OK (890/890 identical names)")
    if ok:
        print("DATASET DOWNLOAD VERIFIED OK")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-tarball", action="store_true",
                    help="Keep the 1.6 GB mirror tarball after extraction (default: delete).")
    args = ap.parse_args()
    try:
        download_tarball()
        extract_and_organise()
    except (subprocess.CalledProcessError, tarfile.TarError, OSError) as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    ok = verify()
    if ok and not args.keep_tarball and TARBALL.exists():
        TARBALL.unlink()
        print("Removed tarball to save space.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
