"""
AquaTrust v3 — Step 43: Deep Dataset Storage Audit & Fresh Holdout Feasibility
==============================================================================
Scans:
  1. Primary Raw Storage  : D:/AquaTrust/data/raw
  2. Original Source Path : D:/24574879 (if present)
  3. Image Hashes (MD5)   : Identifies duplicates and unindexed imagery
  4. Annotation Mapping   : Audits bounding box coordinate distributions

Outputs:
  - D:/AquaTrust/results_v2/raw_storage_audit_ledger.csv
"""

import os
import hashlib
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"D:\AquaTrust")
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
SOURCE_DIR   = Path(r"D:\24574879")
RESULTS_DIR  = PROJECT_ROOT / "results_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def compute_md5(file_path: Path) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def find_images(directory: Path):
    if not directory.exists():
        return []
    imgs = []
    for ext in IMG_EXTS:
        imgs.extend(directory.rglob(f"*{ext}"))
        imgs.extend(directory.rglob(f"*{ext.upper()}"))
    return sorted(set(imgs))

def get_label_info(img_p: Path, raw_root: Path):
    lbl_p = img_p.with_suffix(".txt")
    if not lbl_p.exists():
        lbl_p = raw_root / f"{img_p.stem}.txt"
    if not lbl_p.exists():
        return 0, 0, 0, False

    milco, nombo, total = 0, 0, 0
    try:
        lines = lbl_p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        for l in lines:
            parts = l.strip().split()
            if len(parts) >= 5:
                total += 1
                cid = int(float(parts[0]))
                if cid == 0: milco += 1
                elif cid == 1: nombo += 1
        return total, milco, nombo, True
    except Exception:
        return 0, 0, 0, False

def main():
    print("=" * 95)
    print("  AQUATRUST v3 — STEP 43: RAW DATASET STORAGE & HOLDOUT FEASIBILITY AUDIT")
    print("=" * 95)

    raw_images = find_images(RAW_DIR)
    source_images = find_images(SOURCE_DIR)

    print(f"📁 Images in D:/AquaTrust/data/raw : {len(raw_images)}")
    print(f"📁 Images in D:/24574879 (Source)  : {len(source_images)}\n")

    all_scanned = sorted(set(raw_images + source_images))
    print(f"🔍 Total Unique Filepaths Found: {len(all_scanned)}")

    records = []
    seen_hashes = {}

    for idx, img_p in enumerate(all_scanned, 1):
        md5_hash = compute_md5(img_p)
        is_duplicate = md5_hash in seen_hashes

        if not is_duplicate:
            seen_hashes[md5_hash] = img_p.name

        # Validate image readable
        img = cv2.imread(str(img_p))
        if img is not None:
            h, w = img.shape[:2]
            corrupt = False
        else:
            h, w = 0, 0
            corrupt = True

        total_b, m_b, n_b, has_lbl = get_label_info(img_p, RAW_DIR)

        records.append({
            "filename": img_p.name,
            "filepath": str(img_p),
            "directory": img_p.parent.name,
            "md5_hash": md5_hash,
            "is_duplicate_hash": is_duplicate,
            "img_width": w,
            "img_height": h,
            "is_corrupt": corrupt,
            "has_label_file": has_lbl,
            "total_boxes": total_b,
            "milco_boxes": m_b,
            "nombo_boxes": n_b
        })

    df = pd.DataFrame(records)
    out_csv = RESULTS_DIR / "raw_storage_audit_ledger.csv"
    df.to_csv(out_csv, index=False)

    # ── Summary Metrics ────────────────────────────────────────
    unique_hashes = df["md5_hash"].nunique()
    total_corrupt = df["is_corrupt"].sum()
    with_labels   = df[df["has_label_file"] == True]
    bg_images     = df[df["total_boxes"] == 0]
    target_images = df[df["total_boxes"] > 0]

    print("─" * 95)
    print("  📊 AUDIT SUMMARY")
    print("─" * 95)
    print(f"  • Total Scanned Files              : {len(df)}")
    print(f"  • Cryptographically Unique Images  : {unique_hashes}")
    print(f"  • Total Corrupted / Unreadable     : {total_corrupt}")
    print(f"  • Images with Paired Labels        : {len(with_labels)}")
    print(f"  • Target Images (Annotated Objects): {len(target_images)} ({len(target_images)/len(df)*100:.1f}%)")
    print(f"  • Pure Background / Clutter Images : {len(bg_images)} ({len(bg_images)/len(df)*100:.1f}%)")
    print(f"  • Total Bounding Boxes Counted     : {df['total_boxes'].sum()} (MILCO: {df['milco_boxes'].sum()}, NOMBO: {df['nombo_boxes'].sum()})")

    print("\n" + "─" * 95)
    print("  🔒 PROTOCOL DECISION FOR AQUATRUST v3")
    print("─" * 95)
    if unique_hashes == 1170:
        print("  [STATUS] The dataset pool is exactly 1,170 unique physical side-scan sonar images.")
        print("  [RECOMMENDATION] For publication-grade AquaTrust v3, we must implement an:")
        print("                   'Outer 5-Fold Stratified Nested Cross-Validation Protocol'")
        print("                   where each fold acts as an independent, untouched locked test set (N=234).")
        print("                   This yields an aggregate locked test size of N=1,170 with zero test leakage!")
    else:
        print(f"  [STATUS] Found {unique_hashes - 1170} additional unindexed unique images!")
        print("  [RECOMMENDATION] Construct a completely fresh, previously unexposed held-out test split.")

    print("\n" + "=" * 95)
    print(f"  ✅ STEP 43 COMPLETE — Manifest saved to: {out_csv}")
    print("=" * 95)

if __name__ == "__main__":
    main()
