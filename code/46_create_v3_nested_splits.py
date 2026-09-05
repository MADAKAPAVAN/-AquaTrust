"""
AquaTrust v3 — Step 46: Nested 5-Fold Stratified Partitioning & Directory Generator
==================================================================================
Partitions 1,170 unique physical side-scan sonar images into 5 stratified outer folds.
For each fold j in {1..5}:
  - test/  (20% - 234 imgs): Untouched final test split
  - train/ (70% - 819 imgs): YOLO training split
  - val/   (10% - 117 imgs): YOLO validation and calibration split

Stratification categories:
  1. background (no objects)
  2. MILCO_only
  3. NOMBO_only
  4. both_classes

Outputs:
  - D:/AquaTrust/data/split_v3/fold_j/ (Directory tree)
  - D:/AquaTrust/data/split_v3/fold_j/data.yaml (YOLO config)
  - D:/AquaTrust/data/split_v3/nested_5fold_manifest.csv (Master audit manifest)
"""

import os
import shutil
import random
import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

# ── Configuration ──────────────────────────────────────────────
PROJECT_ROOT = Path(r"D:\AquaTrust")
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
SPLIT_ROOT   = PROJECT_ROOT / "data" / "split_v3"

SEED         = 2026   # Fixed scientific seed
CLASS_NAMES  = {0: "MILCO", 1: "NOMBO"}
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def find_all_images(directory: Path) -> list:
    imgs = []
    for ext in IMG_EXTS:
        imgs.extend(directory.rglob(f"*{ext}"))
        imgs.extend(directory.rglob(f"*{ext.upper()}"))
    return sorted(set(imgs))

def get_label_path(img_path: Path) -> Path:
    cand = img_path.with_suffix(".txt")
    if cand.exists(): return cand
    cand = RAW_DIR / f"{img_path.stem}.txt"
    if cand.exists(): return cand
    cand = img_path.parent / f"{img_path.stem}.txt"
    return cand

def read_classes(img_path: Path):
    lbl_path = get_label_path(img_path)
    if not lbl_path.exists():
        return set(), 0, 0, []
    
    classes = set()
    milco, nombo = 0, 0
    boxes = []
    
    try:
        lines = lbl_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        for l in lines:
            parts = l.strip().split()
            if len(parts) >= 5:
                cid = int(float(parts[0]))
                classes.add(cid)
                boxes.append(parts)
                if cid == 0: milco += 1
                elif cid == 1: nombo += 1
    except Exception:
        pass
        
    return classes, milco, nombo, boxes

def get_strat_group(img_path: Path) -> str:
    classes, _, _, _ = read_classes(img_path)
    if len(classes) == 0: return "background"
    if classes == {0}:    return "MILCO_only"
    if classes == {1}:    return "NOMBO_only"
    return "both_classes"

def main():
    print("=" * 75)
    print("  AQUATRUST v3 — NESTED 5-FOLD STRATIFIED PARTITION GENERATOR")
    print("=" * 75)

    images = find_all_images(RAW_DIR)
    print(f"📁 Source Images Found in raw: {len(images)}")
    if len(images) == 0:
        print(f"❌ No images found in {RAW_DIR}")
        return

    # 1. Stratification Grouping
    rng = random.Random(SEED)
    img_data = []
    for img in images:
        grp = get_strat_group(img)
        total, m_cnt, n_cnt, _ = read_classes(img)
        img_data.append({
            "path": img,
            "name": img.name,
            "group": grp,
            "total_boxes": total,
            "milco_boxes": m_cnt,
            "nombo_boxes": n_cnt
        })

    df_img = pd.DataFrame(img_data)
    print("\n🔍 Class Stratification Groups:")
    for g_name, g_df in df_img.groupby("group"):
        print(f"   • {g_name:18s}: {len(g_df):4d} images")

    # 2. Build 5 Stratified Outer Folds (20% Test, 80% Dev pool)
    skf_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    # Track assignments for the master manifest
    manifest_rows = []
    for f in range(1, 6):
        df_img[f"fold_{f}_role"] = ""

    # Generate Outer Folds
    for fold_idx, (dev_indices, test_idx) in enumerate(skf_outer.split(df_img["path"], df_img["group"]), 1):
        # Mark Outer Test
        df_img.loc[test_idx, f"fold_{fold_idx}_role"] = "test"
        
        # Segment Dev Pool (80%) into Train (70% of total) and Val (10% of total)
        # Train / Val split ratio inside Dev Pool = 70/80 = 87.5% Train, 12.5% Val
        df_dev = df_img.iloc[dev_indices].reset_index()
        skf_inner = StratifiedKFold(n_splits=8, shuffle=True, random_state=SEED)
        
        # Take 1st split for val (1/8 ≈ 12.5%), remaining 7/8 for train
        for train_pool_idx, val_pool_idx in skf_inner.split(df_dev["path"], df_dev["group"]):
            val_orig_idx = df_dev.iloc[val_pool_idx]["index"].values
            train_orig_idx = df_dev.iloc[train_pool_idx]["index"].values
            df_img.loc[val_orig_idx, f"fold_{fold_idx}_role"] = "val"
            df_img.loc[train_orig_idx, f"fold_{fold_idx}_role"] = "train"
            break

    # Clean prior split folders
    if SPLIT_ROOT.exists():
        shutil.rmtree(SPLIT_ROOT)
    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)

    # 3. Create Directories and Copy File Pairs
    print("\n📦 Generating fold directories and copying image pairs...")
    
    for f in range(1, 6):
        fold_dir = SPLIT_ROOT / f"fold_{f}"
        for split in ["train", "val", "test"]:
            (fold_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (fold_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        # Copy Files
        for _, row in df_img.iterrows():
            img_p = row["path"]
            role  = row[f"fold_{f}_role"]
            
            dst_img = fold_dir / role / "images" / img_p.name
            dst_lbl = fold_dir / role / "labels" / f"{img_p.stem}.txt"
            
            shutil.copy2(str(img_p), str(dst_img))
            src_lbl = get_label_path(img_p)
            if src_lbl.exists():
                shutil.copy2(str(src_lbl), str(dst_lbl))
            else:
                dst_lbl.touch()

        # Write data.yaml for this fold
        yaml_content = {
            "path": str(fold_dir.resolve()).replace("\\", "/"),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "names": {i: name for i, name in CLASS_NAMES.items()}
        }
        with open(fold_dir / "data.yaml", "w", encoding="utf-8") as yf:
            yaml.dump(yaml_content, yf, default_flow_style=False, sort_keys=False)

        print(f"   ✅ Created Fold {f} directory + data.yaml")

    # 4. Save Master Audit Manifest
    manifest_df = df_img.drop(columns=["path"])
    manifest_path = SPLIT_ROOT / "nested_5fold_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\n📝 Master Nested-5Fold Manifest Saved: {manifest_path}")

    # 5. Output Verification Report
    print("\n" + "=" * 75)
    print("  📊 AQUATRUST v3 PARTITION SUMMARY REPORT")
    print("=" * 75)
    for f in range(1, 6):
        print(f"\n👉 Fold {f} Role Allocations:")
        summary = manifest_df.groupby(f"fold_{f}_role").agg(
            Images=("name", "count"),
            Total_Boxes=("total_boxes", "sum"),
            MILCO=("milco_boxes", "sum"),
            NOMBO=("nombo_boxes", "sum")
        ).reset_index()
        print(summary.to_string(index=False))

    print("\n" + "=" * 75)
    print("  ✅ NESTED 5-FOLD INFRASTRUCTURE COMPLETE")
    print("=" * 75)

if __name__ == "__main__":
    main()
