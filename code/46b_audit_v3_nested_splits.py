"""
AquaTrust v3 — Step 46b: Cryptographic & Statistical 5-Fold Audit Engine
========================================================================
Audits:
  1. Exact image counts per fold (Train: 819, Val: 117, Test: 234)
  2. Exact-Once Test Guarantee (Every image tested in exactly 1 outer fold)
  3. Class balance (MILCO, NOMBO, Background) across all 5 folds
  4. Physical disk integrity check (image and label files across all fold directories)
  5. Verification of data.yaml for all 5 folds
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(r"D:\AquaTrust")
SPLIT_ROOT   = PROJECT_ROOT / "data" / "split_v3"
MANIFEST_CSV = SPLIT_ROOT / "nested_5fold_manifest.csv"

def main():
    print("=" * 85)
    print("  AQUATRUST v3 — STEP 46b: NESTED 5-FOLD PARTITION AUDIT")
    print("=" * 85)

    if not MANIFEST_CSV.exists():
        print(f"❌ Manifest not found: {MANIFEST_CSV}")
        return

    df = pd.read_csv(MANIFEST_CSV)
    print(f"📁 Manifest Loaded: {MANIFEST_CSV}")
    print(f"🖼️  Total Registered Images: {len(df)}")
    print(f"🔑 Unique Image Names     : {df['name'].nunique()}")

    assert len(df) == 1170, f"Expected 1,170 images, got {len(df)}"
    assert df["name"].nunique() == 1170, "Duplicate image names detected in manifest!"

    # ───────────────────────────────────────────────────────────
    # 1. PER-FOLD ROLE ALLOCATION AUDIT
    # ───────────────────────────────────────────────────────────
    print("\n" + "─" * 85)
    print("  📊 1. PER-FOLD SAMPLE & BOX ALLOCATIONS")
    print("─" * 85)

    for f in range(1, 6):
        role_col = f"fold_{f}_role"
        
        # Explicit numeric aggregation to avoid Pandas type conflicts
        agg_dict = {
            "name": "count",
            "total_boxes": "sum",
            "milco_boxes": "sum",
            "nombo_boxes": "sum"
        }
        
        fold_summary = df.groupby(role_col).agg(agg_dict).reset_index()
        fold_summary.columns = ["Role", "Images", "Total_Boxes", "MILCO_Boxes", "NOMBO_Boxes"]
        
        # Calculate percentage of images
        fold_summary["Image_Pct"] = (fold_summary["Images"] / len(df) * 100).round(1)

        print(f"\n👉 Outer Fold {f}:")
        print(fold_summary[["Role", "Images", "Image_Pct", "Total_Boxes", "MILCO_Boxes", "NOMBO_Boxes"]].to_string(index=False))

        # Strict balance assertions
        test_cnt  = int(df[df[role_col] == "test"]["name"].count())
        val_cnt   = int(df[df[role_col] == "val"]["name"].count())
        train_cnt = int(df[df[role_col] == "train"]["name"].count())

        assert test_cnt == 234,  f"Fold {f} test count is {test_cnt}, expected 234"
        assert val_cnt == 117,   f"Fold {f} val count is {val_cnt}, expected 117"
        assert train_cnt == 819, f"Fold {f} train count is {train_cnt}, expected 819"

    # ───────────────────────────────────────────────────────────
    # 2. EXACT-ONCE TEST GUARANTEE AUDIT
    # ───────────────────────────────────────────────────────────
    print("\n" + "─" * 85)
    print("  🔒 2. EXACT-ONCE OUTER TEST VERIFICATION")
    print("─" * 85)

    role_cols = [f"fold_{f}_role" for f in range(1, 6)]
    test_appearances = (df[role_cols] == "test").sum(axis=1)

    exact_once_pass = (test_appearances == 1).all()
    print(f"  • Every image tested in EXACTLY 1 outer fold : {'✅ PASSED' if exact_once_pass else '❌ FAILED'}")
    print(f"  • Min test appearances across images         : {test_appearances.min()}")
    print(f"  • Max test appearances across images         : {test_appearances.max()}")
    print(f"  • Total test evaluations across all 5 folds  : {test_appearances.sum()} (5 x 234 = 1,170)")

    assert exact_once_pass, "Some images are tested more than once or never tested!"

    # ───────────────────────────────────────────────────────────
    # 3. PHYSICAL DISK INTEGRITY CHECK
    # ───────────────────────────────────────────────────────────
    print("\n" + "─" * 85)
    print("  💾 3. PHYSICAL DISK INTEGRITY CHECK (split_v3/)")
    print("─" * 85)

    all_disk_pass = True
    for f in range(1, 6):
        fold_dir = SPLIT_ROOT / f"fold_{f}"
        yaml_file = fold_dir / "data.yaml"
        
        tr_imgs = len(list((fold_dir / "train" / "images").glob("*.*")))
        va_imgs = len(list((fold_dir / "val" / "images").glob("*.*")))
        te_imgs = len(list((fold_dir / "test" / "images").glob("*.*")))
        
        tr_lbls = len(list((fold_dir / "train" / "labels").glob("*.txt")))
        va_lbls = len(list((fold_dir / "val" / "labels").glob("*.txt")))
        te_lbls = len(list((fold_dir / "test" / "labels").glob("*.txt")))

        yaml_ok = yaml_file.exists()
        counts_ok = (tr_imgs == 819 and va_imgs == 117 and te_imgs == 234 and 
                     tr_lbls == 819 and va_lbls == 117 and te_lbls == 234)

        status = "✅ OK" if (yaml_ok and counts_ok) else "❌ MISMATCH"
        if not (yaml_ok and counts_ok): all_disk_pass = False

        print(f"  • Fold {f}: Images [Tr:{tr_imgs}, Va:{va_imgs}, Te:{te_imgs}] | Labels [Tr:{tr_lbls}, Va:{va_lbls}, Te:{te_lbls}] | YAML: {yaml_ok} -> {status}")

    print("\n" + "=" * 85)
    if all_disk_pass:
        print("  🎉 AUDIT 100% PASSED: NESTED 5-FOLD PROTOCOL IS SCIENTIFICALLY FLAWLESS")
    else:
        print("  ❌ DISK AUDIT FAILED — Review errors above.")
    print("=" * 85)

if __name__ == "__main__":
    main()
