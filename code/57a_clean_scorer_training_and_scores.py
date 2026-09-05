"""
AquaTrust v3 — Step 57A: Scorer Training on Train & Clean Score Generation
==========================================================================
Enforces clean separation between scoring model training and CRC calibration:
  1. Extracts detection features on fold_{j}/train (819 images)
  2. Fits M2 (Conf-Only), M4 (Consistency), and M6 (Class-Interaction) strictly on train
  3. Predicts clean out-of-sample scores on:
     - clean_scores_fold_{j}_calib.csv (from fold_{j}/val, N=117 images)
     - clean_scores_fold_{j}_test.csv  (from fold_{j}/test, N=234 images)

Outputs:
  - D:/AquaTrust/results_v3/clean_scores_fold_{j}_calib.csv
  - D:/AquaTrust/results_v3/clean_scores_fold_{j}_test.csv
  - D:/AquaTrust/results_v3/clean_master_test_manifest.csv
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Configuration ──────────────────────────────────────────────
PROJECT_ROOT = Path(r"D:\AquaTrust")
SPLIT_ROOT   = PROJECT_ROOT / "data" / "split_v3"
RESULTS_DIR  = PROJECT_ROOT / "results_v3"
MODELS_DIR   = PROJECT_ROOT / "models" / "v3_nested"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE       = 832
SEED           = 2026
CONF_THRESHOLD = 0.15
IOU_GT_MATCH   = 0.50
K_PERTURBS     = 3
IMG_EXTS       = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ── Feature Extraction Functions ───────────────────────────────
def apply_degradation(img: np.ndarray, condition: str) -> np.ndarray:
    if condition == "C0_Clean": return img.copy()
    h, w = img.shape[:2]
    img_f = img.astype(np.float32) / 255.0
    if condition == "C1_Speckle":
        noise = np.random.normal(0, 0.28, img.shape).astype(np.float32)
        return np.clip(img_f * (1.0 + noise) * 255.0, 0, 255).astype(np.uint8)
    elif condition == "C2_Contrast":
        mean_val = np.mean(img_f)
        return np.clip((mean_val + 0.38 * (img_f - mean_val)) * 255.0, 0, 255).astype(np.uint8)
    elif condition == "C3_Dropout":
        deg = img.copy()
        for _ in range(np.random.randint(2, 5)):
            y_s = np.random.randint(0, max(1, h - 30))
            b_h = np.random.randint(8, 25)
            deg[y_s : y_s + b_h, :] = (deg[y_s : y_s + b_h, :] * 0.08).astype(np.uint8)
        return deg
    elif condition == "C4_Combined":
        noise = np.random.normal(0, 0.20, img.shape).astype(np.float32)
        deg_s = img_f * (1.0 + noise)
        mean_val = np.mean(deg_s)
        deg_c = np.clip((mean_val + 0.50 * (deg_s - mean_val)) * 255.0, 0, 255).astype(np.uint8)
        for _ in range(2):
            y_s = np.random.randint(0, max(1, h - 20))
            b_h = np.random.randint(6, 18)
            deg_c[y_s : y_s + b_h, :] = (deg_c[y_s : y_s + b_h, :] * 0.1).astype(np.uint8)
        return deg_c
    return img.copy()

def extract_quality(img_bgr: np.ndarray, box_xyxy: list) -> tuple:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, max(x1 + 4, x2)), min(h, max(y1 + 4, y2))
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    patch = img_gray[y1:y2, x1:x2].astype(np.float32)
    if patch.size < 4: return 0.0, 0.0
    mu, sigma = float(np.mean(patch)), float(np.std(patch))
    return round(float(sigma / 128.0), 5), round(float(sigma / (mu + 1e-4)), 5)

def compute_stability(model, base_img: np.ndarray, anchor_box: list, dev) -> tuple:
    x1_a, y1_a, x2_a, y2_a = anchor_box
    area_a = max(0, x2_a - x1_a) * max(0, y2_a - y1_a)
    ious, confs = [], []
    perturbations = [
        lambda im: np.clip(im.astype(np.float32) * 1.10, 0, 255).astype(np.uint8),
        lambda im: np.clip((im.astype(np.float32) - 128) * 1.15 + 128, 0, 255).astype(np.uint8),
        lambda im: np.clip(im + np.random.normal(0, 10, im.shape), 0, 255).astype(np.uint8)
    ]
    for p_fn in perturbations[:K_PERTURBS]:
        p_img = p_fn(base_img)
        res = model.predict(p_img, conf=0.10, iou=0.45, imgsz=IMG_SIZE, verbose=False, device=dev)
        best_iou, best_conf = 0.0, 0.0
        if len(res[0].boxes) > 0:
            for b in res[0].boxes:
                bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy()
                b_conf = float(b.conf[0].cpu().numpy())
                ix1, iy1 = max(x1_a, bx1), max(y1_a, by1)
                ix2, iy2 = min(x2_a, bx2), min(y2_a, by2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = area_a + (max(0, bx2 - bx1) * max(0, by2 - by1)) - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou: best_iou, best_conf = iou, b_conf
        ious.append(best_iou)
        confs.append(best_conf if best_iou > 0.1 else 0.0)
    return round(float(np.mean(ious)) if ious else 0.0, 4), round(float(np.var(confs)) if confs else 0.0, 5)

def load_gt(lbl_path: Path, img_w: int, img_h: int) -> list:
    if not lbl_path.exists(): return []
    gt = []
    lines = lbl_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    for l in lines:
        parts = l.strip().split()
        if len(parts) >= 5:
            cid = int(float(parts[0]))
            xc, yc, w, h = [float(v) for v in parts[1:5]]
            gt.append([(xc - w/2)*img_w, (yc - h/2)*img_h, (xc + w/2)*img_w, (yc + h/2)*img_h, cid])
    return gt

def match_gt(det_xyxy, det_cls, gt_boxes):
    if not gt_boxes: return 0, 0.0
    x1_d, y1_d, x2_d, y2_d = det_xyxy
    area_d = max(0, x2_d - x1_d) * max(0, y2_d - y1_d)
    best_iou, matched = 0.0, 0
    for gx1, gy1, gx2, gy2, gcls in gt_boxes:
        ix1, iy1 = max(x1_d, gx1), max(y1_d, gy1)
        ix2, iy2 = min(x2_d, gx2), min(y2_d, gy2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = area_d + ((gx2 - gx1) * (gy2 - gy1)) - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            if iou >= IOU_GT_MATCH and det_cls == gcls: matched = 1
    return matched, round(best_iou, 4)

def extract_features_from_images(split_dir: Path, role: str, fold_idx: int, model, dev) -> pd.DataFrame:
    cached_csv = RESULTS_DIR / f"features_fold_{fold_idx}_{role}.csv"
    if cached_csv.exists():
        return pd.read_csv(cached_csv)

    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    img_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

    conditions = ["C0_Clean", "C1_Speckle", "C2_Contrast", "C3_Dropout", "C4_Combined"]
    rows = []

    for idx, img_p in enumerate(img_paths, 1):
        base_img = cv2.imread(str(img_p))
        if base_img is None: continue
        h, w = base_img.shape[:2]
        gt_boxes = load_gt(lbl_dir / f"{img_p.stem}.txt", w, h)

        for cond in conditions:
            deg_img = apply_degradation(base_img, cond)
            results = model.predict(deg_img, conf=CONF_THRESHOLD, iou=0.45, imgsz=IMG_SIZE, verbose=False, device=dev)
            if len(results[0].boxes) == 0: continue

            for b in results[0].boxes:
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                conf = float(b.conf[0].cpu().numpy())
                pred_cls = int(b.cls[0].cpu().numpy())
                is_corr, iou_val = match_gt(xyxy, pred_cls, gt_boxes)
                qc, qv = extract_quality(deg_img, xyxy)
                si, vari = compute_stability(model, deg_img, xyxy, dev)

                rows.append({
                    "fold": fold_idx, "split_role": role, "image_name": img_p.name,
                    "condition": cond, "pred_class": pred_cls, "confidence": round(conf, 4),
                    "q_contrast": qc, "q_variability": qv, "box_consistency": si,
                    "conf_variance": vari, "is_correct": is_corr, "iou_gt": iou_val
                })

    df = pd.DataFrame(rows)
    df.to_csv(cached_csv, index=False)
    return df

def make_m6_feats(d: pd.DataFrame) -> np.ndarray:
    c  = d["confidence"].values
    qc = d["q_contrast"].values
    qv = d["q_variability"].values
    k  = d["pred_class"].values
    return np.column_stack([c, qc, qv, k, c*k, qc*k, qv*k])

def main():
    print("=" * 105)
    print("  AQUATRUST v3 — STEP 57A: SCORER TRAINING ON TRAIN & CLEAN SCORE GENERATION")
    print("=" * 105)

    dev = 0 if torch.cuda.is_available() else "cpu"
    master_test_dfs = []

    for f in range(1, 6):
        print(f"\n⚡ Processing Fold {f}/5...")
        fold_dir = SPLIT_ROOT / f"fold_{f}"
        weights_p = MODELS_DIR / f"fold_{f}" / "weights" / "best.pt"

        if not weights_p.exists():
            print(f"❌ Weights missing for Fold {f} at {weights_p}")
            return

        model = YOLO(str(weights_p))

        # 1. Extract / Load Features
        print(f"   • Loading Train Features (819 imgs)...")
        df_tr = extract_features_from_images(fold_dir / "train", "train", f, model, dev)
        print(f"   • Loading Val (CRC Calib) Features (117 imgs)...")
        df_va = extract_features_from_images(fold_dir / "val", "val", f, model, dev)
        print(f"   • Loading Test (Locked) Features (234 imgs)...")
        df_te = extract_features_from_images(fold_dir / "test", "test", f, model, dev)

        y_tr = df_tr["is_correct"].values
        y_va = df_va["is_correct"].values
        y_te = df_te["is_correct"].values

        # 2. Fit Scoring Models STRICTLY on Train Detections
        # M2: Conf-Only Platt
        sc_m2 = StandardScaler()
        X_tr_m2 = sc_m2.fit_transform(df_tr[["confidence"]])
        clf_m2 = LogisticRegression(C=1.0, solver="lbfgs", random_state=SEED).fit(X_tr_m2, y_tr)

        # M4: Consistency Specialist
        cons_cols = ["confidence", "box_consistency", "conf_variance"]
        sc_m4 = StandardScaler()
        X_tr_m4 = sc_m4.fit_transform(df_tr[cons_cols])
        clf_m4 = LogisticRegression(C=0.5, solver="lbfgs", random_state=SEED).fit(X_tr_m4, y_tr)

        # M6: AquaTrust Class-Interaction Champion
        sc_m6 = StandardScaler()
        X_tr_m6 = sc_m6.fit_transform(make_m6_feats(df_tr))
        clf_m6 = LogisticRegression(C=0.5, solver="lbfgs", random_state=SEED).fit(X_tr_m6, y_tr)

        # 3. Generate Clean Out-of-Sample Scores on Val (CRC Calib)
        df_calib_clean = df_va.copy()
        df_calib_clean["score_m1_raw"]  = df_va["confidence"].values
        df_calib_clean["score_m2_conf"] = clf_m2.predict_proba(sc_m2.transform(df_va[["confidence"]]))[:, 1]
        df_calib_clean["score_m4_cons"] = clf_m4.predict_proba(sc_m4.transform(df_va[cons_cols]))[:, 1]
        df_calib_clean["score_m6_inter"]= clf_m6.predict_proba(sc_m6.transform(make_m6_feats(df_va)))[:, 1]
        df_calib_clean.to_csv(RESULTS_DIR / f"clean_scores_fold_{f}_calib.csv", index=False)

        # 4. Generate Clean Out-of-Sample Scores on Test (Locked Test)
        df_test_clean = df_te.copy()
        df_test_clean["score_m1_raw"]  = df_te["confidence"].values
        df_test_clean["score_m2_conf"] = clf_m2.predict_proba(sc_m2.transform(df_te[["confidence"]]))[:, 1]
        df_test_clean["score_m4_cons"] = clf_m4.predict_proba(sc_m4.transform(df_te[cons_cols]))[:, 1]
        df_test_clean["score_m6_inter"]= clf_m6.predict_proba(sc_m6.transform(make_m6_feats(df_te)))[:, 1]
        df_test_clean.to_csv(RESULTS_DIR / f"clean_scores_fold_{f}_test.csv", index=False)

        master_test_dfs.append(df_test_clean)
        print(f"   ✅ Clean Scores Saved: Calib (N={len(df_calib_clean)}), Test (N={len(df_test_clean)})")

    # Save Master 5-Fold Test Manifest
    df_master = pd.concat(master_test_dfs, ignore_index=True)
    df_master.to_csv(RESULTS_DIR / "clean_master_test_manifest.csv", index=False)

    print("\n" + "=" * 105)
    print(f"  🎉 STEP 57A COMPLETE — All Scoring Models Trained on Train (N_total_test = {len(df_master)})")
    print(f"  Saved to: {RESULTS_DIR}")
    print("=" * 105)

if __name__ == "__main__":
    main()
