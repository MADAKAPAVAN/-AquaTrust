"""
AquaTrust v3 — Step 47: Robust Resumable 5-Fold Training & Feature Extractor
=============================================================================
Features:
  - Automatically skips folds that are already completed
  - Explicit VRAM cache purging (torch.cuda.empty_cache()) to prevent memory leaks
  - Safe worker thread allocation for Windows
  - Resumes cleanly from where it stopped
"""

import os
import gc
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(r"D:\AquaTrust")
SPLIT_ROOT   = PROJECT_ROOT / "data" / "split_v3"
OUTPUT_DIR   = PROJECT_ROOT / "results_v3"
MODELS_DIR   = PROJECT_ROOT / "models" / "v3_nested"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE       = 832
BATCH_SIZE     = 8
EPOCHS         = 50
PATIENCE       = 12
SEED           = 2026
CONF_THRESHOLD = 0.15
IOU_GT_MATCH   = 0.50
K_PERTURBS     = 3
IMG_EXTS       = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def apply_degradation(img: np.ndarray, condition: str) -> np.ndarray:
    if condition == "C0_Clean":
        return img.copy()

    h, w = img.shape[:2]
    img_f = img.astype(np.float32) / 255.0

    if condition == "C1_Speckle":
        noise = np.random.normal(0, 0.28, img.shape).astype(np.float32)
        degraded = img_f * (1.0 + noise)
        return np.clip(degraded * 255.0, 0, 255).astype(np.uint8)

    elif condition == "C2_Contrast":
        mean_val = np.mean(img_f)
        degraded = mean_val + 0.38 * (img_f - mean_val)
        return np.clip(degraded * 255.0, 0, 255).astype(np.uint8)

    elif condition == "C3_Dropout":
        degraded = img.copy()
        num_bands = np.random.randint(2, 5)
        for _ in range(num_bands):
            y_start = np.random.randint(0, max(1, h - 30))
            band_h  = np.random.randint(8, 25)
            degraded[y_start : y_start + band_h, :] = (degraded[y_start : y_start + band_h, :] * 0.08).astype(np.uint8)
        return degraded

    elif condition == "C4_Combined":
        noise = np.random.normal(0, 0.20, img.shape).astype(np.float32)
        deg_speckle = img_f * (1.0 + noise)
        mean_val = np.mean(deg_speckle)
        deg_contrast = mean_val + 0.50 * (deg_speckle - mean_val)
        deg_uint8 = np.clip(deg_contrast * 255.0, 0, 255).astype(np.uint8)
        for _ in range(2):
            y_start = np.random.randint(0, max(1, h - 20))
            band_h = np.random.randint(6, 18)
            deg_uint8[y_start : y_start + band_h, :] = (deg_uint8[y_start : y_start + band_h, :] * 0.1).astype(np.uint8)
        return deg_uint8

    return img.copy()

def extract_quality(img_bgr: np.ndarray, box_xyxy: list) -> tuple:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, max(x1 + 4, x2)), min(h, max(y1 + 4, y2))

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    patch = img_gray[y1:y2, x1:x2].astype(np.float32)

    if patch.size < 4:
        return 0.0, 0.0

    mu = float(np.mean(patch))
    sigma = float(np.std(patch))
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
                area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
                union = area_a + area_b - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_conf = b_conf

        ious.append(best_iou)
        confs.append(best_conf if best_iou > 0.1 else 0.0)

    s_i = float(np.mean(ious)) if ious else 0.0
    var_i = float(np.var(confs)) if confs else 0.0
    return round(s_i, 4), round(var_i, 5)

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
        area_g = max(0, gx2 - gx1) * max(0, gy2 - gy1)
        iou = inter / (area_d + area_g - inter) if (area_d + area_g - inter) > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            if iou >= IOU_GT_MATCH and det_cls == gcls:
                matched = 1
    return matched, round(best_iou, 4)

def extract_split_features(split_dir: Path, role: str, fold_idx: int, model, dev):
    out_csv = OUTPUT_DIR / f"features_fold_{fold_idx}_{role}.csv"
    if out_csv.exists():
        print(f"     ℹ️ Manifest already exists: {out_csv.name} (Skipping extraction)")
        return pd.read_csv(out_csv)

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
                    "fold": fold_idx,
                    "split_role": role,
                    "image_name": img_p.name,
                    "condition": cond,
                    "pred_class": pred_cls,
                    "confidence": round(conf, 4),
                    "q_contrast": qc,
                    "q_variability": qv,
                    "box_consistency": si,
                    "conf_variance": vari,
                    "is_correct": is_corr,
                    "iou_gt": iou_val
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"     ✅ Saved {role.upper()} ({len(df)} detections) -> {out_csv.name}")
    return df

def main():
    print("=" * 85)
    print("  AQUATRUST v3 — RESUMABLE 5-FOLD TRAINING & EXTRACTION ENGINE")
    print("=" * 85)

    cuda_avail = torch.cuda.is_available()
    dev = 0 if cuda_avail else "cpu"
    dev_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"

    print(f"🖥️  Compute Device : {dev_name} (device={dev})")
    print(f"⚙️  Architecture   : YOLOv8s @ {IMG_SIZE}x{IMG_SIZE}")
    print(f"📦 Batch / Epochs : Batch {BATCH_SIZE}, Epochs {EPOCHS} (Patience {PATIENCE})\n")

    for f in range(1, 6):
        print("─" * 85)
        print(f"  🔥 FOLD {f}/5")
        print("─" * 85)

        fold_dir   = SPLIT_ROOT / f"fold_{f}"
        yaml_path  = fold_dir / "data.yaml"
        weights_p  = MODELS_DIR / f"fold_{f}" / "weights" / "best.pt"

        # Explicit GPU memory cleanup before each fold
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not weights_p.exists():
            print(f"🚀 Training YOLOv8s on Fold {f}...")
            model = YOLO("yolov8s.pt")
            model.train(
                data=str(yaml_path),
                epochs=EPOCHS,
                imgsz=IMG_SIZE,
                batch=BATCH_SIZE,
                patience=PATIENCE,
                device=dev,
                workers=4,
                project=str(MODELS_DIR),
                name=f"fold_{f}",
                exist_ok=True,
                mosaic=0.5,
                mixup=0.0,
                degrees=10.0,
                fliplr=0.5,
                flipud=0.0,
                seed=SEED,
                deterministic=True,
                save=True,
                plots=False
            )
            # Free model from memory after training
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(f"📦 Weights already exist: {weights_p}")

        # Load checkpoint for feature extraction
        trained_model = YOLO(str(weights_p))
        print(f"🔍 Extracting features for Fold {f}...")
        extract_split_features(fold_dir / "val",  "val",  f, trained_model, dev)
        extract_split_features(fold_dir / "test", "test", f, trained_model, dev)

        del trained_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 85)
    print("  🎉 ALL 5 FOLDS SUCCESSFULLY TRAINED & EXTRACTED")
    print(f"  Feature manifests saved to: {OUTPUT_DIR}")
    print("=" * 85)

if __name__ == "__main__":
    main()
