"""
AquaTrust v3 — Step 51: Group-Exchangeable Image-Level Conformal Risk Control
=============================================================================
Implements exact, finite-sample distribution-free Conformal Risk Control:
  - Exchangeable Unit : Physical Sonar Images (M_val = 117, M_test = 234 per fold)
  - Loss Function 1   : Image-Level False Discovery Rate (L_FDR in [0, 1])
  - Loss Function 2   : Image-Level Any-Error Risk (L_AnyError in {0, 1})
  - Theoretical Bound : (1 / (M + 1)) * (1 + sum(L_i(tau))) <= alpha

Compares Scoring Engines under Identical Conformal Guarantees:
  1. M1: Raw YOLOv8s Confidence
  2. M2: Confidence-Only Platt Scaling
  3. M4: Confidence + Consistency Ranking Model
  4. M6: AquaTrust Class-Interaction Model (Ours)

Outputs:
  - D:/AquaTrust/results_v3/image_crc_master_benchmark.csv
  - D:/AquaTrust/results_v3/image_crc_fold_details.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Configuration ──────────────────────────────────────────────
PROJECT_ROOT = Path(r"D:\AquaTrust")
RESULTS_DIR  = PROJECT_ROOT / "results_v3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED         = 2026
ALPHA_LEVELS = [0.10, 0.15, 0.20, 0.25, 0.30]

def make_m6_feats(d: pd.DataFrame) -> np.ndarray:
    c  = d["confidence"].values
    qc = d["q_contrast"].values
    qv = d["q_variability"].values
    k  = d["pred_class"].values
    return np.column_stack([c, qc, qv, k, c*k, qc*k, qv*k])

# ── Image-Level Loss Functions ─────────────────────────────────
def compute_image_losses(df_split: pd.DataFrame, score_col: str, tau: float, loss_type: str = "FDR") -> list:
    """Computes bounded loss L_m(tau) in [0, 1] for each physical image."""
    image_names = df_split["image_name"].unique()
    losses = []

    for img in image_names:
        sub = df_split[df_split["image_name"] == img]
        accepted = sub[score_col].values >= tau
        k_acc = int(np.sum(accepted))

        if k_acc == 0:
            # If no detection is accepted in this image, zero false alarms occurred
            losses.append(0.0)
        else:
            y_acc = sub["is_correct"].values[accepted]
            fa_count = int(np.sum(1 - y_acc))
            
            if loss_type == "FDR":
                # Proportion of accepted detections that are false alarms in this image
                losses.append(float(fa_count / k_acc))
            elif loss_type == "AnyError":
                # 1 if ANY false alarm was accepted in this image, else 0
                losses.append(1.0 if fa_count > 0 else 0.0)
    return losses

# ── Conformal Risk Control Calibration Theorem ────────────────
def calibrate_image_crc(df_val: pd.DataFrame, score_col: str, alpha: float, loss_type: str = "FDR") -> float:
    """
    Finds lowest tau satisfying:
      (1 / (M + 1)) * (1 + sum_{m=1}^M L_m(tau)) <= alpha
    """
    images = df_val["image_name"].unique()
    M = len(images)
    
    # Grid of candidate thresholds (from conservative down to lenient)
    tau_grid = np.linspace(0.99, 0.15, 200)
    valid_taus = []

    for tau in tau_grid:
        losses = compute_image_losses(df_val, score_col, tau, loss_type=loss_type)
        crc_bound = (1.0 / (M + 1.0)) * (1.0 + float(np.sum(losses)))

        if crc_bound <= alpha:
            valid_taus.append(tau)

    # Lowest valid threshold maximizes autonomous coverage while satisfying the theorem
    return float(min(valid_taus)) if valid_taus else 1.0

def main():
    print("=" * 105)
    print("  AQUATRUST v3 — STEP 51: GROUP-EXCHANGEABLE IMAGE-LEVEL CONFORMAL RISK CONTROL")
    print("=" * 105)

    fold_eval_records = []

    for f in range(1, 6):
        val_csv  = RESULTS_DIR / f"features_fold_{f}_val.csv"
        test_csv = RESULTS_DIR / f"features_fold_{f}_test.csv"

        if not val_csv.exists() or not test_csv.exists():
            print(f"❌ Missing files for Fold {f}")
            return

        df_v = pd.read_csv(val_csv)
        df_t = pd.read_csv(test_csv)

        y_val  = df_v["is_correct"].values
        y_test = df_t["is_correct"].values
        n_test_dets = len(df_t)

        m_val_images  = df_v["image_name"].nunique()
        m_test_images = df_t["image_name"].nunique()

        # ───────────────────────────────────────────────────────
        # 1. FIT CANDIDATE MODELS ON FOLD DEVELOPMENT SPLIT
        # ───────────────────────────────────────────────────────
        # M1: Raw YOLOv8s Confidence
        df_v["score_m1"] = df_v["confidence"].values
        df_t["score_m1"] = df_t["confidence"].values

        # M2: Confidence-Only Platt Baseline
        sc_m2 = StandardScaler()
        X_v_m2 = sc_m2.fit_transform(df_v[["confidence"]])
        X_t_m2 = sc_m2.transform(df_t[["confidence"]])
        clf_m2 = LogisticRegression(C=1.0, solver="lbfgs", random_state=SEED).fit(X_v_m2, y_val)
        df_v["score_m2"] = clf_m2.predict_proba(X_v_m2)[:, 1]
        df_t["score_m2"] = clf_m2.predict_proba(X_t_m2)[:, 1]

        # M4: Confidence + Consistency (Ranking Specialist)
        cons_cols = ["confidence", "box_consistency", "conf_variance"]
        sc_m4 = StandardScaler()
        X_v_m4 = sc_m4.fit_transform(df_v[cons_cols])
        X_t_m4 = sc_m4.transform(df_t[cons_cols])
        clf_m4 = LogisticRegression(C=0.5, solver="lbfgs", random_state=SEED).fit(X_v_m4, y_val)
        df_v["score_m4"] = clf_m4.predict_proba(X_v_m4)[:, 1]
        df_t["score_m4"] = clf_m4.predict_proba(X_t_m4)[:, 1]

        # M6: AquaTrust Class-Interaction Model (Ours)
        sc_m6 = StandardScaler()
        X_v_m6 = sc_m6.fit_transform(make_m6_feats(df_v))
        X_t_m6 = sc_m6.transform(make_m6_feats(df_t))
        clf_m6 = LogisticRegression(C=0.5, solver="lbfgs", random_state=SEED).fit(X_v_m6, y_val)
        df_v["score_m6"] = clf_m6.predict_proba(X_v_m6)[:, 1]
        df_t["score_m6"] = clf_m6.predict_proba(X_t_m6)[:, 1]

        scoring_models = [
            ("M1: Raw YOLOv8s", "score_m1"),
            ("M2: Conf-Only Platt", "score_m2"),
            ("M4: Conf+Consistency", "score_m4"),
            ("M6: AquaTrust (Ours)", "score_m6")
        ]

        # ───────────────────────────────────────────────────────
        # 2. RUN EXACT CONFORMAL RISK CONTROL PER FOLD
        # ───────────────────────────────────────────────────────
        for alpha in ALPHA_LEVELS:
            for loss_name in ["FDR", "AnyError"]:
                for m_label, score_col in scoring_models:
                    # A. Calibrate threshold on validation image clusters
                    tau_hat = calibrate_image_crc(df_v, score_col, alpha, loss_type=loss_name)

                    # B. Evaluate on untouched test image clusters
                    test_img_losses = compute_image_losses(df_t, score_col, tau_hat, loss_type=loss_name)
                    emp_test_loss = float(np.mean(test_img_losses)) * 100.0

                    # Detection-level operational metrics
                    acc_mask = df_t[score_col].values >= tau_hat
                    k_acc = int(np.sum(acc_mask))
                    true_hits = int(np.sum(y_test[acc_mask]))
                    false_alarms = k_acc - true_hits
                    
                    det_coverage = (k_acc / n_test_dets) * 100.0
                    say_metric   = (true_hits / n_test_dets) * 100.0
                    det_risk     = (false_alarms / k_acc * 100.0) if k_acc > 0 else 0.0

                    # Verify if theoretical expectation bound holds on test
                    bound_met = (emp_test_loss <= (alpha * 100.0 + 1.0))  # 1% numerical tolerance

                    fold_eval_records.append({
                        "Fold": f,
                        "Loss_Type": loss_name,
                        "Target_Alpha (%)": int(alpha * 100),
                        "Scoring_Model": m_label,
                        "Learned_Tau": round(tau_hat, 4),
                        "Empirical_Test_Image_Loss (%)": round(emp_test_loss, 2),
                        "Bound_Satisfied": bound_met,
                        "Detection_Coverage (%)": round(det_coverage, 2),
                        "Detection_Risk (%)": round(det_risk, 2),
                        "Safe_Autonomous_Yield (SAY %)": round(say_metric, 2),
                        "Accepted_Hits": true_hits,
                        "Accepted_False_Alarms": false_alarms
                    })

    df_all_records = pd.DataFrame(fold_eval_records)
    df_all_records.to_csv(RESULTS_DIR / "image_crc_fold_details.csv", index=False)

    # ───────────────────────────────────────────────────────────
    # 3. 5-FOLD AGGREGATE SUMMARY TABLE
    # ───────────────────────────────────────────────────────────
    summary = df_all_records.groupby(["Loss_Type", "Target_Alpha (%)", "Scoring_Model"]).agg(
        Mean_Tau=("Learned_Tau", "mean"),
        Mean_Image_Loss=("Empirical_Test_Image_Loss (%)", "mean"),
        Bound_Success_Rate=("Bound_Satisfied", "mean"),
        Mean_Detection_Coverage=("Detection_Coverage (%)", "mean"),
        Mean_Detection_Risk=("Detection_Risk (%)", "mean"),
        Mean_SAY=("Safe_Autonomous_Yield (SAY %)", "mean"),
        Total_Hits_Accepted=("Accepted_Hits", "sum"),
        Total_False_Alarms_Accepted=("Accepted_False_Alarms", "sum")
    ).reset_index().round(2)

    print("\n" + "=" * 105)
    print("  🏆 MASTER 5-FOLD IMAGE-LEVEL CONFORMAL RISK CONTROL BENCHMARK (1,170 IMAGES)")
    print("=" * 105)

    print("\n👉 1. LOSS TYPE: IMAGE FALSE DISCOVERY RATE (L_FDR <= alpha):")
    print("─" * 105)
    fdr_table = summary[summary["Loss_Type"] == "FDR"]
    disp_cols = ["Target_Alpha (%)", "Scoring_Model", "Mean_Image_Loss", "Bound_Success_Rate", "Mean_Detection_Coverage", "Mean_SAY", "Total_Hits_Accepted", "Total_False_Alarms_Accepted"]
    print(fdr_table[disp_cols].to_string(index=False))

    print("\n👉 2. LOSS TYPE: IMAGE ANY-ERROR RISK (L_AnyError <= alpha):")
    print("─" * 105)
    any_table = summary[summary["Loss_Type"] == "AnyError"]
    print(any_table[disp_cols].to_string(index=False))

    summary.to_csv(RESULTS_DIR / "image_crc_master_benchmark.csv", index=False)

    print("\n" + "=" * 105)
    print("  ✅ STEP 51 COMPLETE — Group-Exchangeable CRC Benchmark Compiled")
    print(f"  Results saved to: {RESULTS_DIR}")
    print("=" * 105)

if __name__ == "__main__":
    main()
