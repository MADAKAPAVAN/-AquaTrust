"""
AquaTrust v3 — Step 57B: Exact Image Any-Error Conformal Risk Control (CRC)
===========================================================================
- Calibration Split : clean_scores_fold_{j}_calib.csv (117 exchangeable images)
- Test Split        : clean_scores_fold_{j}_test.csv  (234 untouched images)
- Loss Function     : L_m(tau) = 1 if ANY false alarm accepted in image m, else 0
- Threshold Grid    : Exact sorted empirical score boundaries from calibration set
- Bounding Theorem  : inf { tau : (1 / (M + 1)) * (1 + sum(L_i(tau))) <= alpha }
- Primary Guarantee : E_{V, T}[ L_test(tau_hat) ] <= alpha

Outputs:
  - D:/AquaTrust/results_v3/crc_exact_anyerror_benchmark.csv
  - D:/AquaTrust/results_v3/crc_exact_anyerror_fold_details.csv
  - D:/AquaTrust/results_v3/crc_exact_anyerror_utility_gains.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(r"D:\AquaTrust")
RESULTS_DIR  = PROJECT_ROOT / "results_v3"
ALPHA_LEVELS = [0.10, 0.15, 0.20, 0.25, 0.30]

def compute_image_anyerror_losses(df_split: pd.DataFrame, score_col: str, tau: float) -> np.ndarray:
    """Computes exact L_m(tau) in {0, 1} for each unique physical image."""
    images = df_split["image_name"].unique()
    losses = []
    for img in images:
        sub = df_split[df_split["image_name"] == img]
        accepted = sub[score_col].values >= tau
        if np.sum(accepted) == 0:
            losses.append(0.0)
        else:
            y_acc = sub["is_correct"].values[accepted]
            has_error = np.any(y_acc == 0)
            losses.append(1.0 if has_error else 0.0)
    return np.array(losses, dtype=np.float64)

def calibrate_exact_crc(df_cal: pd.DataFrame, score_col: str, alpha: float) -> tuple:
    """
    Evaluates exact empirical score boundaries:
      (1 / (M + 1)) * (1 + sum(L_i(tau))) <= alpha
    """
    images = df_cal["image_name"].unique()
    M = len(images)
    
    # Exact unique candidate thresholds from calibration data (descending order)
    unique_scores = np.sort(df_cal[score_col].dropna().unique())[::-1]
    
    valid_taus = []
    for tau in unique_scores:
        losses = compute_image_anyerror_losses(df_cal, score_col, tau)
        crc_upper_bound = (1.0 + np.sum(losses)) / (M + 1.0)
        
        if crc_upper_bound <= alpha:
            valid_taus.append((tau, crc_upper_bound))

    if not valid_taus:
        return 1.0, 1.0

    # Most permissive threshold maximizes coverage while strictly satisfying bound
    best_tau, bound_val = min(valid_taus, key=lambda x: x[0])
    return float(best_tau), float(bound_val)

def main():
    print("=" * 105)
    print("  AQUATRUST v3 — STEP 57B: EXACT IMAGE ANY-ERROR CONFORMAL RISK CONTROL")
    print("=" * 105)

    scoring_models = [
        ("M1: Raw YOLOv8s", "score_m1_raw"),
        ("M2: Conf-Only Platt", "score_m2_conf"),
        ("M4: Conf+Consistency", "score_m4_cons"),
        ("M6: AquaTrust (Ours)", "score_m6_inter")
    ]

    all_fold_evals = []

    for f in range(1, 6):
        cal_csv  = RESULTS_DIR / f"clean_scores_fold_{f}_calib.csv"
        test_csv = RESULTS_DIR / f"clean_scores_fold_{f}_test.csv"

        if not cal_csv.exists() or not test_csv.exists():
            print(f"❌ Clean score CSVs missing for Fold {f}. Run Step 57A first.")
            return

        df_cal = pd.read_csv(cal_csv)
        df_tst = pd.read_csv(test_csv)

        y_tst = df_tst["is_correct"].values
        n_tst_dets = len(df_tst)

        for alpha in ALPHA_LEVELS:
            for m_label, col_name in scoring_models:
                # 1. Calibrate Exact Threshold on Calibration Images
                tau_hat, cal_bound = calibrate_exact_crc(df_cal, col_name, alpha)

                # 2. Evaluate on Untouched Outer Test Images
                test_losses = compute_image_anyerror_losses(df_tst, col_name, tau_hat)
                test_image_loss = float(np.mean(test_losses)) * 100.0

                # 3. Detection-Level Operational Yield Metrics
                acc_mask = df_tst[col_name].values >= tau_hat
                k_acc = int(np.sum(acc_mask))
                hits  = int(np.sum(y_tst[acc_mask]))
                fa    = k_acc - hits

                cov_pct = (k_acc / n_tst_dets) * 100.0
                say_pct = (hits / n_tst_dets) * 100.0
                det_risk = (fa / k_acc * 100.0) if k_acc > 0 else 0.0

                all_fold_evals.append({
                    "Fold": f,
                    "Target_Alpha (%)": int(alpha * 100),
                    "Scoring_Model": m_label,
                    "Calibrated_Tau": round(tau_hat, 4),
                    "Calib_CRC_Bound (%)": round(cal_bound * 100.0, 2),
                    "Test_Image_Loss (%)": round(test_image_loss, 2),
                    "Detection_Coverage (%)": round(cov_pct, 2),
                    "Safe_Autonomous_Yield (SAY %)": round(say_pct, 2),
                    "Accepted_Hits": hits,
                    "Accepted_False_Alarms": fa
                })

    df_evals = pd.DataFrame(all_fold_evals)
    df_evals.to_csv(RESULTS_DIR / "crc_exact_anyerror_fold_details.csv", index=False)

    # ───────────────────────────────────────────────────────────
    # 5-FOLD AGGREGATE BENCHMARK SUMMARY (The Definitive Table)
    # ───────────────────────────────────────────────────────────
    summary = df_evals.groupby(["Target_Alpha (%)", "Scoring_Model"]).agg(
        Mean_Tau=("Calibrated_Tau", "mean"),
        Expected_Image_Risk=("Test_Image_Loss (%)", "mean"),
        Risk_Std=("Test_Image_Loss (%)", lambda x: np.std(x, ddof=1)),
        Coverage_Pct=("Detection_Coverage (%)", "mean"),
        SAY_Pct=("Safe_Autonomous_Yield (SAY %)", "mean"),
        Total_Hits=("Accepted_Hits", "sum"),
        Total_False_Alarms=("Accepted_False_Alarms", "sum")
    ).reset_index()

    # Calculate 95% Confidence Interval for Expected Risk across the 5 outer folds
    # t_crit = 2.776 for n=5 (df=4)
    summary["Risk_95%_CI"] = summary.apply(
        lambda r: f"[{max(0.0, r['Expected_Image_Risk'] - 2.776 * (r['Risk_Std'] / np.sqrt(5))):.2f}%, {r['Expected_Image_Risk'] + 2.776 * (r['Risk_Std'] / np.sqrt(5)):.2f}%]", axis=1
    )
    summary["Marginal_Bound_Holds"] = summary["Expected_Image_Risk"] <= summary["Target_Alpha (%)"]

    print("\n" + "─" * 105)
    print("  🏆 EXACT CONFORMAL RISK CONTROL BENCHMARK (Primary Safety Loss: L_AnyError in {0, 1})")
    print("─" * 105)
    disp_cols = ["Target_Alpha (%)", "Scoring_Model", "Expected_Image_Risk", "Marginal_Bound_Holds", "Risk_95%_CI", "Coverage_Pct", "SAY_Pct", "Total_Hits", "Total_False_Alarms"]
    print(summary[disp_cols].round(2).to_string(index=False))
    summary.to_csv(RESULTS_DIR / "crc_exact_anyerror_benchmark.csv", index=False)

    # ───────────────────────────────────────────────────────────
    # UTILITY GAINS: AQUATRUST M6 VS BASLEINES UNDER IDENTICAL ALPHA
    # ───────────────────────────────────────────────────────────
    print("\n" + "─" * 105)
    print("  🔬 CONFORMAL UTILITY GAINS: AQUATRUST M6 VS. RAW YOLO UNDER EQUAL RISK BUDGETS")
    print("─" * 105)

    gains = []
    for alpha in ALPHA_LEVELS:
        alpha_int = int(alpha * 100)
        sub_a = summary[summary["Target_Alpha (%)"] == alpha_int]
        raw_row = sub_a[sub_a["Scoring_Model"].str.contains("M1")].iloc[0]
        m6_row  = sub_a[sub_a["Scoring_Model"].str.contains("M6")].iloc[0]

        say_gain_rel = (m6_row["SAY_Pct"] - raw_row["SAY_Pct"]) / raw_row["SAY_Pct"] * 100.0 if raw_row["SAY_Pct"] > 0 else 0.0
        cov_gain_rel = (m6_row["Coverage_Pct"] - raw_row["Coverage_Pct"]) / raw_row["Coverage_Pct"] * 100.0 if raw_row["Coverage_Pct"] > 0 else 0.0

        gains.append({
            "Risk_Budget_Alpha (%)": alpha_int,
            "Raw_Expected_Risk (%)": raw_row["Expected_Image_Risk"],
            "Aqua_Expected_Risk (%)": m6_row["Expected_Image_Risk"],
            "Raw_Coverage (%)": raw_row["Coverage_Pct"],
            "Aqua_Coverage (%)": m6_row["Coverage_Pct"],
            "Relative_Coverage_Gain": f"+{cov_gain_rel:.1f}%",
            "Raw_SAY (%)": raw_row["SAY_Pct"],
            "Aqua_SAY (%)": m6_row["SAY_Pct"],
            "Relative_SAY_Gain": f"+{say_gain_rel:.1f}%"
        })

    df_gains = pd.DataFrame(gains)
    print(df_gains.to_string(index=False))
    df_gains.to_csv(RESULTS_DIR / "crc_exact_anyerror_utility_gains.csv", index=False)

    print("\n" + "=" * 105)
    print("  ✅ STEP 57B COMPLETE — Exact Conformal Risk Control Ledger Exported")
    print(f"  Results saved to: {RESULTS_DIR}")
    print("=" * 105)

if __name__ == "__main__":
    main()
