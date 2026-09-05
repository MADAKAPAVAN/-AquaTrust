"""
AquaTrust v3 — Step 59: High-Confidence (PAC) Conformal Risk Control & Audit
=============================================================================
Evaluates:
  1. Tier 1: Marginal Conformal Risk Control (E[L] <= alpha)
  2. Tier 2: High-Probability PAC-CRC (P(Risk <= alpha) >= 1 - delta)
     - Exact Clopper-Pearson Binomial Inversion
     - Zero-penalty Fixed-Sequence Testing (tau descending from 1.0)
     - Confidence Levels: 1 - delta in {0.90, 0.95}

Compares Models on Locked Test Splits across Folds 1 to 5:
  - M1: Raw YOLOv8s Confidence
  - M2: Conf-Only Platt Baseline
  - M6: AquaTrust Class-Interaction Model (Ours)

Outputs:
  - D:/AquaTrust/results_v3/pac_crc_audit_benchmark.csv
  - D:/AquaTrust/results_v3/pac_vs_marginal_comparison.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import beta

# ── Configuration ──────────────────────────────────────────────
PROJECT_ROOT = Path(r"D:\AquaTrust")
RESULTS_DIR  = PROJECT_ROOT / "results_v3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_LEVELS     = [0.15, 0.20, 0.25, 0.30]
DELTA_LEVELS     = [0.10, 0.05]  # 90% and 95% PAC Confidence
SEED             = 2026

def compute_image_anyerror(df_split: pd.DataFrame, score_col: str, tau: float) -> np.ndarray:
    """Computes exact L_m(tau) in {0, 1} for each unique physical image."""
    images = df_split["image_name"].unique()
    losses = []
    for img in images:
        sub = df_split[df_split["image_name"] == img]
        accepted = sub[score_col].values >= tau
        k_acc = int(np.sum(accepted))
        if k_acc == 0:
            losses.append(0.0)
        else:
            y_acc = sub["is_correct"].values[accepted]
            has_error = np.any(y_acc == 0)
            losses.append(1.0 if has_error else 0.0)
    return np.array(losses, dtype=np.float64)

# ── 1. Tier 1: Marginal Conformal Risk Control Calibration ────
def calibrate_marginal_crc(df_cal: pd.DataFrame, score_col: str, alpha: float) -> tuple:
    images = df_cal["image_name"].unique()
    M = len(images)
    unique_scores = np.sort(df_cal[score_col].dropna().unique())[::-1]
    
    valid_taus = []
    for tau in unique_scores:
        losses = compute_image_anyerror(df_cal, score_col, tau)
        crc_bound = (1.0 + np.sum(losses)) / (M + 1.0)
        if crc_bound <= alpha:
            valid_taus.append((tau, crc_bound))
            
    if not valid_taus:
        return 1.0, 1.0, False
    best_tau, bound_val = min(valid_taus, key=lambda x: x[0])
    return float(best_tau), float(bound_val), True

# ── 2. Tier 2: Exact Clopper-Pearson Fixed-Sequence PAC-CRC ────
def clopper_pearson_ucb(errors: int, total_m: int, delta: float) -> float:
    if errors >= total_m: return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, total_m - errors))

def calibrate_pac_ltt(df_cal: pd.DataFrame, score_col: str, alpha: float, delta: float) -> tuple:
    images = df_cal["image_name"].unique()
    M = len(images)
    unique_scores = np.sort(df_cal[score_col].dropna().unique())[::-1]

    valid_tau = np.nan
    best_ucb  = 1.0

    for tau in unique_scores:
        losses = compute_image_anyerror(df_cal, score_col, tau)
        error_count = int(np.sum(losses))
        ucb = clopper_pearson_ucb(error_count, M, delta)

        if ucb <= alpha:
            valid_tau = tau
            best_ucb = ucb
        else:
            # Fixed-sequence testing stops at the first non-rejected threshold
            break

    if np.isnan(valid_tau):
        return 1.0, 1.0, False
    return float(valid_tau), float(best_ucb), True

def main():
    print("=" * 105)
    print("  AQUATRUST v3 — STEP 59: HIGH-CONFIDENCE (PAC) CONFORMAL RISK CONTROL AUDIT")
    print("=" * 105)

    scoring_models = [
        ("M1: Raw YOLOv8s", "score_m1_raw"),
        ("M2: Conf Platt Baseline", "score_m2_conf"),
        ("M6: AquaTrust (Ours)", "score_m6_inter")
    ]

    all_pac_evals = []

    for f in range(1, 6):
        cal_csv  = RESULTS_DIR / f"clean_scores_fold_{f}_calib.csv"
        test_csv = RESULTS_DIR / f"clean_scores_fold_{f}_test.csv"

        if not cal_csv.exists() or not test_csv.exists():
            print(f"❌ Clean score files missing for Fold {f}")
            return

        df_cal = pd.read_csv(cal_csv)
        df_tst = pd.read_csv(test_csv)
        y_tst  = df_tst["is_correct"].values
        n_tst_dets = len(df_tst)

        for alpha in ALPHA_LEVELS:
            for m_label, col_name in scoring_models:
                # ── Evaluate Tier 1: Marginal CRC ──────────────
                tau_marg, bound_m, cert_m = calibrate_marginal_crc(df_cal, col_name, alpha)
                test_loss_m = float(np.mean(compute_image_anyerror(df_tst, col_name, tau_marg))) * 100.0
                
                acc_m = df_tst[col_name].values >= tau_marg
                k_m   = int(np.sum(acc_m))
                hits_m = int(np.sum(y_tst[acc_m]))
                cov_m = (k_m / n_tst_dets) * 100.0
                say_m = (hits_m / n_tst_dets) * 100.0

                all_pac_evals.append({
                    "Fold": f, "Scoring_Model": m_label, "Target_Alpha (%)": int(alpha * 100),
                    "Guarantee_Tier": "Tier 1: Marginal CRC (E[L] <= alpha)",
                    "Delta_Param": "N/A", "Confidence_Level": "Marginal (Expected)",
                    "Calibrated_Tau": round(tau_marg, 4), "Certified_Calib": cert_m,
                    "HeldOut_Test_Risk (%)": round(test_loss_m, 2),
                    "Detection_Coverage (%)": round(cov_m, 2),
                    "Safe_Autonomous_Yield (SAY %)": round(say_m, 2),
                    "Accepted_Hits": hits_m, "Accepted_False_Alarms": k_m - hits_m
                })

                # ── Evaluate Tier 2: PAC-CRC (90% and 95% Confidence) ──
                for delta in DELTA_LEVELS:
                    conf_pct = int((1.0 - delta) * 100)
                    tau_pac, ucb_val, cert_pac = calibrate_pac_ltt(df_cal, col_name, alpha, delta)
                    
                    if cert_pac:
                        test_loss_pac = float(np.mean(compute_image_anyerror(df_tst, col_name, tau_pac))) * 100.0
                        acc_pac = df_tst[col_name].values >= tau_pac
                        k_pac   = int(np.sum(acc_pac))
                        hits_pac = int(np.sum(y_tst[acc_pac]))
                        cov_pac = (k_pac / n_tst_dets) * 100.0
                        say_pac = (hits_pac / n_tst_dets) * 100.0
                    else:
                        tau_pac, test_loss_pac, cov_pac, say_pac, hits_pac, k_pac = 1.0, 0.0, 0.0, 0.0, 0, 0

                    all_pac_evals.append({
                        "Fold": f, "Scoring_Model": m_label, "Target_Alpha (%)": int(alpha * 100),
                        "Guarantee_Tier": f"Tier 2: PAC-CRC ({conf_pct}% Confidence)",
                        "Delta_Param": f"delta={delta}", "Confidence_Level": f"{conf_pct}% High-Prob",
                        "Calibrated_Tau": round(tau_pac, 4), "Certified_Calib": cert_pac,
                        "HeldOut_Test_Risk (%)": round(test_loss_pac, 2),
                        "Detection_Coverage (%)": round(cov_pac, 2),
                        "Safe_Autonomous_Yield (SAY %)": round(say_pac, 2),
                        "Accepted_Hits": hits_pac, "Accepted_False_Alarms": k_pac - hits_pac
                    })

    df_pac = pd.DataFrame(all_pac_evals)
    df_pac.to_csv(RESULTS_DIR / "pac_crc_audit_benchmark.csv", index=False)

    # ───────────────────────────────────────────────────────────
    # 1. COMPILE SUMMARY COMPARISON TABLE
    # ───────────────────────────────────────────────────────────
    summary = df_pac.groupby(["Guarantee_Tier", "Target_Alpha (%)", "Scoring_Model"]).agg(
        Mean_Tau=("Calibrated_Tau", "mean"),
        Certification_Rate=("Certified_Calib", "mean"),
        Mean_Test_Risk=("HeldOut_Test_Risk (%)", "mean"),
        Mean_Coverage=("Detection_Coverage (%)", "mean"),
        Mean_SAY=("Safe_Autonomous_Yield (SAY %)", "mean"),
        Total_Hits=("Accepted_Hits", "sum"),
        Total_False_Alarms=("Accepted_False_Alarms", "sum")
    ).reset_index().round(2)

    print("─" * 105)
    print("  🏆 1. CONFORMAL GUARANTEE COMPARISON: MARGINAL CRC VS. HIGH-PROBABILITY PAC-CRC")
    print("─" * 105)
    disp_cols = ["Guarantee_Tier", "Target_Alpha (%)", "Scoring_Model", "Certification_Rate", "Mean_Test_Risk", "Mean_Coverage", "Mean_SAY", "Total_Hits", "Total_False_Alarms"]
    print(summary[disp_cols].to_string(index=False))
    summary.to_csv(RESULTS_DIR / "pac_vs_marginal_comparison.csv", index=False)

    # ───────────────────────────────────────────────────────────
    # 2. THE PRICE OF HIGH CONFIDENCE (Utility Trade-Off Analysis)
    # ───────────────────────────────────────────────────────────
    print("\n" + "─" * 105)
    print("  📉 2. THE PRICE OF HIGH CONFIDENCE: MARGINAL (𝔼[L]≤α) VS. PAC 90% CONFIDENCE (AquaTrust M6)")
    print("─" * 105)

    price_rows = []
    for alpha_int in [15, 20, 25, 30]:
        sub_m = summary[(summary["Target_Alpha (%)"] == alpha_int) & (summary["Scoring_Model"].str.contains("M6")) & (summary["Guarantee_Tier"].str.contains("Marginal"))].iloc[0]
        sub_p = summary[(summary["Target_Alpha (%)"] == alpha_int) & (summary["Scoring_Model"].str.contains("M6")) & (summary["Guarantee_Tier"].str.contains("90%"))].iloc[0]

        delta_say = sub_p["Mean_SAY"] - sub_m["Mean_SAY"]
        delta_cov = sub_p["Mean_Coverage"] - sub_m["Mean_Coverage"]

        price_rows.append({
            "Target_Alpha (%)": alpha_int,
            "Marginal_Risk (%)": sub_m["Mean_Test_Risk"],
            "Marginal_SAY (%)": sub_m["Mean_SAY"],
            "PAC_90_Risk (%)": sub_p["Mean_Test_Risk"],
            "PAC_90_SAY (%)": sub_p["Mean_SAY"],
            "SAY_Penalty (PAC - Marg)": f"{delta_say:+.2f}%",
            "Marginal_Coverage (%)": sub_m["Mean_Coverage"],
            "PAC_90_Coverage (%)": sub_p["Mean_Coverage"],
            "Coverage_Penalty": f"{delta_cov:+.2f}%"
        })

    df_price = pd.DataFrame(price_rows)
    print(df_price.to_string(index=False))

    print("\n" + "=" * 105)
    print("  ✅ STEP 59 COMPLETE — PAC-CRC Audit & Comparison Ledger Compiled")
    print(f"  Results saved to: {RESULTS_DIR}")
    print("=" * 105)

if __name__ == "__main__":
    main()
