"""
AquaTrust v3 — Step 49B: CRC Feasibility & Calibration Pooling Study
====================================================================
Evaluates 3 Conformal Risk Control Formulations across Folds 1 to 5:
  - B1: Strict Mondrian CRC (Class-Conditional: tau_MILCO, tau_NOMBO)
  - B2: Marginal Pooled CRC (Global Fleet Risk: tau_global)
  - B3: Class-Shrinkage CRC (Global Base with Minority Confidence Margin)

Risk Targets Evaluated:
  - alpha in {0.10, 0.15, 0.20, 0.25, 0.30}

Outputs:
  - D:/AquaTrust/results_v3/crc_feasibility_benchmark.csv
  - D:/AquaTrust/results_v3/crc_risk_coverage_tradeoff.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Configuration ──────────────────────────────────────────────
PROJECT_ROOT = Path(r"D:\AquaTrust")
RESULTS_DIR  = PROJECT_ROOT / "results_v3"

SEED         = 2026
CLASS_NAMES  = {0: "MILCO", 1: "NOMBO"}
ALPHA_LEVELS = [0.10, 0.15, 0.20, 0.25, 0.30]

def make_m6_feats(d: pd.DataFrame) -> np.ndarray:
    c  = d["confidence"].values
    qc = d["q_contrast"].values
    qv = d["q_variability"].values
    k  = d["pred_class"].values
    return np.column_stack([c, qc, qv, k, c*k, qc*k, qv*k])

def fit_threshold_for_risk(scores: np.ndarray, labels: np.ndarray, alpha: float) -> float:
    """Finds the lowest score threshold tau such that empirical error rate <= alpha."""
    if len(scores) == 0:
        return 1.0
    thresholds = np.linspace(scores.min(), scores.max(), 300)
    valid_taus = []
    for tau in thresholds:
        accepted = scores >= tau
        if np.sum(accepted) == 0:
            continue
        error_rate = 1.0 - np.mean(labels[accepted])
        if error_rate <= alpha:
            valid_taus.append(tau)
    return float(min(valid_taus)) if valid_taus else float(scores.max())

def main():
    print("=" * 105)
    print("  AQUATRUST v3 — STEP 49B: CRC FEASIBILITY & CALIBRATION POOLING STUDY")
    print("=" * 105)

    all_fold_records = []
    coverage_tradeoff_records = []

    for f in range(1, 6):
        val_csv  = RESULTS_DIR / f"features_fold_{f}_val.csv"
        test_csv = RESULTS_DIR / f"features_fold_{f}_test.csv"

        if not val_csv.exists() or not test_csv.exists():
            print(f"❌ Missing files for Fold {f}")
            return

        df_val  = pd.read_csv(val_csv)
        df_test = pd.read_csv(test_csv)

        y_val  = df_val["is_correct"].values
        y_test = df_test["is_correct"].values

        # 1. Fit M6 Model on fold validation set
        sc = StandardScaler()
        X_val_sc  = sc.fit_transform(make_m6_feats(df_val))
        X_test_sc = sc.transform(make_m6_feats(df_test))

        clf_m6 = LogisticRegression(C=0.5, solver="lbfgs", random_state=SEED).fit(X_val_sc, y_val)
        p_val  = clf_m6.predict_proba(X_val_sc)[:, 1]
        p_test = clf_m6.predict_proba(X_test_sc)[:, 1]

        df_val_eval = df_val.copy()
        df_val_eval["prob_m6"] = p_val

        df_test_eval = df_test.copy()
        df_test_eval["prob_m6"] = p_test

        # 2. Evaluate Formulations across Alpha Levels
        for alpha in ALPHA_LEVELS:
            # ───────────────────────────────────────────────────
            # CONFIG B1: Strict Mondrian CRC (Class-Specific)
            # ───────────────────────────────────────────────────
            sub_m_val = df_val_eval[df_val_eval["pred_class"] == 0]
            sub_n_val = df_val_eval[df_val_eval["pred_class"] == 1]

            tau_m_b1 = fit_threshold_for_risk(sub_m_val["prob_m6"].values, sub_m_val["is_correct"].values, alpha)
            tau_n_b1 = fit_threshold_for_risk(sub_n_val["prob_m6"].values, sub_n_val["is_correct"].values, alpha)

            # Apply B1 to Test
            acc_b1_m = (df_test_eval["pred_class"] == 0) & (df_test_eval["prob_m6"] >= tau_m_b1)
            acc_b1_n = (df_test_eval["pred_class"] == 1) & (df_test_eval["prob_m6"] >= tau_n_b1)
            acc_b1_all = acc_b1_m | acc_b1_n

            # ───────────────────────────────────────────────────
            # CONFIG B2: Marginal Pooled CRC (Single Global tau)
            # ───────────────────────────────────────────────────
            tau_b2_global = fit_threshold_for_risk(p_val, y_val, alpha)

            # Apply B2 to Test
            acc_b2_all = df_test_eval["prob_m6"] >= tau_b2_global
            acc_b2_m   = acc_b2_all & (df_test_eval["pred_class"] == 0)
            acc_b2_n   = acc_b2_all & (df_test_eval["pred_class"] == 1)

            # ───────────────────────────────────────────────────
            # CONFIG B3: Class-Shrinkage CRC (Base + NOMBO Margin)
            # ───────────────────────────────────────────────────
            # Base threshold from pooled data, with conservative offset for NOMBO
            tau_b3_m = tau_b2_global
            tau_b3_n = min(1.0, tau_b2_global + 0.10)  # +10% confidence safety margin for minority

            acc_b3_m   = (df_test_eval["pred_class"] == 0) & (df_test_eval["prob_m6"] >= tau_b3_m)
            acc_b3_n   = (df_test_eval["pred_class"] == 1) & (df_test_eval["prob_m6"] >= tau_b3_n)
            acc_b3_all = acc_b3_m | acc_b3_n

            # ───────────────────────────────────────────────────
            # Compute Test Metrics for All Configurations
            # ───────────────────────────────────────────────────
            configs = [
                ("B1: Strict Mondrian CRC", acc_b1_all, acc_b1_m, acc_b1_n, f"M:{tau_m_b1:.3f}, N:{tau_n_b1:.3f}"),
                ("B2: Marginal Pooled CRC",  acc_b2_all, acc_b2_m, acc_b2_n, f"Global:{tau_b2_global:.3f}"),
                ("B3: Class-Shrinkage CRC",  acc_b3_all, acc_b3_m, acc_b3_n, f"M:{tau_b3_m:.3f}, N:{tau_b3_n:.3f}")
            ]

            n_tst = len(df_test_eval)
            for cfg_name, m_all, m_m, m_n, tau_str in configs:
                # All classes
                k_all = int(np.sum(m_all))
                risk_all = (1.0 - np.mean(y_test[m_all])) * 100.0 if k_all > 0 else 0.0
                cov_all  = (k_all / n_tst) * 100.0

                # MILCO subgroup
                k_m = int(np.sum(m_m))
                risk_m = (1.0 - np.mean(y_test[m_m])) * 100.0 if k_m > 0 else 0.0

                # NOMBO subgroup
                k_n = int(np.sum(m_n))
                risk_n = (1.0 - np.mean(y_test[m_n])) * 100.0 if k_n > 0 else 0.0

                all_fold_records.append({
                    "Fold": f,
                    "Target_Alpha (%)": int(alpha * 100),
                    "Configuration": cfg_name,
                    "Thresholds": tau_str,
                    "Total_Accepted": k_all,
                    "Coverage (%)": round(cov_all, 1),
                    "Overall_Risk (%)": round(risk_all, 2),
                    "MILCO_Accepted": k_m,
                    "MILCO_Risk (%)": round(risk_m, 2),
                    "NOMBO_Accepted": k_n,
                    "NOMBO_Risk (%)": round(risk_n, 2),
                    "Bound_Satisfied": risk_all <= (alpha * 100.0 + 3.0)  # Within 3% tolerance
                })

    df_records = pd.DataFrame(all_fold_records)

    # Aggregate over all 5 folds
    summary = df_records.groupby(["Target_Alpha (%)", "Configuration"]).agg(
        Mean_Accepted_Detections=("Total_Accepted", "mean"),
        Mean_Coverage_Pct=("Coverage (%)", "mean"),
        Mean_Overall_Test_Risk=("Overall_Risk (%)", "mean"),
        Mean_MILCO_Risk=("MILCO_Risk (%)", "mean"),
        Mean_NOMBO_Risk=("NOMBO_Risk (%)", "mean"),
        Bound_Success_Rate=("Bound_Satisfied", "mean")
    ).reset_index().round(2)

    print("\n" + "─" * 105)
    print("  🏆 5-FOLD AGGREGATE CRC FEASIBILITY & POOLING BENCHMARK")
    print("─" * 105)
    print(summary.to_string(index=False))

    df_records.to_csv(RESULTS_DIR / "crc_feasibility_benchmark.csv", index=False)
    summary.to_csv(RESULTS_DIR / "crc_risk_coverage_tradeoff.csv", index=False)

    print("\n" + "=" * 105)
    print("  ✅ STEP 49B COMPLETE — Feasibility manifests saved to results_v3")
    print("=" * 105)

if __name__ == "__main__":
    main()
