"""
AquaTrust v3 — Step 49A: Conformal Risk Control (CRC) Methodology & Diagnostic Audit
=====================================================================================
Analyzes why Step 48 empirical thresholding violated finite-sample error bounds:
  1. Per-Fold, Per-Class Sample Size Audit (N_val vs N_test for MILCO and NOMBO)
  2. Valuation vs. Test Risk Generalization Gap across tau in [0.20, 0.95]
  3. Evaluates Learn-Then-Test (LTT) Upper Confidence Bounds (Hoeffding/Bentkus)
  4. Evaluates Conformal Target Coverage (Linear Loss Formulations)

Outputs:
  - D:/AquaTrust/results_v3/crc_audit_sample_sizes.csv
  - D:/AquaTrust/results_v3/crc_audit_generalization_gap.csv
  - D:/AquaTrust/results_v3/crc_audit_ltt_bounds.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import binom

# ── Configuration ──────────────────────────────────────────────
PROJECT_ROOT = Path(r"D:\AquaTrust")
RESULTS_DIR  = PROJECT_ROOT / "results_v3"

CLASS_NAMES = {0: "MILCO", 1: "NOMBO"}
ALPHA_TARGETS = [0.10, 0.15, 0.20]
DELTA_CONFIDENCE = 0.10  # 90% confidence that risk <= alpha

# ── Statistical Bounding Functions ─────────────────────────────
def hoeffding_ucb(emp_risk: float, n: int, delta: float = 0.10) -> float:
    """Computes Hoeffding Upper Confidence Bound for bounded loss [0, 1]."""
    if n <= 0: return 1.0
    bound = emp_risk + np.sqrt(np.log(1.0 / delta) / (2.0 * n))
    return min(1.0, float(bound))

def bentkus_ucb(emp_risk: float, n: int, delta: float = 0.10) -> float:
    """Inverts Binomial tail for sharp finite-sample UCB on 0-1 loss."""
    if n <= 0: return 1.0
    k = int(np.ceil(emp_risk * n))
    # Find smallest p such that P(Binomial(n, p) <= k) <= delta
    p_grid = np.linspace(emp_risk, 1.0, 500)
    for p in p_grid:
        if binom.cdf(k, n, p) <= delta:
            return float(p)
    return 1.0

def main():
    print("=" * 100)
    print("  AQUATRUST v3 — STEP 49A: CONFORMAL RISK CONTROL (CRC) METHODOLOGY AUDIT")
    print("=" * 100)

    # ───────────────────────────────────────────────────────────
    # 1. PER-FOLD SAMPLE SIZE & DETECTION AUDIT
    # ───────────────────────────────────────────────────────────
    print("\n📊 1. CALIBRATION (VAL) VS. EVALUATION (TEST) DETECTION SIZES")
    print("─" * 100)

    sample_audit = []
    for f in range(1, 6):
        val_csv  = RESULTS_DIR / f"features_fold_{f}_val.csv"
        test_csv = RESULTS_DIR / f"features_fold_{f}_test.csv"

        if not val_csv.exists() or not test_csv.exists():
            print(f"❌ Missing files for Fold {f}")
            return

        df_v = pd.read_csv(val_csv)
        df_t = pd.read_csv(test_csv)

        for cid, cname in CLASS_NAMES.items():
            v_sub = df_v[df_v["pred_class"] == cid]
            t_sub = df_t[df_t["pred_class"] == cid]

            sample_audit.append({
                "Fold": f,
                "Class": cname,
                "Val_Total_Det": len(v_sub),
                "Val_True_Hits (y=1)": int(v_sub["is_correct"].sum()),
                "Val_False_Alarms (y=0)": int(len(v_sub) - v_sub["is_correct"].sum()),
                "Test_Total_Det": len(t_sub),
                "Test_True_Hits (y=1)": int(t_sub["is_correct"].sum()),
                "Test_False_Alarms (y=0)": int(len(t_sub) - t_sub["is_correct"].sum()),
            })

    df_sample = pd.DataFrame(sample_audit)
    print(df_sample.to_string(index=False))
    df_sample.to_csv(RESULTS_DIR / "crc_audit_sample_sizes.csv", index=False)

    # ───────────────────────────────────────────────────────────
    # 2. GENERALIZATION GAP ACROSS THRESHOLD SPECTRUM (tau)
    # ───────────────────────────────────────────────────────────
    print("\n" + "─" * 100)
    print("  🔬 2. EMPIRICAL RISK GENERALIZATION GAP (Val Error vs. Test Error across tau)")
    print("─" * 100)

    gap_records = []
    taus = [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]

    master_preds_csv = RESULTS_DIR / "nested_5fold_master_predictions.csv"
    df_all_test = pd.read_csv(master_preds_csv)

    for f in range(1, 6):
        df_v = pd.read_csv(RESULTS_DIR / f"features_fold_{f}_val.csv")
        df_t = pd.read_csv(RESULTS_DIR / f"features_fold_{f}_test.csv")

        # Use M6 score for consistency
        s_val  = df_v["confidence"].values   # Base confidence check
        s_test = df_t["confidence"].values
        y_val  = df_v["is_correct"].values
        y_test = df_t["is_correct"].values

        for cid, cname in CLASS_NAMES.items():
            v_mask = (df_v["pred_class"] == cid).values
            t_mask = (df_t["pred_class"] == cid).values

            for tau in taus:
                acc_v = v_mask & (s_val >= tau)
                acc_t = t_mask & (s_test >= tau)

                risk_v = (1.0 - np.mean(y_val[acc_v])) * 100.0 if np.sum(acc_v) > 0 else 0.0
                risk_t = (1.0 - np.mean(y_test[acc_t])) * 100.0 if np.sum(acc_t) > 0 else 0.0

                gap_records.append({
                    "Fold": f,
                    "Class": cname,
                    "Threshold_tau": tau,
                    "Val_Accepted_N": int(np.sum(acc_v)),
                    "Val_Risk (%)": round(risk_v, 2),
                    "Test_Accepted_N": int(np.sum(acc_t)),
                    "Test_Risk (%)": round(risk_t, 2),
                    "Risk_Gap (Test - Val)": round(risk_t - risk_v, 2)
                })

    df_gap = pd.DataFrame(gap_records)
    gap_summary = df_gap.groupby(["Class", "Threshold_tau"]).agg(
        Mean_Val_N=("Val_Accepted_N", "mean"),
        Mean_Val_Risk=("Val_Risk (%)", "mean"),
        Mean_Test_N=("Test_Accepted_N", "mean"),
        Mean_Test_Risk=("Test_Risk (%)", "mean"),
        Mean_Generalization_Gap=("Risk_Gap (Test - Val)", "mean")
    ).reset_index()

    print(gap_summary.round(2).to_string(index=False))
    gap_summary.to_csv(RESULTS_DIR / "crc_audit_generalization_gap.csv", index=False)

    # ───────────────────────────────────────────────────────────
    # 3. LEARN-THEN-TEST (LTT) UPPER BOUND AUDIT
    # ───────────────────────────────────────────────────────────
    print("\n" + "─" * 100)
    print("  🛡️ 3. STATISTICAL BOUND AUDIT: EMPIRICAL SEARCH VS. CONFORMAL UCB (LTT)")
    print("─" * 100)

    ltt_records = []
    # Evaluate Fold 1 as primary representative
    df_v1 = pd.read_csv(RESULTS_DIR / "features_fold_1_val.csv")
    df_t1 = pd.read_csv(RESULTS_DIR / "features_fold_1_test.csv")

    for cid, cname in CLASS_NAMES.items():
        v_sub = df_v1[df_v1["pred_class"] == cid]
        t_sub = df_t1[df_t1["pred_class"] == cid]

        scores_v = v_sub["confidence"].values
        labels_v = v_sub["is_correct"].values
        scores_t = t_sub["confidence"].values
        labels_t = t_sub["is_correct"].values

        for alpha in ALPHA_TARGETS:
            # A. Naive Empirical Search (Step 48 approach)
            valid_naive = []
            valid_ltt_hoeff = []
            valid_ltt_bentkus = []

            for tau in np.linspace(0.20, 0.95, 100):
                acc_mask = scores_v >= tau
                n_acc = int(np.sum(acc_mask))
                if n_acc < 5: continue

                emp_err = 1.0 - np.mean(labels_v[acc_mask])
                if emp_err <= alpha:
                    valid_naive.append((tau, emp_err))

                # Compute UCB bounds
                ucb_h = hoeffding_ucb(emp_err, n_acc, delta=DELTA_CONFIDENCE)
                if ucb_h <= alpha:
                    valid_ltt_hoeff.append((tau, ucb_h))

                ucb_b = bentkus_ucb(emp_err, n_acc, delta=DELTA_CONFIDENCE)
                if ucb_b <= alpha:
                    valid_ltt_bentkus.append((tau, ucb_b))

            tau_naive = min([t for t, _ in valid_naive]) if valid_naive else np.nan
            tau_hoeff = min([t for t, _ in valid_ltt_hoeff]) if valid_ltt_hoeff else np.nan
            tau_bentk = min([t for t, _ in valid_ltt_bentkus]) if valid_ltt_bentkus else np.nan

            # Evaluate Test Error for each method
            def eval_test(tau_val):
                if np.isnan(tau_val): return np.nan, 0
                t_acc = scores_t >= tau_val
                if np.sum(t_acc) == 0: return 0.0, 0
                return round(float((1.0 - np.mean(labels_t[t_acc])) * 100.0), 2), int(np.sum(t_acc))

            err_naive, k_naive = eval_test(tau_naive)
            err_hoeff, k_hoeff = eval_test(tau_hoeff)
            err_bentk, k_bentk = eval_test(tau_bentk)

            ltt_records.append({
                "Class": cname,
                "Target_Alpha": alpha,
                "Naive_Tau": round(tau_naive, 4) if not np.isnan(tau_naive) else "None",
                "Naive_Test_Error (%)": err_naive,
                "Naive_Accepted_K": k_naive,
                "LTT_Bentkus_Tau": round(tau_bentk, 4) if not np.isnan(tau_bentk) else "None (Unreachable)",
                "LTT_Test_Error (%)": err_bentk if not np.isnan(tau_bentk) else "N/A",
                "LTT_Accepted_K": k_bentk
            })

    df_ltt = pd.DataFrame(ltt_records)
    print(df_ltt.to_string(index=False))
    df_ltt.to_csv(RESULTS_DIR / "crc_audit_ltt_bounds.csv", index=False)

    print("\n" + "=" * 100)
    print("  ✅ STEP 49A AUDIT COMPLETE — Audit manifests saved to results_v3")
    print("=" * 100)

if __name__ == "__main__":
    main()
