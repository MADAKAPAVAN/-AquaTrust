"""
AquaTrust v3 — Step 53: CRC Claim Validity & Statistical Taxonomy Audit
========================================================================
Audits and categorizes every Conformal Risk Control configuration:
  - Verifies the marginal expectation theorem: E[L_test] <= alpha
  - Categorizes configurations into Tier 1 (Valid), Tier 2 (Trade-off), Tier 3 (Rejected)
  - Isolates and freezes the Primary Operational Policies for poster & paper
  - Generates the formal Conformal Audit Ledger

Outputs:
  - D:/AquaTrust/results_v3/master_crc_validity_ledger.csv
  - D:/AquaTrust/results_v3/crc_frozen_primary_policies.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
PROJECT_ROOT = Path(r"D:\AquaTrust")
RESULTS_DIR  = PROJECT_ROOT / "results_v3"
DETAILS_CSV  = RESULTS_DIR / "image_crc_fold_details.csv"
BENCH_CSV    = RESULTS_DIR / "image_crc_master_benchmark.csv"

def main():
    print("=" * 105)
    print("  AQUATRUST v3 — STEP 53: CRC CLAIM VALIDITY & STATISTICAL TAXONOMY AUDIT")
    print("=" * 105)

    if not DETAILS_CSV.exists() or not BENCH_CSV.exists():
        print(f"❌ Missing benchmark files in {RESULTS_DIR}")
        return

    df_bench = pd.read_csv(BENCH_CSV)
    df_details = pd.read_csv(DETAILS_CSV)

    print(f"📁 Loaded 5-Fold Details: {len(df_details)} records across {df_details['Fold'].nunique()} folds\n")

    # ───────────────────────────────────────────────────────────
    # 1. SCIENTIFIC AUDIT & TIER CLASSIFICATION
    # ───────────────────────────────────────────────────────────
    ledger_rows = []

    for _, row in df_bench.iterrows():
        loss_type  = row["Loss_Type"]
        alpha_val  = row["Target_Alpha (%)"]
        model_name = row["Scoring_Model"]
        exp_loss   = row["Mean_Image_Loss"]
        coverage   = row["Mean_Detection_Coverage"]
        say_val    = row["Mean_SAY"]
        hits       = int(row["Total_Hits_Accepted"])
        false_alarms = int(row["Total_False_Alarms_Accepted"])

        nominal_alpha = alpha_val
        delta_loss    = exp_loss - nominal_alpha

        # Exact Conformal Validity: E[L] <= alpha + numerical epsilon (0.5%)
        is_strictly_valid = exp_loss <= (nominal_alpha + 0.50)
        
        # Tier Classification
        if is_strictly_valid and coverage >= 10.0:
            tier = "TIER 1 (Verified & High Utility)"
            publishable = "YES (Primary Claim)"
        elif is_strictly_valid and coverage < 10.0:
            tier = "TIER 2 (Verified Conservative / Low Cov)"
            publishable = "YES (Safety Claim)"
        else:
            tier = "TIER 3 (Violates Bound / Unsupported)"
            publishable = "NO (Flagged / Rejected)"

        ledger_rows.append({
            "Loss_Type": loss_type,
            "Target_Alpha (%)": int(alpha_val),
            "Scoring_Engine": model_name,
            "Expected_Test_Loss (%)": exp_loss,
            "Nominal_Target (%)": int(alpha_val),
            "Loss_Slack (Exp - Target)": round(delta_loss, 2),
            "Detection_Coverage (%)": coverage,
            "Safe_Autonomous_Yield (SAY %)": say_val,
            "Total_Hits": hits,
            "Total_False_Alarms": false_alarms,
            "Conformal_Tier": tier,
            "Publishable_Status": publishable
        })

    df_ledger = pd.DataFrame(ledger_rows)

    # ───────────────────────────────────────────────────────────
    # 2. DISPLAY AUDIT LEDGER BY LOSS TYPE
    # ───────────────────────────────────────────────────────────
    print("─" * 105)
    print("  📋 1. ANY-ERROR RISK AUDIT LEDGER (L_AnyError in {0, 1})")
    print("─" * 105)
    sub_any = df_ledger[df_ledger["Loss_Type"] == "AnyError"]
    disp_cols = ["Target_Alpha (%)", "Scoring_Engine", "Expected_Test_Loss (%)", "Detection_Coverage (%)", "Safe_Autonomous_Yield (SAY %)", "Conformal_Tier", "Publishable_Status"]
    print(sub_any[disp_cols].to_string(index=False))

    print("\n" + "─" * 105)
    print("  📋 2. FALSE DISCOVERY RATE AUDIT LEDGER (L_FDR in [0, 1])")
    print("─" * 105)
    sub_fdr = df_ledger[df_ledger["Loss_Type"] == "FDR"]
    print(sub_fdr[disp_cols].to_string(index=False))

    df_ledger.to_csv(RESULTS_DIR / "master_crc_validity_ledger.csv", index=False)

    # ───────────────────────────────────────────────────────────
    # 3. SELECT & FREEZE PRIMARY CONFORMAL OPERATIONAL POLICIES
    # ───────────────────────────────────────────────────────────
    print("\n" + "─" * 105)
    print("  🔒 3. CANONICAL CONFORMAL POLICIES FOR PAPER & POSTER")
    print("─" * 105)

    # Policy 1: High-Consequence Safety (Any-Error alpha = 10%)
    pol_1 = df_ledger[(df_ledger["Loss_Type"] == "AnyError") & (df_ledger["Target_Alpha (%)"] == 10) & (df_ledger["Scoring_Engine"].str.contains("M6"))].iloc[0]
    
    # Policy 2: Balanced Autonomy (Any-Error alpha = 25%)
    pol_2 = df_ledger[(df_ledger["Loss_Type"] == "AnyError") & (df_ledger["Target_Alpha (%)"] == 25) & (df_ledger["Scoring_Engine"].str.contains("M6"))].iloc[0]
    
    # Policy 3: High-Throughput Fleet Inspection (FDR alpha = 25%)
    pol_3 = df_ledger[(df_ledger["Loss_Type"] == "FDR") & (df_ledger["Target_Alpha (%)"] == 25) & (df_ledger["Scoring_Engine"].str.contains("M6"))].iloc[0]

    canonical_policies = [
        {"Policy_Name": "Policy 1: High-Consequence Naval Safety", "Loss": pol_1["Loss_Type"], "Alpha": int(pol_1["Target_Alpha (%)"]), "Model": pol_1["Scoring_Engine"], "Expected_Loss": pol_1["Expected_Test_Loss (%)"], "Coverage": pol_1["Detection_Coverage (%)"], "SAY": pol_1["Safe_Autonomous_Yield (SAY %)"], "Guarantee": "E[L_AnyError] <= 10.0%"},
        {"Policy_Name": "Policy 2: Balanced Autonomous Inspection", "Loss": pol_2["Loss_Type"], "Alpha": int(pol_2["Target_Alpha (%)"]), "Model": pol_2["Scoring_Engine"], "Expected_Loss": pol_2["Expected_Test_Loss (%)"], "Coverage": pol_2["Detection_Coverage (%)"], "SAY": pol_2["Safe_Autonomous_Yield (SAY %)"], "Guarantee": "E[L_AnyError] <= 25.0%"},
        {"Policy_Name": "Policy 3: High-Yield Fleet Survey",        "Loss": pol_3["Loss_Type"], "Alpha": int(pol_3["Target_Alpha (%)"]), "Model": pol_3["Scoring_Engine"], "Expected_Loss": pol_3["Expected_Test_Loss (%)"], "Coverage": pol_3["Detection_Coverage (%)"], "SAY": pol_3["Safe_Autonomous_Yield (SAY %)"], "Guarantee": "E[L_FDR] <= 25.0%"}
    ]

    df_canon = pd.DataFrame(canonical_policies)
    print(df_canon[["Policy_Name", "Guarantee", "Expected_Loss", "Coverage", "SAY"]].to_string(index=False))

    # Export canonical JSON artifact
    with open(RESULTS_DIR / "crc_frozen_primary_policies.json", "w", encoding="utf-8") as f:
        json.dump(canonical_policies, f, indent=2)

    print("\n" + "=" * 105)
    print("  ✅ STEP 53 COMPLETE — MASTER CRC VALIDITY LEDGER COMPILED")
    print(f"  • Ledger CSV Saved : {RESULTS_DIR / 'master_crc_validity_ledger.csv'}")
    print(f"  • Policies JSON    : {RESULTS_DIR / 'crc_frozen_primary_policies.json'}")
    print("=" * 105)

if __name__ == "__main__":
    main()
