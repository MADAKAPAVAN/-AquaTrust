"""
AquaTrust v3 — Step 63: Canonical Poster Specification & Content Freeze
========================================================================
Compiles and freezes the official narrative text, panel specifications, and
figure mappings for the Smart Amrita Hackathon (SAH) 2026 poster presentation.

Outputs:
  - D:/AquaTrust/results_v3/canonical_poster_narrative_spec.txt
  - D:/AquaTrust/results_v3/poster_copy_blocks.json
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(r"D:\AquaTrust")
RESULTS_DIR  = PROJECT_ROOT / "results_v3"
FIGURES_DIR  = PROJECT_ROOT / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

poster_spec = {
    "metadata": {
        "title": "AquaTrust: Calibrated Reliability and Conformal Risk Control for Side-Scan Sonar Object Detection",
        "track": "Track T3: AI & Applied Machine Learning",
        "competition": "Smart Amrita Hackathon (SAH) 2026",
        "thematic_anchor": "SIH Problem Statement 26057",
        "presenter": "Pavan (Solo Project)",
        "protocol": "5-Fold Nested Cross-Validation (1,170 SSS Images, N=2,115 Detections)"
    },
    "panels": {
        "panel_1_problem": {
            "title": "Problem Statement & Motivation",
            "body": "Deep neural object detectors (YOLOv8) on Autonomous Underwater Vehicles (AUVs) suffer from severe perceptual overconfidence due to acoustic speckle, contrast loss, and benthic clutter, outputting >90% confidence on false alarms. This triggers costly autonomous mission disruptions."
        },
        "panel_2_protocol": {
            "title": "Leakage-Safe 5-Fold Nested Protocol",
            "body": "1,170 physical SSS images partitioned into 5 disjoint outer test folds. Geometric diagnostics revealed that 69.0% of targets are severely small (<1,024 px^2). High-resolution YOLOv8s @ 832x832 achieved mAP50 = 0.4585 (+65.4% relative gain). Every image is evaluated exactly once as an untouched test sample."
        },
        "panel_3_methodology": {
            "title": "Multi-Evidence Formulation & Architecture",
            "body": "AquaTrust extracts raw confidence c_i, patch contrast Q_c, speckle variability Q_v, and perturbation consistency S_i. The Class-Interaction Model (M6) modulates acoustic features by predicted class k_hat to avoid penalizing specular MILCO returns while suppressing high-variance NOMBO clutter."
        },
        "panel_4_benchmark": {
            "title": "Canonical 5-Fold Results Benchmark",
            "table_ref": "canonical_poster_table.csv",
            "summary": "AquaTrust M6 achieves ECE-10 = 0.0689 (-52.7% vs Raw), ROC-AUC = 0.7047 (p < 0.001, 100% Bootstrap Win Rate), Risk@80% = 49.65% (p < 0.01 vs Raw and Platt), Coverage@20% = 24.48%, and SAY@20% = 17.92%."
        },
        "panel_5_insights": {
            "title": "Calibration & Selective Risk Insights",
            "figures": ["Figure_1_Master_Reliability_Diagram.png", "Figure_2_Ablation_Forest_Plot.png", "Figure_3_Selective_Risk_Coverage.png"],
            "body": "Platt scaling alone drops discrimination (AUC=0.6467). AquaTrust M6 resolves this trade-off. At 80% coverage, AquaTrust reduces false alarm risk from 51.18% to 49.65% (p < 0.01 across 2,000 paired bootstrap resamples)."
        },
        "panel_6_conformal": {
            "title": "Conformal Risk Control & Utility Frontier",
            "figure": "Figure_4_Conformal_Utility_Frontier.png",
            "body": "Image-level CRC guarantees E[L_AnyError] <= alpha. At alpha=20%, AquaTrust delivers a +25.5% relative gain in autonomous coverage and a +20.3% gain in Safe Autonomous Yield (SAY) over raw YOLOv8s (p < 0.001)."
        },
        "panel_7_pac_and_deployment": {
            "title": "High-Confidence PAC-CRC & Deployment",
            "figures": ["Figure_5_Marginal_vs_PAC_CRC_Tradeoff.png", "Figure_6_Operational_Disagreement_Case.png"],
            "body": "Exact Clopper-Pearson LTT testing provides 90% and 95% PAC high-probability guarantees for high-consequence naval operations, quantifying the exact coverage trade-off required for provable safety."
        },
        "panel_8_conclusion": {
            "title": "Limitations & Conclusion",
            "body": "Evaluated on naval SSS mine-detection imagery as a structural sensing proxy. AquaTrust proves that auxiliary acoustic quality features, perturbation consistency, and conformal calibration enable safe, verifiable autonomous marine inspection."
        }
    }
}

spec_text = f"""================================================================================
          AQUATRUST v3 — OFFICIAL CANONICAL POSTER SPECIFICATION
================================================================================
Title       : {poster_spec['metadata']['title']}
Track       : {poster_spec['metadata']['track']}
Event       : {poster_spec['metadata']['competition']}
Presenter   : {poster_spec['metadata']['presenter']}
Protocol    : {poster_spec['metadata']['protocol']}
================================================================================

PANEL 1: {poster_spec['panels']['panel_1_problem']['title']}
{poster_spec['panels']['panel_1_problem']['body']}

PANEL 2: {poster_spec['panels']['panel_2_protocol']['title']}
{poster_spec['panels']['panel_2_protocol']['body']}

PANEL 3: {poster_spec['panels']['panel_3_methodology']['title']}
{poster_spec['panels']['panel_3_methodology']['body']}

PANEL 4: {poster_spec['panels']['panel_4_benchmark']['title']}
{poster_spec['panels']['panel_4_benchmark']['summary']}

PANEL 5: {poster_spec['panels']['panel_5_insights']['title']}
{poster_spec['panels']['panel_5_insights']['body']}
Figures: {', '.join(poster_spec['panels']['panel_5_insights']['figures'])}

PANEL 6: {poster_spec['panels']['panel_6_conformal']['title']}
{poster_spec['panels']['panel_6_conformal']['body']}
Figure: {poster_spec['panels']['panel_6_conformal']['figure']}

PANEL 7: {poster_spec['panels']['panel_7_pac_and_deployment']['title']}
{poster_spec['panels']['panel_7_pac_and_deployment']['body']}
Figures: {', '.join(poster_spec['panels']['panel_7_pac_and_deployment']['figures'])}

PANEL 8: {poster_spec['panels']['panel_8_conclusion']['title']}
{poster_spec['panels']['panel_8_conclusion']['body']}
================================================================================
"""

(RESULTS_DIR / "canonical_poster_narrative_spec.txt").write_text(spec_text, encoding="utf-8")
with open(RESULTS_DIR / "poster_copy_blocks.json", "w", encoding="utf-8") as jf:
    json.dump(poster_spec, jf, indent=2)

print("=" * 95)
print("  ✅ STEP 63 COMPLETE — Canonical Poster Specification Exported")
print(f"  • Plain Text Spec : {RESULTS_DIR / 'canonical_poster_narrative_spec.txt'}")
print(f"  • JSON Copy Blocks: {RESULTS_DIR / 'poster_copy_blocks.json'}")
print("=" * 95)
