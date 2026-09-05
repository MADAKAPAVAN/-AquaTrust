"""
AquaTrust v3 — Step 45: Joint Multi-Objective Differentiable Neural Engine
==========================================================================
Architecture:
  - Input Layer : 6-Dimensional Evidence Vector [c, Qc, Qv, S, Var, k_hat]
  - Shared Trunk: Linear(6 -> 16) + GELU + LayerNorm
  - Head 1 (Cal): Differentiable Soft-ECE + Brier Loss with Class-Gating
  - Head 2 (Rnk): Pairwise Margin Ranking Loss (Maximizes AUC & AURC)

Training Protocol:
  - Training   : features_calib_fit.csv (N=290)
  - Validation : features_calib_select.csv (N=317)
  - Test Set   : features_test_locked.csv (Evaluated only once locked)

Outputs:
  - D:/AquaTrust/results_v2/v3_training_convergence_log.csv
  - D:/AquaTrust/results_v2/v3_locked_test_benchmark.csv
  - D:/AquaTrust/models/aquatrust_v3_joint_engine.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

# ── Configuration ──────────────────────────────────────────────
PROJECT_ROOT = Path(r"D:\AquaTrust")
RESULTS_DIR  = PROJECT_ROOT / "results_v2"
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CALIB_FIT_CSV    = RESULTS_DIR / "features_calib_fit.csv"
CALIB_SELECT_CSV = RESULTS_DIR / "features_calib_select.csv"
TEST_LOCKED_CSV  = RESULTS_DIR / "features_test_locked.csv"

SEED        = 2026
EPOCHS      = 250
LR          = 0.005
MARGIN      = 0.25
ALPHA_ECE   = 0.40
BETA_RANK   = 1.00
WEIGHT_DECAY = 0.001

torch.manual_seed(SEED)
np.random.seed(SEED)

def compute_fixed_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for low, high in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (probs >= low) & (probs <= high) if high == 1.0 else (probs >= low) & (probs < high)
        prop = np.mean(in_bin)
        if prop > 0:
            acc = np.mean(labels[in_bin])
            conf = np.mean(probs[in_bin])
            ece += np.abs(acc - conf) * prop
    return round(float(ece), 4)

def compute_aurc(probs: np.ndarray, labels: np.ndarray) -> float:
    coverages = np.linspace(0.1, 1.0, 50)
    order = np.argsort(probs)[::-1]
    risks = []
    n = len(labels)
    for c in coverages:
        k = max(1, int(round(c * n)))
        acc = np.mean(labels[order[:k]])
        risks.append(1.0 - acc)
    try: return float(np.trapezoid(risks, coverages))
    except AttributeError: return float(np.trapz(risks, coverages))

def compute_risk_at_cov(probs: np.ndarray, labels: np.ndarray, cov: float = 0.80) -> float:
    order = np.argsort(probs)[::-1]
    k = max(1, int(round(cov * len(labels))))
    return float(1.0 - np.mean(labels[order[:k]]))

# ── Differentiable Soft-ECE Loss ───────────────────────────────
class SoftECELoss(nn.Module):
    def __init__(self, n_bins=10, sigma=0.05):
        super().__init__()
        self.n_bins = n_bins
        self.sigma = sigma
        self.bin_centers = torch.linspace(1.0 / (2 * n_bins), 1.0 - 1.0 / (2 * n_bins), n_bins)

    def forward(self, probs, labels):
        ece = torch.tensor(0.0, device=probs.device)
        self.bin_centers = self.bin_centers.to(probs.device)
        
        for center in self.bin_centers:
            # Gaussian soft bin membership
            weights = torch.exp(-((probs - center) ** 2) / (2 * self.sigma ** 2))
            weight_sum = weights.sum() + 1e-6
            
            bin_acc = (weights * labels).sum() / weight_sum
            bin_conf = (weights * probs).sum() / weight_sum
            prop = weight_sum / len(probs)
            
            ece += torch.abs(bin_acc - bin_conf) * prop
        return ece

# ── AquaTrust v3 Neural Architecture ───────────────────────────
class AquaTrustV3Engine(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=16):
        super().__init__()
        # Shared Representation Backbone
        self.shared_trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10)
        )

        # Head 1: Probability Calibration Head (with Class Interaction)
        self.cal_main  = nn.Linear(hidden_dim, 1)
        self.cal_inter = nn.Linear(hidden_dim, 1)

        # Head 2: Discriminative Ranking Head
        self.rank_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, k_class):
        h = self.shared_trunk(x)
        
        # Class-gated logit: main + inter * k
        cal_logit = self.cal_main(h) + self.cal_inter(h) * k_class
        p_cal = torch.sigmoid(cal_logit).squeeze(-1)
        
        # Ranking Score
        s_rank = self.rank_head(h).squeeze(-1)
        
        return p_cal, s_rank

def main():
    print("=" * 95)
    print("  AQUATRUST v3 — JOINT MULTI-OBJECTIVE DIFFERENTIABLE NEURAL ENGINE")
    print("=" * 95)

    df_fit = pd.read_csv(CALIB_FIT_CSV)
    df_sel = pd.read_csv(CALIB_SELECT_CSV)
    df_tst = pd.read_csv(TEST_LOCKED_CSV)

    feature_cols = ["confidence", "q_contrast", "q_variability", "box_consistency", "conf_variance", "pred_class"]

    scaler = StandardScaler()
    X_fit_sc = scaler.fit_transform(df_fit[feature_cols].values)
    X_sel_sc = scaler.transform(df_sel[feature_cols].values)
    X_tst_sc = scaler.transform(df_tst[feature_cols].values)

    # Convert to PyTorch Tensors
    t_X_fit = torch.tensor(X_fit_sc, dtype=torch.float32)
    t_y_fit = torch.tensor(df_fit["is_correct"].values, dtype=torch.float32)
    t_k_fit = torch.tensor(df_fit["pred_class"].values, dtype=torch.float32).unsqueeze(-1)

    t_X_sel = torch.tensor(X_sel_sc, dtype=torch.float32)
    t_y_sel = torch.tensor(df_sel["is_correct"].values, dtype=torch.float32)
    t_k_sel = torch.tensor(df_sel["pred_class"].values, dtype=torch.float32).unsqueeze(-1)

    t_X_tst = torch.tensor(X_tst_sc, dtype=torch.float32)
    t_y_tst = torch.tensor(df_tst["is_correct"].values, dtype=torch.float32)
    t_k_tst = torch.tensor(df_tst["pred_class"].values, dtype=torch.float32).unsqueeze(-1)

    # Instantiate Model, Losses & Optimizer
    model = AquaTrustV3Engine(input_dim=len(feature_cols), hidden_dim=16)
    brier_loss_fn = nn.MSELoss()
    soft_ece_fn   = SoftECELoss(n_bins=10, sigma=0.05)
    optimizer     = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_score = 999.0
    best_weights = None
    convergence_log = []

    print(f"\n🚀 Training AquaTrust v3 Engine for {EPOCHS} epochs with Joint Differentiable Loss...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        p_cal_fit, s_rank_fit = model(t_X_fit, t_k_fit)

        # 1. Calibration Loss
        loss_brier = brier_loss_fn(p_cal_fit, t_y_fit)
        loss_ece   = soft_ece_fn(p_cal_fit, t_y_fit)

        # 2. Pairwise Margin Ranking Loss
        pos_mask = (t_y_fit == 1)
        neg_mask = (t_y_fit == 0)
        s_pos = s_rank_fit[pos_mask]
        s_neg = s_rank_fit[neg_mask]
        
        # Pairwise differences: s_pos - s_neg
        diff_matrix = s_pos.unsqueeze(1) - s_neg.unsqueeze(0)
        loss_rank = torch.mean(torch.relu(MARGIN - diff_matrix) ** 2)

        # Total Joint Objective
        total_loss = loss_brier + ALPHA_ECE * loss_ece + BETA_RANK * loss_rank

        total_loss.backward()
        optimizer.step()

        # Validation Step
        model.eval()
        with torch.no_grad():
            p_sel, s_sel = model(t_X_sel, t_k_sel)
            np_p_sel = p_sel.numpy()
            np_s_sel = s_sel.numpy()
            np_y_sel = t_y_sel.numpy()

            val_brier = float(brier_score_loss(np_y_sel, np_p_sel))
            val_ece10 = float(compute_fixed_ece(np_p_sel, np_y_sel, 10))
            val_auc   = float(roc_auc_score(np_y_sel, np_s_sel))
            val_risk  = float(compute_risk_at_cov(np_s_sel, np_y_sel, 0.80))

            # Composite Validation Objective (Lower is better)
            val_composite = 0.35 * val_brier + 0.30 * val_ece10 + 0.20 * (1.0 - val_auc) + 0.15 * val_risk

            if val_composite < best_val_score:
                best_val_score = val_composite
                best_weights = model.state_dict()

        if epoch % 50 == 0 or epoch == EPOCHS:
            print(f"  Epoch [{epoch:3d}/{EPOCHS:3d}] | Total Loss: {total_loss.item():.4f} | Val Brier: {val_brier:.4f} | Val ECE: {val_ece10:.4f} | Val AUC: {val_auc:.4f} | Val Risk@80%: {val_risk:.4f}")

        convergence_log.append({
            "epoch": epoch, "total_loss": total_loss.item(), "loss_brier": loss_brier.item(),
            "loss_ece": loss_ece.item(), "loss_rank": loss_rank.item(), "val_brier": val_brier,
            "val_ece10": val_ece10, "val_auc": val_auc, "val_risk80": val_risk
        })

    pd.DataFrame(convergence_log).to_csv(RESULTS_DIR / "v3_training_convergence_log.csv", index=False)
    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), MODELS_DIR / "aquatrust_v3_joint_engine.pth")

    # ── Final Locked Test Evaluation (N=277) ───────────────────
    print("\n" + "=" * 95)
    print("  🏆 FINAL SCIENTIFIC BENCHMARK ON LOCKED TEST SET (N=277)")
    print("=" * 95)

    model.eval()
    with torch.no_grad():
        p_tst, s_tst = model(t_X_tst, t_k_tst)
        np_p_tst = np.clip(p_tst.numpy(), 1e-6, 1 - 1e-6)
        np_s_tst = s_tst.numpy()
        np_y_tst = t_y_tst.numpy()

    # Metrics
    tst_brier = round(float(brier_score_loss(np_y_tst, np_p_tst)), 4)
    tst_ece10 = compute_fixed_ece(np_p_tst, np_y_tst, 10)
    tst_auc   = round(float(roc_auc_score(np_y_tst, np_s_tst)), 4)
    tst_prauc = round(float(average_precision_score(np_y_tst, np_s_tst)), 4)
    tst_risk  = round(compute_risk_at_cov(np_s_tst, np_y_tst, 0.80), 4)
    tst_aurc  = round(compute_aurc(np_s_tst, np_y_tst), 4)
    tst_loss  = round(float(log_loss(np_y_tst, np_p_tst)), 4)

    bench_summary = pd.DataFrame([{
        "Architecture": "AquaTrust v3 (Joint Multi-Objective Engine)",
        "ECE-10 ↓": tst_ece10,
        "Risk@80% ↓": tst_risk,
        "Brier ↓": tst_brier,
        "LogLoss ↓": tst_loss,
        "AUC ↑": tst_auc,
        "PR-AUC ↑": tst_prauc,
        "AURC ↓": tst_aurc
    }])

    print(bench_summary.to_string(index=False))
    bench_summary.to_csv(RESULTS_DIR / "v3_locked_test_benchmark.csv", index=False)

    print("\n" + "=" * 95)
    print("  ✅ STEP 45 COMPLETE — AquaTrust v3 Engine Successfully Evaluated")
    print("=" * 95)

if __name__ == "__main__":
    main()
