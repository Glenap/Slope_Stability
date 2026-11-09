"""
=============================================================================
MODULE 6: Physical Consistency Audit for ML Slope Stability Surrogates
=============================================================================
Audits whether machine learning models adhere to fundamental geotechnical laws:
  1. Cohesion (c):     increasing c MUST increase FoS (dFoS/dc >= 0)
  2. Friction (phi):   increasing phi MUST increase FoS (dFoS/dphi >= 0)
  3. Slope angle (beta): increasing beta MUST decrease FoS (dFoS/dbeta <= 0)
  4. Slope height (H):   increasing H MUST decrease FoS (dFoS/dH <= 0)
  5. Unit weight (gamma): increasing gamma MUST decrease FoS (dFoS/dgamma <= 0)

Computes the Physical Violation Rate (PVR) across test cases and produces
empirical evidence of unconstrained ML failure modes.
=============================================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_clean_data(data_path="data/clean_dataset.csv"):
    df = pd.read_csv(data_path)
    features = ['c', 'phi', 'E', 'gamma', 'beta', 'H']
    return df, features

def audit_model_physics(model_pipeline, X, features, delta_c=3.0, delta_phi=3.0, delta_beta=3.0, delta_H=3.0, delta_gamma=1.0):
    """
    Computes Physical Violation Rates for all fundamental geotechnical laws.
    """
    N = len(X)
    pred_base = model_pipeline.predict(X)

    # 1. Cohesion: c + delta_c should increase FoS
    X_c = X.copy()
    c_idx = features.index('c')
    X_c[:, c_idx] += delta_c
    pred_c = model_pipeline.predict(X_c)
    viol_c = np.sum(pred_c < pred_base)

    # 2. Friction angle: phi + delta_phi should increase FoS
    X_phi = X.copy()
    phi_idx = features.index('phi')
    X_phi[:, phi_idx] += delta_phi
    pred_phi = model_pipeline.predict(X_phi)
    viol_phi = np.sum(pred_phi < pred_base)

    # 3. Slope angle: beta + delta_beta should decrease FoS
    X_beta = X.copy()
    beta_idx = features.index('beta')
    X_beta[:, beta_idx] += delta_beta
    pred_beta = model_pipeline.predict(X_beta)
    viol_beta = np.sum(pred_beta > pred_base)

    # 4. Slope height: H + delta_H should decrease FoS
    X_H = X.copy()
    H_idx = features.index('H')
    X_H[:, H_idx] += delta_H
    pred_H = model_pipeline.predict(X_H)
    viol_H = np.sum(pred_H > pred_base)

    # 5. Unit weight: gamma + delta_gamma should decrease FoS
    X_gamma = X.copy()
    gamma_idx = features.index('gamma')
    X_gamma[:, gamma_idx] += delta_gamma
    pred_gamma = model_pipeline.predict(X_gamma)
    viol_gamma = np.sum(pred_gamma > pred_base)

    total_checks = N * 5
    total_violations = viol_c + viol_phi + viol_beta + viol_H + viol_gamma
    overall_pvr = (total_violations / total_checks) * 100.0

    return {
        "N_samples": N,
        "viol_c_pct": (viol_c / N) * 100.0,
        "viol_phi_pct": (viol_phi / N) * 100.0,
        "viol_beta_pct": (viol_beta / N) * 100.0,
        "viol_H_pct": (viol_H / N) * 100.0,
        "viol_gamma_pct": (viol_gamma / N) * 100.0,
        "overall_PVR_pct": overall_pvr
    }

def run_physical_audit(models_dir="models", data_path="data/clean_dataset.csv"):
    df, features = load_clean_data(data_path)
    X = df[features].values
    
    # Audit available models
    models_to_test = {
        "XGBoost": "XGBoost_FoS_bishop.pkl",
        "LightGBM": "LightGBM_FoS_bishop.pkl",
        "MLP": "MLP_FoS_bishop.pkl"
    }

    results = []
    for name, pkl_name in models_to_test.items():
        pkl_path = os.path.join(models_dir, pkl_name)
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, "rb") as f:
            pipe = pickle.load(f)
        
        audit = audit_model_physics(pipe, X, features)
        audit["Model"] = name
        results.append(audit)

    audit_df = pd.DataFrame(results)
    out_csv = os.path.join(models_dir, "physics_audit_metrics.csv")
    audit_df.to_csv(out_csv, index=False)
    print(f"[+] Physical Consistency Audit saved to {out_csv}")
    print(audit_df.to_string(index=False))

    # Plot audit comparison
    plot_dir = os.path.join(models_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(results))
    width = 0.15

    labels = ["c (Viol %)", "phi (Viol %)", "beta (Viol %)", "H (Viol %)", "gamma (Viol %)"]
    keys = ["viol_c_pct", "viol_phi_pct", "viol_beta_pct", "viol_H_pct", "viol_gamma_pct"]

    for i, (k, l) in enumerate(zip(keys, labels)):
        vals = [r[k] for r in results]
        plt.bar(x + (i - 2) * width, vals, width, label=l)

    plt.xlabel("Model Architecture", fontweight="bold", fontsize=11)
    plt.ylabel("Physical Violation Rate (%)", fontweight="bold", fontsize=11)
    plt.title("Physical Consistency Audit: Geotechnical Law Violation Rates in Unconstrained ML", fontweight="bold", fontsize=12)
    plt.xticks(x, [r["Model"] for r in results])
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_png = os.path.join(plot_dir, "physics_violation_audit.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[+] Audit chart saved to {out_png}")

if __name__ == "__main__":
    run_physical_audit()
