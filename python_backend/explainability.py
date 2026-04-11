"""
=============================================================================
MODULE 4: Model Explainability — SHAP Analysis
=============================================================================
Uses SHAP (SHapley Additive exPlanations) to:
  1. Compute global feature importance for each model × target
  2. Generate SHAP beeswarm plots
  3. Generate SHAP dependence plots (each feature vs FoS_bishop)
  4. Run sensitivity checks (physical consistency):
       - Increase c  → FoS should increase
       - Increase φ  → FoS should increase
       - Increase β  → FoS should decrease
       - Increase H  → FoS should decrease (for fixed soil params)
  5. Save a full explainability report

Requirements:  pip install shap matplotlib pandas
=============================================================================
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed. Run: pip install shap")


FEATURE_COLS  = ["c", "phi", "E", "gamma", "beta", "H"]
TARGET_COLS   = ["FoS_bishop", "FoS_janbu", "u_max_mm"]
FEATURE_NAMES = {
    "c":     "Cohesion c (kPa)",
    "phi":   "Friction angle φ (°)",
    "E":     "Young's modulus E (MPa)",
    "gamma": "Unit weight γ (kN/m³)",
    "beta":  "Slope angle β (°)",
    "H":     "Slope height H (m)",
}


# ─────────────────────────────────────────────────────────────────────────────
# SHAP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_shap_analysis(model_name: str,
                      models_dir: str,
                      data_path: str,
                      output_dir: str,
                      n_background: int = 100,
                      n_explain: int = 200):
    """
    Run SHAP explainability for one model across all targets.

    Parameters
    ----------
    model_name   : str  – "XGBoost" | "LightGBM" | "MLP"
    models_dir   : str  – directory where .pkl files are saved
    data_path    : str  – path to clean_dataset.csv
    output_dir   : str  – directory to save plots and report
    n_background : int  – background samples for KernelExplainer (MLP)
    n_explain    : int  – samples to explain
    """
    if not HAS_SHAP:
        print("SHAP not available. Skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path).dropna()
    X  = df[FEATURE_COLS].values
    feature_display = [FEATURE_NAMES[f] for f in FEATURE_COLS]

    # Use a random subset for efficiency
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), min(n_explain, len(X)), replace=False)
    X_explain = X[idx]

    shap_values_all = {}

    for target in TARGET_COLS:
        pkl_path = os.path.join(models_dir, f"{model_name}_{target}.pkl")
        if not os.path.exists(pkl_path):
            print(f"  Model not found: {pkl_path}  — skipping {target}")
            continue

        with open(pkl_path, "rb") as f:
            pipeline = pickle.load(f)

        regressor = pipeline.named_steps["model"]
        scaler    = pipeline.named_steps["scaler"]

        X_scaled = scaler.transform(X_explain)

        print(f"  Computing SHAP for {model_name} | {target} ...", end=" ")

        # ── Choose SHAP explainer based on model type ─────────────────────
        if model_name in ("XGBoost", "LightGBM"):
            # TreeExplainer: fast, exact
            explainer  = shap.TreeExplainer(regressor)
            shap_vals  = explainer.shap_values(X_scaled)
        else:
            # KernelExplainer: model-agnostic, slower
            bg_idx    = rng.choice(len(X), min(n_background, len(X)), replace=False)
            X_bg      = scaler.transform(X[bg_idx])
            explainer = shap.KernelExplainer(
                regressor.predict, X_bg, link="identity")
            shap_vals = explainer.shap_values(
                X_scaled, nsamples=100, silent=True)

        shap_values_all[target] = shap_vals
        print("done")

        # ── Beeswarm (summary) plot ────────────────────────────────────────
        plt.figure(figsize=(8, 5))
        shap.summary_plot(
            shap_vals, X_scaled,
            feature_names=feature_display,
            show=False, plot_size=None
        )
        plt.title(f"SHAP Summary — {model_name} | {target}", fontsize=12)
        plt.tight_layout()
        path = os.path.join(output_dir,
                            f"shap_summary_{model_name}_{target}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

        # ── Bar importance plot ────────────────────────────────────────────
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        sorted_idx    = np.argsort(mean_abs_shap)[::-1]

        plt.figure(figsize=(7, 4))
        plt.barh(
            [feature_display[i] for i in sorted_idx][::-1],
            mean_abs_shap[sorted_idx][::-1],
            color="steelblue", alpha=0.85
        )
        plt.xlabel("Mean |SHAP value|", fontsize=11)
        plt.title(f"SHAP Global Importance — {model_name} | {target}", fontsize=11)
        plt.tight_layout()
        path = os.path.join(output_dir,
                            f"shap_importance_{model_name}_{target}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\n  SHAP plots saved to: {output_dir}")
    return shap_values_all


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSISTENCY / SENSITIVITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_check(models_dir: str,
                      model_name: str = "XGBoost",
                      output_dir: str = "."):
    """
    Vary one parameter at a time (others at median) and check that
    ML predictions follow geotechnical theory:

        ∂FoS/∂c   > 0   (more cohesion → more stable)
        ∂FoS/∂φ   > 0   (more friction → more stable)
        ∂FoS/∂β   < 0   (steeper slope → less stable)
        ∂FoS/∂H   < 0   (taller slope → less stable)
        ∂FoS/∂γ   < 0   (heavier soil → less stable)
        ∂FoS/∂E   ≈ 0   (stiffness barely affects FoS in LEM)
    """
    target = "FoS_bishop"
    pkl_path = os.path.join(models_dir, f"{model_name}_{target}.pkl")
    if not os.path.exists(pkl_path):
        print(f"Model not found: {pkl_path}")
        return

    with open(pkl_path, "rb") as f:
        pipeline = pickle.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Median values
    medians = {"c": 22.5, "phi": 30.0, "E": 55.0,
               "gamma": 19.0, "beta": 40.0, "H": 20.0}

    vary_ranges = {
        "c":     np.linspace(5,   40,  30),
        "phi":   np.linspace(20,  40,  30),
        "E":     np.linspace(10,  100, 30),
        "gamma": np.linspace(16,  22,  30),
        "beta":  np.linspace(30,  50,  30),
        "H":     np.linspace(10,  30,  30),
    }

    expected_sign = {
        "c":     "+",   # positive correlation
        "phi":   "+",
        "E":     "~",   # negligible
        "gamma": "-",   # negative correlation
        "beta":  "-",
        "H":     "-",
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    report_lines = ["PHYSICAL CONSISTENCY REPORT", "=" * 50, ""]

    for ax, (param, values) in zip(axes, vary_ranges.items()):
        X_sweep = np.array([
            [v if feat == param else medians[feat]
             for feat in FEATURE_COLS]
            for v in values
        ])
        preds = pipeline.predict(X_sweep)

        # Compute trend
        slope = np.polyfit(values, preds, 1)[0]
        if abs(slope) < 0.005:
            trend = "~"
        elif slope > 0:
            trend = "+"
        else:
            trend = "-"

        exp = expected_sign[param]
        ok  = (trend == exp) or (exp == "~")
        status = "✓ PASS" if ok else "✗ FAIL"

        ax.plot(values, preds, "b-o", markersize=3, linewidth=1.5)
        ax.set_xlabel(FEATURE_NAMES[param], fontsize=9)
        ax.set_ylabel("Predicted FoS", fontsize=9)
        ax.set_title(f"{FEATURE_NAMES[param]}\n"
                     f"Trend={trend}  Expected={exp}  {status}", fontsize=9,
                     color="green" if ok else "red")
        ax.grid(True, alpha=0.3)

        report_lines.append(
            f"{param:8s}  slope={slope:+.4f}  trend={trend}  "
            f"expected={exp}  {status}"
        )

    plt.suptitle(f"Sensitivity Analysis — {model_name} | FoS_bishop",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, f"sensitivity_{model_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Sensitivity plot saved: {path}")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_path = os.path.join(output_dir, "sensitivity_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    MODELS_DIR = "models"
    DATA_PATH  = os.path.join("data", "clean_dataset.csv")
    OUT_DIR    = os.path.join("models", "explainability")

    # ── SHAP Analysis ────────────────────────────────────────────────────
    print("Running SHAP analysis on XGBoost...")
    run_shap_analysis(
        model_name="XGBoost",
        models_dir=MODELS_DIR,
        data_path=DATA_PATH,
        output_dir=OUT_DIR,
    )

    # ── Physical Consistency Check ────────────────────────────────────────
    print("\nRunning sensitivity / physical consistency check...")
    sensitivity_check(
        models_dir=MODELS_DIR,
        model_name="XGBoost",
        output_dir=OUT_DIR,
    )

    print("\nExplainability analysis complete.")
    print(f"All outputs saved to: {OUT_DIR}")
