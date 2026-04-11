"""
=============================================================================
MODULE 5: Visualization, EDA & Final Report Generation
=============================================================================
Covers:
  1. Exploratory Data Analysis (EDA) plots
  2. Failure circle visualizer (for a single slope case)
  3. Model comparison radar chart
  4. Speed-up comparison table
  5. Utility: predict with a saved model (for deployment/demo)
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


FEATURE_COLS = ["c", "phi", "E", "gamma", "beta", "H"]
TARGET_COLS  = ["FoS_bishop", "FoS_janbu", "u_max_mm"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. EDA PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_eda(data_path: str, output_dir: str = "."):
    """
    Exploratory Data Analysis:
      - Histograms of all inputs and outputs
      - Correlation heatmap
      - FoS distribution
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path).dropna()

    # ── Histograms ─────────────────────────────────────────────────────────
    all_cols = FEATURE_COLS + TARGET_COLS
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()

    for ax, col in zip(axes, all_cols):
        df[col].hist(ax=ax, bins=30, color="steelblue",
                     edgecolor="white", alpha=0.85)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    for ax in axes[len(all_cols):]:
        ax.set_visible(False)

    plt.suptitle("Dataset — Input & Output Distributions", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_histograms.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Correlation heatmap ────────────────────────────────────────────────
    corr = df[all_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(all_cols)))
    ax.set_yticks(range(len(all_cols)))
    ax.set_xticklabels(all_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(all_cols, fontsize=9)

    for i in range(len(all_cols)):
        for j in range(len(all_cols)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(val) > 0.6 else "black")

    ax.set_title("Pearson Correlation Matrix", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_correlation.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── FoS stability class distribution ──────────────────────────────────
    df["stability"] = pd.cut(df["FoS_bishop"],
                             bins=[0, 1.0, 1.25, 1.5, np.inf],
                             labels=["Unstable (<1.0)",
                                     "Critical (1.0–1.25)",
                                     "Marginal (1.25–1.5)",
                                     "Stable (>1.5)"])
    counts = df["stability"].value_counts().sort_index()
    colors = ["#d73027", "#fc8d59", "#fee090", "#91bfdb"]

    plt.figure(figsize=(7, 4))
    plt.bar(counts.index, counts.values, color=colors, edgecolor="white")
    plt.xlabel("Stability Class", fontsize=11)
    plt.ylabel("Count", fontsize=11)
    plt.title("Slope Stability Class Distribution (FoS Bishop)", fontsize=12)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_stability_classes.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    print(f"EDA plots saved to: {output_dir}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. FAILURE CIRCLE VISUALIZER
# ─────────────────────────────────────────────────────────────────────────────

def visualize_slip_circle(c, phi, E, gamma, beta, H,
                          output_path: str = "slip_circle.png"):
    """
    Draw the slope geometry and the critical slip circle for a given parameter set.
    """
    from lem_solver import (
        build_slope_geometry, find_critical_fos, discretize_circle)

    geo = build_slope_geometry(H, beta)
    res = find_critical_fos(c, phi, gamma, H, beta, n_grid=12, n_random=100)

    cx = res["critical_cx"]
    cy = res["critical_cy"]
    R  = res["critical_R"]
    fos_b = res["FoS_bishop"]

    # Slope outline
    toe_x   = geo["toe"][0]
    crest_x = geo["crest"][0]
    pad = 0.3 * H

    x_profile = [toe_x - pad, toe_x, crest_x, crest_x + pad]
    y_profile = [0, 0, H, H]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Fill slope body
    ax.fill_between([toe_x - pad, toe_x, crest_x, crest_x + pad],
                    [0, 0, H, H],
                    y2=-0.15 * H,
                    color="#c8a96e", alpha=0.6, label="Soil body")
    ax.plot(x_profile, y_profile, "k-", linewidth=2, label="Slope surface")

    # Critical slip circle arc
    theta_arr = np.linspace(0, 2 * np.pi, 360)
    x_circ = cx + R * np.cos(theta_arr)
    y_circ = cy - R * np.sin(theta_arr)

    # Only plot the part below the slope surface and above y=0
    mask = (y_circ >= -0.05 * H) & (y_circ <= H + 0.1 * H)
    ax.plot(x_circ[mask], y_circ[mask], "r--",
            linewidth=2, label=f"Critical slip circle (FoS={fos_b:.3f})")

    # Circle centre
    ax.plot(cx, cy, "r^", markersize=10, label="Circle centre")

    ax.set_xlim(toe_x - pad * 1.5, crest_x + pad * 1.5)
    ax.set_ylim(-0.3 * H, cy + R * 0.3)
    ax.set_xlabel("Horizontal distance (m)", fontsize=11)
    ax.set_ylabel("Elevation (m)", fontsize=11)
    ax.set_title(
        f"Critical Slip Circle  |  c={c} kPa, φ={phi}°, β={beta}°, H={H} m\n"
        f"FoS (Bishop) = {fos_b:.3f}",
        fontsize=12
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Slip circle plot saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL COMPARISON CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(metrics_csv: str, output_dir: str = "."):
    """
    Bar chart comparing R² and RMSE across models for FoS_bishop.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(metrics_csv)
    target_df = df[df["target"] == "FoS_bishop"].copy()

    if target_df.empty:
        print("No FoS_bishop metrics found.")
        return

    models = target_df["model"].tolist()
    r2s    = target_df["R2"].tolist()
    rmses  = target_df["RMSE"].tolist()
    speeds = target_df["infer_ms_per_sample"].tolist()

    x = np.arange(len(models))
    width = 0.3

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # R²
    axes[0].bar(x, r2s, width=0.5, color="steelblue", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15, ha="right")
    axes[0].set_ylabel("R²")
    axes[0].set_title("Coefficient of Determination (R²)\nFoS_bishop")
    axes[0].axhline(0.99, color="red", linestyle="--",
                    linewidth=1, label="R²=0.99")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    # RMSE
    axes[1].bar(x, rmses, width=0.5, color="tomato", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15, ha="right")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Root Mean Square Error (RMSE)\nFoS_bishop")
    axes[1].grid(True, axis="y", alpha=0.3)

    # Speed
    axes[2].bar(x, speeds, width=0.5, color="teal", alpha=0.85)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=15, ha="right")
    axes[2].set_ylabel("ms / sample")
    axes[2].set_title("Inference Speed (ms per sample)")
    axes[2].grid(True, axis="y", alpha=0.3)

    plt.suptitle("Model Comparison — FoS_bishop", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Model comparison plot saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. QUICK PREDICTION UTILITY (for deployment)
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(c, phi, E, gamma, beta, H,
                   model_name: str = "XGBoost",
                   models_dir: str = "models"):
    """
    Predict FoS and displacement for a single input using a saved model.

    Parameters
    ----------
    c, phi, E, gamma, beta, H : float – geotechnical parameters
    model_name : str  – model to use ("XGBoost", "LightGBM", "MLP")
    models_dir : str  – directory where .pkl files are stored

    Returns
    -------
    dict : {FoS_bishop, FoS_janbu, u_max_mm}
    """
    X = np.array([[c, phi, E, gamma, beta, H]])
    results = {}

    for target in TARGET_COLS:
        pkl_path = os.path.join(models_dir, f"{model_name}_{target}.pkl")
        if not os.path.exists(pkl_path):
            results[target] = None
            continue
        with open(pkl_path, "rb") as f:
            pipeline = pickle.load(f)
        results[target] = round(float(pipeline.predict(X)[0]), 4)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH   = os.path.join("data", "clean_dataset.csv")
    METRICS_CSV = os.path.join("models", "metrics.csv")
    PLOTS_DIR   = os.path.join("models", "plots")

    # ── EDA ───────────────────────────────────────────────────────────────
    if os.path.exists(DATA_PATH):
        print("Generating EDA plots...")
        plot_eda(DATA_PATH, output_dir=PLOTS_DIR)

    # ── Model comparison ─────────────────────────────────────────────────
    if os.path.exists(METRICS_CSV):
        print("Generating model comparison chart...")
        plot_model_comparison(METRICS_CSV, output_dir=PLOTS_DIR)

    # ── Slip circle visualizer ────────────────────────────────────────────
    print("\nVisualizing critical slip circle for a sample case...")
    visualize_slip_circle(
        c=20, phi=30, E=50, gamma=18, beta=45, H=15,
        output_path=os.path.join(PLOTS_DIR, "sample_slip_circle.png")
    )

    # ── Quick prediction demo ─────────────────────────────────────────────
    if os.path.exists(os.path.join("models", "XGBoost_FoS_bishop.pkl")):
        print("\nQuick prediction (XGBoost):")
        res = predict_single(c=20, phi=30, E=50, gamma=18, beta=45, H=15)
        for k, v in res.items():
            print(f"  {k} = {v}")
