"""
=============================================================================
MODULE 3: Machine Learning Surrogate Model Training & Evaluation
=============================================================================
Trains 4 models on the LEM dataset:
  1. Artificial Neural Network (MLP)
  2. Gradient Boosting (XGBoost)
  3. Gradient Boosting (LightGBM)
  4. Gaussian Process Regression

Targets: FoS_bishop, FoS_janbu, u_max_mm

Outputs:
  - Trained model files (.pkl / .pt)
  - Evaluation metrics CSV
  - Prediction vs Actual plots
  - Speed-up comparison
=============================================================================
"""

import numpy as np
import pandas as pd
import os
import time
import pickle
import warnings
warnings.filterwarnings("ignore")

# ─── Sklearn ────────────────────────────────────────────────────────────────
from sklearn.model_selection   import train_test_split, KFold, cross_val_score
from sklearn.preprocessing     import StandardScaler
from sklearn.neural_network    import MLPRegressor
from sklearn.gaussian_process  import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.metrics           import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline          import Pipeline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── XGBoost / LightGBM ──────────────────────────────────────────────────────
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed. Skipping XGBoost model.")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("WARNING: lightgbm not installed. Skipping LightGBM model.")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = ["c", "phi", "E", "gamma", "beta", "H"]
TARGET_COLS  = ["FoS_bishop", "FoS_janbu", "u_max_mm"]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & SPLITTING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: str, test_size: float = 0.20,
              val_size: float = 0.10, seed: int = 42):
    """
    Load clean dataset and split into train / validation / test.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    Each y is a DataFrame with all target columns.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=FEATURE_COLS + TARGET_COLS)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COLS]

    # First split: train+val vs test
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)

    # Second split: train vs validation
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, random_state=seed)

    print(f"Dataset split:  train={len(X_train)}  "
          f"val={len(X_val)}  test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def build_models():
    """
    Return a dict of {model_name: sklearn Pipeline}.
    Each pipeline = StandardScaler + Regressor.
    """
    models = {}

    # ── 1. Multi-Layer Perceptron ──────────────────────────────────────────
    models["MLP"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=42,
        ))
    ])

    # ── 2. XGBoost ────────────────────────────────────────────────────────
    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  xgb.XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.01,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
            ))
        ])

    # ── 3. LightGBM ───────────────────────────────────────────────────────
    if HAS_LGB:
        models["LightGBM"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  lgb.LGBMRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.01,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1,
            ))
        ])

    # ── 4. Gaussian Process Regression ────────────────────────────────────
    # Note: GPR is slow for large datasets; use on a subset if n > 500
    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(6)) + WhiteKernel()
    models["GPR"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            normalize_y=True,
            random_state=42,
        ))
    ])

    return models


# ─────────────────────────────────────────────────────────────────────────────
# METRICS CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> dict:
    """Compute RMSE, MAE, R² for a single target."""
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {
        "target": target,
        "RMSE":   round(float(np.sqrt(mse)), 5),
        "MAE":    round(float(mae), 5),
        "R2":     round(float(r2), 5),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test,
                       output_dir: str = "."):
    """
    Train all models on all targets, evaluate, save results.

    For GPR: use at most 400 training samples (speed constraint).
    """
    os.makedirs(output_dir, exist_ok=True)
    models      = build_models()
    all_metrics = []
    trained     = {}

    for model_name, pipeline in models.items():
        print(f"\n{'='*50}")
        print(f"Training: {model_name}")
        print(f"{'='*50}")
        trained[model_name] = {}

        for target in TARGET_COLS:
            print(f"  Target: {target}", end=" ... ")

            y_tr = y_train[target].values
            y_vl = y_val[target].values
            y_te = y_test[target].values

            # GPR: subsample for speed
            if model_name == "GPR" and len(X_train) > 400:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(X_train), 400, replace=False)
                X_fit, y_fit = X_train[idx], y_tr[idx]
            else:
                X_fit, y_fit = X_train, y_tr

            # ── Fit ────────────────────────────────────────────────────────
            t0 = time.perf_counter()
            pipeline.fit(X_fit, y_fit)
            train_time = time.perf_counter() - t0

            # ── Predict (measure inference time) ──────────────────────────
            t0 = time.perf_counter()
            y_pred_test = pipeline.predict(X_test)
            infer_time  = time.perf_counter() - t0

            # ── Metrics ────────────────────────────────────────────────────
            met = compute_metrics(y_te, y_pred_test, target)
            met["model"]       = model_name
            met["train_time_s"] = round(train_time, 3)
            met["infer_ms_per_sample"] = round(infer_time / len(X_test) * 1000, 4)
            all_metrics.append(met)

            print(f"R²={met['R2']:.4f}  RMSE={met['RMSE']:.4f}  "
                  f"train={train_time:.1f}s  infer={met['infer_ms_per_sample']:.3f}ms/sample")

            # ── Save model ─────────────────────────────────────────────────
            model_path = os.path.join(
                output_dir, f"{model_name}_{target}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(pipeline, f)

            trained[model_name][target] = {
                "pipeline":   pipeline,
                "y_pred":     y_pred_test,
                "y_true":     y_te,
            }

    # ── Save metrics table ────────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)[
        ["model", "target", "R2", "RMSE", "MAE",
         "train_time_s", "infer_ms_per_sample"]]
    metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    print("\n\nAll Metrics:\n")
    print(metrics_df.to_string(index=False))

    return trained, metrics_df


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION vs ACTUAL PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_predictions(trained: dict, output_dir: str = "."):
    """
    Generate Predicted vs Actual scatter plots for each model × target.
    Saved as PNG files.
    """
    os.makedirs(output_dir, exist_ok=True)

    for model_name, targets in trained.items():
        n_targets = len(targets)
        fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5))
        if n_targets == 1:
            axes = [axes]

        for ax, (target, data) in zip(axes, targets.items()):
            y_true = data["y_true"]
            y_pred = data["y_pred"]
            r2     = r2_score(y_true, y_pred)

            vmin = min(y_true.min(), y_pred.min())
            vmax = max(y_true.max(), y_pred.max())

            ax.scatter(y_true, y_pred, alpha=0.5, s=15,
                       color="steelblue", edgecolors="none")
            ax.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=1.5,
                    label="1:1 line")
            ax.set_xlabel(f"LEM (True) — {target}", fontsize=11)
            ax.set_ylabel(f"ML Predicted — {target}", fontsize=11)
            ax.set_title(f"{model_name} | {target}\nR² = {r2:.4f}", fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"pred_vs_actual_{model_name}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved: {plot_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE PLOT (XGBoost / LightGBM)
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(trained: dict, output_dir: str = "."):
    """Plot feature importances for tree-based models."""
    for model_name in ["XGBoost", "LightGBM"]:
        if model_name not in trained:
            continue

        fig, axes = plt.subplots(1, len(TARGET_COLS),
                                 figsize=(5 * len(TARGET_COLS), 4))
        if len(TARGET_COLS) == 1:
            axes = [axes]

        for ax, target in zip(axes, TARGET_COLS):
            if target not in trained[model_name]:
                continue
            pipeline = trained[model_name][target]["pipeline"]
            regressor = pipeline.named_steps["model"]

            importances = regressor.feature_importances_
            sorted_idx  = np.argsort(importances)[::-1]

            ax.bar(range(len(FEATURE_COLS)),
                   importances[sorted_idx],
                   color="teal", alpha=0.8)
            ax.set_xticks(range(len(FEATURE_COLS)))
            ax.set_xticklabels(
                [FEATURE_COLS[i] for i in sorted_idx],
                rotation=30, ha="right")
            ax.set_title(f"{model_name} | {target}", fontsize=11)
            ax.set_ylabel("Feature Importance", fontsize=10)
            ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, f"feature_importance_{model_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Feature importance plot saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH  = os.path.join("data", "clean_dataset.csv")
    OUTPUT_DIR = "models"

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("Run 02_dataset_generation.py first.")
        exit(1)

    # ── Load data ──────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(DATA_PATH)

    # ── Train & evaluate ───────────────────────────────────────────────────
    trained, metrics_df = train_and_evaluate(
        X_train, X_val, X_test, y_train, y_val, y_test,
        output_dir=OUTPUT_DIR
    )

    # ── Plots ──────────────────────────────────────────────────────────────
    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    print("\nGenerating prediction plots...")
    plot_predictions(trained, output_dir=plots_dir)
    print("Generating feature importance plots...")
    plot_feature_importance(trained, output_dir=plots_dir)

    print("\nDone! All models saved to:", OUTPUT_DIR)
