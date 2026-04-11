"""
=============================================================================
run_all.py  — Master Script: Run the entire pipeline end-to-end
=============================================================================
Order of execution:
  Step 1: Validate LEM solver (quick sanity check)
  Step 2: Generate dataset (1000 LEM simulations)
  Step 3: Train ML models
  Step 4: SHAP explainability analysis
  Step 5: Visualization & EDA
=============================================================================
"""

import os
import sys
import time

# ─── Path Setup ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from lem_solver         import run_simulation
from dataset_generation import run_dataset
from ml_training        import load_data, train_and_evaluate, plot_predictions, plot_feature_importance
from explainability     import run_shap_analysis, sensitivity_check
from visualization      import plot_eda, plot_model_comparison, visualize_slip_circle


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

N_SAMPLES   = 1000   # number of LEM simulations (increase to 1500 for final)
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
PLOTS_DIR   = os.path.join(MODELS_DIR, "plots")
EXPL_DIR    = os.path.join(MODELS_DIR, "explainability")

DATA_PATH   = os.path.join(DATA_DIR,   "clean_dataset.csv")
METRICS_CSV = os.path.join(MODELS_DIR, "metrics.csv")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def section(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Validate LEM Solver
# ─────────────────────────────────────────────────────────────────────────────

section("STEP 1: LEM Solver Validation")

test_cases = [
    dict(c=20, phi=30, E=50, gamma=18, beta=45, H=10),
    dict(c=5,  phi=20, E=20, gamma=20, beta=50, H=20),
    dict(c=40, phi=40, E=80, gamma=16, beta=30, H=15),
]

all_ok = True
for i, tc in enumerate(test_cases, 1):
    res = run_simulation(**tc)
    ok = res["converged"] and res["FoS_bishop"] is not None
    status = "PASS" if ok else "FAIL"
    print(f"  Test {i}: FoS={res.get('FoS_bishop','N/A')}  "
          f"u_max={res.get('u_max_mm','N/A')} mm  [{status}]")
    if not ok:
        all_ok = False

if not all_ok:
    print("WARNING: Some validation cases failed. Check lem_solver.py")
else:
    print("All validation tests passed.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Dataset Generation
# ─────────────────────────────────────────────────────────────────────────────

section("STEP 2: Dataset Generation (Latin Hypercube Sampling)")

if os.path.exists(DATA_PATH):
    print(f"Dataset already exists at {DATA_PATH}. Skipping generation.")
    print("Delete the file to regenerate.")
else:
    t0 = time.time()
    df = run_dataset(n_samples=N_SAMPLES, output_dir=DATA_DIR, seed=42)
    print(f"\nDataset generated in {time.time()-t0:.1f}s. "
          f"Shape: {df.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — ML Training & Evaluation
# ─────────────────────────────────────────────────────────────────────────────

section("STEP 3: ML Surrogate Model Training")

X_train, X_val, X_test, y_train, y_val, y_test = load_data(DATA_PATH)

trained, metrics_df = train_and_evaluate(
    X_train, X_val, X_test, y_train, y_val, y_test,
    output_dir=MODELS_DIR
)

print("\nGenerating prediction plots...")
plot_predictions(trained, output_dir=PLOTS_DIR)
plot_feature_importance(trained, output_dir=PLOTS_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Explainability
# ─────────────────────────────────────────────────────────────────────────────

section("STEP 4: SHAP Explainability Analysis")

best_model = "XGBoost"  # change to "LightGBM" if XGBoost not installed

run_shap_analysis(
    model_name=best_model,
    models_dir=MODELS_DIR,
    data_path=DATA_PATH,
    output_dir=EXPL_DIR,
    n_background=100,
    n_explain=200,
)

print("\nRunning physical consistency / sensitivity check...")
sensitivity_check(
    models_dir=MODELS_DIR,
    model_name=best_model,
    output_dir=EXPL_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Visualizations
# ─────────────────────────────────────────────────────────────────────────────

section("STEP 5: EDA & Visualization")

print("Generating EDA plots...")
plot_eda(DATA_PATH, output_dir=PLOTS_DIR)

print("Generating model comparison chart...")
plot_model_comparison(METRICS_CSV, output_dir=PLOTS_DIR)

print("Visualizing a sample critical slip circle...")
visualize_slip_circle(
    c=20, phi=30, E=50, gamma=18, beta=45, H=15,
    output_path=os.path.join(PLOTS_DIR, "sample_slip_circle.png")
)


# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────

section("ALL STEPS COMPLETE")
print(f"""
Output summary:
  Dataset (raw)     : {DATA_DIR}/raw_dataset.csv
  Dataset (clean)   : {DATA_PATH}
  Trained models    : {MODELS_DIR}/*.pkl
  Metrics           : {METRICS_CSV}
  Plots             : {PLOTS_DIR}/
  Explainability    : {EXPL_DIR}/

To predict with the surrogate:
  from visualization import predict_single
  result = predict_single(c=20, phi=30, E=50, gamma=18, beta=45, H=15)
  print(result)
""")
