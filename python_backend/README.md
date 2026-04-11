# ML Surrogate for LEM-Based Slope Stability Analysis
## Complete Project Guide

---

## What Changed from the Original Proposal

| Aspect | Original (FEM) | This Implementation (LEM) |
|--------|---------------|--------------------------|
| **Solver** | PLAXIS 2D (paid) — Finite Element + SSR | Pure Python — Limit Equilibrium (Bishop + Janbu) |
| **Failure surface** | Internally computed by FEM convergence | Critical circular slip surface (grid + random search) |
| **FoS method** | Shear Strength Reduction (SSR) | Bishop's Simplified + Janbu's Simplified |
| **Displacement** | FEM nodal displacements | Empirical formula (γH²/E scaled by FoS) |
| **Cost** | Expensive software license | Free, runs on any PC |
| **Speed per sim** | 30–300 seconds | < 1 second |
| **ML targets** | FoS_SSR, u_max (FEM) | FoS_bishop, FoS_janbu, u_max (empirical) |
| **Everything else** | Same | Same (LHS, ML algorithms, SHAP, timeline) |

---

## Project Structure

```
slope_stability_surrogate/
│
├── 01_lem_solver.py          ← Bishop + Janbu LEM solver (Module 1)
├── 02_dataset_generation.py  ← LHS sampling + batch simulation (Module 2)
├── 03_ml_training.py         ← MLP, XGBoost, LightGBM, GPR training (Module 3)
├── 04_explainability.py      ← SHAP + sensitivity analysis (Module 4)
├── 05_visualization.py       ← EDA, slip circle plot, model comparison (Module 5)
├── run_all.py                ← Master script (runs everything)
├── requirements.txt          ← Python dependencies
│
├── data/                     ← [auto-created] datasets
│   ├── raw_dataset.csv
│   └── clean_dataset.csv
│
└── models/                   ← [auto-created] trained models + plots
    ├── XGBoost_FoS_bishop.pkl
    ├── XGBoost_FoS_janbu.pkl
    ├── XGBoost_u_max_mm.pkl
    ├── LightGBM_*.pkl
    ├── MLP_*.pkl
    ├── GPR_*.pkl
    ├── metrics.csv
    ├── plots/
    └── explainability/
```

---

## Setup Instructions

### 1. Install Python (if not already installed)
Download from https://python.org — use Python 3.10 or 3.11.

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Rename files to module names (important!)
The files must be importable as a Python package. Create this structure:

```
your_project_folder/
├── slope_stability_surrogate/
│   ├── __init__.py               ← create empty file
│   ├── lem_solver.py             ← rename from 01_lem_solver.py
│   ├── dataset_generation.py     ← rename from 02_dataset_generation.py
│   ├── ml_training.py            ← rename from 03_ml_training.py
│   ├── explainability.py         ← rename from 04_explainability.py
│   └── visualization.py         ← rename from 05_visualization.py
├── run_all.py
└── requirements.txt
```

Create an empty `__init__.py`:
```bash
# Windows:
type nul > slope_stability_surrogate/__init__.py
# Linux/Mac:
touch slope_stability_surrogate/__init__.py
```

---

## Running the Project

### Option A: Run everything at once
```bash
python run_all.py
```

### Option B: Run step by step

**Step 1 — Test the LEM solver:**
```bash
python -m slope_stability_surrogate.lem_solver
```
Expected output: 3 test cases with FoS values around 1.0–2.5.

**Step 2 — Generate the dataset:**
```bash
python -m slope_stability_surrogate.dataset_generation
```
This runs 1000 LEM simulations (takes ~5–15 minutes).
Output: `data/clean_dataset.csv`

**Step 3 — Train ML models:**
```bash
python -m slope_stability_surrogate.ml_training
```
Trains MLP, XGBoost, LightGBM, GPR.
Output: `models/*.pkl` and `models/metrics.csv`

**Step 4 — Explainability:**
```bash
python -m slope_stability_surrogate.explainability
```
Output: SHAP plots in `models/explainability/`

**Step 5 — Visualize:**
```bash
python -m slope_stability_surrogate.visualization
```
Output: EDA plots, slip circle figure, comparison charts.

---

## Understanding the LEM Theory

### Bishop's Simplified Method
Divides the failure mass into vertical slices. For each slice:
- Weight W = γ × h × b
- Base inclination angle α
- FoS (iterative):

```
FoS = Σ[(c·b + W·tanφ) / mα] / Σ[W·sinα]

where mα = cosα + sinα·tanφ/FoS
```

### Janbu's Simplified Method
Similar to Bishop but uses horizontal force equilibrium instead of moment equilibrium. Includes a correction factor f₀ for curvature of the slip surface:

```
FoS_Janbu = f₀ × Σ[(c·b + W·tanφ)/nα] / Σ[W·tanα]

where nα = cos²α × (1 + tanα·tanφ/FoS)
and f₀ ≈ 1 + 0.031 × (d/L)
```

### Critical Circle Search
The code searches for the circle with minimum FoS using:
1. **Coarse grid** (15×15 grid of centre positions)
2. **Fine random search** (200 random perturbations around best grid point)

---

## Parameter Reference

| Parameter | Symbol | Range | Unit |
|-----------|--------|-------|------|
| Cohesion | c | 5–40 | kPa |
| Friction angle | φ | 20–40 | ° |
| Young's modulus | E | 10–100 | MPa |
| Unit weight | γ | 16–22 | kN/m³ |
| Slope angle | β | 30–50 | ° |
| Slope height | H | 10–30 | m |

**Outputs:**
| Target | Description |
|--------|-------------|
| FoS_bishop | Factor of Safety (Bishop's method) |
| FoS_janbu | Factor of Safety (Janbu's method) |
| u_max_mm | Estimated max displacement (mm) |

---

## Expected Results

After training on 1000 simulations, expect:

| Model | R² (FoS_bishop) | RMSE |
|-------|----------------|------|
| XGBoost | 0.97–0.99 | 0.03–0.06 |
| LightGBM | 0.97–0.99 | 0.03–0.06 |
| MLP | 0.94–0.98 | 0.05–0.10 |
| GPR | 0.90–0.96 | 0.07–0.12 |

**Speed-up**: LEM ≈ 0.5 seconds/simulation. ML inference ≈ 0.01 ms/sample.
That's a **50,000× speed-up** for prediction after training.

---

## Physical Consistency Checks

The sensitivity analysis (Module 4) verifies:

| Parameter | Expected Effect on FoS | Reason |
|-----------|------------------------|--------|
| ↑ Cohesion (c) | ↑ FoS | More shear resistance |
| ↑ Friction angle (φ) | ↑ FoS | More shear resistance |
| ↑ Slope angle (β) | ↓ FoS | Steeper = more driving force |
| ↑ Slope height (H) | ↓ FoS | Taller = more weight |
| ↑ Unit weight (γ) | ↓ FoS | Heavier soil increases driving force more than resistance |
| ↑ Young's modulus (E) | ≈ no change | LEM FoS doesn't depend on stiffness |

---

## Common Issues

**"Module not found" error:**
Make sure you created `__init__.py` and named files correctly (see Setup Step 4).

**GPR is very slow:**
This is normal. GPR is trained on only 400 samples max. Reduce `n_background` in the explainability script if needed.

**FoS values seem too high or too low:**
Check parameter units — c must be in kPa, E in MPa, gamma in kN/m³.

**SHAP not working:**
Run `pip install shap` or upgrade: `pip install --upgrade shap`

---

## Extending the Project

1. **Add more samples**: Change `N_SAMPLES = 1500` in `run_all.py`
2. **Add seismic loading**: Modify `bishop_fos()` to include horizontal seismic coefficient kh
3. **Non-circular failure**: Replace circle search with Spencer's method for non-circular surfaces
4. **Probabilistic analysis**: Monte Carlo sampling over uncertain parameters using the trained surrogate

---

## References

1. Bishop, A.W. (1955). The use of slip circle in the stability analysis of slopes. *Géotechnique*, 5(1), 7–17.
2. Janbu, N. (1954). Application of composite slip surfaces for stability analysis. *Proc. European Conf. on Stability of Earth Slopes*, Stockholm.
3. Griffiths, D.V. & Lane, P.A. (1999). Slope stability analysis by finite elements. *Géotechnique*, 49(3), 387–403.
4. Lundberg, S.M. & Lee, S.I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
