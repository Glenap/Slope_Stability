# SlopeML — ML Surrogate Slope Stability Analyzer
## Full Project (Frontend + Backend)

---

## Project Structure

```
SlopeML/
│
├── index.html                   ← Main frontend (open this in a browser)
│
├── css/
│   └── style.css                ← Full stylesheet (dark industrial theme)
│
├── js/
│   ├── lem.js                   ← LEM solver in JavaScript (Bishop + Janbu)
│   ├── draw.js                  ← Canvas drawing (slope + slip circle)
│   ├── chart.js                 ← Sensitivity sweep line chart
│   └── app.js                   ← Main UI controller
│
└── python_backend/
    ├── 01_lem_solver.py         ← LEM solver (Python) — dataset ground truth
    ├── 02_dataset_generation.py ← Latin Hypercube Sampling + batch runs
    ├── 03_ml_training.py        ← Train XGBoost / LightGBM / MLP / GPR
    ├── 04_explainability.py     ← SHAP analysis + sensitivity check
    ├── 05_visualization.py      ← EDA plots + model comparison charts
    ├── run_all.py               ← Master pipeline script
    ├── requirements.txt         ← pip dependencies
    └── README.md                ← Full backend documentation
```

---

## Running the Frontend

**No server required.** Just open `index.html` in any modern browser:
- Double-click `index.html`, OR
- Right-click → Open With → Chrome / Firefox / Edge

### What the frontend does:
| Feature | Description |
|---------|-------------|
| **Simulator** | Enter soil + geometry parameters, click "Compute FoS" — instant results |
| **Slip Circle Plot** | Draws the critical failure surface on a slope cross-section |
| **Bishop + Janbu** | Both methods computed, FoS displayed with stability color coding |
| **Displacement** | Estimated max displacement (mm) shown in real time |
| **Sensitivity Sweep** | Vary one parameter across its full range → line chart of FoS response |
| **Theory section** | Explains the LEM methodology and ML surrogate pipeline |

### FoS Stability Thresholds:
| Color | Range | Status |
|-------|-------|--------|
| 🟢 Green | FoS ≥ 1.5 | Stable |
| 🟡 Yellow | 1.0 ≤ FoS < 1.5 | Marginal / Critical |
| 🔴 Red | FoS < 1.0 | Failure |

---

## Running the Python Backend

```bash
cd python_backend
pip install -r requirements.txt

# Option A: run everything
python run_all.py

# Option B: step by step
python 01_lem_solver.py          # test solver
python 02_dataset_generation.py  # generate 1000 LEM simulations
python 03_ml_training.py         # train ML surrogate models
python 04_explainability.py      # SHAP analysis
python 05_visualization.py       # plots and EDA
```

---

## Architecture

```
User Browser (index.html)
  └── js/lem.js          — Pure JS port of Bishop/Janbu solver
  └── js/draw.js         — Canvas rendering for slope visualization
  └── js/chart.js        — Sensitivity sweep chart
  └── js/app.js          — UI controller

Python Backend (offline, for training)
  └── LEM simulations → clean_dataset.csv
  └── ML training     → XGBoost/LightGBM/MLP models (.pkl)
  └── SHAP analysis   → explainability plots
```

The **frontend solver** (JavaScript) uses identical mathematics to the
Python backend — same Bishop formula, same Janbu correction factor, same
critical circle search strategy. This means the browser computes real LEM
results, not approximations.

---

## Browser Compatibility
Chrome 90+, Firefox 88+, Edge 90+, Safari 14+
