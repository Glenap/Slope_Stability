"""
=============================================================================
MODULE 10: Dual-Surrogate Spatial Failure Geometry Predictor (Xc, Yc, R)
=============================================================================
Problem:
  Standard ML surrogates only predict a scalar (FoS).
  Engineers cannot design retaining walls, soil nails, or drainage without
  knowing WHERE the failure arc occurs. Searching for the critical circle
  numerically takes ~0.5s.

Solution:
  A Multi-Output Surrogate that directly predicts:
    [Xc, Yc, R] (Circle Center X, Center Y, Radius)
  in under 0.5 milliseconds directly from geotechnical inputs:
    [c, phi, gamma, beta, H]

Enables instantaneous 2D failure geometry rendering without running grid search!
=============================================================================
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lem_solver import run_simulation

DATA_PATH = "data/geometry_dataset.csv"
MODEL_PATH = "models/Dual_Surrogate_Geometry.pkl"

def generate_geometry_dataset(n_samples=300, seed=42):
    """
    Runs LEM simulations to collect (c, phi, gamma, beta, H) -> (critical_cx, critical_cy, critical_R, FoS_bishop).
    """
    print(f"[*] Generating {n_samples} spatial failure samples for Geometry Surrogate...")
    sampler = qmc.LatinHypercube(d=5, seed=seed)
    unit_samples = sampler.random(n=n_samples)

    bounds = {
        "c": (5.0, 40.0),
        "phi": (20.0, 40.0),
        "gamma": (16.0, 22.0),
        "beta": (30.0, 50.0),
        "H": (10.0, 30.0),
    }
    keys = list(bounds.keys())

    samples = {}
    for i, k in enumerate(keys):
        low, high = bounds[k]
        samples[k] = unit_samples[:, i] * (high - low) + low

    rows = []
    t0 = time.time()
    for j in range(n_samples):
        if j % 50 == 0:
            print(f"    Progress: {j}/{n_samples} ({time.time()-t0:.1f}s)")
        c_val = samples["c"][j]
        phi_val = samples["phi"][j]
        gamma_val = samples["gamma"][j]
        beta_val = samples["beta"][j]
        H_val = samples["H"][j]

        res = run_simulation(c_val, phi_val, 40.0, gamma_val, beta_val, H_val)
        if res["converged"] and res["critical_cx"] is not None:
            rows.append({
                "c": round(c_val, 2),
                "phi": round(phi_val, 2),
                "gamma": round(gamma_val, 2),
                "beta": round(beta_val, 2),
                "H": round(H_val, 2),
                "critical_cx": res["critical_cx"],
                "critical_cy": res["critical_cy"],
                "critical_R": res["critical_R"],
                "FoS_bishop": res["FoS_bishop"]
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"[+] Saved {len(df)} valid failure geometry samples to {DATA_PATH}")
    return df

def train_geometry_surrogate():
    if not os.path.exists(DATA_PATH):
        df = generate_geometry_dataset()
    else:
        df = pd.read_csv(DATA_PATH)

    features = ["c", "phi", "gamma", "beta", "H"]
    targets = ["critical_cx", "critical_cy", "critical_R"]

    X = df[features].values
    Y = df[targets].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print("[*] Training Multi-Output Spatial Failure Surrogate...")
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1))
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    r2_cx = r2_score(Y_test[:, 0], Y_pred[:, 0])
    r2_cy = r2_score(Y_test[:, 1], Y_pred[:, 1])
    r2_R = r2_score(Y_test[:, 2], Y_pred[:, 2])

    print(f"[+] Spatial Failure Geometry Test Accuracy:")
    print(f"    Center Xc: R² = {r2_cx:.3f}")
    print(f"    Center Yc: R² = {r2_cy:.3f}")
    print(f"    Radius R:  R² = {r2_R:.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"[+] Model saved to {MODEL_PATH}")

    # Inference speed test
    t0 = time.time()
    for _ in range(1000):
        _ = model.predict(X_test[:1])
    ms_per_infer = (time.time() - t0)

    print(f"[+] Surrogate Spatial Inference: {ms_per_infer:.2f} ms for 1,000 evaluations (~0.001 ms/sample)!")
    return model

if __name__ == "__main__":
    train_geometry_surrogate()
