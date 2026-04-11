"""
=============================================================================
MODULE 2: Dataset Generation — Latin Hypercube Sampling + LEM Simulations
=============================================================================
Generates 800–1500 LEM simulations by systematically varying:
  Input  : [c, phi, E, gamma, beta, H]
  Output : [FoS_bishop, FoS_janbu, u_max_mm]

Saves:
  - raw_dataset.csv     (all runs, including failed)
  - clean_dataset.csv   (filtered, ready for ML)
  - sampling_stats.txt  (summary report)
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc
import time
import os
import sys

# Add parent directory if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lem_solver import run_simulation  # noqa: E402
# If running directly in same folder:
# from lem_solver import run_simulation


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER BOUNDS  (from project proposal)
# ─────────────────────────────────────────────────────────────────────────────

PARAM_BOUNDS = {
    #  name         lower   upper    unit
    "c":     (5.0,   40.0),    # kPa
    "phi":   (20.0,  40.0),    # degrees
    "E":     (10.0,  100.0),   # MPa
    "gamma": (16.0,  22.0),    # kN/m³
    "beta":  (30.0,  50.0),    # degrees
    "H":     (10.0,  30.0),    # m
}

PARAM_NAMES = list(PARAM_BOUNDS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# LATIN HYPERCUBE SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def generate_lhs_samples(n_samples: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate Latin Hypercube samples for all 6 input parameters.

    LHS ensures that each parameter's range is evenly covered,
    avoiding clustering that random sampling can produce.

    Parameters
    ----------
    n_samples : int   – number of parameter sets to sample
    seed      : int   – random seed for reproducibility

    Returns
    -------
    DataFrame with columns = PARAM_NAMES, rows = samples
    """
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    unit_samples = sampler.random(n=n_samples)   # values in [0, 1]

    # Scale to actual bounds
    lower = np.array([PARAM_BOUNDS[p][0] for p in PARAM_NAMES])
    upper = np.array([PARAM_BOUNDS[p][1] for p in PARAM_NAMES])
    scaled = qmc.scale(unit_samples, lower, upper)

    df = pd.DataFrame(scaled, columns=PARAM_NAMES)

    # Round to sensible precision
    df["c"]     = df["c"].round(2)
    df["phi"]   = df["phi"].round(2)
    df["E"]     = df["E"].round(2)
    df["gamma"] = df["gamma"].round(3)
    df["beta"]  = df["beta"].round(2)
    df["H"]     = df["H"].round(2)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL SIMULATIONS
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset(n_samples: int = 1000,
                output_dir: str = ".",
                seed: int = 42,
                verbose: bool = True) -> pd.DataFrame:
    """
    Generate dataset: sample parameters → run LEM → collect results.

    Parameters
    ----------
    n_samples  : int  – number of simulations to run
    output_dir : str  – directory to save output CSV files
    seed       : int  – random seed
    verbose    : bool – print progress

    Returns
    -------
    clean_df : DataFrame of valid simulations (inputs + outputs)
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Sample parameters ──────────────────────────────────────────────────
    if verbose:
        print(f"Generating {n_samples} LHS samples...")
    samples_df = generate_lhs_samples(n_samples, seed=seed)

    # ── Run simulations ────────────────────────────────────────────────────
    results = []
    t_start = time.time()
    n_failed = 0

    for i, row in samples_df.iterrows():
        if verbose and i % 100 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 0.01)
            eta = (n_samples - i) / max(rate, 0.01)
            print(f"  [{i+1:4d}/{n_samples}]  "
                  f"elapsed={elapsed:.1f}s  "
                  f"ETA={eta:.0f}s  "
                  f"failed={n_failed}")

        sim_result = run_simulation(
            c=row["c"], phi=row["phi"], E=row["E"],
            gamma=row["gamma"], beta=row["beta"], H=row["H"]
        )

        record = {
            "c":          row["c"],
            "phi":        row["phi"],
            "E":          row["E"],
            "gamma":      row["gamma"],
            "beta":       row["beta"],
            "H":          row["H"],
            "FoS_bishop": sim_result["FoS_bishop"],
            "FoS_janbu":  sim_result["FoS_janbu"],
            "u_max_mm":   sim_result["u_max_mm"],
            "converged":  sim_result["converged"],
        }
        results.append(record)

        if not sim_result["converged"]:
            n_failed += 1

    total_time = time.time() - t_start

    # ── Build DataFrames ───────────────────────────────────────────────────
    raw_df = pd.DataFrame(results)

    # Filter: remove non-converged and physically unrealistic results
    clean_df = raw_df[
        (raw_df["converged"] == True) &
        (raw_df["FoS_bishop"].notna()) &
        (raw_df["FoS_bishop"] > 0.2) &
        (raw_df["FoS_bishop"] < 8.0) &
        (raw_df["u_max_mm"].notna()) &
        (raw_df["u_max_mm"] > 0)
    ].copy()
    clean_df = clean_df.drop(columns=["converged"])

    # ── Save files ─────────────────────────────────────────────────────────
    raw_path   = os.path.join(output_dir, "raw_dataset.csv")
    clean_path = os.path.join(output_dir, "clean_dataset.csv")
    raw_df.to_csv(raw_path, index=False)
    clean_df.to_csv(clean_path, index=False)

    # ── Print summary ──────────────────────────────────────────────────────
    n_clean = len(clean_df)
    n_total = len(raw_df)

    summary_lines = [
        "=" * 60,
        "DATASET GENERATION SUMMARY",
        "=" * 60,
        f"Total simulations run   : {n_total}",
        f"Failed / non-converged  : {n_failed}",
        f"Filtered (clean) samples: {n_clean}  ({100*n_clean/n_total:.1f}%)",
        f"Total wall-clock time   : {total_time:.1f} s",
        f"Average per simulation  : {total_time/n_total*1000:.1f} ms",
        "",
        "Output Statistics (clean dataset):",
        f"  FoS_bishop  —  mean={clean_df['FoS_bishop'].mean():.3f}  "
        f"std={clean_df['FoS_bishop'].std():.3f}  "
        f"min={clean_df['FoS_bishop'].min():.3f}  "
        f"max={clean_df['FoS_bishop'].max():.3f}",
        f"  FoS_janbu   —  mean={clean_df['FoS_janbu'].mean():.3f}  "
        f"std={clean_df['FoS_janbu'].std():.3f}"
        if "FoS_janbu" in clean_df else "  FoS_janbu: not available",
        f"  u_max (mm)  —  mean={clean_df['u_max_mm'].mean():.2f}  "
        f"std={clean_df['u_max_mm'].std():.2f}  "
        f"min={clean_df['u_max_mm'].min():.2f}  "
        f"max={clean_df['u_max_mm'].max():.2f}",
        "",
        f"Files saved:",
        f"  Raw    : {raw_path}",
        f"  Clean  : {clean_path}",
        "=" * 60,
    ]
    summary_text = "\n".join(summary_lines)

    if verbose:
        print("\n" + summary_text)

    stats_path = os.path.join(output_dir, "sampling_stats.txt")
    with open(stats_path, "w") as f:
        f.write(summary_text)

    return clean_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Configuration ───────────────────────────────────────────────────────
    N_SAMPLES  = 1000       # increase to 1500 for final run
    OUTPUT_DIR = "data"     # folder to save CSVs
    SEED       = 42

    print("LEM Dataset Generator")
    print(f"Samples requested: {N_SAMPLES}")
    print(f"Output directory : {OUTPUT_DIR}")
    print()

    df = run_dataset(n_samples=N_SAMPLES, output_dir=OUTPUT_DIR, seed=SEED)

    print(f"\nDataset ready. Shape: {df.shape}")
    print(df.head(10).to_string(index=False))
