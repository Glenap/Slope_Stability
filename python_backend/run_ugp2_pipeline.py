"""
=============================================================================
MASTER PIPELINE: Semester 2 UGP Novelty Engine (SlopeML-Plus)
=============================================================================
Orchestrates all novel geotechnical & AI modules:
  1. Module 6: Physical Consistency Audit (detects non-physical artifacts)
  2. Module 7: Physics-Informed Monotonic Machine Learning (PIML-XGBoost)
  3. Module 8: Probabilistic Reliability & Monte Carlo Simulation Engine
  4. Module 9: Landslide Forensic Inverse Back-Analysis Engine
=============================================================================
"""

import sys
import os
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from importlib import import_module

def main():
    print("=" * 75)
    print("SLOPEML-PLUS: ADVANCED GEOTECHNICAL & AI RESEARCH SUITE")
    print("Undergraduate Project (UGP-II) Novelty Engine Execution")
    print("=" * 75)

    start_time = time.time()

    # Step 1: Physics Audit
    print("\n>>> STEP 1: Running Physical Consistency Audit on Baseline Models...")
    audit_mod = import_module("06_physics_audit")
    audit_mod.run_physical_audit()

    # Step 2: PIML Training
    print("\n>>> STEP 2: Training Physics-Informed Monotonic ML (PIML) Surrogates...")
    piml_mod = import_module("07_physics_constrained_ml")
    piml_mod.train_and_evaluate_piml()

    # Step 3: Probabilistic Monte Carlo Reliability
    print("\n>>> STEP 3: Executing Ultra-Fast Monte Carlo Probabilistic Reliability...")
    reliability_mod = import_module("08_probabilistic_reliability")
    reliability_mod.run_reliability_suite()

    # Step 4: Landslide Forensic Back-Analysis
    print("\n>>> STEP 4: Executing Real-Time Forensic Inverse Back-Analysis...")
    forensic_mod = import_module("09_inverse_back_analysis")
    forensic_mod.run_forensic_case_studies()

    # Step 5: Dual-Surrogate Spatial Failure Geometry
    print("\n>>> STEP 5: Verifying Dual-Surrogate Spatial Failure Geometry (Xc, Yc, R)...")
    geom_mod = import_module("10_geometry_surrogate")
    geom_mod.train_geometry_surrogate()

    total_time = time.time() - start_time
    print("\n" + "=" * 75)
    print(f"SUCCESS: Complete UGP-II Novelty Suite executed in {total_time:.2f} seconds!")
    print("Deliverables generated:")
    print("  [1] models/physics_audit_metrics.csv & plots/physics_violation_audit.png")
    print("  [2] models/piml_comparison_metrics.csv & plots/piml_monotonicity_proof.png")
    print("  [3] models/reliability_summary.csv & plots/reliability_fragility_curves.png")
    print("  [4] models/forensic_back_analysis_benchmarks.csv & plots/forensic_back_analysis.png")
    print("  [5] models/Dual_Surrogate_Geometry.pkl (Predicts Failure Arc in < 0.001ms)")
    print("=" * 75)

if __name__ == "__main__":
    main()
