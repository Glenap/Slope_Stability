"""
=============================================================================
MODULE 1: Limit Equilibrium Method (LEM) Slope Stability Solver
=============================================================================
Implements:
  - Bishop's Simplified Method (circular failure surface)
  - Janbu's Simplified Method (circular failure surface)
  - Automatic critical slip circle search (grid + random search)
  - Maximum displacement estimation (empirical formula)
=============================================================================
"""

import numpy as np
from scipy.optimize import brentq
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_slope_geometry(H, beta_deg):
    """
    Build 2D slope geometry coordinates.

    Slope profile (x, y) from left to right:
      Toe  → Slope face → Crest

    Parameters
    ----------
    H        : float  – slope height (m)
    beta_deg : float  – slope angle from horizontal (degrees)

    Returns
    -------
    dict with 'toe', 'crest', 'slope_length', 'horizontal_extent'
    """
    beta_rad = np.radians(beta_deg)
    horizontal_run = H / np.tan(beta_rad)   # horizontal extent of slope face

    toe_x = 0.0
    toe_y = 0.0
    crest_x = horizontal_run
    crest_y = H

    return {
        "toe":              np.array([toe_x, toe_y]),
        "crest":            np.array([crest_x, crest_y]),
        "H":                H,
        "beta_rad":         beta_rad,
        "beta_deg":         beta_deg,
        "horizontal_run":   horizontal_run,
    }


def slope_surface_y(x, geo):
    """
    Return the y-coordinate of the slope surface at horizontal position x.
    Flat toe region → inclined face → flat crest.
    """
    toe_x   = geo["toe"][0]
    crest_x = geo["crest"][0]
    H       = geo["H"]

    if x <= toe_x:
        return 0.0
    elif x >= crest_x:
        return H
    else:
        frac = (x - toe_x) / (crest_x - toe_x)
        return frac * H


# ─────────────────────────────────────────────────────────────────────────────
# SLICE DISCRETISATION
# ─────────────────────────────────────────────────────────────────────────────

def discretize_circle(cx, cy, R, geo, n_slices=30):
    """
    Divide the failure mass above a circular arc into vertical slices.

    Parameters
    ----------
    cx, cy   : float – centre of slip circle
    R        : float – radius of slip circle
    geo      : dict  – slope geometry from build_slope_geometry()
    n_slices : int   – number of vertical slices

    Returns
    -------
    slices : list of dicts, each with:
        x_mid, width, alpha (base angle), h_soil (soil column height),
        base_y, surface_y
    Returns None if geometry is invalid.
    """
    # x-extent where circle intersects the slope base (y=0) and surface
    # Circle equation: (x-cx)^2 + (y-cy)^2 = R^2
    # Entry and exit x on y=0 plane:
    discriminant = R**2 - cy**2
    if discriminant <= 0:
        return None

    x_left  = cx - np.sqrt(discriminant)
    x_right = cx + np.sqrt(discriminant)

    # Clamp to slope domain
    x_left  = max(x_left,  geo["toe"][0]   - 0.0)
    x_right = min(x_right, geo["crest"][0] + geo["H"] / np.tan(geo["beta_rad"]) * 0.5)

    if x_right - x_left < 0.01 * geo["H"]:
        return None

    x_mids = np.linspace(x_left, x_right, n_slices + 1)
    slices = []

    for i in range(n_slices):
        xl = x_mids[i]
        xr = x_mids[i + 1]
        xm = 0.5 * (xl + xr)
        w  = xr - xl

        # Base of slice on circle
        disc_base = R**2 - (xm - cx)**2
        if disc_base < 0:
            continue
        base_y = cy - np.sqrt(disc_base)   # lower intersection (failure surface)

        # Top of slice = slope surface
        surf_y = slope_surface_y(xm, geo)

        h_soil = surf_y - base_y
        if h_soil <= 0:
            continue

        # Inclination of base (alpha = angle of base tangent to horizontal)
        sin_alpha = (xm - cx) / R
        sin_alpha = np.clip(sin_alpha, -0.98, 0.98)
        alpha = np.arcsin(sin_alpha)

        slices.append({
            "x_mid":    xm,
            "width":    w,
            "alpha":    alpha,         # radians
            "h_soil":   h_soil,
            "base_y":   base_y,
            "surf_y":   surf_y,
        })

    return slices if len(slices) >= 3 else None


# ─────────────────────────────────────────────────────────────────────────────
# BISHOP'S SIMPLIFIED METHOD
# ─────────────────────────────────────────────────────────────────────────────

def bishop_fos(slices, c, phi_deg, gamma):
    """
    Compute FoS by Bishop's Simplified Method (iterative).

    Formula:
      FoS = Σ [ (c·b + W·tan(φ)) / mα ] / Σ [ W·sin(α) ]

    where  mα = cos(α) + sin(α)·tan(φ)/FoS

    Parameters
    ----------
    slices   : list of slice dicts (from discretize_circle)
    c        : float – cohesion (kPa)
    phi_deg  : float – friction angle (degrees)
    gamma    : float – unit weight (kN/m³)

    Returns
    -------
    FoS : float (or None if non-convergent)
    """
    phi = np.radians(phi_deg)
    tan_phi = np.tan(phi)

    # Pre-compute per-slice quantities
    W_list     = [s["h_soil"] * s["width"] * gamma for s in slices]
    alpha_list = [s["alpha"]                        for s in slices]
    b_list     = [s["width"]                        for s in slices]

    sum_driving = sum(W * np.sin(a) for W, a in zip(W_list, alpha_list))
    if sum_driving <= 0:
        return None

    fos = 1.5   # initial guess
    for _ in range(200):
        numerator = 0.0
        for W, alpha, b in zip(W_list, alpha_list, b_list):
            m_alpha = np.cos(alpha) + np.sin(alpha) * tan_phi / fos
            if abs(m_alpha) < 1e-6:
                m_alpha = 1e-6
            numerator += (c * b + W * tan_phi) / m_alpha

        fos_new = numerator / sum_driving
        if abs(fos_new - fos) < 1e-5:
            fos = fos_new
            break
        fos = 0.5 * fos + 0.5 * fos_new   # damped update

    return round(float(fos), 4) if 0.2 < fos < 10.0 else None


# ─────────────────────────────────────────────────────────────────────────────
# JANBU'S SIMPLIFIED METHOD
# ─────────────────────────────────────────────────────────────────────────────

def janbu_fos(slices, c, phi_deg, gamma):
    """
    Compute FoS by Janbu's Simplified Method with correction factor f0.

    Formula:
      FoS_basic = Σ [ (c·b + W·tan(φ)) / n_α ] / Σ [ W·tan(α) ]
      FoS = f0 * FoS_basic

    where n_α = cos²(α) * (1 + tan(α)·tan(φ)/FoS)
    and f0 is Janbu's correction factor ≈ 1 + 0.031*(d/L) for c-φ soils.
    """
    phi = np.radians(phi_deg)
    tan_phi = np.tan(phi)

    W_list     = [s["h_soil"] * s["width"] * gamma for s in slices]
    alpha_list = [s["alpha"]                        for s in slices]
    b_list     = [s["width"]                        for s in slices]

    sum_driving = sum(W * np.tan(a) for W, a in zip(W_list, alpha_list))
    if abs(sum_driving) < 1e-6:
        return None

    fos = 1.5
    for _ in range(200):
        numerator = 0.0
        for W, alpha, b in zip(W_list, alpha_list, b_list):
            n_alpha = np.cos(alpha)**2 * (1 + np.tan(alpha) * tan_phi / fos)
            if abs(n_alpha) < 1e-6:
                n_alpha = 1e-6
            numerator += (c * b + W * tan_phi) / n_alpha

        fos_new = numerator / sum_driving
        if abs(fos_new - fos) < 1e-5:
            fos = fos_new
            break
        fos = 0.5 * fos + 0.5 * fos_new

    # Janbu correction factor f0 (based on d/L, average depth ratio)
    h_avg = np.mean([s["h_soil"] for s in slices])
    x_list = [s["x_mid"] for s in slices]
    L = max(x_list) - min(x_list) if len(x_list) > 1 else 1.0
    d_over_L = h_avg / L if L > 0 else 0.1
    f0 = 1.0 + 0.031 * d_over_L   # simplified correction for c-φ soils

    fos_corrected = float(fos * f0)
    return round(fos_corrected, 4) if 0.2 < fos_corrected < 10.0 else None


# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL SLIP CIRCLE SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def find_critical_fos(c, phi_deg, gamma, H, beta_deg,
                      n_grid=15, n_random=200, n_slices=30,
                      method="bishop"):
    """
    Search for the critical (minimum) FoS by scanning slip circle centres.

    Strategy
    --------
    1. Coarse grid search over (cx, cy) space
    2. Fine random search around best grid point
    3. Return minimum FoS found

    Parameters
    ----------
    method : "bishop" | "janbu" | "both"

    Returns
    -------
    dict with 'FoS_bishop', 'FoS_janbu', 'critical_cx', 'critical_cy',
              'critical_R', 'n_valid_circles'
    """
    geo = build_slope_geometry(H, beta_deg)

    # Search bounds for circle centre
    # cx: from toe to 1.5*horizontal_run
    # cy: from H to 3*H  (above crest)
    cx_min = geo["toe"][0]   - 0.5 * H
    cx_max = geo["crest"][0] + 0.5 * H
    cy_min = H * 0.5
    cy_max = H * 3.0

    fos_solver = bishop_fos if method == "bishop" else janbu_fos

    best_fos   = np.inf
    best_cx    = cx_min
    best_cy    = cy_min
    best_R     = H * 1.5
    n_valid    = 0

    fos_bishop_best = None
    fos_janbu_best  = None

    # ── 1. Grid search ──────────────────────────────────────────────────────
    cx_vals = np.linspace(cx_min, cx_max, n_grid)
    cy_vals = np.linspace(cy_min, cy_max, n_grid)

    for cx in cx_vals:
        for cy in cy_vals:
            # Radius must reach the toe and the crest approximately
            R_min = np.sqrt((cx - geo["toe"][0])**2   + cy**2) * 0.95
            R_max = np.sqrt((cx - geo["toe"][0])**2   + cy**2) * 1.5
            for R in np.linspace(R_min, R_max, 4):
                slices = discretize_circle(cx, cy, R, geo, n_slices)
                if slices is None:
                    continue
                n_valid += 1
                fos_b = bishop_fos(slices, c, phi_deg, gamma)
                if fos_b and fos_b < best_fos:
                    best_fos = fos_b
                    best_cx, best_cy, best_R = cx, cy, R
                    fos_bishop_best = fos_b

    # ── 2. Random search around best grid point ──────────────────────────────
    rng = np.random.default_rng(42)
    spread = H * 0.4
    for _ in range(n_random):
        cx = best_cx + rng.uniform(-spread, spread)
        cy = best_cy + rng.uniform(-spread * 0.5, spread * 0.5)
        cy = max(cy, H * 0.3)
        R_try = np.sqrt((cx - geo["toe"][0])**2 + cy**2) * rng.uniform(0.9, 1.2)
        R_try = max(R_try, H * 0.8)

        slices = discretize_circle(cx, cy, R_try, geo, n_slices)
        if slices is None:
            continue
        n_valid += 1
        fos_b = bishop_fos(slices, c, phi_deg, gamma)
        if fos_b and fos_b < best_fos:
            best_fos = fos_b
            best_cx, best_cy, best_R = cx, cy, R_try
            fos_bishop_best = fos_b

    # ── 3. Compute Janbu FoS at critical circle ───────────────────────────
    slices_best = discretize_circle(best_cx, best_cy, best_R, geo, n_slices)
    if slices_best:
        fos_janbu_best = janbu_fos(slices_best, c, phi_deg, gamma)

    return {
        "FoS_bishop":     fos_bishop_best,
        "FoS_janbu":      fos_janbu_best,
        "critical_cx":    best_cx,
        "critical_cy":    best_cy,
        "critical_R":     best_R,
        "n_valid_circles": n_valid,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DISPLACEMENT ESTIMATION (Empirical)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_max_displacement(FoS, H, E_MPa, gamma, c, phi_deg):
    """
    Empirical estimate of maximum displacement (mm) based on:
      - FoS (proximity to failure)
      - Slope height H
      - Soil stiffness E

    Formula derived from Griffiths & Lane (1999) concept:
      u_max ≈ k * gamma * H² / E  *  f(FoS)

    where f(FoS) = 1/(FoS - 0.9)^0.8  for FoS close to 1
                = small constant       for stable slopes

    This is an approximation; true displacement requires FEM.
    """
    E_kPa = E_MPa * 1000.0   # convert MPa → kPa

    # Base elastic settlement scaling
    u_elastic = (gamma * H**2) / E_kPa   # dimensionless → scale to mm
    u_elastic_mm = u_elastic * 1000.0    # convert m → mm

    # Proximity-to-failure amplification
    if FoS <= 1.0:
        amp = 10.0
    elif FoS < 1.1:
        amp = 5.0 + (1.1 - FoS) / 0.1 * 5.0
    elif FoS < 2.0:
        amp = 1.0 + 4.0 / (FoS - 0.9)**1.2
    else:
        amp = 1.0

    u_max_mm = u_elastic_mm * amp
    return round(float(u_max_mm), 4)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE SIMULATION ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(c, phi, E, gamma, beta, H):
    """
    Run a complete LEM simulation for one parameter set.

    Parameters
    ----------
    c     : cohesion (kPa)
    phi   : friction angle (degrees)
    E     : Young's modulus (MPa)  — used only for displacement estimate
    gamma : unit weight (kN/m³)
    beta  : slope angle (degrees)
    H     : slope height (m)

    Returns
    -------
    dict: {FoS_bishop, FoS_janbu, u_max_mm, converged}
    """
    result = find_critical_fos(c, phi, gamma, H, beta,
                               n_grid=12, n_random=150, n_slices=30)

    fos_b = result["FoS_bishop"]
    fos_j = result["FoS_janbu"]

    if fos_b is None:
        return {"FoS_bishop": None, "FoS_janbu": None,
                "u_max_mm": None, "converged": False}

    u_max = estimate_max_displacement(fos_b, H, E, gamma, c, phi)

    return {
        "FoS_bishop": fos_b,
        "FoS_janbu":  fos_j,
        "u_max_mm":   u_max,
        "converged":  True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("LEM Solver — Quick Validation Test")
    print("=" * 60)

    # Test case: c=20 kPa, φ=30°, E=50 MPa, γ=18 kN/m³, β=45°, H=10 m
    # Expected FoS (Bishop) ≈ 1.3–1.6 for these parameters
    test_cases = [
        dict(c=20, phi=30, E=50, gamma=18, beta=45, H=10),
        dict(c=5,  phi=20, E=20, gamma=20, beta=50, H=20),   # unstable
        dict(c=40, phi=40, E=80, gamma=16, beta=30, H=15),   # stable
    ]

    for i, tc in enumerate(test_cases):
        res = run_simulation(**tc)
        print(f"\nTest {i+1}: c={tc['c']} kPa, φ={tc['phi']}°, "
              f"β={tc['beta']}°, H={tc['H']} m")
        print(f"  FoS (Bishop) = {res['FoS_bishop']}")
        print(f"  FoS (Janbu)  = {res['FoS_janbu']}")
        print(f"  u_max        = {res['u_max_mm']} mm")
        print(f"  Converged    = {res['converged']}")
