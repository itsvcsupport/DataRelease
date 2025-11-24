#!/usr/bin/env python
"""
pantheon_desi_consistency_published.py

Quick consistency test between:
  - Pantheon+ (SN) flat LCDM fit with H0 fixed, and
  - DESI DR1 BAO+BBN+acoustic-scale constraints on (H0, Omega_m).

Steps:
1. Load Pantheon+SH0ES distances and fit flat LCDM with H0 fixed.
2. Use published DESI posterior for (H0, Omega_m) as a 2D Gaussian.
3. Compute Delta chi^2 for:
   - Pantheon+ LCDM point (H0_fixed, Omega_m_best)
   - Pantheon+ CCBH best fit (H0_CCBH, Omega_m_CCBH)
   relative to the DESI posterior.

Edit the CONFIG section to match your file paths and CCBH best-fit.
"""

import os
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

try:
    from scipy.optimize import minimize_scalar
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# ============================================================
# CONFIG – EDIT THESE FOR YOUR SETUP
# ============================================================

# Pantheon+ distances file (relative to DataRelease)
PANTHEON_FILE = "Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat"

# Fixed H0 value for the LCDM fit to Pantheon+
H0_LCDM = 73.04  # km/s/Mpc

# Published DESI DR1 posterior for flat LCDM (BAO + BBN + acoustic scale)
# Adame et al. (DESI 2025):
#   H0 = 68.52 ± 0.62 km/s/Mpc
#   Omega_m = 0.295 ± 0.015
H0_DESI_MEAN = 68.52
SIGMA_H0_DESI = 0.62
OM_DESI_MEAN = 0.295
SIGMA_OM_DESI = 0.015

# Your Pantheon+ CCBH best fit (update these as needed)
CCBH_H0 = 73.2    # km/s/Mpc
CCBH_OM = 0.386   # dimensionless

# Redshift range for the SN subset
ZMIN = 0.01
ZMAX = None  # or e.g. 0.6

# ============================================================
# FUNCTIONS
# ============================================================

def load_pantheon_subset(filename, zmin=0.01, zmax=None):
    """
    Load Pantheon+ distance data and return a cleaned subset.

    We automatically choose a reasonable (mu, mu_err) pair:
      MU / MUERR
      MU_SH0ES / MU_SH0ES_ERR_DIAG
      m_b_corr / m_b_corr_err_DIAG
    """
    df = pd.read_csv(filename, sep=r"\s+", comment="#")

    # --- pick a redshift column ---
    z_col_candidates = ["zHD", "zCMB", "z"]
    z_col = None
    for c in z_col_candidates:
        if c in df.columns:
            z_col = c
            break
    if z_col is None:
        raise RuntimeError(
            f"Could not find a redshift column among {z_col_candidates}. "
            f"Available columns are: {list(df.columns)}"
        )

    # --- pick distance modulus + error columns ---
    mu_col_pairs = [
        ("MU", "MUERR"),
        ("MU_SH0ES", "MU_SH0ES_ERR_DIAG"),
        ("m_b_corr", "m_b_corr_err_DIAG"),
    ]

    mu_col = err_col = None
    for mc, ec in mu_col_pairs:
        if mc in df.columns and ec in df.columns:
            mu_col, err_col = mc, ec
            break

    if mu_col is None:
        raise RuntimeError(
            "Could not find any usable (mu, mu_err) column pair. "
            "Tried: " + ", ".join([f"{m}/{e}" for m, e in mu_col_pairs]) +
            f". Available columns: {list(df.columns)}"
        )

    print(f"Using columns for mu: {mu_col}, mu_err: {err_col}")

    z = df[z_col].values
    mu = df[mu_col].values
    mu_err = df[err_col].values

    mask = np.isfinite(z) & np.isfinite(mu) & np.isfinite(mu_err)
    if zmin is not None:
        mask &= (z >= zmin)
    if zmax is not None:
        mask &= (z <= zmax)

    return z[mask], mu[mask], mu_err[mask]


def chi2_lcdm(Omega_m, z, mu_obs, mu_err, H0):
    """
    Compute chi^2 for flat LCDM with fixed H0 and free Omega_m, M.

    We analytically minimize over the absolute magnitude offset M:

        mu_model(z) = 5 log10(D_L(z; H0, Omega_m) / 10 pc) + M

    For a given Omega_m, the best-fit M is:

        M_best = sum_i [w_i (mu_obs_i - mu_model0_i)] / sum_i w_i

    where mu_model0 is the model with M = 0 and w_i = 1 / sigma_i^2.
    """
    cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m, Tcmb0=2.725)
    dl = cosmo.luminosity_distance(z)  # in Mpc
    mu_model0 = 5 * np.log10(dl.to(u.pc).value / 10.0)  # M = 0

    w = 1.0 / mu_err**2
    numerator = np.sum(w * (mu_obs - mu_model0))
    denominator = np.sum(w)
    M_best = numerator / denominator

    resid = mu_obs - (mu_model0 + M_best)
    chi2 = np.sum((resid / mu_err)**2)

    return chi2, M_best


def fit_Omega_m(z, mu_obs, mu_err, H0, Om_min=0.05, Om_max=0.6):
    """
    Fit Omega_m for flat LCDM with fixed H0, marginalizing over M.
    """
    if HAS_SCIPY:
        def f(Om):
            chi2, _ = chi2_lcdm(Om, z, mu_obs, mu_err, H0)
            return chi2

        res = minimize_scalar(f, bounds=(Om_min, Om_max), method="bounded")
        Om_best = res.x
        chi2_best, M_best = chi2_lcdm(Om_best, z, mu_obs, mu_err, H0)
    else:
        Om_grid = np.linspace(Om_min, Om_max, 500)
        chi2_grid = []
        M_grid = []
        for Om in Om_grid:
            chi2_val, M_val = chi2_lcdm(Om, z, mu_obs, mu_err, H0)
            chi2_grid.append(chi2_val)
            M_grid.append(M_val)
        chi2_grid = np.array(chi2_grid)
        idx = np.argmin(chi2_grid)
        Om_best = float(Om_grid[idx])
        chi2_best = float(chi2_grid[idx])
        M_best = float(M_grid[idx])

    return Om_best, chi2_best, M_best


def delta_chi2(theta, mean, cov):
    """
    Compute Delta chi^2 = (theta - mean)^T Cov^{-1} (theta - mean)
    for a 2D parameter vector theta = [H0, Omega_m].
    """
    theta = np.asarray(theta)
    mean = np.asarray(mean)
    d = theta - mean
    inv_cov = np.linalg.inv(cov)
    return float(d.T @ inv_cov @ d)


# ============================================================
# MAIN
# ============================================================

def main():
    print(">>> Starting Pantheon+ ↔ DESI (published) consistency test")
    print(f"  Using Pantheon file : {PANTHEON_FILE}")
    print()

    if not os.path.exists(PANTHEON_FILE):
        raise FileNotFoundError(f"Pantheon file not found: {PANTHEON_FILE}")

    # 1. Pantheon+ LCDM fit
    print("=== Step 1: Pantheon+ flat LCDM fit (H0 fixed) ===")
    z, mu, mu_err = load_pantheon_subset(PANTHEON_FILE, zmin=ZMIN, zmax=ZMAX)
    print(f"Loaded {len(z)} SNe")

    Om_best, chi2_best, M_best = fit_Omega_m(z, mu, mu_err, H0_LCDM)
    dof = len(z) - 2  # parameters: Omega_m and M

    print(f"Fixed H0           = {H0_LCDM:.3f} km/s/Mpc")
    print(f"Best-fit Omega_m   = {Om_best:.4f}")
    print(f"Best-fit M         = {M_best:.4f}")
    print(f"chi^2              = {chi2_best:.2f} for N = {len(z)}")
    print(f"chi^2 / dof        = {chi2_best / dof:.3f}")
    print()

    # 2. DESI posterior (Gaussian approximation from published constraints)
    print("=== Step 2: DESI DR1 (H0, Omega_m) from published BAO+BBN+θ* result ===")

    mean = np.array([H0_DESI_MEAN, OM_DESI_MEAN])
    cov = np.diag([SIGMA_H0_DESI**2, SIGMA_OM_DESI**2])

    print(f"DESI mean (H0, Omega_m) = ({mean[0]:.3f}, {mean[1]:.4f})")
    print("DESI diagonal covariance matrix (from published 1-sigma errors):")
    print(cov)
    print()

    # 3. Delta chi^2 for LCDM and CCBH best-fit points
    print("=== Step 3: Delta chi^2 relative to DESI ===")

    theta_lcdm = np.array([H0_LCDM, Om_best])
    theta_ccbh = np.array([CCBH_H0, CCBH_OM])

    dchi2_lcdm = delta_chi2(theta_lcdm, mean, cov)
    dchi2_ccbh = delta_chi2(theta_ccbh, mean, cov)

    print("Pantheon+ LCDM point:")
    print(f"  (H0, Omega_m) = ({theta_lcdm[0]:.3f}, {theta_lcdm[1]:.4f})")
    print(f"  Delta chi^2 wrt DESI = {dchi2_lcdm:.3f}")
    print()

    print("Pantheon+ CCBH point:")
    print(f"  (H0, Omega_m) = ({theta_ccbh[0]:.3f}, {theta_ccbh[1]:.4f})")
    print(f"  Delta chi^2 wrt DESI = {dchi2_ccbh:.3f}")
    print()

    print("Interpretation guide for 2 parameters (Gaussian approx):")
    print("  ~68% region: Delta chi^2 ≲ 2.3")
    print("  ~95% region: Delta chi^2 ≲ 6.0")
    print("If both points are below ~6, they are statistically consistent with DESI.")
    print(">>> Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print("\n*** ERROR while running pantheon_desi_consistency_published.py ***")
        traceback.print_exc()
        print("\n(Check PANTHEON_FILE path and CCBH_* values in the CONFIG section.)")
