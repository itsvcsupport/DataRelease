#!/usr/bin/env python
"""
pantheon_desi_consistency_test.py

Steps:
1. Load Pantheon+SH0ES distances and fit flat LCDM with H0 fixed.
2. Load a DESI DR1 BAO iminuit best-fit file and extract (H0, Omega_m)
   with their 1-sigma uncertainties.
3. Build a simple Gaussian prior from DESI and compute Delta chi^2 for:
   - Pantheon+ LCDM point (H0_fixed, Omega_m_best)
   - Pantheon+ CCBH best fit (H0_CCBH, Omega_m_CCBH)

Edit the CONFIG section to match your file paths and DESI parameter names.
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

# DESI iminuit best-fit file
# Adjust this to the actual path you see in your workspace, e.g.:
#   bao-cosmo-params/iminuit/base/desi-bao-all/bestfit.minimum.txt
DESI_IMINUIT_FILE = (
    r"C:\Users\WilliamKellett\iminuit\base\desi-bao-all\bestfit.minimum.txt"
)

# Names of the H0 and Omega_m parameters in that file
# (the loader is a bit flexible and will also look for 'h', 'omegam', 'Omega_m', etc.)
DESI_PARAM_H0 = "H0"
DESI_PARAM_OM = "omegam"

# Insert your Pantheon+ CCBH best fit here
CCBH_H0 = 73.2    # km/s/Mpc
CCBH_OM = 0.386   # dimensionless

# Redshift range for the SN subset
ZMIN = 0.01
ZMAX = None  # or e.g. 0.6

# ============================================================
# FUNCTIONS
# ============================================================

def load_pantheon_subset(filename, zmin=0.01, zmax=None):
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
    # Try several known Pantheon+ / SH0ES variants in order of preference.
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


def load_desi_iminuit_bestfit(path, name_H0="H0", name_Om="omegam"):
    """
    Load DESI iminuit bestfit.minimum.txt and extract mean and (diagonal) covariance
    for (H0, Omega_m).

    Assumes the file is a whitespace-delimited table where:
      - first column is parameter name,
      - second column is best-fit value,
      - third column is 1-sigma error.

    If 'h' is found instead of 'H0', converts to H0 = 100*h.
    """
    df = pd.read_csv(path, delim_whitespace=True, comment="#")

    if df.shape[1] < 2:
        raise RuntimeError(
            f"DESI iminuit file {path} has too few columns to parse. "
            f"Columns: {list(df.columns)}"
        )

    param_col = df.columns[0]
    val_col = df.columns[1]
    err_col = df.columns[2] if df.shape[1] >= 3 else None

    if err_col is None:
        raise RuntimeError(
            f"DESI iminuit file {path} does not appear to contain 1-sigma errors. "
            f"Columns: {list(df.columns)}"
        )

    def find_param_row(name_options):
        for name in name_options:
            mask = df[param_col].astype(str).str.strip().str.lower() == name.lower()
            if mask.any():
                return df[mask].iloc[0]
        names = ", ".join(name_options)
        raise RuntimeError(
            f"Could not find any of [{names}] in column '{param_col}'. "
            f"Available parameter names: {df[param_col].tolist()}"
        )

    # H0 or h
    row_H0 = find_param_row([name_H0, "H0", "h"])
    val_H0 = row_H0[val_col]
    err_H0 = row_H0[err_col]

    # If this is actually 'h', convert to H0
    is_h = str(row_H0[param_col]).strip().lower() == "h"
    if is_h:
        val_H0 = 100.0 * val_H0
        err_H0 = 100.0 * err_H0

    # Omega_m variations
    row_Om = find_param_row([name_Om, "omegam", "Omega_m", "Om_m"])
    val_Om = row_Om[val_col]
    err_Om = row_Om[err_col]

    mean = np.array([float(val_H0), float(val_Om)])
    cov = np.diag([float(err_H0)**2, float(err_Om)**2])

    return mean, cov


def delta_chi2(theta, mean, cov):
    theta = np.asarray(theta)
    mean = np.asarray(mean)
    d = theta - mean
    inv_cov = np.linalg.inv(cov)
    return float(d.T @ inv_cov @ d)


# ============================================================
# MAIN
# ============================================================

def main():
    print(">>> Starting Pantheon+ ↔ DESI consistency test")
    print(f"  Using Pantheon file : {PANTHEON_FILE}")
    print(f"  Using DESI iminuit  : {DESI_IMINUIT_FILE}")
    print()

    # basic file existence checks
    if not os.path.exists(PANTHEON_FILE):
        raise FileNotFoundError(f"Pantheon file not found: {PANTHEON_FILE}")
    if not os.path.exists(DESI_IMINUIT_FILE):
        raise FileNotFoundError(f"DESI iminuit file not found: {DESI_IMINUIT_FILE}")

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

    # 2. DESI posterior (Gaussian approximation from iminuit)
    print("=== Step 2: DESI DR1 (H0, Omega_m) from iminuit bestfit ===")
    mean, cov = load_desi_iminuit_bestfit(
        DESI_IMINUIT_FILE, name_H0=DESI_PARAM_H0, name_Om=DESI_PARAM_OM
    )
    print(f"DESI mean (H0, Omega_m) = ({mean[0]:.3f}, {mean[1]:.4f})")
    print("DESI diagonal covariance matrix (from 1-sigma errors):")
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
        print("\n*** ERROR while running pantheon_desi_consistency_test.py ***")
        traceback.print_exc()
        print("\n(Check DESI_IMINUIT_FILE path and DESI_PARAM_* names in the CONFIG section.)")
