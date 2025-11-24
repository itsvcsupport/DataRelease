#!/usr/bin/env python
"""
pantheon_desi_chain_test.py

Steps:
1. Load Pantheon+SH0ES distances and fit flat LCDM with H0 fixed.
2. Load a DESI DR1 BAO cobaya chain and compute the mean and covariance
   of (H0, Omega_m).
3. Compute Delta chi^2 for:
   - Pantheon+ LCDM point (H0_fixed, Omega_m_best)
   - Pantheon+ CCBH best fit (H0_CCBH, Omega_m_CCBH)
   relative to the DESI posterior.

Edit the CONFIG section to match your file paths and your current CCBH best-fit.
"""

import os
import numpy as np
import pandas as pd
import io
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

# DESI cobaya chain file
# Example (ABSOLUTE path):
#   r"C:\Users\WilliamKellett\cobaya\base\desi-bao-all\chain_1.txt"
DESI_CHAIN_FILE = r"C:\Users\WilliamKellett\cobaya\base\desi-bao-all\chain.1.txt"  # <-- EDIT THIS

# Candidate column names in the DESI chain for H0 and Omega_m
DESI_H0_CANDIDATES = ["H0", "h"]  # still fine, we'll add a fallback
DESI_OM_CANDIDATES = ["omegam", "omm", "Omega_m", "Om_m", "om"]


# Insert your Pantheon+ CCBH best fit here
CCBH_H0 = 73.2    # km/s/Mpc (update if needed)
CCBH_OM = 0.386   # dimensionless (update if needed)

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


def load_desi_cobaya_chain(path, h_candidates, om_candidates):
    """
    Load DESI cobaya chain and compute mean and covariance
    of (H0, Omega_m).

    Assumes a standard Cobaya chain format, e.g.:

        # some comments
        # weight minuslogpost H0 omegam ...
        1  14.9  69.3  0.30 ...

    In your DESI chain we don't have H0/h, but we *do* have:
        rdrag, H0rdrag

    so we can reconstruct H0 via:
        H0 = H0rdrag / rdrag
    """
    # --- read raw lines ---
    with open(path, "r") as f:
        lines = f.readlines()

    header_idx = None
    header_names = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue

        tokens = stripped.lstrip("#").strip().split()
        if "weight" in tokens and "minuslogpost" in tokens:
            header_idx = i
            header_names = tokens
            break

    if header_idx is None or header_names is None:
        raise RuntimeError(
            "Could not find a Cobaya-style header line in DESI chain. "
            "Expected a line like '# weight minuslogpost H0 omegam ...'."
        )

    # --- build a buffer for the numeric data (lines after the header) ---
    data_str = "".join(lines[header_idx + 1 :])
    buf = io.StringIO(data_str)

    # --- read the numeric table using the discovered column names ---
    df = pd.read_csv(buf, sep=r"\s+", header=None, names=header_names)

    cols = list(df.columns)
    cols_stripped = [c.strip() for c in cols]
    col_map = {c.strip(): c for c in cols}  # stripped -> original

    def find_column(candidates, what=""):
        for cand in candidates:
            if cand in col_map:
                return col_map[cand]
        raise RuntimeError(
            f"Could not find any of {candidates} as {what} column. "
            f"Available columns: {cols_stripped}"
        )

    # ---------- H0 handling ----------
    H0_col = None
    h_col = None

    # 1) look for an explicit H0 column
    for cand in h_candidates:
        if cand in col_map and cand.lower().startswith("h0"):
            H0_col = col_map[cand]
            break

    # 2) if not, look for 'h'
    if H0_col is None:
        for cand in h_candidates:
            if cand in col_map and cand.lower() == "h":
                h_col = col_map[cand]
                break

    # 3) if neither present, try to build H0 from H0rdrag / rdrag
    H0_vals = None
    if H0_col is not None:
        H0_vals = df[H0_col].values
    elif h_col is not None:
        h_vals = df[h_col].values
        H0_vals = 100.0 * h_vals
    else:
        # Fallback: require H0rdrag and rdrag
        if "H0rdrag" in col_map and "rdrag" in col_map:
            H0r_col = col_map["H0rdrag"]
            rdrag_col = col_map["rdrag"]
            H0_vals = df[H0r_col].values / df[rdrag_col].values
        else:
            raise RuntimeError(
                "Could not find H0, h, or (H0rdrag & rdrag) in DESI chain. "
                f"Columns: {cols_stripped}"
            )

    # ---------- Omega_m handling ----------
    Om_col = find_column(om_candidates, "Omega_m")
    Om_vals = df[Om_col].values

    print(
        'Using DESI chain: H0 from "{}", Omega_m -> "{}"'.format(
            "H0/h" if (H0_col or h_col) else "H0rdrag/rdrag",
            Om_col,
        )
    )

    # ---------- Stats ----------
    X = np.vstack([H0_vals, Om_vals]).T
    mean = X.mean(axis=0)
    cov = np.cov(X.T)

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
    print(">>> Starting Pantheon+ ↔ DESI (cobaya) consistency test")
    print(f"  Using Pantheon file : {PANTHEON_FILE}")
    print(f"  Using DESI chain    : {DESI_CHAIN_FILE}")
    print()

    # basic file existence checks
    if not os.path.exists(PANTHEON_FILE):
        raise FileNotFoundError(f"Pantheon file not found: {PANTHEON_FILE}")
    if not os.path.exists(DESI_CHAIN_FILE):
        raise FileNotFoundError(f"DESI chain file not found: {DESI_CHAIN_FILE}")

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

    # 2. DESI posterior from cobaya chain
    print("=== Step 2: DESI DR1 (H0, Omega_m) from cobaya chain ===")
    mean, cov = load_desi_cobaya_chain(
        DESI_CHAIN_FILE, DESI_H0_CANDIDATES, DESI_OM_CANDIDATES
    )
    print(f"DESI mean (H0, Omega_m) = ({mean[0]:.3f}, {mean[1]:.4f})")
    print("DESI covariance matrix:")
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

    print("Interpretation guide for 2 parameters:")
    print("  ~68% region: Delta chi^2 ≲ 2.3")
    print("  ~95% region: Delta chi^2 ≲ 6.0")
    print("If both points are below ~6, they are statistically consistent with DESI.")
    print(">>> Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print("\n*** ERROR while running pantheon_desi_chain_test.py ***")
        traceback.print_exc()
        print("\n(Check DESI_CHAIN_FILE path and DESI_*_CANDIDATES names in the CONFIG section.)")
