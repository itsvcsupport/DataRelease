#!/usr/bin/env python
"""
desi_H0rs_CC_BH_test.py

Goal:
  Compare Pantheon+ best-fit ΛCDM and CCBH models against DESI DR1
  in the parameter space DESI actually constrains, namely (H0 * r_s, Omega_m),
  where r_s is the sound horizon at the drag epoch r_s(a*).

Steps:
  1. Load a DESI cobaya chain with columns including (H0rdrag, rdrag, omegam).
  2. Construct DESI mean and covariance in (H0*r_s, Omega_m) using the
     chain's (H0rdrag, omegam).
  3. For each model (LCDM and CCBH):
       - Run CLASS to compute r_s,drag in that cosmology,
       - Compute H0 * r_s,drag and Omega_m,
       - Evaluate Delta chi^2 relative to the DESI (H0*r_s, Omega_m) posterior.

NOTE:
  - This script assumes you have CLASS installed with the Python wrapper
    ("classy") available.
  - The CCBH implementation here is a placeholder: you must edit
    `build_class_params_ccbh` to use your modified CLASS with k(M) coupling.
"""

import os
import io
import numpy as np
import pandas as pd

try:
    from classy import Class
except ImportError:
    raise ImportError(
        "Could not import 'classy'. Make sure CLASS is installed and "
        "the Python wrapper is available in this environment."
    )

# ============================================================
# CONFIG – EDIT THESE FOR YOUR SETUP
# ============================================================

# DESI cobaya chain file (ABSOLUTE path is safest)
# Use "Copy Path" in VS Code on chain.1.txt and paste here:
DESI_CHAIN_FILE = r"C:\Users\WilliamKellett\cobaya\base\desi-bao-all\chain.1.txt"

# Column names we expect in the DESI chain header
DESI_COL_H0RDAG = "H0rdrag"
DESI_COL_OM     = "omegam"   # good
# If error persists, fallback:
# DESI_COL_OM = "omm"


# Pantheon+ best-fit LCDM point (from your SN-only fit)
LCDM_H0  = 73.040   # km/s/Mpc
LCDM_OM  = 0.3492   # dimensionless

# Pantheon+ best-fit CCBH point (from your SN-only CCBH run)
CCBH_H0  = 73.200   # km/s/Mpc
CCBH_OM  = 0.3860   # dimensionless
CCBH_KBH = 3.200    # example CCBH coupling exponent k_BH; edit as needed

# Fixed baryon density for CLASS runs (approx Planck)
OMEGA_B_H2 = 0.02237

# Some fixed nuisance cosmology params for CLASS
NS_DEFAULT       = 0.965
AS_DEFAULT       = 2.1e-9
TAU_REIO_DEFAULT = 0.054
N_EFF_DEFAULT    = 3.046

# ============================================================
# DESI CHAIN LOADER – (H0rdrag, Omega_m)
# ============================================================

def load_desi_H0rs_omegam(path, col_H0r=DESI_COL_H0RDAG, col_Om=DESI_COL_OM):
    """
    Load a DESI cobaya chain and construct the mean and covariance in the
    (H0*r_s, Omega_m) space, using the chain's (H0rdrag, omegam).

    We:
      1) Find the Cobaya header line (starting with '#', containing 'weight'),
      2) Parse column names,
      3) Read numeric data below the header into a DataFrame,
      4) Extract H0rdrag and Omega_m,
      5) Compute mean and covariance of (H0rdrag, Omega_m).

    Returns:
      mean: ndarray shape (2,)  [mean_H0r, mean_Om]
      cov : ndarray shape (2,2)
    """
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
            "Expected a line like '# weight minuslogpost H0rdrag omegam ...'."
        )

    data_str = "".join(lines[header_idx + 1 :])
    buf = io.StringIO(data_str)
    df = pd.read_csv(buf, sep=r"\s+", header=None, names=header_names)

    cols = list(df.columns)
    cols_stripped = [c.strip() for c in cols]
    col_map = {c.strip(): c for c in cols}

    def get_col(name, what=""):
        if name not in col_map:
            raise RuntimeError(
                f"DESI chain missing expected column '{name}' for {what}. "
                f"Available columns: {cols_stripped}"
            )
        return col_map[name]

    H0r_col = get_col(col_H0r, "H0rdrag")
    Om_col  = get_col(col_Om, "Omega_m")

    H0r_vals = df[H0r_col].values  # H0 * r_drag
    Om_vals  = df[Om_col].values   # Omega_m

    X = np.vstack([H0r_vals, Om_vals]).T
    mean = X.mean(axis=0)
    cov  = np.cov(X.T)

    print(f"DESI chain: using columns H0rdrag -> '{H0r_col}', Omega_m -> '{Om_col}'")
    return mean, cov

# ============================================================
# CLASS WRAPPERS – COMPUTE H0 * r_s,drag AND OMEGA_M
# ============================================================

def build_class_params_lcdm(H0, Omega_m):
    """
    Build a CLASS parameter dictionary for a flat LCDM model with given
    H0 and Omega_m, using a fixed Omega_b h^2 and solving for Omega_cdm h^2.

    This is an approximation: we fix omega_b to a Planck-like value and
    adjust omega_cdm so that Omega_m = (omega_b + omega_cdm) / h^2.
    """
    h = H0 / 100.0
    omega_b = OMEGA_B_H2
    omega_m_target = Omega_m * h**2
    omega_cdm = omega_m_target - omega_b
    if omega_cdm <= 0:
        raise ValueError(
            f"Computed omega_cdm <= 0 for H0={H0}, Omega_m={Omega_m}. "
            f"omega_m_target={omega_m_target}, omega_b={omega_b}"
        )

    params = {
        "output": "mPk",
        "h": h,
        "omega_b": omega_b,
        "omega_cdm": omega_cdm,
        "Omega_k": 0.0,
        "N_ur": N_EFF_DEFAULT,
        "YHe": "BBN",
        "A_s": AS_DEFAULT,
        "n_s": NS_DEFAULT,
        "tau_reio": TAU_REIO_DEFAULT,
    }
    return params


def build_class_params_ccbh(H0, Omega_m, k_BH):
    """
    Build a CLASS parameter dictionary for your CCBH model.

    IMPORTANT:
      This is a *placeholder* that currently just uses the same LCDM
      background as build_class_params_lcdm, and adds a dummy 'k_BH'
      parameter. You MUST edit this to match your actual modified CLASS
      implementation of mass-dependent cosmological coupling.

    For example, if you have extended CLASS to include a new fluid with
    parameters (k_BH, Omega_BH, w_BH(a)), you would set those here.

    For now, we treat it as LCDM + a parameter k_BH that your custom CLASS
    build knows how to interpret.
    """
    params = build_class_params_lcdm(H0, Omega_m)

    # Example: adding a custom coupling parameter for your modified CLASS
    # (You must ensure your CLASS build actually uses this.)
    params["k_BH"] = k_BH

    # If you have a non-trivial DE sector, you might add, e.g.:
    # params["Omega_fld"] = ...
    # params["w0_fld"]    = ...
    # params["wa_fld"]    = ...
    # etc.

    return params


def compute_H0rs_Omega_m(params):
    """
    Run CLASS with the given parameter dictionary, compute:
      - H0 (km/s/Mpc),
      - r_s,drag (Mpc),
      - H0 * r_s,drag,
      - Omega_m (from CLASS).

    Returns:
      H0r_s : float
      Omega_m : float
    """
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    # CLASS returns derived parameters; 'H0' and 'Omega_m' should be among them.
    derived = cosmo.get_current_derived_parameters(
        ["H0", "Omega_m", "rs_drag"]
    )

    H0_val   = float(derived["H0"])
    Omega_m  = float(derived["Omega_m"])
    rs_drag  = float(derived["rs_drag"])  # Mpc

    H0rs = H0_val * rs_drag

    cosmo.struct_cleanup()
    cosmo.empty()

    return H0rs, Omega_m

# ============================================================
# CHI^2 UTILITY
# ============================================================

def delta_chi2(theta, mean, cov):
    """
    Compute Delta chi^2 = (theta - mean)^T Cov^{-1} (theta - mean)
    for a 2D parameter vector theta = [H0*r_s, Omega_m].
    """
    theta = np.asarray(theta)
    mean  = np.asarray(mean)
    d = theta - mean
    inv_cov = np.linalg.inv(cov)
    return float(d.T @ inv_cov @ d)

# ============================================================
# MAIN
# ============================================================

def main():
    print(">>> Starting DESI (H0*r_s, Omega_m) consistency test for ΛCDM vs CCBH")
    print(f"  Using DESI chain : {DESI_CHAIN_FILE}")
    print()

    if not os.path.exists(DESI_CHAIN_FILE):
        raise FileNotFoundError(f"DESI chain file not found: {DESI_CHAIN_FILE}")

    # 1. DESI posterior in (H0*r_s, Omega_m)
    print("=== Step 1: DESI DR1 (H0*r_s, Omega_m) from cobaya chain ===")
    mean_desi, cov_desi = load_desi_H0rs_omegam(DESI_CHAIN_FILE)
    print(f"DESI mean (H0*r_s, Omega_m) = ({mean_desi[0]:.3f}, {mean_desi[1]:.5f})")
    print("DESI covariance matrix:")
    print(cov_desi)
    print()

    # 2. Compute (H0*r_s, Omega_m) for Pantheon+ ΛCDM best-fit
    print("=== Step 2: CLASS – Pantheon+ ΛCDM best-fit point ===")
    params_lcdm = build_class_params_lcdm(LCDM_H0, LCDM_OM)
    H0rs_lcdm, Om_lcdm_from_class = compute_H0rs_Omega_m(params_lcdm)
    theta_lcdm = np.array([H0rs_lcdm, Om_lcdm_from_class])

    print(f"Pantheon+ ΛCDM input (H0, Omega_m) = ({LCDM_H0:.3f}, {LCDM_OM:.4f})")
    print(f"CLASS-derived (H0*r_s, Omega_m)    = ({H0rs_lcdm:.3f}, {Om_lcdm_from_class:.5f})")
    print()

    # 3. Compute (H0*r_s, Omega_m) for Pantheon+ CCBH best-fit
    print("=== Step 3: CLASS – Pantheon+ CCBH best-fit point ===")
    params_ccbh = build_class_params_ccbh(CCBH_H0, CCBH_OM, CCBH_KBH)
    H0rs_ccbh, Om_ccbh_from_class = compute_H0rs_Omega_m(params_ccbh)
    theta_ccbh = np.array([H0rs_ccbh, Om_ccbh_from_class])

    print(f"Pantheon+ CCBH input (H0, Omega_m, k_BH) = ({CCBH_H0:.3f}, {CCBH_OM:.4f}, {CCBH_KBH:.3f})")
    print(f"CLASS-derived (H0*r_s, Omega_m)          = ({H0rs_ccbh:.3f}, {Om_ccbh_from_class:.5f})")
    print()

    # 4. Delta chi^2 relative to DESI
    print("=== Step 4: Delta chi^2 relative to DESI in (H0*r_s, Omega_m) ===")
    dchi2_lcdm = delta_chi2(theta_lcdm, mean_desi, cov_desi)
    dchi2_ccbh = delta_chi2(theta_ccbh, mean_desi, cov_desi)

    print("Pantheon+ ΛCDM point:")
    print(f"  (H0*r_s, Omega_m) = ({theta_lcdm[0]:.3f}, {theta_lcdm[1]:.5f})")
    print(f"  Delta chi^2 wrt DESI = {dchi2_lcdm:.3f}")
    print()

    print("Pantheon+ CCBH point:")
    print(f"  (H0*r_s, Omega_m) = ({theta_ccbh[0]:.3f}, {theta_ccbh[1]:.5f})")
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
        print("\n*** ERROR while running desi_H0rs_CC_BH_test.py ***")
        traceback.print_exc()
        print("\nCheck DESI_CHAIN_FILE path and customize build_class_params_ccbh "
              "for your actual CCBH CLASS implementation.")
