#!/usr/bin/env python
"""
desi_H0rs_CC_BH_camb.py

Compare Pantheon+ best-fit ΛCDM and CCBH models against DESI DR1
in the DESI-native parameter space (H0 * r_s, Omega_m), where r_s
is the sound horizon at the drag epoch.

We:
  1. Load a DESI cobaya chain with columns including (H0rdrag, omegam).
  2. Build DESI mean + covariance in (H0*r_s, Omega_m) using H0rdrag, omegam.
  3. For each model (LCDM, CCBH):
       - Run CAMB to compute r_drag and Omega_m,
       - Form H0 * r_drag, Omega_m,
       - Compute Delta chi^2 vs DESI.

NOTE:
  - CCBH implementation here is still a placeholder: we pass k_BH through
    to CAMB but do not yet have a modified CAMB that uses it. You’ll need
    to plug in your own modified CAMB (e.g. MGCAMB/EFTCAMB-style) later.
"""

import os
import io
import numpy as np
import pandas as pd

import camb
from camb import model

# ============================================================
# CONFIG – EDIT THESE FOR YOUR SETUP
# ============================================================

# DESI cobaya chain file (ABSOLUTE path is safest)
DESI_CHAIN_FILE = r"C:\Users\WilliamKellett\cobaya\base\desi-bao-all\chain.1.txt"

# Column names for H0*r_drag and Omega_m in that chain
DESI_COL_H0RDAG = "H0rdrag"
DESI_COL_OM     = "omegam"   # if this errors, try "omm"

# Pantheon+ best-fit LCDM point (your SN-only fit)
LCDM_H0  = 73.040   # km/s/Mpc
LCDM_OM  = 0.3492   # dimensionless

# Pantheon+ best-fit CCBH point (your SN-only CCBH run)
CCBH_H0  = 73.200   # km/s/Mpc
CCBH_OM  = 0.3860   # dimensionless
CCBH_KBH = 3.200    # mass-dependent coupling exponent (placeholder)

# Fixed baryon density for CAMB runs (Planck-like)
OMEGA_B_H2 = 0.02237

# Some fixed nuisance cosmology params for CAMB
NS_DEFAULT       = 0.965
AS_DEFAULT       = 2.1e-9
TAU_REIO_DEFAULT = 0.054
N_EFF_DEFAULT    = 3.046

# ============================================================
# DESI CHAIN LOADER – (H0*r_s, Omega_m)
# ============================================================

def load_desi_H0rs_omegam(path, col_H0r=DESI_COL_H0RDAG, col_Om=DESI_COL_OM):
    """
    Load DESI cobaya chain and build mean + covariance in (H0*r_s, Omega_m)
    using columns (H0rdrag, omegam).

    Returns:
      mean: array([mean_H0r, mean_Om])
      cov : 2x2 covariance matrix
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
            "Could not find Cobaya-style header line in DESI chain "
            "(expected '# weight minuslogpost ...')."
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
    Om_col  = get_col(col_Om,  "Omega_m")

    H0r_vals = df[H0r_col].values
    Om_vals  = df[Om_col].values

    X = np.vstack([H0r_vals, Om_vals]).T
    mean = X.mean(axis=0)
    cov  = np.cov(X.T)

    print(f"DESI chain: using columns H0rdrag -> '{H0r_col}', Omega_m -> '{Om_col}'")
    return mean, cov

# ============================================================
# CAMB WRAPPERS – COMPUTE H0 * r_drag AND Omega_m
# ============================================================

def build_camb_params_lcdm(H0, Omega_m):
    """
    Build a CAMBparams instance for a flat ΛCDM cosmology with given H0, Omega_m.
    We fix omega_b = OMEGA_B_H2 and solve omega_cdm so that:
      Omega_m = (omega_b + omega_cdm)/h^2
    """
    h = H0 / 100.0
    ombh2 = OMEGA_B_H2
    omega_m_target = Omega_m * h**2
    omch2 = omega_m_target - ombh2
    if omch2 <= 0:
        raise ValueError(
            f"Computed omch2 <= 0 for H0={H0}, Omega_m={Omega_m}. "
            f"omega_m_target={omega_m_target}, ombh2={ombh2}"
        )

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        omk=0.0,
        mnu=0.06,
        nnu=N_EFF_DEFAULT,
        tau=TAU_REIO_DEFAULT,
    )
    pars.InitPower.set_params(As=AS_DEFAULT, ns=NS_DEFAULT)
    pars.set_dark_energy(w=-1.0)  # vanilla Λ
    pars.set_matter_power(redshifts=[0.0], kmax=2.0)
    pars.WantCls = False
    return pars


def build_camb_params_ccbh(H0, Omega_m, k_BH):
    """
    Build CAMBparams for a first-pass CCBH background.

    Mapping:
      - Treat the BH-coupled sector as an effective dark-energy-like fluid
        with constant w_BH = -k_BH / 3.
      - Replace Λ with this fluid (i.e., no separate cosmological constant).
      - Matter sector (baryons + CDM) stays at Omega_m as inferred from
        the Pantheon+ fit.
      - Flat universe: Omega_k = 0, so Omega_BH,0 is whatever is needed to
        make the total sum to 1 once radiation is included.

    This uses standard CAMB dark-energy machinery, so no Fortran changes
    are required yet.
    """
    h = H0 / 100.0
    ombh2 = OMEGA_B_H2
    omega_m_target = Omega_m * h**2
    omch2 = omega_m_target - ombh2
    if omch2 <= 0:
        raise ValueError(
            f"Computed omch2 <= 0 for H0={H0}, Omega_m={Omega_m}. "
            f"omega_m_target={omega_m_target}, ombh2={ombh2}"
        )

    # Base cosmology (no explicit Lambda term; DE will be the BH fluid)
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        omk=0.0,
        mnu=0.06,
        nnu=N_EFF_DEFAULT,
        tau=TAU_REIO_DEFAULT,
    )
    pars.InitPower.set_params(As=AS_DEFAULT, ns=NS_DEFAULT)

    # Effective BH "dark energy" fluid
    w_BH = -k_BH / 3.0
    # Use CAMB's fluid dark-energy model with constant w
    pars.set_dark_energy(
        w=w_BH,
        wa=0.0,
        dark_energy_model="fluid"
    )

    # Tag the params so you can see k_BH later if needed
    pars.k_BH = k_BH

    # CAMB will automatically set Omega_de = 1 - Omega_m - Omega_r - Omega_k
    # so the BH fluid plays the role of dark energy.

    pars.set_matter_power(redshifts=[0.0], kmax=2.0)
    pars.WantCls = False

    return pars



def compute_H0rs_Omega_m(pars):
    """
    Run CAMB with the given CAMBparams, compute:
      - H0 (km/s/Mpc),
      - r_drag (Mpc),
      - H0 * r_drag,
      - Omega_m (from derived params or from the input params).

    Returns:
      H0r_s : float
      Omega_m : float
    """
    results = camb.get_results(pars)
    derived = results.get_derived_params()  # dict

    # --- H0 ---
    # Some CAMB builds include 'H0' in derived, some don't.
    if "H0" in derived:
        H0_val = float(derived["H0"])
    else:
        # CAMB always stores H0 on the params object in km/s/Mpc
        H0_val = float(pars.H0)

    # --- Omega_m ---
    if "Om_m" in derived:
        Om_m = float(derived["Om_m"])
    else:
        # Fallback: compute from ombh2 + omch2 and h
        h = H0_val / 100.0
        Om_m = float((pars.ombh2 + pars.omch2) / (h * h))

    # --- r_drag ---
    # DESI uses the sound horizon at the drag epoch; CAMB reports 'rdrag'
    if "rdrag" in derived:
        rdrag = float(derived["rdrag"])
    else:
        raise KeyError(
            "CAMB derived parameters missing 'rdrag'; available keys are: "
            f"{list(derived.keys())}"
        )

    H0rs = H0_val * rdrag
    return H0rs, Om_m

# ============================================================
# CHI^2 UTILITY
# ============================================================

def delta_chi2(theta, mean, cov):
    theta = np.asarray(theta)
    mean  = np.asarray(mean)
    d = theta - mean
    inv_cov = np.linalg.inv(cov)
    return float(d.T @ inv_cov @ d)

# ============================================================
# MAIN
# ============================================================

def main():
    print(">>> Starting DESI (H0*r_s, Omega_m) consistency test with CAMB")
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

    # 2. Pantheon+ ΛCDM
    print("=== Step 2: CAMB – Pantheon+ ΛCDM best-fit ===")
    pars_lcdm = build_camb_params_lcdm(LCDM_H0, LCDM_OM)
    H0rs_lcdm, Om_lcdm = compute_H0rs_Omega_m(pars_lcdm)
    theta_lcdm = np.array([H0rs_lcdm, Om_lcdm])

    print(f"Pantheon+ ΛCDM input (H0, Omega_m)   = ({LCDM_H0:.3f}, {LCDM_OM:.4f})")
    print(f"CAMB-derived (H0*r_s, Omega_m)       = ({H0rs_lcdm:.3f}, {Om_lcdm:.5f})")
    print()

    # 3. Pantheon+ CCBH (placeholder mapping)
    print("=== Step 3: CAMB – Pantheon+ CCBH best-fit (placeholder background) ===")
    pars_ccbh = build_camb_params_ccbh(CCBH_H0, CCBH_OM, CCBH_KBH)
    H0rs_ccbh, Om_ccbh = compute_H0rs_Omega_m(pars_ccbh)
    theta_ccbh = np.array([H0rs_ccbh, Om_ccbh])

    print(f"Pantheon+ CCBH input (H0, Omega_m, k_BH) = ({CCBH_H0:.3f}, {CCBH_OM:.4f}, {CCBH_KBH:.3f})")
    print(f"CAMB-derived (H0*r_s, Omega_m)           = ({H0rs_ccbh:.3f}, {Om_ccbh:.5f})")
    print()

    # 4. Delta chi^2 vs DESI
    print("=== Step 4: Delta chi^2 vs DESI in (H0*r_s, Omega_m) ===")
    dchi2_lcdm = delta_chi2(theta_lcdm, mean_desi, cov_desi)
    dchi2_ccbh = delta_chi2(theta_ccbh, mean_desi, cov_desi)

    print("Pantheon+ ΛCDM point:")
    print(f"  (H0*r_s, Omega_m) = ({theta_lcdm[0]:.3f}, {theta_lcdm[1]:.5f})")
    print(f"  Delta chi^2 wrt DESI = {dchi2_lcdm:.3f}")
    print()

    print("Pantheon+ CCBH point (placeholder background):")
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
        print("\n*** ERROR while running desi_H0rs_CC_BH_camb.py ***")
        traceback.print_exc()
