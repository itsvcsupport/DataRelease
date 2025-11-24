#!/usr/bin/env python
"""
desi_H0rs_CC_BH_emulator.py

Fast, CAMB/CLASS-free emulator to compare Pantheon+ ΛCDM and CCBH
best-fit points against DESI DR1 in the BAO-native parameter space
(H0 * r_s, Omega_m), where r_s is the sound horizon at the drag epoch.

Steps:
  1. Load DESI cobaya chain and build mean + covariance in (H0*r_s, Omega_m)
     using (H0rdrag, omegam) columns.
  2. Implement a CCBH-inspired background:
       - BH fluid with rho_BH obeying
           d ln rho_BH / d ln a = k_eff(a) - 3
         where k_eff(a) transitions from ~0 to ~k_max around a_t ~ 0.07
         (t ~ 300 Myr).
       - Flat universe: Omega_BH,0 = 1 - Omega_m - Omega_r.
  3. Compute r_s via:
       r_s = ∫_0^{a_d} c_s(a) / [a^2 H(a)] da
     using:
       - Eisenstein & Hu 1998 approximation for z_d,
       - c_s(a) = c / sqrt(3 (1 + R(a))),  R(a) = (3 rho_b / 4 rho_gamma).
  4. Evaluate Delta chi^2 vs DESI for:
       - Pantheon+ ΛCDM-like (k_max = 3),
       - Pantheon+ CCBH best-fit (k_max = k_BH ~ 3.2).
"""

import os
import io
import numpy as np
import pandas as pd

# ============================================================
# CONFIG – EDIT THESE FOR YOUR SETUP
# ============================================================

# DESI cobaya chain file (ABSOLUTE path is safest)
DESI_CHAIN_FILE = r"C:\Users\WilliamKellett\cobaya\base\desi-bao-all\chain.1.txt"

# Column names in the DESI chain for H0*r_drag and Omega_m
DESI_COL_H0RDAG = "H0rdrag"
DESI_COL_OM     = "omegam"   # if needed, you could change to "omm"

# Pantheon+ best-fit ΛCDM-like point (your SN-only fit)
LCDM_H0  = 73.040   # km/s/Mpc
LCDM_OM  = 0.3492   # dimensionless

# Pantheon+ best-fit CCBH point (your SN-only CCBH run)
CCBH_H0  = 73.200   # km/s/Mpc
CCBH_OM  = 0.3860   # dimensionless
CCBH_KBH = 3.200    # mass-dependent coupling exponent k_BH (SMBH-like)

# Cosmological constants / choices for emulator
OMEGA_B_H2 = 0.02237   # physical baryon density (Planck-like)
N_EFF      = 3.046     # effective neutrino number
T_CMB      = 2.7255    # K
C_LIGHT    = 299792.458  # km/s

# CCBH transition parameters
A_TRANS   = 0.07   # transition scale factor (t ~ 300 Myr)
GAMMA_KEFF = 5.0   # steepness of k_eff(a) transition

# Integration settings for r_s integral
A_MIN       = 1e-6
N_A_SAMPLES = 4000

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
# COSMOLOGY EMULATOR – H(a), k_eff(a), z_drag, r_s
# ============================================================

def omega_r_h2(T_cmb=T_CMB, N_eff=N_EFF):
    """
    Physical radiation density today: omega_r = omega_gamma * (1 + 0.2271*N_eff).
    omega_gamma ≈ 2.469e-5 (for T_cmb ≈ 2.7255K).
    """
    omega_gamma = 2.469e-5 * (T_cmb / 2.7255) ** 4
    return omega_gamma * (1.0 + 0.2271 * N_eff)


def z_drag_eisenstein_hu(omega_m, omega_b):
    """
    Eisenstein & Hu (1998) approximation for drag epoch redshift z_d.
    Inputs:
      omega_m = Omega_m * h^2
      omega_b = Omega_b * h^2
    """
    b1 = 0.313 * (omega_m ** -0.419) * (1.0 + 0.607 * (omega_m ** 0.674))
    b2 = 0.238 * (omega_m ** 0.223)
    z_d = 1291.0 * (omega_m ** 0.251) / (1.0 + 0.659 * (omega_m ** 0.828))
    z_d *= (1.0 + b1 * (omega_b ** b2))
    return z_d


def k_eff(a, k_max, a_t=A_TRANS, gamma=GAMMA_KEFF):
    """
    Effective coupling k_eff(a) representing the mass-weighted average
    over the BH population:

      - Early times (a << a_t): stellar BHs dominate => k_eff ~ 0.
      - Around a_t (~0.07): SMBH population ramps up => rapid increase.
      - Late times (a >> a_t): SMBHs dominate => k_eff ~ k_max.

    Implemented as a logistic:
      k_eff(a) = k_max / [1 + (a_t / a)^gamma].
    """
    S = 1.0 / (1.0 + (a_t / a)**gamma)
    return k_max * S


def rho_BH_of_a(a_vals, Omega_BH0, k_max, a_t=A_TRANS, gamma=GAMMA_KEFF):
    """
    Compute rho_BH(a) / rho_crit,0 for the BH fluid by integrating:

      d ln rho_BH / d ln a = k_eff(a) - 3,

    with boundary condition rho_BH(a=1) = Omega_BH0.

    We integrate in ln a from 0 (a=1) downwards to smaller a.
    """
    ln_a = np.log(a_vals)
    # Sort descending in a (i.e. descending ln a) so ln a[0] ~ 0 (a ~ 1)
    idx_sort = np.argsort(ln_a)[::-1]
    ln_a_sorted = ln_a[idx_sort]
    a_sorted = np.exp(ln_a_sorted)

    k_vals = k_eff(a_sorted, k_max=k_max, a_t=a_t, gamma=gamma)

    ln_rho = np.zeros_like(ln_a_sorted)
    # At a=1 => ln_rho = ln(Omega_BH0)
    ln_rho[0] = np.log(Omega_BH0)

    for i in range(1, len(ln_a_sorted)):
        dln_a = ln_a_sorted[i] - ln_a_sorted[i-1]
        slope = k_vals[i-1] - 3.0
        ln_rho[i] = ln_rho[i-1] + slope * dln_a

    # Unscramble to original order
    rho_sorted = np.exp(ln_rho)
    rho = np.zeros_like(rho_sorted)
    rho[idx_sort] = rho_sorted

    return rho


def rs_drag_emulator(H0, Omega_m, k_max):
    """
    Compute the sound horizon at drag epoch, r_s,drag, using a CCBH-inspired
    background:

      H(a)^2 / H0^2 = Omega_r * a^-4 + Omega_m * a^-3 + Omega_BH(a),

    where Omega_BH(a) is obtained by integrating:

      d ln rho_BH / d ln a = k_eff(a) - 3,

    with k_eff(a) = k_max / [1 + (a_t / a)^gamma].

    Inputs:
      H0      : Hubble parameter today [km/s/Mpc]
      Omega_m : total matter fraction today
      k_max   : asymptotic coupling exponent (e.g. 3 for Λ-like,
                3.2 for your CCBH best-fit)

    Returns:
      r_s (Mpc)
    """
    h = H0 / 100.0
    omega_b = OMEGA_B_H2
    omega_r = omega_r_h2(T_CMB, N_EFF)
    Omega_r = omega_r / h**2

    # Flat universe: Omega_BH0 picks up the slack
    Omega_BH0 = 1.0 - Omega_m - Omega_r
    if Omega_BH0 <= 0:
        raise ValueError(
            f"Omega_BH0 <= 0 for H0={H0}, Omega_m={Omega_m}. "
            f"Omega_r={Omega_r}, Omega_BH0={Omega_BH0}"
        )

    # Drag redshift and scale factor
    omega_m = Omega_m * h**2
    z_d = z_drag_eisenstein_hu(omega_m, omega_b=omega_b)
    a_d = 1.0 / (1.0 + z_d)

    # Integration grid in a
    a_vals = np.logspace(np.log10(A_MIN), np.log10(a_d), N_A_SAMPLES)

    # Baryon-photon sound speed c_s(a)
    # R(a) = 3 rho_b / (4 rho_gamma) = (3 * omega_b / (4 * omega_gamma)) * a
    omega_gamma = omega_r / (1.0 + 0.2271 * N_EFF)
    R_prefac = (3.0 * omega_b) / (4.0 * omega_gamma)
    R_vals = R_prefac * a_vals
    c_s_vals = C_LIGHT / np.sqrt(3.0 * (1.0 + R_vals))  # km/s

    # BH density evolution
    Omega_BH_vals = rho_BH_of_a(a_vals, Omega_BH0, k_max=k_max,
                                a_t=A_TRANS, gamma=GAMMA_KEFF)

    # H(a) in km/s/Mpc
    H_over_H0_sq = (
        Omega_r * a_vals**-4 +
        Omega_m * a_vals**-3 +
        Omega_BH_vals
    )
    H_vals = H0 * np.sqrt(H_over_H0_sq)

    # Integrand: c_s / (a^2 H(a)) [Mpc]
    integrand = c_s_vals / (a_vals**2 * H_vals)

    # Trapezoidal integration over a
    r_s = np.trapezoid(integrand, a_vals)

    return r_s


def compute_H0rs_emu(H0, Omega_m, k_max):
    """
    Wrapper to compute (H0 * r_s, Omega_m) for the emulator.
    """
    r_s = rs_drag_emulator(H0, Omega_m, k_max)
    H0rs = H0 * r_s
    return H0rs, Omega_m

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
    print(">>> Starting DESI (H0*r_s, Omega_m) CCBH emulator test")
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

    # 2. Pantheon+ ΛCDM-like emulator (k_max = 3 => w_BH ~ -1 at late times)
    print("=== Step 2: Emulator – Pantheon+ ΛCDM-like best-fit (k_max=3) ===")
    H0rs_lcdm, Om_lcdm = compute_H0rs_emu(LCDM_H0, LCDM_OM, k_max=3.0)
    theta_lcdm = np.array([H0rs_lcdm, Om_lcdm])

    print(f"Pantheon+ ΛCDM input (H0, Omega_m) = ({LCDM_H0:.3f}, {LCDM_OM:.4f})")
    print(f"Emulator (H0*r_s, Omega_m)        = ({H0rs_lcdm:.3f}, {Om_lcdm:.5f})")
    print()

    # 3. Pantheon+ CCBH emulator (k_max = CCBH_KBH)
    print(f"=== Step 3: Emulator – Pantheon+ CCBH best-fit (k_max={CCBH_KBH:.3f}) ===")
    H0rs_ccbh, Om_ccbh = compute_H0rs_emu(CCBH_H0, CCBH_OM, k_max=CCBH_KBH)
    theta_ccbh = np.array([H0rs_ccbh, Om_ccbh])

    print(f"Pantheon+ CCBH input (H0, Omega_m, k_BH) = ({CCBH_H0:.3f}, {CCBH_OM:.4f}, {CCBH_KBH:.3f})")
    print(f"Emulator (H0*r_s, Omega_m)               = ({H0rs_ccbh:.3f}, {Om_ccbh:.5f})")
    print()

    # 4. Delta chi^2 vs DESI
    print("=== Step 4: Delta chi^2 vs DESI in (H0*r_s, Omega_m) ===")
    dchi2_lcdm = delta_chi2(theta_lcdm, mean_desi, cov_desi)
    dchi2_ccbh = delta_chi2(theta_ccbh, mean_desi, cov_desi)

    print("Pantheon+ ΛCDM-like point:")
    print(f"  (H0*r_s, Omega_m) = ({theta_lcdm[0]:.3f}, {theta_lcdm[1]:.5f})")
    print(f"  Delta chi^2 wrt DESI = {dchi2_lcdm:.3f}")
    print()

    print(f"Pantheon+ CCBH point (k_max={CCBH_KBH:.3f}):")
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
        print("\n*** ERROR while running desi_H0rs_CC_BH_emulator.py ***")
        traceback.print_exc()
