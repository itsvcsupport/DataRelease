#!/usr/bin/env python
"""
desi_H0rs_CC_BH_emulator.py

Fast, CAMB/CLASS-free emulator to compare Pantheon+ ΛCDM and CCBH
best-fit points against DESI DR1 in the BAO-native parameter space
(H0 * r_s, Omega_m), where r_s is the sound horizon at the drag epoch.

Capabilities:
  1. Load DESI cobaya chain and build mean + covariance in (H0*r_s, Omega_m)
     using (H0rdrag, omegam) columns.
  2. Implement two CCBH-style backgrounds:
       (a) Simple "Λ-like" CCBH with a single k_max (logistic k_eff(a)).
       (b) 3-phase piecewise k_eff(a) with:
            - k1 for a < a1       (early BH era)
            - k2 for a1 <= a < a2 (SMBH ignition / rapid growth)
            - k3 for a >= a2      (post-ignition / current era)
         where rho_BH obeys d ln rho_BH / d ln a = k_eff(a) - 3.
  3. Compute r_s via:
       r_s = ∫_0^{a_d} c_s(a) / [a^2 H(a)] da
     using:
       - Eisenstein & Hu 1998 approximation for z_d,
       - c_s(a) = c / sqrt(3 (1 + R(a))),  R(a) = (3 rho_b / 4 rho_gamma).
  4. Evaluate Delta chi^2 vs DESI for:
       - Pantheon+ ΛCDM-like best-fit (single k_max).
       - Pantheon+ CCBH best-fit (single k_max).
       - 3-phase k_eff(a) patterns (reverse inference).
  5. Load Pantheon+ SNe and compute SN chi^2 (diagonal errors only),
     using the same H(a) background and analytic minimization over the
     absolute magnitude offset M.
  6. Perform a joint SN+DESI grid scan over (H0, Omega_m) for a fixed
     3-phase k_eff(a) pattern, and report the best-fit combined point.
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

# Pantheon+ distances file (relative to your DataRelease root)
PANTHEON_FILE = "Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat"

# Pantheon+ best-fit ΛCDM-like point (your SN-only fit)
LCDM_H0  = 73.040   # km/s/Mpc
LCDM_OM  = 0.3492   # dimensionless

# Pantheon+ best-fit CCBH point (your SN-only CCBH run)
CCBH_H0  = 73.200   # km/s/Mpc
CCBH_OM  = 0.3860   # dimensionless
CCBH_KBH = 3.200    # mass-dependent coupling exponent for SMBH-like behavior

# Cosmological constants / choices for emulator
OMEGA_B_H2 = 0.02237   # physical baryon density (Planck-like)
N_EFF      = 3.046     # effective neutrino number
T_CMB      = 2.7255    # K
C_LIGHT    = 299792.458  # km/s

# "Simple" CCBH logistic transition parameters (for the single-k_max emulator)
A_TRANS    = 0.07   # transition scale factor (t ~ 300 Myr)
GAMMA_KEFF = 5.0    # steepness of k_eff(a) transition

# Integration settings for r_s and distance integrals
A_MIN       = 1e-6
N_A_SAMPLES = 4000     # for early-time integrals (r_s)
N_A_BG      = 1000     # for background a-grid from a_min .. 1 for distances

# Scan settings for the 3-phase piecewise model (reverse DESI inference)
SCAN_K1_RANGE = (0.0, 0.5)      # early BH era (nearly matter-like)
SCAN_K2_RANGE = (1.5, 4.0)      # ignition phase (coupling stronger)
SCAN_K3_RANGE = (0.0, 3.0)      # late-time effective coupling
SCAN_A1_RANGE = (0.03, 0.06)    # start of ignition
SCAN_A2_RANGE = (0.08, 0.20)    # end of ignition
SCAN_N_PER_DIM = 3              # coarse grid resolution
EARLY_BH_FRACTION_MAX = 1e-3    # require Omega_BH at recombination < this × Omega_m

# Joint SN+DESI fitter (H0, Omega_m) scan settings
JOINT_H0_RANGE = (68.0, 75.0)   # km/s/Mpc
JOINT_OM_RANGE = (0.26, 0.40)
JOINT_N_H0     = 8
JOINT_N_OM     = 8

# Fixed 3-phase k_eff(a) pattern for joint fitter (from your best DESI scan)
JOINT_K1 = 0.0
JOINT_K2 = 1.5
JOINT_K3 = 1.5
JOINT_A1 = 0.03
JOINT_A2 = 0.08

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
# COSMOLOGY BASICS – H(a), z_drag, r_s
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

# ============================================================
# SIMPLE LOGISTIC CCBH BACKGROUND (single k_max)
# ============================================================

def k_eff_logistic(a, k_max, a_t=A_TRANS, gamma=GAMMA_KEFF):
    """
    Effective coupling k_eff(a) as a logistic:

      k_eff(a) = k_max / [1 + (a_t / a)^gamma].

    Early a << a_t:  k_eff ~ 0 (BH behaves like matter).
    Late  a >> a_t:  k_eff ~ k_max (BH behaves like DE-like fluid).
    """
    S = 1.0 / (1.0 + (a_t / a)**gamma)
    return k_max * S


def rho_BH_of_a_logistic(a_vals, Omega_BH0, k_max, a_t=A_TRANS, gamma=GAMMA_KEFF):
    """
    Compute rho_BH(a) / rho_crit,0 for the BH fluid under the logistic k_eff(a).
    """
    ln_a = np.log(a_vals)
    idx_sort = np.argsort(ln_a)[::-1]
    ln_a_sorted = ln_a[idx_sort]
    a_sorted = np.exp(ln_a_sorted)

    k_vals = k_eff_logistic(a_sorted, k_max=k_max, a_t=a_t, gamma=gamma)

    ln_rho = np.zeros_like(ln_a_sorted)
    ln_rho[0] = np.log(Omega_BH0)

    for i in range(1, len(ln_a_sorted)):
        dln_a = ln_a_sorted[i] - ln_a_sorted[i-1]
        slope = k_vals[i-1] - 3.0
        ln_rho[i] = ln_rho[i-1] + slope * dln_a

    rho_sorted = np.exp(ln_rho)
    rho = np.zeros_like(rho_sorted)
    rho[idx_sort] = rho_sorted

    return rho


def rs_drag_emulator_logistic(H0, Omega_m, k_max):
    """
    Compute r_s,drag for the logistic k_eff(a) model.
    """
    h = H0 / 100.0
    omega_b = OMEGA_B_H2
    omega_r = omega_r_h2(T_CMB, N_EFF)
    Omega_r = omega_r / h**2

    Omega_BH0 = 1.0 - Omega_m - Omega_r
    if Omega_BH0 <= 0:
        raise ValueError(
            f"Omega_BH0 <= 0 for H0={H0}, Omega_m={Omega_m}. "
            f"Omega_r={Omega_r}, Omega_BH0={Omega_BH0}"
        )

    omega_m = Omega_m * h**2
    z_d = z_drag_eisenstein_hu(omega_m, omega_b=omega_b)
    a_d = 1.0 / (1.0 + z_d)

    a_vals = np.logspace(np.log10(A_MIN), np.log10(a_d), N_A_SAMPLES)

    omega_gamma = omega_r / (1.0 + 0.2271 * N_EFF)
    R_prefac = (3.0 * omega_b) / (4.0 * omega_gamma)
    R_vals = R_prefac * a_vals
    c_s_vals = C_LIGHT / np.sqrt(3.0 * (1.0 + R_vals))

    Omega_BH_vals = rho_BH_of_a_logistic(a_vals, Omega_BH0, k_max=k_max,
                                         a_t=A_TRANS, gamma=GAMMA_KEFF)

    H_over_H0_sq = (
        Omega_r * a_vals**-4 +
        Omega_m * a_vals**-3 +
        Omega_BH_vals
    )
    H_vals = H0 * np.sqrt(H_over_H0_sq)

    integrand = c_s_vals / (a_vals**2 * H_vals)
    r_s = np.trapezoid(integrand, a_vals)

    return r_s


def compute_H0rs_emu_logistic(H0, Omega_m, k_max):
    r_s = rs_drag_emulator_logistic(H0, Omega_m, k_max)
    H0rs = H0 * r_s
    return H0rs, Omega_m

# ============================================================
# 3-PHASE PIECEWISE k_eff(a)
# ============================================================

def k_eff_piecewise(a, k1, k2, k3, a1, a2):
    """
    Piecewise-constant k_eff(a) with 3 phases:

      k1 for a < a1       (early BH era, mostly stellar BHs)
      k2 for a1 <= a < a2 (SMBH ignition / rapid growth)
      k3 for a >= a2      (post-ignition / present era)
    """
    a = np.asarray(a)
    k = np.empty_like(a)

    mask1 = (a < a1)
    mask2 = (a >= a1) & (a < a2)
    mask3 = (a >= a2)

    k[mask1] = k1
    k[mask2] = k2
    k[mask3] = k3

    return k


def rho_BH_of_a_piecewise(a_vals, Omega_BH0, k1, k2, k3, a1, a2):
    """
    Compute rho_BH(a) / rho_crit,0 for the BH fluid under the 3-phase
    piecewise k_eff(a) model.
    """
    ln_a = np.log(a_vals)
    idx_sort = np.argsort(ln_a)[::-1]
    ln_a_sorted = ln_a[idx_sort]
    a_sorted = np.exp(ln_a_sorted)

    k_vals = k_eff_piecewise(a_sorted, k1, k2, k3, a1, a2)

    ln_rho = np.zeros_like(ln_a_sorted)
    ln_rho[0] = np.log(Omega_BH0)

    for i in range(1, len(ln_a_sorted)):
        dln_a = ln_a_sorted[i] - ln_a_sorted[i-1]
        slope = k_vals[i-1] - 3.0
        ln_rho[i] = ln_rho[i-1] + slope * dln_a

    rho_sorted = np.exp(ln_rho)
    rho = np.zeros_like(rho_sorted)
    rho[idx_sort] = rho_sorted

    return rho


def rs_drag_emulator_piecewise(H0, Omega_m, k1, k2, k3, a1, a2):
    """
    Compute the sound horizon at drag epoch, r_s,drag, for the 3-phase
    k_eff(a) model.
    """
    h = H0 / 100.0
    omega_b = OMEGA_B_H2
    omega_r = omega_r_h2(T_CMB, N_EFF)
    Omega_r = omega_r / h**2

    Omega_BH0 = 1.0 - Omega_m - Omega_r
    if Omega_BH0 <= 0:
        raise ValueError(
            f"Omega_BH0 <= 0 for H0={H0}, Omega_m={Omega_m}. "
            f"Omega_r={Omega_r}, Omega_BH0={Omega_BH0}"
        )

    omega_m = Omega_m * h**2
    z_d = z_drag_eisenstein_hu(omega_m, omega_b=omega_b)
    a_d = 1.0 / (1.0 + z_d)

    a_vals = np.logspace(np.log10(A_MIN), np.log10(a_d), N_A_SAMPLES)

    omega_gamma = omega_r / (1.0 + 0.2271 * N_EFF)
    R_prefac = (3.0 * omega_b) / (4.0 * omega_gamma)
    R_vals = R_prefac * a_vals
    c_s_vals = C_LIGHT / np.sqrt(3.0 * (1.0 + R_vals))

    Omega_BH_vals = rho_BH_of_a_piecewise(
        a_vals, Omega_BH0, k1, k2, k3, a1, a2
    )

    H_over_H0_sq = (
        Omega_r * a_vals**-4 +
        Omega_m * a_vals**-3 +
        Omega_BH_vals
    )
    H_vals = H0 * np.sqrt(H_over_H0_sq)

    integrand = c_s_vals / (a_vals**2 * H_vals)
    r_s = np.trapezoid(integrand, a_vals)

    return r_s


def compute_H0rs_emu_piecewise(H0, Omega_m, k1, k2, k3, a1, a2):
    r_s = rs_drag_emulator_piecewise(H0, Omega_m, k1, k2, k3, a1, a2)
    H0rs = H0 * r_s
    return H0rs, Omega_m


def BH_safe_at_recomb(H0, Omega_m, k1, k2, k3, a1, a2,
                      threshold=EARLY_BH_FRACTION_MAX):
    """
    Check that the BH density is negligible at recombination:
      Omega_BH(a_rec) / Omega_m(a_rec) < threshold.

    We use a_rec ~ 1/1100.
    """
    h = H0 / 100.0
    omega_r = omega_r_h2(T_CMB, N_EFF)
    Omega_r = omega_r / h**2

    Omega_BH0 = 1.0 - Omega_m - Omega_r
    if Omega_BH0 <= 0:
        return False

    a_rec = 1.0 / 1100.0
    a_vals = np.array([a_rec, 1.0])

    Omega_BH_vals = rho_BH_of_a_piecewise(
        a_vals, Omega_BH0, k1, k2, k3, a1, a2
    )
    Omega_BH_rec = Omega_BH_vals[0]

    # Matter scales as a^-3
    Omega_m_rec = Omega_m * a_rec**-3

    frac = Omega_BH_rec / Omega_m_rec
    return frac < threshold

# ============================================================
# PANTHEON+ LOADER & SN χ² (diagonal only)
# ============================================================

def load_pantheon_subset(filename, zmin=0.01, zmax=None):
    """
    Load Pantheon+SH0ES file and extract:
      z, mu_obs, mu_err
    using MU_SH0ES / MU_SH0ES_ERR_DIAG if available, else other pairs.
    """
    df = pd.read_csv(filename, sep=r"\s+", comment="#")

    # redshift column
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

    # distance modulus pair
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

    print(f"Pantheon+: using columns for mu: {mu_col}, mu_err: {err_col}")

    z = df[z_col].values
    mu = df[mu_col].values
    mu_err = df[err_col].values

    mask = np.isfinite(z) & np.isfinite(mu) & np.isfinite(mu_err)
    if zmin is not None:
        mask &= (z >= zmin)
    if zmax is not None:
        mask &= (z <= zmax)

    return z[mask], mu[mask], mu_err[mask]


def build_background_for_distances(H0, Omega_m, k1, k2, k3, a1, a2):
    """
    Build a background H(a) and comoving distance integral
    for the 3-phase k_eff(a) model.

    Returns:
      a_grid_desc: array from 1 -> A_MIN (descending)
      Dint_desc  : array of ∫_a^1 c/(a'^2 H(a')) da' (Mpc) evaluated on a_grid_desc
    """
    h = H0 / 100.0
    omega_b = OMEGA_B_H2
    omega_r = omega_r_h2(T_CMB, N_EFF)
    Omega_r = omega_r / h**2

    Omega_BH0 = 1.0 - Omega_m - Omega_r
    if Omega_BH0 <= 0:
        raise ValueError(
            f"Omega_BH0 <= 0 for H0={H0}, Omega_m={Omega_m}. "
            f"Omega_r={Omega_r}, Omega_BH0={Omega_BH0}"
        )

    # a-grid from 1 down to A_MIN
    a_grid_desc = np.logspace(0.0, np.log10(A_MIN), N_A_BG)

    Omega_BH_vals = rho_BH_of_a_piecewise(
        a_grid_desc, Omega_BH0, k1, k2, k3, a1, a2
    )

    H_over_H0_sq = (
        Omega_r * a_grid_desc**-4 +
        Omega_m * a_grid_desc**-3 +
        Omega_BH_vals
    )
    H_vals = H0 * np.sqrt(H_over_H0_sq)

    integrand = C_LIGHT / (a_grid_desc**2 * H_vals)

    # cumulative integral from a=1 -> A_MIN:
    # Dint(1) = 0, Dint(a_i) = ∫_{a_i}^1 integrand da'
    Dint_desc = np.zeros_like(a_grid_desc)
    for i in range(1, len(a_grid_desc)):
        da = a_grid_desc[i-1] - a_grid_desc[i]  # positive
        Dint_desc[i] = Dint_desc[i-1] + 0.5 * (integrand[i] + integrand[i-1]) * da

    return a_grid_desc, Dint_desc


def sn_chi2_piecewise(H0, Omega_m, k1, k2, k3, a1, a2,
                      z_sn, mu_sn, mu_err_sn):
    """
    Compute Pantheon+ chi^2 for a given (H0, Omega_m, k1,k2,k3,a1,a2)
    using the 3-phase k_eff(a) background. We analytically minimize
    over an additive offset M (absolute magnitude + H0 degeneracy).
    """
    a_grid_desc, Dint_desc = build_background_for_distances(
        H0, Omega_m, k1, k2, k3, a1, a2
    )

    # Interpolator for comoving distance D_C(z) = ∫_a^1 c/(a'^2 H(a')) da'
    # a_grid_desc is descending, we can still use np.interp if we reverse.
    a_rev = a_grid_desc[::-1]
    Dint_rev = Dint_desc[::-1]

    def D_C_of_z(z):
        a = 1.0 / (1.0 + z)
        # clamp to grid range
        a_clipped = np.clip(a, a_rev[0], a_rev[-1])
        return np.interp(a_clipped, a_rev, Dint_rev)

    D_C_vals = D_C_of_z(z_sn)
    D_L_vals = (1.0 + z_sn) * D_C_vals  # Mpc

    # model distance modulus without M offset:
    mu_model0 = 5.0 * np.log10(D_L_vals) + 25.0

    # analytic best-fit offset M_best
    w = 1.0 / mu_err_sn**2
    num = np.sum(w * (mu_sn - mu_model0))
    den = np.sum(w)
    M_best = num / den

    resid = mu_sn - (mu_model0 + M_best)
    chi2 = np.sum((resid / mu_err_sn)**2)

    return chi2, M_best

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
# GRID SCAN FOR PIECEWISE k_eff(a) (DESI-ONLY)
# ============================================================

def scan_piecewise_k(
    H0, Omega_m, desi_mean, desi_cov,
    k1_range=SCAN_K1_RANGE,
    k2_range=SCAN_K2_RANGE,
    k3_range=SCAN_K3_RANGE,
    a1_range=SCAN_A1_RANGE,
    a2_range=SCAN_A2_RANGE,
    n_per_dim=SCAN_N_PER_DIM
):
    """
    Coarse grid scan over (k1, k2, k3, a1, a2) to find the best-fit
    3-phase k_eff(a) pattern that minimizes Delta chi^2 vs DESI
    in (H0*r_s, Omega_m), with H0 and Omega_m fixed.

    Returns:
      best_params = (k1, k2, k3, a1, a2, H0rs_best, Om_best)
      best_chi2   = minimum Delta chi^2
    """
    k1_vals = np.linspace(k1_range[0], k1_range[1], n_per_dim)
    k2_vals = np.linspace(k2_range[0], k2_range[1], n_per_dim)
    k3_vals = np.linspace(k3_range[0], k3_range[1], n_per_dim)
    a1_vals = np.linspace(a1_range[0], a1_range[1], n_per_dim)
    a2_vals = np.linspace(a2_range[0], a2_range[1], n_per_dim)

    best = None
    best_chi2 = np.inf

    for k1 in k1_vals:
        for k2 in k2_vals:
            for k3 in k3_vals:
                for a1 in a1_vals:
                    for a2 in a2_vals:
                        if a2 <= a1:
                            continue

                        if not BH_safe_at_recomb(H0, Omega_m, k1, k2, k3, a1, a2):
                            continue

                        try:
                            H0rs, Om = compute_H0rs_emu_piecewise(
                                H0, Omega_m, k1, k2, k3, a1, a2
                            )
                        except Exception:
                            continue

                        theta = np.array([H0rs, Om])
                        chi2 = delta_chi2(theta, desi_mean, desi_cov)

                        if chi2 < best_chi2:
                            best_chi2 = chi2
                            best = (k1, k2, k3, a1, a2, H0rs, Om)

    return best, best_chi2

# ============================================================
# JOINT SN+DESI SCAN OVER (H0, Omega_m)
# ============================================================

def joint_scan_H0_Om(
    z_sn, mu_sn, mu_err_sn,
    desi_mean, desi_cov,
    k1, k2, k3, a1, a2,
    H0_range=JOINT_H0_RANGE,
    Om_range=JOINT_OM_RANGE,
    n_H0=JOINT_N_H0,
    n_Om=JOINT_N_OM
):
    """
    Joint SN+DESI grid scan over (H0, Omega_m) for fixed 3-phase
    k_eff(a) parameters (k1,k2,k3,a1,a2).

    Returns:
      best_params = (H0_best, Omega_m_best, M_best_SN, H0rs_best)
      best_chi2_SN
      best_chi2_DESI
      best_chi2_total
    """
    H0_vals = np.linspace(H0_range[0], H0_range[1], n_H0)
    Om_vals = np.linspace(Om_range[0], Om_range[1], n_Om)

    best = None
    best_chi2_tot = np.inf

    for H0 in H0_vals:
        for Om in Om_vals:
            # early-time safety
            if not BH_safe_at_recomb(H0, Om, k1, k2, k3, a1, a2):
                continue

            # SN chi^2
            try:
                chi2_SN, M_best = sn_chi2_piecewise(H0, Om, k1, k2, k3, a1, a2,
                                                    z_sn, mu_sn, mu_err_sn)
            except Exception:
                continue

            # DESI chi^2
            try:
                H0rs, Om_eff = compute_H0rs_emu_piecewise(H0, Om, k1, k2, k3, a1, a2)
            except Exception:
                continue

            theta = np.array([H0rs, Om_eff])
            chi2_DESI = delta_chi2(theta, desi_mean, desi_cov)

            chi2_tot = chi2_SN + chi2_DESI

            if chi2_tot < best_chi2_tot:
                best_chi2_tot = chi2_tot
                best = (H0, Om, M_best, H0rs, chi2_SN, chi2_DESI, chi2_tot)

    return best

# ============================================================
# MAIN
# ============================================================

def main():
    print(">>> Starting DESI (H0*r_s, Omega_m) CCBH emulator & SN+DESI joint fit")
    print(f"  Using DESI chain : {DESI_CHAIN_FILE}")
    print(f"  Using Pantheon   : {PANTHEON_FILE}")
    print()

    if not os.path.exists(DESI_CHAIN_FILE):
        raise FileNotFoundError(f"DESI chain file not found: {DESI_CHAIN_FILE}")
    if not os.path.exists(PANTHEON_FILE):
        raise FileNotFoundError(f"Pantheon file not found: {PANTHEON_FILE}")

    # 1. DESI posterior in (H0*r_s, Omega_m)
    print("=== Step 1: DESI DR1 (H0*r_s, Omega_m) from cobaya chain ===")
    mean_desi, cov_desi = load_desi_H0rs_omegam(DESI_CHAIN_FILE)
    print(f"DESI mean (H0*r_s, Omega_m) = ({mean_desi[0]:.3f}, {mean_desi[1]:.5f})")
    print("DESI covariance matrix:")
    print(cov_desi)
    print()

    # 2. Pantheon+ ΛCDM-like emulator (logistic single k_max=3)
    print("=== Step 2: Simple logistic emulator – Pantheon+ ΛCDM-like (k_max=3) ===")
    H0rs_lcdm, Om_lcdm = compute_H0rs_emu_logistic(LCDM_H0, LCDM_OM, k_max=3.0)
    theta_lcdm = np.array([H0rs_lcdm, Om_lcdm])
    dchi2_lcdm = delta_chi2(theta_lcdm, mean_desi, cov_desi)

    print(f"Pantheon+ ΛCDM input (H0, Omega_m) = ({LCDM_H0:.3f}, {LCDM_OM:.4f})")
    print(f"Emulator (H0*r_s, Omega_m)        = ({H0rs_lcdm:.3f}, {Om_lcdm:.5f})")
    print(f"Delta chi^2 wrt DESI (ΛCDM-like)  = {dchi2_lcdm:.3f}")
    print()

    # 3. Pantheon+ CCBH emulator (logistic single k_max=CCBH_KBH)
    print(f"=== Step 3: Simple logistic emulator – Pantheon+ CCBH (k_max={CCBH_KBH:.3f}) ===")
    H0rs_ccbh_simple, Om_ccbh_simple = compute_H0rs_emu_logistic(CCBH_H0, CCBH_OM, k_max=CCBH_KBH)
    theta_ccbh_simple = np.array([H0rs_ccbh_simple, Om_ccbh_simple])
    dchi2_ccbh_simple = delta_chi2(theta_ccbh_simple, mean_desi, cov_desi)

    print(f"Pantheon+ CCBH input (H0, Omega_m, k_BH) = ({CCBH_H0:.3f}, {CCBH_OM:.4f}, {CCBH_KBH:.3f})")
    print(f"Emulator (H0*r_s, Omega_m)               = ({H0rs_ccbh_simple:.3f}, {Om_ccbh_simple:.5f})")
    print(f"Delta chi^2 wrt DESI (CCBH simple)       = {dchi2_ccbh_simple:.3f}")
    print()

    # 4. Reverse inference: scan 3-phase k_eff(a) for best DESI match
    print("=== Step 4: Reverse inference – scan 3-phase k_eff(a) to match DESI ===")
    print(f"  Using Pantheon+ CCBH H0={CCBH_H0:.3f}, Omega_m={CCBH_OM:.4f} as anchors")
    best_desik, best_chi2_desik = scan_piecewise_k(
        CCBH_H0, CCBH_OM, mean_desi, cov_desi,
        k1_range=SCAN_K1_RANGE,
        k2_range=SCAN_K2_RANGE,
        k3_range=SCAN_K3_RANGE,
        a1_range=SCAN_A1_RANGE,
        a2_range=SCAN_A2_RANGE,
        n_per_dim=SCAN_N_PER_DIM
    )

    if best_desik is None:
        print("No viable 3-phase k_eff(a) found that passes early-time safety cuts.")
    else:
        k1b, k2b, k3b, a1b, a2b, H0rs_bestk, Om_bestk = best_desik
        print("Best-fit 3-phase k_eff(a) parameters (DESI-only, coarse scan):")
        print(f"  k1 (early)      = {k1b:.3f}")
        print(f"  k2 (ignition)   = {k2b:.3f}")
        print(f"  k3 (late)       = {k3b:.3f}")
        print(f"  a1 (start ign.) = {a1b:.4f}  (z ~ {1/a1b - 1:.1f})")
        print(f"  a2 (end ign.)   = {a2b:.4f}  (z ~ {1/a2b - 1:.1f})")
        print()
        print("Resulting (H0*r_s, Omega_m):")
        print(f"  (H0*r_s, Omega_m) = ({H0rs_bestk:.3f}, {Om_bestk:.5f})")
        print(f"  Delta chi^2 wrt DESI (3-phase CCBH) = {best_chi2_desik:.3f}")
        print()
        print("Rough sigma-levels (sqrt(Delta chi^2)) for reference:")
        print(f"  ΛCDM-like simple      : ~{np.sqrt(dchi2_lcdm):.2f}σ")
        print(f"  CCBH simple (k_max)   : ~{np.sqrt(dchi2_ccbh_simple):.2f}σ")
        print(f"  3-phase CCBH (best)   : ~{np.sqrt(best_chi2_desik):.2f}σ")
    print()

    # 5. Load Pantheon+ for SN chi^2
    print("=== Step 5: Load Pantheon+ and run joint SN+DESI scan over (H0, Omega_m) ===")
    z_sn, mu_sn, mu_err_sn = load_pantheon_subset(PANTHEON_FILE, zmin=0.01, zmax=None)
    print(f"Loaded {len(z_sn)} SNe for joint fit")
    print()

    # Use fixed 3-phase pattern from JOINT_* constants for joint scan
    print("Scanning H0 and Omega_m with fixed 3-phase k_eff(a):")
    print(f"  k1={JOINT_K1}, k2={JOINT_K2}, k3={JOINT_K3}, a1={JOINT_A1}, a2={JOINT_A2}")
    best_joint = joint_scan_H0_Om(
        z_sn, mu_sn, mu_err_sn,
        mean_desi, cov_desi,
        k1=JOINT_K1, k2=JOINT_K2, k3=JOINT_K3,
        a1=JOINT_A1, a2=JOINT_A2,
        H0_range=JOINT_H0_RANGE,
        Om_range=JOINT_OM_RANGE,
        n_H0=JOINT_N_H0,
        n_Om=JOINT_N_OM
    )

    if best_joint is None:
        print("No joint SN+DESI solution found in the scanned (H0, Omega_m) region.")
    else:
        H0_best, Om_best, M_best_SN, H0rs_best, chi2_SN_best, chi2_DESI_best, chi2_tot_best = best_joint
        print("Best-fit joint SN+DESI (coarse scan, fixed 3-phase k_eff):")
        print(f"  H0       = {H0_best:.3f} km/s/Mpc")
        print(f"  Omega_m  = {Om_best:.4f}")
        print(f"  M_best   = {M_best_SN:.4f}  (SN absolute magnitude offset)")
        print(f"  H0*r_s   = {H0rs_best:.3f}")
        print(f"  chi^2_SN   = {chi2_SN_best:.2f}")
        print(f"  chi^2_DESI = {chi2_DESI_best:.2f}")
        print(f"  chi^2_tot  = {chi2_tot_best:.2f}")
        print(f"  sqrt(chi^2_DESI) ~ {np.sqrt(chi2_DESI_best):.2f}σ tension in DESI space alone")
        print()

    print("Interpretation guide for 2 parameters (Gaussian approx):")
    print("  For DESI-only chi^2 in (H0*r_s, Omega_m):")
    print("    ~68% region: Delta chi^2 ≲ 2.3")
    print("    ~95% region: Delta chi^2 ≲ 6.0")
    print("If both points are below ~6, they are statistically consistent with DESI.")
    print(">>> Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print("\n*** ERROR while running desi_H0rs_CC_BH_emulator.py ***")
        traceback.print_exc()
