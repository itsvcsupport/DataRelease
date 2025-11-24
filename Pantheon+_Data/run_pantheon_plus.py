import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import FlatwCDM
import astropy.units as u

# ---------------------------
# 1. Load the data table
# ---------------------------

data_file = "4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat"

# Pantheon+SH0ES.dat is whitespace-delimited with a header row
df = pd.read_csv(
    data_file,
    sep=r"\s+",
    comment="#"
)

# Quick sanity check
print("Columns in data file:")
print(df.columns.tolist())
print("Number of rows:", len(df))

# ---------------------------
# 2. Select the subset you want
#    Here: SNe used in the SH0ES Hubble-flow sample
# ---------------------------

mask = df["USED_IN_SH0ES_HF"] == 1

df_sub = df[mask].reset_index(drop=True)
N = len(df_sub)
print(f"Using {N} objects with USED_IN_SH0ES_HF == 1")

# Redshift array (Pantheon+ recommends zHD for Hubble diagram)
z = df_sub["zHD"].values

# Observed distance modulus from SH0ES calibration
mu_obs = df_sub["MU_SH0ES"].values

# ---------------------------
# 3. Load the full covariance matrix
#    Pantheon+SH0ES_STAT+SYS.cov format:
#      first line: N_total (here 1701)
#      remaining: N_total^2 numbers, row-major
# ---------------------------

cov_file = "4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov"

with open(cov_file, "r") as f:
    # First line gives total dimension
    first_line = f.readline().strip()
    N_tot = int(first_line)

    # Read the remaining numbers
    data_flat = np.fromfile(f, sep=" ")

if data_flat.size != N_tot * N_tot:
    raise RuntimeError(
        f"Expected {N_tot * N_tot} covariance elements, "
        f"got {data_flat.size}"
    )

C_full = data_flat.reshape((N_tot, N_tot))

print("Full covariance shape:", C_full.shape)

# ---------------------------
# 4. Subselect covariance for the same subset of SNe
# ---------------------------

# Indices of the selected SNe in the original full table
indices = np.where(mask.values)[0]

# Use np.ix_ to slice rows and columns
C_sub = C_full[np.ix_(indices, indices)]

print("Subselected covariance shape:", C_sub.shape)

# Precompute the inverse once for speed
Cinv_sub = np.linalg.inv(C_sub)

# ---------------------------
# 5. Define CCBH cosmology and theory distance modulus
#    CCBH effective: flat, w_BH = -k/3
# ---------------------------

def mu_th_ccbh(z_arr, H0, Om0, k):
    """
    CCBH late-time effective model:
      - flat universe: Om0 + Ode0 = 1
      - BH dark energy behaves as constant-w with w_BH = -k/3
    Implemented via FlatwCDM as a pure distance engine.
    """
    w_BH = -k / 3.0
    cosmo = FlatwCDM(H0=H0 * u.km/u.s/u.Mpc, Om0=Om0, w0=w_BH)
    return cosmo.distmod(z_arr).value  # magnitudes

# ---------------------------
# 6. Compute chi-square with full covariance
# ---------------------------

def chi2_ccbh(H0, Om0, k):
    mu_model = mu_th_ccbh(z, H0, Om0, k)
    delta = mu_obs - mu_model
    # chi2 = delta^T C^{-1} delta
    return float(delta @ (Cinv_sub @ delta))

# ---------------------------
# 7. Likelihood function for flat CCBH effective model
# ---------------------------

def pantheon_loglike_ccbh(H0_val, Om0_val, k_val):
    """Pantheon+SH0ES log-likelihood for flat CCBH effective model."""
    mu_model = mu_th_ccbh(z, H0_val, Om0_val, k_val)
    delta = mu_obs - mu_model
    chi2_local = float(delta @ (Cinv_sub @ delta))
    return -0.5 * chi2_local

# ---------------------------
# 8. Grid-search for flat CCBH
# ---------------------------

def compute_logL_grid_ccbh(H0_grid, Om0_grid, k_grid):
    """Compute log-likelihood on a grid of (H0, Om0, k)."""
    Z = np.zeros((len(H0_grid), len(Om0_grid), len(k_grid)))
    for i, H in enumerate(H0_grid):
        for j, Om in enumerate(Om0_grid):
            # Flatness prior: 0 < Om < 1
            if Om <= 0.0 or Om >= 1.0:
                Z[i, j, :] = -np.inf
                continue
            for k_idx, kval in enumerate(k_grid):
                Z[i, j, k_idx] = pantheon_loglike_ccbh(H, Om, kval)
    return Z

def find_best_fit(H0_grid, Om0_grid, k_grid, logL_grid):
    """Find best-fit parameters from a precomputed logL grid."""
    idx = np.unravel_index(np.argmax(logL_grid), logL_grid.shape)
    i_best, j_best, k_best = idx
    return {
        "H0": float(H0_grid[i_best]),
        "Om0": float(Om0_grid[j_best]),
        "k": float(k_grid[k_best]),
        "w_BH": -float(k_grid[k_best]) / 3.0,
        "logL": float(logL_grid[i_best, j_best, k_best]),
        "indices": idx
    }

# ---------------------------
# 9. Plotting utilities
# ---------------------------

def plot_contours_Om_k(Om0_grid, k_grid, logL_2d, best, H0_fixed,
                       save_basename=None):
    """
    Plot likelihood contours in the (Omega_m, k) plane for fixed H0.
    logL_2d is a 2D array over (Omega_m, k) for that fixed H0.
    """
    Om0_vals = np.array(Om0_grid)
    k_vals   = np.array(k_grid)

    logL_max = np.max(logL_2d)
    dlogL = logL_2d - logL_max
    levels = np.sort([-0.5 * 2.30, -0.5 * 6.18, -0.5 * 11.83])

    Om_mesh, k_mesh = np.meshgrid(Om0_vals, k_vals, indexing="ij")

    plt.figure()
    cs = plt.contour(Om_mesh, k_mesh, dlogL, levels=levels)
    plt.clabel(cs, inline=True, fontsize=8)
    plt.scatter(best["Om0"], best["k"], marker="x", label="Best fit")
    plt.xlabel(r"$\Omega_m$")
    plt.ylabel(r"$k$")
    plt.title(
        rf"Pantheon+ Likelihood (CCBH) in $(\Omega_m,k)$ "
        rf"$\ (H_0={H0_fixed:.1f}\ \mathrm{{km\,s^{-1}\,Mpc^{-1}}})$"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_basename is not None:
        plt.savefig(f"{save_basename}.png")
        plt.savefig(f"{save_basename}.pdf")

    plt.show()

def plot_contours_H0_k(H0_grid, k_grid, logL_2d, best, Om_fixed,
                       save_basename=None):
    """
    Plot likelihood contours in the (H0, k) plane for fixed Omega_m.
    logL_2d is a 2D array over (H0, k) for that fixed Omega_m.
    """
    H0_vals = np.array(H0_grid)
    k_vals  = np.array(k_grid)

    # Normalize so max logL = 0
    logL_max = np.max(logL_2d)
    dlogL = logL_2d - logL_max

    # 1σ, 2σ, 3σ for 2 parameters: Δχ² = 2.30, 6.18, 11.83 → ΔlogL = -0.5 Δχ²
    levels = np.sort([-0.5 * 2.30, -0.5 * 6.18, -0.5 * 11.83])

    H0_mesh, k_mesh = np.meshgrid(H0_vals, k_vals, indexing="ij")

    plt.figure()
    cs = plt.contour(H0_mesh, k_mesh, dlogL, levels=levels)
    plt.clabel(cs, inline=True, fontsize=8)
    plt.scatter(best["H0"], best["k"], marker="x", label="Best fit")
    plt.xlabel(r"$H_0\ \mathrm{[km\,s^{-1}\,Mpc^{-1}]}$")
    plt.ylabel(r"$k$")
    plt.title(
        rf"Pantheon+ Likelihood (CCBH) in $(H_0,k)$ "
        rf"$\ (\Omega_m={Om_fixed:.3f})$"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_basename is not None:
        plt.savefig(f"{save_basename}.png")
        plt.savefig(f"{save_basename}.pdf")

    plt.show()

def plot_residuals(H0_val, Om0_val, k_val):
    """Plot μ_obs - μ_th vs z for given (H0, Om0, k) in CCBH."""
    mu_model = mu_th_ccbh(z, H0_val, Om0_val, k_val)
    delta = mu_obs - mu_model

    w_val = -k_val / 3.0

    plt.figure(figsize=(8, 5))
    plt.scatter(z, delta, s=10)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Redshift z")
    plt.ylabel(r"$\mu_{\mathrm{obs}} - \mu_{\mathrm{th}}$ (mag)")
    plt.title(
        f"Pantheon+ Residuals (CCBH; H0={H0_val:.1f}, Ωm={Om0_val:.3f}, "
        f"k={k_val:.3f}, w_BH={w_val:.3f})"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_likelihood_contours(H0_grid, Om0_grid, logL_2d, best, k_fixed,
                             save_basename=None):
    """
    Plot likelihood contours in the (H0, Om0) plane for a fixed k
    (equivalently fixed w_BH = -k/3).

    If save_basename is provided, saves PNG and PDF with that basename.
    """
    H0_vals = np.array(H0_grid)
    Om0_vals = np.array(Om0_grid)

    # Normalize so max logL = 0
    logL_max = np.max(logL_2d)
    dlogL = logL_2d - logL_max

    # For 2D Gaussian, Δχ² = 2.30, 6.18, 11.83 → ΔlogL = -0.5 Δχ²
    levels = np.sort([-0.5 * 2.30, -0.5 * 6.18, -0.5 * 11.83])

    H0_mesh, Om0_mesh = np.meshgrid(H0_vals, Om0_vals, indexing="ij")

    w_fixed = -k_fixed / 3.0

    plt.figure()
    cs = plt.contour(H0_mesh, Om0_mesh, dlogL, levels=levels)
    plt.clabel(cs, inline=True, fontsize=8)
    plt.scatter(best["H0"], best["Om0"], marker="x", label="Best fit")
    plt.xlabel(r"$H_0\ \mathrm{[km\,s^{-1}\,Mpc^{-1}]}$")
    plt.ylabel(r"$\Omega_m$")
    plt.title(
        rf"Pantheon+ Likelihood (CCBH) in $(H_0,\Omega_m)$ "
        rf"$\ (k={k_fixed:.3f},\, w_{{BH}}={w_fixed:.3f})$"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_basename is not None:
        plt.savefig(f"{save_basename}.png")
        plt.savefig(f"{save_basename}.pdf")

    plt.show()

# ---------------------------
# 10. Main execution block
# ---------------------------

if __name__ == "__main__":
    # --- sanity check at a CCBH-like starting point (optional) ---
    H0_start  = 73.0
    Om0_start = 0.40
    k_start   = 3.05   # from coarse run

    chi2_start = chi2_ccbh(H0_start, Om0_start, k_start)
    dof = N - 3  # three free params: H0, Om0, k
    print(f"Chi-square (start): {chi2_start:.2f} for N={N} (dof ≈ {dof})")
    print(f"Chi2/dof ≈ {chi2_start/dof:.2f}")
    print("logL(H0, Om0, k) at starting values =",
          pantheon_loglike_ccbh(H0_start, Om0_start, k_start))

    # ============================================================
    # HIGH-RESOLUTION ZOOM GRID AROUND COARSE BEST FIT
    #   H0 : 72 → 74 in steps of 0.1
    #   Ωm : 0.35 → 0.45 in steps of 0.002
    #   k  : 2.9 → 3.2 in steps of 0.01
    # ============================================================

    H0_grid_fine  = np.arange(72.0, 74.0 + 1e-9, 0.1)
    Om0_grid_fine = np.arange(0.35, 0.45 + 1e-9, 0.002)
    k_grid_fine   = np.arange(2.9,  3.2  + 1e-9, 0.01)

    print("Computing HIGH-RES logL grid for flat CCBH effective model...")
    logL_grid_fine = compute_logL_grid_ccbh(H0_grid_fine, Om0_grid_fine, k_grid_fine)

    best_fine = find_best_fit(H0_grid_fine, Om0_grid_fine, k_grid_fine, logL_grid_fine)
    print("\n===== High-resolution CCBH best fit (Pantheon+ only) =====")
    print(f"H0      = {best_fine['H0']:.3f} km/s/Mpc")
    print(f"Omega_m = {best_fine['Om0']:.3f}")
    print(f"k       = {best_fine['k']:.3f}")
    print(f"w_BH    = {best_fine['w_BH']:.3f}")
    print(f"logL    = {best_fine['logL']:.3f}")
    chi2_best = -2.0 * best_fine["logL"]
    print(f"chi2    = {chi2_best:.3f} for N={N}")
    print(f"chi2/dof ≈ {chi2_best/dof:.3f}")

    # ------------------------------------------------------------
    # Residuals for best-fit parameters (not saved, visual sanity)
    # ------------------------------------------------------------
    plot_residuals(best_fine["H0"], best_fine["Om0"], best_fine["k"])

    # ------------------------------------------------------------
    # 1) (H0, Omega_m) at fixed k = k_best
    # ------------------------------------------------------------
    i_best, j_best, k_best = best_fine["indices"]

    logL_slice_H0_Om = logL_grid_fine[:, :, k_best]
    k_fixed = k_grid_fine[k_best]

    plot_likelihood_contours(
        H0_grid_fine, Om0_grid_fine, logL_slice_H0_Om, best_fine, k_fixed,
        save_basename="pantheon_ccbh_H0_Om"
    )

    # ------------------------------------------------------------
    # 2) (H0, k) at fixed Omega_m = Om_best
    # ------------------------------------------------------------
    logL_slice_H0_k = logL_grid_fine[:, j_best, :]  # shape: (len(H0), len(k))
    Om_fixed = Om0_grid_fine[j_best]

    plot_contours_H0_k(
        H0_grid_fine, k_grid_fine, logL_slice_H0_k, best_fine, Om_fixed,
        save_basename="pantheon_ccbh_H0_k"
    )

    # ------------------------------------------------------------
    # 3) (Omega_m, k) at fixed H0 = H0_best
    # ------------------------------------------------------------
    logL_slice_Om_k = logL_grid_fine[i_best, :, :]  # shape: (len(Om0), len(k))
    H0_fixed = H0_grid_fine[i_best]

    plot_contours_Om_k(
        Om0_grid_fine, k_grid_fine, logL_slice_Om_k, best_fine, H0_fixed,
        save_basename="pantheon_ccbh_Om_k"
    )
