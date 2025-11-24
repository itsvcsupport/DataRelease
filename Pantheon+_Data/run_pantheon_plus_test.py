import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM, FlatwCDM
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

# ---------------------------
# 5. Define a cosmology and compute theory distance modulus
#    TEST 1: Flat wCDM sanity check (constant w)
# ---------------------------

# Example starting parameters (can be changed or replaced by best fit)
H0 = 70.0   # km/s/Mpc
Om0 = 0.3
w = -1.0    # ΛCDM = -1 ; your theory expects slightly phantom ~ -1.03

cosmo = FlatwCDM(H0=H0 * u.km/u.s/u.Mpc, Om0=Om0, w0=w)

# Astropy gives distance modulus directly
mu_th = cosmo.distmod(z).value  # in magnitudes

# ---------------------------
# 6. Compute chi-square with full covariance
# ---------------------------

delta = mu_obs - mu_th  # data - theory

# Solve C_sub * x = delta
# chi2 = delta^T C^{-1} delta = x^T delta where x = C^{-1} delta
x = np.linalg.solve(C_sub, delta)
chi2 = float(delta @ x)

dof = N - 3  # three free params: H0, Om0, w
print(f"Chi-square: {chi2:.2f} for N={N} (dof ≈ {dof})")
print(f"Chi2/dof ≈ {chi2/dof:.2f}")

# ---------------------------
# 7. Likelihood function for flat wCDM
# ---------------------------

def pantheon_loglike(H0_val, Om0_val, w_val):
    """Pantheon+SH0ES log-likelihood for flat wCDM with constant w."""
    cosmo_local = FlatwCDM(H0=H0_val * u.km/u.s/u.Mpc, Om0=Om0_val, w0=w_val)
    mu_th_local = cosmo_local.distmod(z).value
    delta_local = mu_obs - mu_th_local
    x_local = np.linalg.solve(C_sub, delta_local)
    chi2_local = float(delta_local @ x_local)
    return -0.5 * chi2_local

# ---------------------------
# 8. Rough grid-search for flat wCDM
# ---------------------------

def compute_logL_grid(H0_grid, Om0_grid, w_grid):
    """Compute log-likelihood on a grid of (H0, Om0, w)."""
    Z = np.zeros((len(H0_grid), len(Om0_grid), len(w_grid)))
    for i, H in enumerate(H0_grid):
        for j, Om in enumerate(Om0_grid):
            for k, wv in enumerate(w_grid):
                Z[i, j, k] = pantheon_loglike(H, Om, wv)
    return Z

def find_best_fit(H0_grid, Om0_grid, w_grid, logL_grid):
    """Find best-fit parameters from a precomputed logL grid."""
    idx = np.unravel_index(np.argmax(logL_grid), logL_grid.shape)
    i_best, j_best, k_best = idx
    return {
        "H0": H0_grid[i_best],
        "Om0": Om0_grid[j_best],
        "w": w_grid[k_best],
        "logL": logL_grid[i_best, j_best, k_best],
        "indices": idx
    }

# ---------------------------
# 9. Plotting utilities
# ---------------------------

def plot_residuals(H0_val, Om0_val, w_val):
    """Plot μ_obs - μ_th vs z for given (H0, Om0, w)."""
    cosmo_local = FlatwCDM(H0=H0_val * u.km/u.s/u.Mpc, Om0=Om0_val, w0=w_val)
    mu_th_local = cosmo_local.distmod(z).value
    delta_local = mu_obs - mu_th_local

    plt.figure(figsize=(8, 5))
    plt.scatter(z, delta_local, s=10)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Redshift z")
    plt.ylabel(r"$\mu_{\mathrm{obs}} - \mu_{\mathrm{th}}$ (mag)")
    plt.title(
        f"Pantheon+ Residuals (H0={H0_val:.1f}, Ωm={Om0_val:.3f}, w={w_val:.3f})"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_likelihood_contours(H0_grid, Om0_grid, logL_2d, best, w_fixed):
    """
    Plot likelihood contours in the (H0, Om0) plane for a fixed w.
    Uses ΔlogL levels corresponding to ~1σ, 2σ, 3σ for 2 parameters.
    """
    H0_vals = np.array(H0_grid)
    Om0_vals = np.array(Om0_grid)

    # Normalize so max logL = 0
    logL_max = np.max(logL_2d)
    dlogL = logL_2d - logL_max

    # For 2D Gaussian, Δχ² = 2.30, 6.18, 11.83 → ΔlogL = -0.5 Δχ²
    levels = np.sort([-0.5 * 2.30, -0.5 * 6.18, -0.5 * 11.83])

    H0_mesh, Om0_mesh = np.meshgrid(H0_vals, Om0_vals, indexing="ij")

    plt.figure(figsize=(8, 6))
    cs = plt.contour(H0_mesh, Om0_mesh, dlogL, levels=levels)
    plt.clabel(cs, inline=True, fontsize=8)
    plt.scatter(best["H0"], best["Om0"], marker="x", label="Best fit")
    plt.xlabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
    plt.ylabel(r"$\Omega_m$")
    plt.title(f"Pantheon+ Likelihood Contours (flat wCDM, w={w_fixed:.3f})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------------------
# 10. Main execution block
# ---------------------------

if __name__ == "__main__":
    # Single point check at starting values
    print("logL(H0, Om0, w) at starting values =",
          pantheon_loglike(H0, Om0, w))

    # Define grids for rough best fit (can tighten later)
    H0_grid = np.linspace(60.0, 80.0, 41)     # 0.5 km/s/Mpc steps
    Om0_grid = np.linspace(0.20, 0.40, 41)    # 0.005 steps
    w_grid   = np.linspace(-1.3, -0.7, 41)    # includes phantom & quintessence

    print("Computing logL grid for flat wCDM (this may take a bit)...")
    logL_grid = compute_logL_grid(H0_grid, Om0_grid, w_grid)

    best = find_best_fit(H0_grid, Om0_grid, w_grid, logL_grid)
    print("Rough best-fit from grid search:")
    print(best)

    # Plot residuals for best-fit parameters
    plot_residuals(best["H0"], best["Om0"], best["w"])

    # Take 2D slice at best-fit w for contour plot
    i_best, j_best, k_best = best["indices"]
    logL_slice = logL_grid[:, :, k_best]
    w_fixed = w_grid[k_best]

    plot_likelihood_contours(H0_grid, Om0_grid, logL_slice, best, w_fixed)
