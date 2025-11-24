import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import ccbh_theory as c

# -----------------------------------
# Configuration: your Pantheon+ path
# -----------------------------------
PANTHEON_PATH = (
    r"C:\Users\WilliamKellett\Documents\GitHub\DataRelease\Pantheon+_Data"
    r"\4_DISTANCES_AND_COVAR\Pantheon+SH0ES.dat"
)

# -----------------------------------
# Planck ΛCDM baseline (your baseline)
# -----------------------------------
H0_pl    = 67.4
ombh2_pl = 0.0224
omch2_pl = 0.1200
c_light  = 299792.458  # km/s

def omega_m_from_cosmo(H0, ombh2, omch2):
    h = H0 / 100.0
    return (ombh2 + omch2) / h**2

Omega_m_pl = omega_m_from_cosmo(H0_pl, ombh2_pl, omch2_pl)

def H_lcdm_of_z(z, H0, Omega_m):
    z = np.asarray(z)
    zp1 = 1.0 + z
    return H0 * np.sqrt(Omega_m * zp1**3 + (1.0 - Omega_m))


def dL_LCDM(z):
    """Luminosity distance in ΛCDM, Mpc."""
    z = np.asarray(z)
    d = []
    for zi in z:
        zs = np.linspace(0.0, zi, 2000)
        Hs = H_lcdm_of_z(zs, H0_pl, Omega_m_pl)
        integral = np.trapezoid(1.0 / Hs, zs)
        d.append(c_light * (1.0 + zi) * integral)
    return np.array(d)


# -----------------------------------
# CCBH parameters – your best-fit
# -----------------------------------
H0_cc    = 70.8
ombh2_cc = 0.0224
# Adjusted to give Omega_m ≈ 0.269
omch2_cc = 0.1124

k1       = 0.99
k2       = 2.58
k3       = 4.23
z_t1     = 5.55
z_t2     = 0.128

def dL_CCBH(z):
    """Luminosity distance in your CCBH model, Mpc."""
    z = np.asarray(z)
    d = []
    for zi in z:
        zs = np.linspace(0.0, zi, 2000)
        # Vectorized call: H_of_z over the whole zs array
        Hs = c.H_of_z(zs, H0_cc, ombh2_cc, omch2_cc,
                      k1, k2, k3, z_t1, z_t2)
        Hs = np.asarray(Hs).reshape(-1)  # ensure 1D
        integral = np.trapezoid(1.0 / Hs, zs)
        d.append(c_light * (1.0 + zi) * integral)
    return np.array(d)



def pick_column(df, candidates, what):
    """
    Pick the first column in `candidates` that exists in df.
    If none exist, raise with a helpful message.
    """
    for name in candidates:
        if name in df.columns:
            return df[name].values, name
    raise KeyError(
        f"Could not find any of the candidate columns for {what}: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def main():
    # -----------------------------
    # Load Pantheon+ SN data
    # -----------------------------
    df = pd.read_csv(PANTHEON_PATH, sep=r"\s+", comment="#")
    print("Pantheon+ columns:", list(df.columns))

    # Try to guess reasonable column names for z, mu, mu_err
    z_vals, z_col = pick_column(
        df,
        ["zHD", "zCMB", "z", "zHEL"],
        "redshift"
    )

    mu_obs, mu_col = pick_column(
        df,
        ["MU_SH0ES", "MU", "DISTMOD", "MU_SK", "m_b_corr"],
        "distance modulus μ"
    )

    # For error, some files may have MUERR_SH0ES, MUERR, or something similar;
    # if no error column exists, we can set a dummy small error just to make plots.
    try:
        mu_err, muerr_col = pick_column(
            df,
            ["MU_SH0ES_ERR_DIAG", "MUERR_SH0ES", "MUERR", "MUERR_SK", "DISTMOD_ERR"],
            "distance modulus error"
        )
    except KeyError as e:
        print("WARNING: Could not find a μ error column. Using zeros for mu_err.")
        print(e)
        mu_err = np.zeros_like(mu_obs)

    print(f"Using z column:  {z_col}")
    print(f"Using μ column:  {mu_col}")

    # -----------------------------
    # Compute model distance moduli
    # -----------------------------
    dL_lcdm = dL_LCDM(z_vals)
    mu_lcdm = 5.0 * np.log10(dL_lcdm) + 25.0

    dL_ccbh = dL_CCBH(z_vals)
    mu_ccbh = 5.0 * np.log10(dL_ccbh) + 25.0

    # -----------------------------
    # Plot 1: μ(z) curves
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.errorbar(z_vals, mu_obs, yerr=mu_err, fmt=".", label=f"Pantheon+ ({mu_col})")
    plt.plot(z_vals, mu_lcdm, label="ΛCDM")
    plt.plot(z_vals, mu_ccbh, label="CCBH")
    plt.xlabel("z")
    plt.ylabel("Distance Modulus μ")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title("Pantheon+ SN Distance Modulus: ΛCDM vs CCBH")
    plt.savefig("chains/sn_mu_lcdm_vs_ccbh.png", bbox_inches="tight")
    print("Saved chains/sn_mu_lcdm_vs_ccbh.png")

    # -----------------------------
    # Plot 2: Residuals μ_obs – μ_LCDM
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.errorbar(z_vals, mu_obs - mu_lcdm, yerr=mu_err, fmt=".", label="μ_obs − μ_ΛCDM")
    plt.axhline(0.0, color="k", ls="--", alpha=0.5)
    plt.xlabel("z")
    plt.ylabel("Residual (mag)")
    plt.title("Pantheon+ Residuals: SN − ΛCDM")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig("chains/sn_residuals_lcdm.png", bbox_inches="tight")
    print("Saved chains/sn_residuals_lcdm.png")

    # -----------------------------
    # Plot 3: Residuals μ_obs – μ_CCBH
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.errorbar(z_vals, mu_obs - mu_ccbh, yerr=mu_err, fmt=".", label="μ_obs − μ_CCBH")
    plt.axhline(0.0, color="k", ls="--", alpha=0.5)
    plt.xlabel("z")
    plt.ylabel("Residual (mag)")
    plt.title("Pantheon+ Residuals: SN − CCBH")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig("chains/sn_residuals_ccbh.png", bbox_inches="tight")
    print("Saved chains/sn_residuals_ccbh.png")

    # -----------------------------
    # Plot 4: Δμ = μ_CCBH − μ_LCDM
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(z_vals, mu_ccbh - mu_lcdm, ".", label="μ_CCBH − μ_ΛCDM")
    plt.axhline(0.0, color="k", ls="--", alpha=0.5)
    plt.xlabel("z")
    plt.ylabel("Δμ (mag)")
    plt.title("Model Difference: CCBH − ΛCDM")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig("chains/sn_delta_mu_ccbh_vs_lcdm.png", bbox_inches="tight")
    print("Saved chains/sn_delta_mu_ccbh_vs_lcdm.png")


if __name__ == "__main__":
    main()
