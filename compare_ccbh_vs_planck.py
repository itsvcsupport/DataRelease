import numpy as np
import matplotlib.pyplot as plt

import ccbh_theory as c  # your CCBH background code


# --------- helpers --------- #

def omega_m_from_cosmo(H0, ombh2, omch2):
    h = H0 / 100.0
    return (ombh2 + omch2) / h**2


def H_lcdm_of_z(z, H0, Omega_m, Omega_r=0.0):
    """
    Flat LCDM: H(z) = H0 * sqrt( Omega_r (1+z)^4 + Omega_m (1+z)^3 + (1-Omega_m-Omega_r) )
    """
    zp1 = 1.0 + z
    return H0 * np.sqrt(Omega_r * zp1**4 + Omega_m * zp1**3 + 1.0 - Omega_m - Omega_r)


# --------- main comparison --------- #

def main():
    # --------------------------------------------------
    # 1) Planck ΛCDM baseline (edit if you want)
    # --------------------------------------------------
    H0_pl    = 67.4      # km/s/Mpc
    ombh2_pl = 0.0224
    omch2_pl = 0.1200
    Omega_m_pl = omega_m_from_cosmo(H0_pl, ombh2_pl, omch2_pl)

    # --------------------------------------------------
    # 2) CCBH parameters — EDIT THESE TO YOUR ACTUAL NUMBERS
    # --------------------------------------------------
    # These should be your best-fit / tuned CCBH parameters from SN+DESI(+Planck distance).
    # Just change the values below and re-run to see updated H(z) and ratios.

    H0_cc    = 70.8      # example: 70 km/s/Mpc
    ombh2_cc = 0.0224    # example: similar to Planck
    omch2_cc = 0.1124    # example: similar to Planck

    k1       = .99       # example values; replace with your actual best-fit
    k2       = 2.58
    k3       = 4.23
    z_t1     = 5.55
    z_t2     = .128

    # --------------------------------------------------
    # 3) Compute CCBH background summary
    # --------------------------------------------------
    Omega_m_cc = omega_m_from_cosmo(H0_cc, ombh2_cc, omch2_cc)
    Omega_BH0_cc = c.omega_ccbh0(H0_cc, ombh2_cc, omch2_cc,
                                  k1, k2, k3, z_t1, z_t2)

    # Print summaries
    print("=== Planck ΛCDM baseline ===")
    print(f"H0       = {H0_pl:.3f}")
    print(f"Omega_m  = {Omega_m_pl:.4f}")
    print(f"ombh2    = {ombh2_pl:.5f}")
    print(f"omch2    = {omch2_pl:.5f}")
    print()

    print("=== CCBH (manual parameters) ===")
    print(f"H0          = {H0_cc:.3f}")
    print(f"Omega_m     = {Omega_m_cc:.4f}")
    print(f"Omega_BH0   = {Omega_BH0_cc:.4f}")
    print(f"k1,k2,k3    = {k1:.3f}, {k2:.3f}, {k3:.3f}")
    print(f"z_t1, z_t2  = {z_t1:.3f}, {z_t2:.3f}")
    print()

    # --------------------------------------------------
    # 4) H(z) comparison
    # --------------------------------------------------
    z = np.linspace(0, 3, 300)

    H_lcdm = H_lcdm_of_z(z, H0_pl, Omega_m_pl)

    H_ccbh = np.array([
        c.H_of_z(zi, H0_cc, ombh2_cc, omch2_cc,
                 k1, k2, k3, z_t1, z_t2)
        for zi in z
    ])

    # Plot H(z)
    plt.figure()
    plt.plot(z, H_lcdm, label="ΛCDM (Planck baseline)")
    plt.plot(z, H_ccbh, label="CCBH (manual params)")
    plt.xlabel("z")
    plt.ylabel("H(z) [km/s/Mpc]")
    plt.legend()
    plt.title("Expansion history: Planck ΛCDM vs CCBH")
    plt.grid(True, alpha=0.3)
    plt.savefig("chains/compare_Hz_planck_vs_ccbh.png", bbox_inches="tight")
    print("Saved H(z) comparison to chains/compare_Hz_planck_vs_ccbh.png")

    # Plot H_CCBH / H_LCDM
    plt.figure()
    plt.plot(z, H_ccbh / H_lcdm)
    plt.xlabel("z")
    plt.ylabel("H_CCBH(z) / H_ΛCDM(z)")
    plt.title("Relative deviation in H(z)")
    plt.axhline(1.0, color="k", linestyle="--", alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.savefig("chains/compare_Hz_ratio_planck_vs_ccbh.png", bbox_inches="tight")
    print("Saved H(z) ratio to chains/compare_Hz_ratio_planck_vs_ccbh.png")


if __name__ == "__main__":
    main()
