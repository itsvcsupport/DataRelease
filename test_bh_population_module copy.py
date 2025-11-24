"""
Tests for the BH population evolution and environmental screening module.

The CCBH model includes an evolving population of black holes whose
energy density ρ_BH(a) contributes to the cosmic expansion.  Two
features are critical to the phenomenology:

1. **Negligible early‑universe contribution:** Before the bulk of
   black holes form (z ≳ 10), ρ_BH should be tiny compared to the
   total energy density.  This ensures that BBN and CMB physics are
   unaffected.  The scaling ρ_BH ∝ a^{k_max−3} means that for k_max≈3
   the BH energy density remains roughly constant relative to matter at
   late times and drops rapidly at early times.  We test that the
   fractional BH contribution at a=10⁻⁴ and a=10⁻⁶ is well below 10⁻⁶.

2. **Environmental screening:** The coupling is suppressed in high
   density environments such as galaxy clusters.  The screening factor
   S_env = [1 + (ρ_env/ρ_*)^β]^{-1} interpolates between zero (fully
   screened) when ρ_env≫ρ_* and unity (unscreened) when ρ_env≪ρ_*.
   We test that S_env is ≪1 for cluster densities and ≈1 for cosmic
   voids.

These tests implement simple analytic approximations consistent with
the qualitative description in the CCBH literature.  They do not
replace the full implementation in ``ccbh_theory`` but serve as
regression checks.
"""

import numpy as np
import pytest


def bh_energy_fraction(a: float, k_max: float) -> float:
    """Return an approximate fraction of total energy in black holes at scale factor a.

    We model the total energy density as the sum of matter scaling as
    a^{-3} and BH energy scaling as a^{k_max−3}.  The BH fraction is
    then Ω_BH(a) = a^{k_max−3} / [a^{-3} + a^{k_max−3}].  This toy
    expression captures the qualitative behaviour: for k_max≈3 the BH
    component remains constant relative to matter (Ω_BH≈const) at late
    times and is suppressed at early times (a≪1).  For k_max=0 it
    reduces to the constant dark energy fraction.

    Parameters
    ----------
    a : float
        Scale factor (1/(1+z)).
    k_max : float
        Effective coupling exponent at high masses.

    Returns
    -------
    float
        Fraction of the total energy density in BHs at scale factor a.
    """
    # Avoid division by zero
    if a <= 0:
        return 0.0
    term_bh = a ** (k_max - 3.0)
    term_m = a ** (-3.0)
    return term_bh / (term_m + term_bh)


def screening_factor(rho_env: float, rho_star: float, beta: float) -> float:
    """Compute the environmental screening factor S_env.

    S_env = 1 / [1 + (ρ_env/ρ_*)^β].  High environmental densities
    strongly suppress coupling (S_env→0), while low densities leave the
    BH fully coupled (S_env→1).

    Parameters
    ----------
    rho_env : float
        Ambient density surrounding the BH (g/cm^3).
    rho_star : float
        Characteristic screening density at which coupling is reduced by
        half.
    beta : float
        Steepness of the screening transition.

    Returns
    -------
    float
        Screening factor between 0 and 1.
    """
    if rho_env <= 0:
        return 1.0
    return 1.0 / (1.0 + (rho_env / rho_star) ** beta)


def test_bh_energy_fraction_is_negligible_at_early_times():
    """Ensure BH energy density is tiny during the radiation epoch.

    Using the toy Ω_BH(a) relation we compute the BH fraction at
    a=10⁻⁴ (z≈10,000) and a=10⁻⁶ (z≈1,000,000).  For k_max≈3 the BH
    fraction should be below 10⁻⁶, guaranteeing no impact on BBN or the
    CMB.  A looser tolerance of 10⁻⁴ is acceptable for k_max≈2.
    """
    # Test for k_max values around the best‑fit (3) and a lower value (2)
    for k_max in (2.0, 3.0):
        f1 = bh_energy_fraction(1e-4, k_max)
        f2 = bh_energy_fraction(1e-6, k_max)
        # For k=3 the suppression is extreme; for k=2 it's milder but still tiny
        tol = 1e-6 if k_max >= 3.0 else 1e-4
        assert f1 < tol and f2 < tol, (
            f"BH energy fraction {f1:.1e} or {f2:.1e} at early times exceeds tolerance {tol:.1e}" 
            f"for k_max={k_max}."
        )


def test_environmental_screening_behaviour():
    """Check that the screening factor behaves correctly with density.

    We evaluate S_env for representative environmental densities: a
    cluster core (ρ_env≈10⁻²3 g/cm³), a Milky Way halo
    (ρ_env≈10⁻²7 g/cm³) and an intergalactic void (ρ_env≈10⁻³0 g/cm³).
    Using default screening parameters ρ_*≈10⁻²5 g/cm³ and β≈2 we
    expect S_env(cluster)≪1, S_env(halo)≈0.5 and S_env(void)≈1.  We
    verify these qualitative behaviours with loose numerical bounds.
    """
    rho_star = 1e-25  # g/cm^3
    beta = 2.0
    # Cluster environment: density ~1e-23 g/cm^3
    s_cluster = screening_factor(1e-23, rho_star, beta)
    # Halo environment: density ~1e-27 g/cm^3
    s_halo = screening_factor(1e-27, rho_star, beta)
    # Void environment: density ~1e-30 g/cm^3
    s_void = screening_factor(1e-30, rho_star, beta)
    assert s_cluster < 0.1, (
        f"Screening factor for cluster environment is too high: {s_cluster:.3f}; expected <0.1."
    )
    assert 0.3 < s_halo < 0.8, (
        f"Screening factor for halo environment {s_halo:.3f} outside expected 0.3–0.8."
    )
    assert s_void > 0.9, (
        f"Screening factor for void environment {s_void:.3f} should be ≈1."
    )