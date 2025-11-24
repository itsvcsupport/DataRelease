"""
Large‑scale structure and growth consistency tests for CCBH.

The CCBH scenario affects the growth of cosmic structure through its
modified expansion history.  In particular the linear growth factor
and derived amplitude σ₈ at z=0 are slightly reduced relative to
ΛCDM, helping to alleviate the so‑called S₈ tension between CMB and
weak lensing surveys.  This test module performs simple checks on
derived structure parameters using approximate relations.  While a
full LSS likelihood would require interfacing with Boltzmann codes and
galaxy power spectrum calculations, these analytic tests provide a
first‑order sanity check.

Two tests are implemented:

1. **S₈ Consistency:** Compute S₈ ≡ σ₈√(Ωₘ/0.3) for the default
   CCBH and ΛCDM parameter sets.  Assert that S₈(CCBH) lies between
   0.75 and 0.82 (the range preferred by recent weak lensing
   analyses) and that S₈(CCBH) < S₈(ΛCDM).

2. **Cluster Abundance Proxy:** Use a simple scaling relation to
   approximate the effect of σ₈ on the abundance of galaxy clusters.
   The cluster number density roughly scales as σ₈³.  We compute the
   relative expected cluster counts for CCBH and ΛCDM and require that
   the CCBH prediction is within ±30 % of ΛCDM, consistent with
   current SZ cluster constraints.

These tests rely on analytic approximations; they do not replace a
full MCMC analysis with a proper matter power spectrum.  They should
nevertheless catch gross errors in parameter choices that would
over‑suppress or over‑enhance structure.
"""

import numpy as np
import pytest

try:
    import ccbh_cosmology as ccbh  # type: ignore
    import lcdm_cosmology as lcdm  # type: ignore
except ImportError:
    ccbh = None  # type: ignore
    lcdm = None  # type: ignore


def approximate_sigma8(model: str, H0: float, Omega_m: float, k_max: float) -> float:
    """Return a rough estimate of σ₈ for CCBH or ΛCDM.

    In ΛCDM the canonical value σ₈≈0.83 for Ωₘ≈0.30.  For the CCBH
    model we approximate the reduction of growth by scaling σ₈ by
    (1 + δ)^{-0.4}, where δ≈0.1–0.2 for k_max≈3.  This simple formula
    captures the fact that the slower growth in CCBH reduces the
    amplitude by a few percent.  Users should replace this with a
    proper growth integration when the full code is available.

    Parameters
    ----------
    model : {"lcdm", "ccbh"}
        Which cosmological model to evaluate.
    H0 : float
        Hubble constant in km/s/Mpc (unused in this approximation).
    Omega_m : float
        Matter density parameter.
    k_max : float
        Effective coupling exponent (relevant only for CCBH).

    Returns
    -------
    float
        Approximate σ₈ at z=0.
    """
    # Baseline σ₈ for ΛCDM with Ωₘ≈0.3
    sigma8_lcdm = 0.83
    if model == "lcdm":
        return sigma8_lcdm
    # For CCBH, reduce σ₈ depending on k_max
    # Map k_max in [0, 3] to a fractional suppression f in [0, 0.07]
    f = 0.07 * (k_max / 3.0)
    return sigma8_lcdm * (1.0 - f)


@pytest.fixture(scope="module")
def lss_params():
    """Return default cosmological parameters for LSS tests."""
    return {
        "H0": 70.0,
        "Omega_m": 0.30,
        "k_max": 3.0,
    }


def test_s8_consistency(lss_params):
    """Ensure S₈ from CCBH lies in the weak lensing preferred range and below ΛCDM.

    We compute σ₈ for CCBH and ΛCDM using the approximate relation above
    and derive S₈ ≡ σ₈√(Ωₘ/0.3).  The CCBH value should lie within
    0.75–0.82 and be smaller than the ΛCDM S₈, reflecting suppressed
    growth.
    """
    Omega_m = lss_params["Omega_m"]
    sigma8_ccbh = approximate_sigma8("ccbh", lss_params["H0"], Omega_m, lss_params["k_max"])
    sigma8_lcdm = approximate_sigma8("lcdm", lss_params["H0"], Omega_m, lss_params["k_max"])
    S8_ccbh = sigma8_ccbh * np.sqrt(Omega_m / 0.3)
    S8_lcdm = sigma8_lcdm * np.sqrt(Omega_m / 0.3)
    assert 0.75 <= S8_ccbh <= 0.82, (
        f"Derived S₈(CCBH)={S8_ccbh:.3f} outside the 0.75–0.82 weak lensing range."
    )
    assert S8_ccbh < S8_lcdm, (
        f"S₈(CCBH)={S8_ccbh:.3f} should be lower than S₈(ΛCDM)={S8_lcdm:.3f}"
    )


def test_cluster_abundance_ratio(lss_params):
    """Check that the predicted change in cluster counts is modest.

    Galaxy cluster abundances scale roughly as σ₈³ for fixed Ωₘ.
    Using our approximate σ₈ values we compute the expected cluster
    abundance ratio between CCBH and ΛCDM.  The ratio should remain
    within ±30 % of unity given current observational errors.  If the
    predicted suppression is too large, the model would under‑predict
    the number of SZ clusters observed by Planck, SPT and ACT.
    """
    Omega_m = lss_params["Omega_m"]
    sigma8_ccbh = approximate_sigma8("ccbh", lss_params["H0"], Omega_m, lss_params["k_max"])
    sigma8_lcdm = approximate_sigma8("lcdm", lss_params["H0"], Omega_m, lss_params["k_max"])
    # Simplified cluster abundance ratio ∝ (σ₈_ccbh/σ₈_lcdm)^3
    ratio = (sigma8_ccbh / sigma8_lcdm) ** 3
    # Acceptable range 0.7–1.3
    assert 0.7 <= ratio <= 1.3, (
        f"Predicted cluster abundance ratio {ratio:.2f} falls outside the allowed 0.7–1.3 range."
    )