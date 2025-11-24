"""
Null ΛCDM consistency test for the CCBH model.

One of the most basic sanity checks of any extended cosmological model
is that it reduces exactly to the reference ΛCDM model when its extra
parameters are set to their null values.  For the CCBH framework this
corresponds to switching off the coupling of black holes to cosmic
expansion (k_low = k_max = 0) or, equivalently, taking the transition
mass M_c → ∞ so that no BH ever couples.  In this limit the CCBH
Hubble rate should match the standard flat ΛCDM relation to within
machine precision.

This test compares the Hubble function computed using the CCBH model
with ``k_max`` set to zero against the analytic ΛCDM expression over
a range of redshifts.  It asserts that the fractional difference
between the two is <10⁻⁶ across the sampled redshifts.  A looser
tolerance can be adopted depending on numerical precision of the
implementation.

If a real CCBH implementation is unavailable, the test falls back to
the analytic forms defined in :mod:`test_cosmic_chronometers`.  In
that case the test is trivial but still useful as a template.
"""

import numpy as np
import pytest

try:
    import ccbh_cosmology as ccbh  # type: ignore
    import lcdm_cosmology as lcdm  # type: ignore
except ImportError:
    ccbh = None  # type: ignore
    lcdm = None  # type: ignore

# Import fallback Hubble functions from the chronometer test if available.
try:
    from .test_cosmic_chronometers import hubble_lcdm_fallback, hubble_ccbh_fallback
except Exception:
    # Define minimal fallback here
    def hubble_lcdm_fallback(z: np.ndarray, H0: float, Omega_m: float) -> np.ndarray:
        return H0 * np.sqrt(Omega_m * (1 + z) ** 3 + (1 - Omega_m))

    def hubble_ccbh_fallback(z: np.ndarray, H0: float, Omega_m: float, k_eff: float) -> np.ndarray:
        Omega_bh0 = 1.0 - Omega_m
        return H0 * np.sqrt(Omega_m * (1 + z) ** 3 + Omega_bh0 * (1 + z) ** (-k_eff))


@pytest.fixture(scope="module")
def null_params():
    """Return a dictionary of cosmological parameters corresponding to ΛCDM.

    By setting the coupling exponent k_max to zero and taking M_c to
    infinity (or equivalently leaving it unused), the CCBH model should
    collapse to ΛCDM.  The matter density Ω_m and Hubble constant H0
    match typical values used in the analysis.
    """
    return {
        "H0": 70.0,
        "Omega_m": 0.30,
        "k_max": 0.0,
        "M_c": np.inf,
        "beta": 2.0,
    }


def compute_hubble_ccbh_null(z: np.ndarray, params: dict) -> np.ndarray:
    """Evaluate the CCBH Hubble function in the null coupling limit.

    If a proper CCBH implementation exists, call it with k_max=0.  If
    not, use the fallback analytic expression.  Only H0 and Ω_m are
    relevant when k_max=0.
    """
    if ccbh is not None:
        return ccbh.hubble_parameter(z, **params)  # type: ignore
    return hubble_ccbh_fallback(z, params["H0"], params["Omega_m"], 0.0)


def compute_hubble_lcdm(z: np.ndarray, params: dict) -> np.ndarray:
    """Evaluate the reference ΛCDM Hubble function.

    Use the imported ``lcdm_cosmology`` module if available, otherwise
    revert to the analytic fallback.
    """
    if lcdm is not None:
        return lcdm.hubble_parameter(z, H0=params["H0"], Omega_m=params["Omega_m"])  # type: ignore
    return hubble_lcdm_fallback(z, params["H0"], params["Omega_m"])


def test_ccbh_reduces_to_lcdm(null_params):
    """Check that the CCBH H(z) matches ΛCDM when k_max=0.

    This test samples redshifts from 0 to 3 and computes the fractional
    difference between the CCBH and ΛCDM Hubble functions.  The
    difference should be consistent with numerical noise (<10⁻⁶).  A
    failure here would indicate that the implementation does not turn
    off coupling correctly when requested.
    """
    z_vals = np.linspace(0.0, 3.0, 13)
    H_ccbh = compute_hubble_ccbh_null(z_vals, null_params)
    H_lcdm = compute_hubble_lcdm(z_vals, null_params)
    # Avoid division by zero at z=-1 by restricting to z>=0 (already enforced)
    frac = (H_ccbh - H_lcdm) / H_lcdm
    max_frac = float(np.max(np.abs(frac)))
    assert max_frac < 1e-6, (
        f"CCBH H(z) does not reduce to ΛCDM in the null limit; max fractional difference {max_frac:.2e}"
    )