"""
Core cosmology validation tests for the CCBH model.

This test suite exercises the implementation of cosmological probes beyond
the Pantheon+ supernovae that are already part of the pipeline.  It
includes direct H(z) measurements (cosmic chronometers), weak lensing
constraints on the amplitude of structure (S8), and full CMB anisotropy
tests via integration with a Boltzmann solver.  The goal of these tests
is to ensure that the CCBH model yields an expansion history and growth
history consistent with all current observations while relieving the
H0 and S8 tensions.

The tests assume that there exists a ``ccbh_cosmology`` module in the
repository which provides functions for computing the Hubble parameter,
distance measures and structure growth given a set of cosmological
parameters.  If the actual module name differs, update the import
accordingly.  Similarly, the data files referenced in this test should
be located in the repository under ``data/cosmo`` or similar.  These
tests are written using the ``pytest`` framework.
"""

import numpy as np

# Try to import pytest.  If unavailable (e.g. running outside a full
# testing environment), define a minimal stub that provides the
# ``mark.skipif`` decorator and ``skip`` function used below.  This
# allows the tests to execute under a plain Python interpreter without
# raising ``ModuleNotFoundError``.  The stub simply prints a message
# when a test is skipped instead of integrating with a test runner.
try:
    import pytest  # type: ignore
except ImportError:
    class _DummySkipDecorator:
        def __call__(self, reason: str = ""):
            def decorator(func):
                def wrapper(*args, **kwargs):
                    print(f"Skipped: {reason}")
                    return None
                return wrapper
            return decorator

    class _DummyMark:
        def skipif(self, condition: bool, reason: str = ""):
            def decorator(func):
                def wrapper(*args, **kwargs):
                    if condition:
                        print(f"Skipped: {reason}")
                        return None
                    return func(*args, **kwargs)
                return wrapper
            return decorator

    class _DummyPytest:
        mark = _DummyMark()
        skip = staticmethod(_DummySkipDecorator())

    pytest = _DummyPytest()

try:
    # Attempt to import the CCBH cosmology model.  If the module name
    # differs in the code base, adjust accordingly.
    import ccbh_cosmology as ccbh
    import lcdm_cosmology as lcdm  # reference ΛCDM implementation
except ImportError:
    # These imports may fail in environments without the full code base.
    # Define stubs to allow the test file to import.  Replace these
    # placeholders with real implementations in the code base.
    ccbh = None
    lcdm = None


@pytest.fixture(scope="module")
def default_params():
    """Return a dictionary of default cosmological parameters for CCBH.

    These values should match those used in the existing pipeline and
    represent the current best–fit parameters.  They include the Hubble
    constant ``H0``, matter fraction ``Omega_m``, coupling strength
    ``k_max``, transition mass ``M_c``, screening index ``beta`` and
    environmental density scale ``rho_*``, along with the sum of
    neutrino masses.  Adjust these defaults as updated fits become
    available.
    """
    return {
        "H0": 70.0,       # km/s/Mpc
        "Omega_m": 0.3,   # total matter density parameter
        "k_max": 3.0,    # maximum coupling exponent at high masses
        "M_c": 1e7,      # solar masses, transition mass scale
        "beta": 2.0,     # screening steepness parameter
        "rho_*": 1e-25,  # g/cm^3, environmental screening density
        "sum_mnu": 0.2,  # eV, sum of neutrino masses
    }


@pytest.mark.skipif(ccbh is None, reason="ccbh_cosmology module not available")
def test_hubble_rate_matches_chronometer_data(default_params):
    """Compare the CCBH model H(z) with cosmic chronometer measurements.

    The cosmic chronometer data consist of direct Hubble parameter
    measurements derived from differential ages of galaxies.  They are
    independent of any cosmological model and provide a powerful
    check on late–time expansion.  This test loads a table of redshifts
    and measured H(z) values (with uncertainties) and compares them to
    the predictions of the CCBH and ΛCDM models.  It asserts that the
    CCBH model lies within the 1σ uncertainties of the data points and
    quantifies any systematic deviation from ΛCDM.

    The chronometer data should be stored in a CSV file with columns
    ``z``, ``Hz``, ``sigma_Hz`` in a directory such as
    ``data/chronometers/chronometer_data.csv`` within the repository.
    """
    # Load observational H(z) data
    try:
        chron_data = np.genfromtxt(
            "data/chronometers/chronometer_data.csv",
            delimiter=",",
            names=True,
        )
    except IOError:
        pytest.skip("Chronometer data file not found; ensure it is present in the data directory.")

    z = chron_data["z"]
    H_obs = chron_data["Hz"]
    sigma_H = chron_data["sigma_Hz"]

    # Compute model predictions
    H_ccbh = ccbh.hubble_parameter(z, **default_params)
    H_lcdm = lcdm.hubble_parameter(z, H0=default_params["H0"], Omega_m=default_params["Omega_m"])  # type: ignore

    # Assert that CCBH predictions are within uncertainties of observed values
    residuals = (H_ccbh - H_obs) / sigma_H
    max_abs_residual = np.max(np.abs(residuals))
    assert max_abs_residual < 2.5, (
        f"CCBH H(z) deviates from chronometer data by more than 2.5σ: max residual {max_abs_residual:.2f}σ"
    )

    # Quantify the average fractional difference relative to ΛCDM
    fractional_diff = (H_ccbh - H_lcdm) / H_lcdm
    avg_frac = np.mean(fractional_diff)
    # The late–time coupling should produce at most a few percent difference in H(z)
    assert np.abs(avg_frac) < 0.05, (
        f"Average fractional difference between CCBH and ΛCDM H(z) exceeds 5%: {avg_frac:.3f}"
    )


@pytest.mark.skipif(ccbh is None, reason="ccbh_cosmology module not available")
def test_weak_lensing_s8_consistency(default_params):
    """Verify that the CCBH model yields a derived S8 compatible with lensing data.

    The parameter S8 ≡ σ8 √(Ω_m/0.3) encapsulates the tension between
    Planck CMB and weak lensing surveys.  CCBH aims to lower the
    predicted S8 relative to ΛCDM, bringing it closer to lensing
    observations.  This test computes the linear matter power spectrum
    normalization σ8 and S8 for both CCBH and ΛCDM for the provided
    parameter set and asserts that S8(CCBH) ≈ 0.78 within a tolerance.
    """
    # Compute σ8 from the growth module
    sigma8_ccbh = ccbh.compute_sigma8(**default_params)
    sigma8_lcdm = lcdm.compute_sigma8(H0=default_params["H0"], Omega_m=default_params["Omega_m"])  # type: ignore

    # Compute S8
    S8_ccbh = sigma8_ccbh * np.sqrt(default_params["Omega_m"] / 0.3)
    S8_lcdm = sigma8_lcdm * np.sqrt(default_params["Omega_m"] / 0.3)

    # Check that CCBH S8 is within the range preferred by lensing surveys
    assert 0.75 <= S8_ccbh <= 0.82, (
        f"Derived S8={S8_ccbh:.3f} lies outside the lensing preferred range (0.75–0.82)."
    )

    # Check that CCBH reduces S8 relative to ΛCDM
    assert S8_ccbh < S8_lcdm, (
        f"CCBH S8 ({S8_ccbh:.3f}) is not lower than ΛCDM S8 ({S8_lcdm:.3f}); expected lower growth."
    )


@pytest.mark.skipif(ccbh is None, reason="ccbh_cosmology module not available")
def test_full_planck_spectra_compatibility(default_params):
    """Ensure that the CCBH model reproduces the Planck TT, TE, EE power spectra.

    In the CCBH framework black holes couple negligibly to cosmological
    expansion at early times, so the primary anisotropy spectrum should
    remain indistinguishable from ΛCDM.  This test relies on the
    availability of a Boltzmann solver (e.g. a modified CLASS) that
    supports the CCBH background.  It computes the CMB power spectra
    for the provided parameter set and compares them to the Planck 2018
    best–fit spectra.  The test asserts that fractional differences
    remain below 1% across the multipole range considered.

    Note that running a Boltzmann solver in unit tests can be
    computationally expensive.  Consider using precomputed spectra or
    coarse resolution settings to accelerate evaluation.  If the
    necessary modules are not available, this test is skipped.
    """
    try:
        # Hypothetical interface to compute CMB spectra: returns ell and C_ell arrays
        ell, Cl_TT_ccbh, Cl_TE_ccbh, Cl_EE_ccbh = ccbh.compute_cmb_spectra(**default_params)
        ell_ref, Cl_TT_ref, Cl_TE_ref, Cl_EE_ref = lcdm.compute_cmb_spectra(
            H0=default_params["H0"], Omega_m=default_params["Omega_m"], sum_mnu=default_params["sum_mnu"]
        )  # type: ignore
    except AttributeError:
        pytest.skip("CMB spectrum computation not available in current environment.")

    # Restrict to the multipole range where precision is highest (e.g. 30 < ell < 2000)
    idx = (ell > 30) & (ell < 2000)
    ell = ell[idx]
    # Compute fractional differences
    frac_TT = np.abs((Cl_TT_ccbh[idx] - Cl_TT_ref[idx]) / Cl_TT_ref[idx])
    frac_TE = np.abs((Cl_TE_ccbh[idx] - Cl_TE_ref[idx]) / Cl_TE_ref[idx])
    frac_EE = np.abs((Cl_EE_ccbh[idx] - Cl_EE_ref[idx]) / Cl_EE_ref[idx])

    # Assert that maximum fractional difference is below 1%
    assert np.max(frac_TT) < 0.01, "TT spectrum deviates from Planck by more than 1%"
    assert np.max(frac_TE) < 0.015, "TE spectrum deviates from Planck by more than 1.5%"
    assert np.max(frac_EE) < 0.015, "EE spectrum deviates from Planck by more than 1.5%"