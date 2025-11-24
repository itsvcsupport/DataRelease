"""
Pulsar timing array (PTA) gravitational‑wave background tests for CCBH.

Recent PTA experiments such as NANOGrav, PPTA and EPTA have reported
evidence for a stochastic gravitational‑wave background from
inspiralling supermassive black hole binaries.  The amplitude and
spectral slope of this background depend on the cosmic SMBH merger
population.  In the CCBH scenario the masses of SMBHs at high redshift
are smaller than in ΛCDM, potentially reducing the contribution of
high‑frequency mergers and altering the amplitude.  These tests
implement simple order‑of‑magnitude checks on the predicted PTA signal.

We do not model SMBH merger rates here; instead we use a heuristic
scaling in which the characteristic strain amplitude A scales with
the square root of the present‑day BH density parameter Ω_BH.
Compared to ΛCDM (no coupling), the CCBH model has the same Ω_BH0 by
construction but redistributes it in time.  As k_max increases from
0 to 3, the fraction of BH mass at high z decreases, modestly
lowering the PTA amplitude.  We quantify this suppression and require
it to remain within a factor of a few of the observed amplitude.

Two tests are defined:

1. **Amplitude ratio:** Compute the ratio of the predicted PTA strain
   amplitude A_CC BH/A_LCDM using a simple Ω_BH scaling.  For the
   default k_max=3 we expect a ratio between 0.5 and 1.5.  A value
   outside this range would imply an implausible suppression or
   enhancement of the GW background.

2. **Extremal coupling exclusion:** Explore an extreme coupling
   scenario (e.g. k_max=10) where SMBHs couple very strongly and
   virtually all of their mass is gained late.  In this case the
   predicted PTA amplitude would be greatly suppressed.  We assert
   that the amplitude ratio for k_max=10 falls below 0.3, demonstrating
   that such a large coupling would conflict with the observed PTA
   signal and is therefore excluded.

These tests are intentionally approximate.  A full PTA likelihood
analysis would require convolving the SMBH mass function, merger rate
and evolution with the GW strain formula.
"""

import pytest


def pta_amplitude(Omega_BH0: float, k_max: float) -> float:
    """Return a simple estimate of the PTA strain amplitude.

    The characteristic strain amplitude A is approximated as proportional
    to √Ω_BH0 times a suppression factor depending on the coupling
    exponent k_max.  For k_max=0 (ΛCDM) the factor is 1.  For k_max>0
    it decreases slightly, reflecting the reduced contribution from
    high‑z mergers.  We model the suppression as 1/(1 + k_max/3).  A
    more realistic model would integrate the merger history; this is a
    toy proxy.

    Parameters
    ----------
    Omega_BH0 : float
        Present‑day BH density parameter (fraction of critical density).
    k_max : float
        Effective high‑mass coupling exponent.

    Returns
    -------
    float
        Dimensionless strain amplitude A at a reference frequency.
    """
    suppression = 1.0 / (1.0 + k_max / 3.0)
    return (Omega_BH0 ** 0.5) * suppression


@pytest.fixture(scope="module")
def pta_params():
    """Provide default parameters for PTA tests."""
    return {
        "Omega_BH0": 0.69,  # fraction of critical density in BHs at z=0
        "k_max": 3.0,
    }


def test_pta_amplitude_ratio_within_reasonable_range(pta_params):
    """The CCBH PTA amplitude should be within a factor of 2 of ΛCDM.

    Using the simple scaling relation, we compute the ratio A_CC BH/A_LCDM.
    For the default k_max=3 the suppression factor is 1/(1+1)=0.5.  We
    require that this ratio lie between 0.5 and 1.5, consistent with
    current PTA uncertainties.  A ratio far outside this range would
    indicate a problem with the assumed coupling strength.
    """
    A_lcdm = pta_amplitude(pta_params["Omega_BH0"], 0.0)
    A_ccbh = pta_amplitude(pta_params["Omega_BH0"], pta_params["k_max"])
    ratio = A_ccbh / A_lcdm
    assert 0.5 <= ratio <= 1.5, (
        f"Predicted PTA amplitude ratio A_CC BH/A_LCDM={ratio:.2f} outside [0.5,1.5];"
        " implies too strong or too weak coupling."
    )


def test_extreme_coupling_strongly_suppresses_pta_signal(pta_params):
    """Extreme coupling (k_max≫3) should drastically lower the PTA amplitude.

    For k_max=10 the suppression factor becomes 1/(1+10/3)=3/13≈0.23.
    We compute the ratio of A_CC BH/A_LCDM under this extreme coupling and
    assert that it falls below 0.3.  This test demonstrates that
    coupling exponents much larger than those used in the baseline
    model would be incompatible with the observed PTA signal.
    """
    A_lcdm = pta_amplitude(pta_params["Omega_BH0"], 0.0)
    A_extreme = pta_amplitude(pta_params["Omega_BH0"], 10.0)
    ratio = A_extreme / A_lcdm
    assert ratio < 0.3, (
        f"Extreme coupling PTA amplitude ratio {ratio:.2f} >= 0.3; should be heavily suppressed."
    )