"""
Gravitational‑wave population and waveform tests for the CCBH model.

Black holes that grow by coupling to the cosmic expansion will have
smaller masses in the past than those observed today.  This could
leave imprints on the gravitational‑wave signals detected by LIGO/Virgo
and LISA.  While the predicted mass growth over the duration of an
individual GW event is minuscule, the integrated effect across
cosmic time may be observable in population statistics.  These tests
implement simple diagnostics inspired by the roadmap:

1. **Stellar BH birth masses:** Given observed merger component
   masses and redshifts for a few LIGO/Virgo events, compute the
   inferred birth masses assuming a coupling exponent k_low≈0.5 in
   the 10–50 M⊙ mass range.  The birth mass is M_birth = M_obs / a^k,
   where a=1/(1+z).  The birth masses should exceed the
   neutron star upper limit (~2.3 M⊙) for all events, ensuring the
   coupling does not force stellar BHs to originate from unphysical
   progenitors.

2. **LISA chirp‑mass redshift trend:** Simulate a small ensemble of
   supermassive BH binary mergers at redshifts z=0.5 and z=3 with
   present‑day masses 10⁶–10⁷ M⊙.  Under the CCBH hypothesis with
   k_max≈3 the intrinsic masses at merger scale as M_birth = M_obs / a^3.
   We verify that the average chirp mass of high‑z events is
   significantly smaller than that of low‑z events (>factor 2), a
   qualitative prediction that future LISA data could test.

3. **EMRI phase drift:** Estimate the fractional mass change of a
   10⁶ M⊙ SMBH over a one‑year inspiral due to coupling.  Even with
   k_max=3 the fractional change Ṁ/M ≈ k H0 is ~6×10⁻¹⁸ s⁻¹, leading
   to a relative mass change of <10⁻¹⁰ per year.  We assert that this
   effect is negligible, implying that standard GR waveforms remain
   valid for CCBH in the EMRI regime.
"""

import numpy as np
import pytest


def birth_mass(m_obs: float, z: float, k_eff: float) -> float:
    """Compute the birth (pre‑growth) mass of a BH given its observed mass and redshift.

    The cosmological scale factor at merger is a = 1/(1+z).  If the
    effective coupling exponent governing the BH mass growth in the
    relevant mass range is k_eff, then the birth mass is M_birth =
    M_obs / a^k_eff.  A value k_eff=0 corresponds to no coupling.
    """
    a = 1.0 / (1.0 + z)
    return m_obs / (a ** k_eff)


def test_stellar_bh_birth_masses():
    """Check that inferred birth masses of LIGO BHs exceed 2 M⊙.

    We take a few representative LIGO/Virgo BH mergers with component
    masses and redshifts (z ≲ 1) from O1–O3 runs.  Using an effective
    coupling k_low=0.5 appropriate for 10–50 M⊙ BHs we compute the
    birth masses and assert they remain above 3 M⊙.  Smaller birth
    masses would imply implausible progenitors, disfavoured by
    astrophysical constraints.
    """
    # Sample (m1, m2, z) for a few events: GW150914, GW170104, GW190521
    events = [
        (36.0, 29.0, 0.09),
        (50.0, 30.0, 0.20),
        (85.0, 66.0, 0.50),
    ]
    k_low = 0.5
    for m1, m2, z in events:
        for m in (m1, m2):
            m_birth = birth_mass(m, z, k_low)
            # Require birth mass above 3 solar masses
            assert m_birth > 3.0, (
                f"Inferred birth mass {m_birth:.1f} M⊙ for component {m} M⊙ at z={z}"
                " is too low; indicates overly strong coupling."
            )


def test_lisa_chirp_mass_trend():
    """Ensure high‑z SMBH mergers have smaller intrinsic masses than low‑z ones.

    We simulate two ensembles of BH binaries with present‑day total
    masses uniformly distributed between 10⁶ and 10⁷ M⊙.  The low‑z
    ensemble is placed at z=0.5 (a=0.667) and the high‑z ensemble at
    z=3 (a=0.25).  We assume the same k_max=3 coupling applies to
    these supermassive BHs.  We compute the chirp mass (M_chirp =
    (m1 m2)^{3/5} / (m1 + m2)^{1/5}) for each ensemble and compare
    their medians.  The median chirp mass at z=3 should be at least
    twice smaller than at z=0.5, reflecting the significant growth
    expected between those epochs.
    """
    rng = np.random.default_rng(seed=54321)
    # Generate 50 random total masses and mass ratios for z=0.5 and z=3
    n = 50
    # total observed mass (today) distribution 10^6–10^7 M⊙
    m_tot = 10 ** rng.uniform(6, 7, size=n)
    # Random mass ratios q in [0.3, 1]
    q = rng.uniform(0.3, 1.0, size=n)
    # Low‑z (z=0.5) intrinsic masses
    z_low = 0.5
    a_low = 1.0 / (1.0 + z_low)
    k_max = 3.0
    m_tot_birth_low = m_tot / (a_low ** k_max)
    m1_low = m_tot_birth_low * q / (1.0 + q)
    m2_low = m_tot_birth_low / (1.0 + q)
    chirp_low = (m1_low * m2_low) ** (3.0 / 5.0) / (m1_low + m2_low) ** (1.0 / 5.0)
    # High‑z (z=3) intrinsic masses
    z_high = 3.0
    a_high = 1.0 / (1.0 + z_high)
    m_tot_birth_high = m_tot / (a_high ** k_max)
    m1_high = m_tot_birth_high * q / (1.0 + q)
    m2_high = m_tot_birth_high / (1.0 + q)
    chirp_high = (m1_high * m2_high) ** (3.0 / 5.0) / (m1_high + m2_high) ** (1.0 / 5.0)
    median_low = np.median(chirp_low)
    median_high = np.median(chirp_high)
    # Expect the high‑z median to be at least 2× smaller than low‑z median
    ratio = median_low / median_high
    assert ratio >= 2.0, (
        f"Median chirp mass ratio low/high = {ratio:.2f} < 2.0;"
        " high‑z mergers should have significantly smaller intrinsic masses."
    )


def test_emri_phase_drift_is_negligible():
    """Check that BH mass growth over an EMRI timescale is tiny.

    For a 10⁶ M⊙ SMBH at z=0 the fractional mass growth rate is k H0.
    With k≈3 and H0≈70 km/s/Mpc (2.3×10⁻¹⁸ s⁻¹) the rate is
    ~6×10⁻¹⁸ s⁻¹.  Over one year (≈3.15×10⁷ s) the fractional change
    ΔM/M ≈ 2×10⁻¹⁰.  Such a small change is far below LISA's
    sensitivity.  We compute this quantity and assert it falls below
    10⁻⁹ to be conservative.
    """
    H0 = 70.0  # km/s/Mpc
    # Convert H0 to s⁻¹: 1 km/s/Mpc = 3.24078×10⁻²⁰ s⁻¹
    H0_SI = H0 * 3.24078e-20
    k = 3.0
    rate = k * H0_SI  # fractional mass change per second
    year_seconds = 365.25 * 24 * 3600
    delta_M_over_M = rate * year_seconds
    assert delta_M_over_M < 1e-9, (
        f"Fractional mass change over one year {delta_M_over_M:.2e} exceeds expected threshold;"
        " EMRI waveforms could be affected."
    )