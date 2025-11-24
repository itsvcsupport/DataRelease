#!/usr/bin/env python
"""
CCBH MCMC - FAST VERSION with caching and vectorization
"""
import numpy as np
import pandas as pd
import emcee
import corner
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore')


class CCBHEmulatorFast:
    def __init__(self):
        self.c_km_s = 299792.458
        self.h_fid = 0.6766
        self.Omega_b_h2 = 0.02242
        self.Omega_r_h2 = 4.18e-5
        self.load_data()
        self.setup_grids()

    def load_data(self):
        """Load Pantheon+ subsample and DESI H0*r_s, Omega_m chain info."""
        # Pantheon+ - subsample for speed during MCMC
        df = pd.read_csv(
            "Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat",
            delim_whitespace=True,
            comment="#",
        )

        # Use every 3rd SN for MCMC (still 567 SNe)
        indices = np.arange(0, len(df), 3)

        self.z_sn = df["zHD"].values[indices]
        self.mu_obs = df["MU_SH0ES"].values[indices]
        self.mu_err = df["MU_SH0ES_ERR_DIAG"].values[indices]

        # Scale factor for chi2 (to account for subsampling)
        self.n_sn_full = len(df)
        self.n_sn_sample = len(indices)
        self.scale_factor = self.n_sn_full / self.n_sn_sample

        print(
            f"Using {self.n_sn_sample} SNe for MCMC (subsampled from {self.n_sn_full})"
        )
        print(
            "mu_err stats: min =",
            np.min(self.mu_err),
            "max =",
            np.max(self.mu_err),
            "n_zero_or_neg =",
            np.sum(self.mu_err <= 0),
        )

        # DESI H0*r_s, Omega_m from Cobaya chain
        chain = np.loadtxt(
            "C:/Users/WilliamKellett/cobaya/base/desi-bao-all/chain.1.txt"
        )
        weights = chain[:, 0]
        H0rs = chain[:, 18]
        omegam = chain[:, 3]

        self.desi_mean = np.array(
            [
                np.average(H0rs, weights=weights),
                np.average(omegam, weights=weights),
            ]
        )

        data = np.column_stack([H0rs, omegam])
        self.desi_cov = np.cov(data.T, aweights=weights)
        self.desi_invcov = np.linalg.inv(self.desi_cov)

        print(
            f"DESI: H0*rs={self.desi_mean[0]:.1f}, Omega_m={self.desi_mean[1]:.5f}"
        )

    def setup_grids(self):
        """Pre-compute integration grids in log(a)."""
        self.a_grid = np.logspace(-4, 0, 500)  # a from 1e-4 to 1
        self.ln_a_grid = np.log(self.a_grid)

    def k_eff(self, a, k1, k2, k3, a1, a2):
        """Piecewise k(a) = k1 (a<a1), k2 (a1<=a<a2), k3 (a>=a2)."""
        if a < a1:
            return k1
        elif a < a2:
            return k2
        else:
            return k3

    def compute_H_grid(self, H0, omegam, k1, k2, k3, a1, a2):
        """Compute H(a) on the pre-defined grid, with BH fluid turning on at a1."""
        Omega_r = self.Omega_r_h2 / self.h_fid**2
        Omega_BH_0 = 1.0 - omegam - Omega_r

        # If today's BH density would be negative, reject
        if Omega_BH_0 < 0:
            return None

        H_vals = []

        for a in self.a_grid:
            # No BH dark energy before a1
            if a < a1:
                Omega_BH_a = 0.0
            else:
                # Integrate BH evolution only from a1 up to current a
                mask = (self.a_grid >= a1) & (self.a_grid <= a)
                ln_a_sub = self.ln_a_grid[mask]

                if ln_a_sub.size < 2:
                    Omega_BH_a = Omega_BH_0
                else:
                    k_vals = np.array(
                        [
                            self.k_eff(a_p, k1, k2, k3, a1, a2)
                            for a_p in self.a_grid[mask]
                        ]
                    )
                    integrand_vals = k_vals - 3.0
                    integral = np.trapz(integrand_vals, ln_a_sub)
                    Omega_BH_a = Omega_BH_0 * np.exp(integral)

            Omega_m_a = omegam / a**3
            Omega_r_a = Omega_r / a**4

            H_squared = H0**2 * (Omega_m_a + Omega_r_a + Omega_BH_a)

            if H_squared <= 0 or not np.isfinite(H_squared):
                return None

            H_vals.append(np.sqrt(H_squared))

        return interp1d(
            self.a_grid,
            H_vals,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )

    def compute_r_s_fast(self, H_interp, omegam):
        """Compute sound horizon r_s using H(a) and Eisenstein & Hu-like z_d."""
        Omega_b = self.Omega_b_h2 / self.h_fid**2
        Omega_m = omegam

        b1 = (
            0.313
            * (Omega_m * self.h_fid**2) ** (-0.419)
            * (1 + 0.607 * (Omega_m * self.h_fid**2) ** 0.674)
        )
        b2 = 0.238 * (Omega_m * self.h_fid**2) ** 0.223
        z_d = (
            1291
            * (Omega_m * self.h_fid**2) ** 0.251
            / (1 + 0.659 * (Omega_m * self.h_fid**2) ** 0.828)
            * (1 + b1 * (Omega_b * self.h_fid**2) ** b2)
        )

        a_d = 1.0 / (1.0 + z_d)

        # Integrate from a=0 to a_d on the precomputed grid
        a_sub = self.a_grid[self.a_grid <= a_d]
        R_vals = 3.0 * Omega_b / (4.0 * self.Omega_r_h2 / self.h_fid**2) * a_sub
        c_s_vals = self.c_km_s / np.sqrt(3.0 * (1.0 + R_vals))
        H_vals = H_interp(a_sub)

        integrand_vals = c_s_vals / (a_sub**2 * H_vals)
        r_s = np.trapz(integrand_vals, a_sub)

        return r_s

    def compute_distances_fast(self, H_interp, z_array):
        """Compute distance moduli μ(z) for an array of redshifts."""
        distances = []

        for z in z_array:
            a = 1.0 / (1.0 + z)
            # Use a fresh dense linear grid from a(z) to 1.0
            a_sub = np.linspace(a, 1.0, 400)

            if len(a_sub) < 2:
                return None

            H_vals = H_interp(a_sub)
            if not np.all(np.isfinite(H_vals)):
                return None

            integrand_vals = self.c_km_s / (a_sub**2 * H_vals)
            d_c = np.trapz(integrand_vals, a_sub)
            d_L = d_c * (1.0 + z)
            mu = 5.0 * np.log10(d_L) + 25.0
            distances.append(mu)

        return np.array(distances)

    def log_likelihood(self, params):
        """Joint SN + DESI log-likelihood."""
        H0, omegam, k1, k2, k3, a1, a2 = params

        try:
            Omega_r = self.Omega_r_h2 / self.h_fid**2
            Omega_BH_0 = 1.0 - omegam - Omega_r

            # Basic sanity: today's BH density must be positive
            if Omega_BH_0 <= 0:
                return -np.inf

            # Compute H(a) grid with BH turned on only after a1
            H_interp = self.compute_H_grid(H0, omegam, k1, k2, k3, a1, a2)
            if H_interp is None:
                return -np.inf

            # Pantheon+ SN chi^2
            mu_theory = self.compute_distances_fast(H_interp, self.z_sn)
            if mu_theory is None or not np.all(np.isfinite(mu_theory)):
                return -np.inf

            residuals = self.mu_obs - mu_theory
            inv_var = 1.0 / self.mu_err**2
            if not np.all(np.isfinite(inv_var)):
                return -np.inf

            # Nuisance absolute magnitude M
            M_best = np.sum(residuals * inv_var) / np.sum(inv_var)
            chi2_sn = np.sum(((residuals - M_best) * np.sqrt(inv_var)) ** 2)

            if not np.isfinite(chi2_sn):
                return -np.inf

            # Scale back up to full Pantheon+ sample size
            chi2_sn *= self.scale_factor

            # DESI H0 * r_s chi^2
            r_s = self.compute_r_s_fast(H_interp, omegam)
            if not np.isfinite(r_s) or r_s <= 0.0:
                return -np.inf

            model_vec = np.array([H0 * r_s, omegam])
            delta = model_vec - self.desi_mean
            chi2_desi = delta @ self.desi_invcov @ delta

            chi2_tot = chi2_sn + chi2_desi
            if not np.isfinite(chi2_tot):
                return -np.inf

            return -0.5 * chi2_tot

        except Exception:
            return -np.inf

    def chi2_components(self, params):
        """
        Return (chi2_sn, chi2_desi, chi2_tot) for a given parameter vector.
        Useful for diagnostics and reporting.
        """
        H0, omegam, k1, k2, k3, a1, a2 = params

        Omega_r = self.Omega_r_h2 / self.h_fid**2
        Omega_BH_0 = 1.0 - omegam - Omega_r
        if Omega_BH_0 <= 0:
            return np.inf, np.inf, np.inf

        H_interp = self.compute_H_grid(H0, omegam, k1, k2, k3, a1, a2)
        if H_interp is None:
            return np.inf, np.inf, np.inf

        mu_theory = self.compute_distances_fast(H_interp, self.z_sn)
        if mu_theory is None or not np.all(np.isfinite(mu_theory)):
            return np.inf, np.inf, np.inf

        residuals = self.mu_obs - mu_theory
        inv_var = 1.0 / self.mu_err**2

        M_best = np.sum(residuals * inv_var) / np.sum(inv_var)
        chi2_sn = np.sum(((residuals - M_best) * np.sqrt(inv_var)) ** 2)
        chi2_sn *= self.scale_factor  # rescale to full Pantheon+

        r_s = self.compute_r_s_fast(H_interp, omegam)
        if not np.isfinite(r_s) or r_s <= 0.0:
            return chi2_sn, np.inf, np.inf

        model_vec = np.array([H0 * r_s, omegam])
        delta = model_vec - self.desi_mean
        chi2_desi = delta @ self.desi_invcov @ delta

        chi2_tot = chi2_sn + chi2_desi
        return chi2_sn, chi2_desi, chi2_tot

    def log_prior(self, params):
        """Priors on the cosmological and CCBH parameters."""
        H0, omegam, k1, k2, k3, a1, a2 = params

        if not (60.0 < H0 < 80.0):
            return -np.inf
        # Low-Ωm branch preferred by CCBH
        if not (0.05 < omegam < 0.25):
            return -np.inf
        if not (0.0 <= k1 < 1.0):
            return -np.inf
        if not (0.5 < k2 < 4.0):
            return -np.inf
        if not (0.0 < k3 < 5.0):
            return -np.inf
        if not (0.02 < a1 < 0.08):
            return -np.inf
        if not (0.06 < a2 < 0.30):
            return -np.inf
        if not (a1 < a2):
            return -np.inf

        return 0.0

    def log_probability(self, params):
        """Posterior = prior + likelihood."""
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf

        ll = self.log_likelihood(params)
        return lp + ll if np.isfinite(ll) else -np.inf


def run_mcmc(nwalkers=32, nsteps_burn=500, nsteps_prod=2000):
    print("=" * 70)
    print("CCBH MCMC - FAST VERSION")
    print("=" * 70)
    print()

    emulator = CCBHEmulatorFast()
    print()

    ndim = 7
    param_names = ["H0", "Omega_m", "k1", "k2", "k3", "a1", "a2"]

    # Use the good joint-fit best parameters as the new starting point
    # From the successful run with Ωm ≈ 0.119, H0 ≈ 73.7
    p0 = np.array(
        [
            73.6985,  # H0
            0.1188,   # Omega_m
            0.0148,   # k1
            2.8807,   # k2
            2.5337,   # k3
            0.0516,   # a1
            0.1987,   # a2
        ]
    )

    print("Sanity check on starting point p0:", p0)
    lp0 = emulator.log_prior(p0)
    ll0 = emulator.log_likelihood(p0)
    lpp0 = lp0 + ll0 if np.isfinite(lp0) and np.isfinite(ll0) else -np.inf
    print("  log_prior(p0)      =", lp0)
    print("  log_likelihood(p0) =", ll0)
    print("  log_prob(p0)       =", lpp0)
    print()

    # Initialize walkers around p0
    # Slightly conservative scales so walkers stay in the good region
    scales = np.array([0.5, 0.01, 0.15, 0.5, 0.5, 0.01, 0.05])

    print(f"Initializing {nwalkers} walkers...")
    pos = p0 + scales * np.random.randn(nwalkers, ndim)

    # Clip to priors
    pos[:, 0] = np.clip(pos[:, 0], 61.0, 79.0)
    pos[:, 1] = np.clip(pos[:, 1], 0.06, 0.24)   # match tightened Ωm prior band
    pos[:, 2] = np.clip(pos[:, 2], 0.0, 0.99)
    pos[:, 3] = np.clip(pos[:, 3], 0.51, 3.99)
    pos[:, 4] = np.clip(pos[:, 4], 0.01, 4.99)
    pos[:, 5] = np.clip(pos[:, 5], 0.021, 0.079)
    pos[:, 6] = np.clip(pos[:, 6], 0.061, 0.29)

    for i in range(nwalkers):
        if pos[i, 5] >= pos[i, 6]:
            pos[i, 6] = pos[i, 5] + 0.01

    print("Done!")
    print()

    sampler = emcee.EnsembleSampler(nwalkers, ndim, emulator.log_probability)

    # Burn-in
    print("=" * 70)
    print("BURN-IN")
    print("=" * 70)
    state = sampler.run_mcmc(pos, nsteps_burn, progress=True)
    print(f"\nAcceptance: {np.mean(sampler.acceptance_fraction):.3f}")

    sampler.reset()

    # Production
    print()
    print("=" * 70)
    print("PRODUCTION")
    print("=" * 70)
    sampler.run_mcmc(state, nsteps_prod, progress=True)

    samples = sampler.get_chain(flat=True)
    log_prob = sampler.get_log_prob(flat=True)

    print(f"\nAcceptance: {np.mean(sampler.acceptance_fraction):.3f}")

    # Save chains
    np.save("ccbh_fast_samples.npy", samples)
    np.save("ccbh_fast_logprob.npy", log_prob)

    # Results
    best_idx = np.argmax(log_prob)
    best_params = samples[best_idx]
    best_chi2 = -2.0 * log_prob[best_idx]

    print()
    print("=" * 70)
    print("BEST-FIT")
    print("=" * 70)
    for name, val in zip(param_names, best_params):
        print(f"  {name:10s} = {val:.6f}")
    print(f"  chi2_total (from log_prob) = {best_chi2:.2f}")

    # Decompose chi^2 into SN and DESI pieces
    chi2_sn, chi2_desi, chi2_tot_check = emulator.chi2_components(best_params)
    print("  chi2_SN                  =", chi2_sn)
    print("  chi2_DESI                =", chi2_desi)
    print("  chi2_total (recomputed)  =", chi2_tot_check)
    print()

    print("=" * 70)
    print("POSTERIORS")
    print("=" * 70)
    for i, name in enumerate(param_names):
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
        print(f"  {name:10s} = {q50:.6f} +{q84 - q50:.6f} -{q50 - q16:.6f}")
    print()

    fig = corner.corner(
        samples,
        labels=param_names,
        truths=best_params,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
    )
    plt.savefig("ccbh_fast_corner.png", dpi=150, bbox_inches="tight")
    print("Saved corner plot")

    return sampler, samples, log_prob


if __name__ == "__main__":
    sampler, samples, log_prob = run_mcmc()
