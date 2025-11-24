#!/usr/bin/env python
"""
CCBH MCMC with optimized initialization
"""
import numpy as np
import pandas as pd
import emcee
import corner
import matplotlib.pyplot as plt
from scipy.integrate import quad
import warnings

warnings.filterwarnings('ignore')

class CCBHEmulator:
    def __init__(self):
        self.c_km_s = 299792.458
        self.h_fid = 0.6766
        self.Omega_b_h2 = 0.02242
        self.Omega_r_h2 = 4.18e-5
        self.load_data()
    
    def load_data(self):
        df = pd.read_csv('Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat',
                         delim_whitespace=True, comment='#')
        self.z_sn = df['zHD'].values
        self.mu_obs = df['MU_SH0ES'].values
        self.mu_err = df['MU_SH0ES_ERR_DIAG'].values
        
        chain = np.loadtxt('C:/Users/WilliamKellett/cobaya/base/desi-bao-all/chain.1.txt')
        weights = chain[:, 0]
        H0rs = chain[:, 1]
        omegam = chain[:, 2]
        
        self.desi_mean = np.array([
            np.average(H0rs, weights=weights),
            np.average(omegam, weights=weights)
        ])
        
        data = np.column_stack([H0rs, omegam])
        self.desi_cov = np.cov(data.T, aweights=weights)
        self.desi_invcov = np.linalg.inv(self.desi_cov)
        
        print(f"Loaded {len(self.z_sn)} SNe")
        print(f"DESI: H0*rs={self.desi_mean[0]:.1f}, Omega_m={self.desi_mean[1]:.4f}")
    
    def k_eff(self, a, k1, k2, k3, a1, a2):
        if a < a1: return k1
        elif a < a2: return k2
        else: return k3
    
    def H_evolution(self, a, H0, omegam, k1, k2, k3, a1, a2):
        try:
            Omega_r = self.Omega_r_h2 / self.h_fid**2
            Omega_BH_0 = 1.0 - omegam - Omega_r
            
            if Omega_BH_0 < 0:
                return np.nan
            
            def integrand(ln_a_prime):
                a_prime = np.exp(ln_a_prime)
                k = self.k_eff(a_prime, k1, k2, k3, a1, a2)
                return k - 3
            
            ln_a = np.log(a)
            integral, _ = quad(integrand, 0, ln_a, limit=100, epsabs=1e-10, epsrel=1e-10)
            
            Omega_BH_a = Omega_BH_0 * np.exp(integral)
            Omega_m_a = omegam / a**3
            Omega_r_a = Omega_r / a**4
            
            H_squared = H0**2 * (Omega_m_a + Omega_r_a + Omega_BH_a)
            
            if H_squared <= 0:
                return np.nan
            
            return np.sqrt(H_squared)
        except:
            return np.nan
    
    def compute_r_s(self, H0, omegam, k1, k2, k3, a1, a2):
        try:
            Omega_b = self.Omega_b_h2 / self.h_fid**2
            Omega_m = omegam
            
            b1 = 0.313 * (Omega_m * self.h_fid**2)**(-0.419) * \
                 (1 + 0.607 * (Omega_m * self.h_fid**2)**0.674)
            b2 = 0.238 * (Omega_m * self.h_fid**2)**0.223
            z_d = 1291 * (Omega_m * self.h_fid**2)**0.251 / \
                  (1 + 0.659 * (Omega_m * self.h_fid**2)**0.828) * \
                  (1 + b1 * (Omega_b * self.h_fid**2)**b2)
            
            a_d = 1 / (1 + z_d)
            
            def integrand(a_prime):
                R = 3 * Omega_b / (4 * self.Omega_r_h2 / self.h_fid**2) * a_prime
                c_s = self.c_km_s / np.sqrt(3 * (1 + R))
                H_a = self.H_evolution(a_prime, H0, omegam, k1, k2, k3, a1, a2)
                if not np.isfinite(H_a) or H_a <= 0:
                    raise ValueError("Bad H")
                return c_s / (a_prime**2 * H_a)
            
            r_s, _ = quad(integrand, 0, a_d, limit=100, epsabs=1e-8, epsrel=1e-8)
            
            if not np.isfinite(r_s) or r_s <= 0:
                return np.nan
            
            return r_s
        except:
            return np.nan
    
    def compute_mu(self, z, H0, omegam, k1, k2, k3, a1, a2):
        try:
            a = 1 / (1 + z)
            
            def integrand(a_prime):
                H_a = self.H_evolution(a_prime, H0, omegam, k1, k2, k3, a1, a2)
                if not np.isfinite(H_a) or H_a <= 0:
                    raise ValueError("Bad H")
                return self.c_km_s / (a_prime**2 * H_a)
            
            d_c, _ = quad(integrand, a, 1.0, limit=100, epsabs=1e-8, epsrel=1e-8)
            
            if not np.isfinite(d_c) or d_c <= 0:
                return np.nan
            
            d_L = d_c * (1 + z)
            mu = 5 * np.log10(d_L) + 25
            
            return mu if np.isfinite(mu) else np.nan
        except:
            return np.nan
    
    def chi2_sn(self, H0, omegam, k1, k2, k3, a1, a2):
        # Sample only 100 SNe for faster initialization checks
        indices = np.random.choice(len(self.z_sn), size=min(100, len(self.z_sn)), replace=False)
        
        mu_theory = np.array([
            self.compute_mu(self.z_sn[i], H0, omegam, k1, k2, k3, a1, a2) 
            for i in indices
        ])
        
        if not np.all(np.isfinite(mu_theory)):
            return np.nan
        
        residuals = self.mu_obs[indices] - mu_theory
        inv_var = 1 / self.mu_err[indices]**2
        
        M_best = np.sum(residuals * inv_var) / np.sum(inv_var)
        chi2 = np.sum(((residuals - M_best) * np.sqrt(inv_var))**2)
        
        # Scale to full sample
        return chi2 * (len(self.z_sn) / len(indices))
    
    def chi2_desi(self, H0, omegam, k1, k2, k3, a1, a2):
        r_s = self.compute_r_s(H0, omegam, k1, k2, k3, a1, a2)
        
        if not np.isfinite(r_s):
            return np.nan
        
        model = np.array([H0 * r_s, omegam])
        delta = model - self.desi_mean
        return delta @ self.desi_invcov @ delta
    
    def log_likelihood(self, params):
        H0, omegam, k1, k2, k3, a1, a2 = params
        
        try:
            # Early-time check
            a_rec = 1/1100
            Omega_r = self.Omega_r_h2 / self.h_fid**2
            Omega_BH_0 = 1.0 - omegam - Omega_r
            
            if Omega_BH_0 < 0:
                return -np.inf
            
            def integrand(ln_a_prime):
                a_prime = np.exp(ln_a_prime)
                k = self.k_eff(a_prime, k1, k2, k3, a1, a2)
                return k - 3
            
            integral, _ = quad(integrand, 0, np.log(a_rec), limit=100)
            Omega_BH_rec = Omega_BH_0 * np.exp(integral)
            
            if Omega_BH_rec / omegam > 0.001:
                return -np.inf
            
            chi2_sn_val = self.chi2_sn(H0, omegam, k1, k2, k3, a1, a2)
            chi2_desi_val = self.chi2_desi(H0, omegam, k1, k2, k3, a1, a2)
            
            if not np.isfinite(chi2_sn_val) or not np.isfinite(chi2_desi_val):
                return -np.inf
            
            chi2_tot = chi2_sn_val + chi2_desi_val
            
            return -0.5 * chi2_tot
        except:
            return -np.inf
    
    def log_prior(self, params):
        H0, omegam, k1, k2, k3, a1, a2 = params
        
        if not (60 < H0 < 80): return -np.inf
        if not (0.20 < omegam < 0.50): return -np.inf
        if not (0.0 <= k1 < 1.0): return -np.inf
        if not (0.5 < k2 < 4.0): return -np.inf
        if not (0.0 < k3 < 5.0): return -np.inf
        if not (0.02 < a1 < 0.08): return -np.inf
        if not (0.06 < a2 < 0.30): return -np.inf
        if not (a1 < a2): return -np.inf
        
        return 0.0
    
    def log_probability(self, params):
        try:
            lp = self.log_prior(params)
            if not np.isfinite(lp):
                return -np.inf
            
            ll = self.log_likelihood(params)
            if not np.isfinite(ll):
                return -np.inf
            
            return lp + ll
        except:
            return -np.inf


def initialize_walkers_smart(emulator, nwalkers, p0, scales):
    """
    Smart walker initialization using Latin hypercube sampling
    """
    print("Initializing walkers with Latin hypercube sampling...")
    
    from scipy.stats import qmc
    
    ndim = len(p0)
    
    # Define parameter bounds
    bounds = np.array([
        [60, 80],      # H0
        [0.20, 0.50],  # omegam
        [0.0, 1.0],    # k1
        [0.5, 4.0],    # k2
        [0.0, 5.0],    # k3
        [0.02, 0.08],  # a1
        [0.06, 0.30]   # a2
    ])
    
    # Latin hypercube sample
    sampler = qmc.LatinHypercube(d=ndim)
    unit_samples = sampler.random(n=nwalkers * 3)  # Oversample
    
    # Scale to bounds, biased toward best-fit
    samples = []
    for unit_sample in unit_samples:
        # Mix LHS with Gaussian around best-fit
        if np.random.rand() < 0.7:  # 70% around best-fit
            sample = p0 + scales * np.random.randn(ndim)
        else:  # 30% uniform across space
            sample = bounds[:, 0] + unit_sample * (bounds[:, 1] - bounds[:, 0])
        
        # Clip to bounds
        sample = np.clip(sample, bounds[:, 0], bounds[:, 1])
        
        # Ensure a1 < a2
        if sample[5] >= sample[6]:
            sample[6] = sample[5] + 0.01
        
        samples.append(sample)
    
    # Test all samples and keep valid ones
    valid_samples = []
    print("Testing walker positions...")
    
    for i, sample in enumerate(samples):
        if i % 10 == 0:
            print(f"  Tested {i}/{len(samples)}, found {len(valid_samples)} valid")
        
        logp = emulator.log_probability(sample)
        if np.isfinite(logp) and logp > -1e10:
            valid_samples.append(sample)
        
        if len(valid_samples) >= nwalkers:
            break
    
    if len(valid_samples) < nwalkers:
        print(f"Warning: Only found {len(valid_samples)} valid positions")
        # Pad with best-fit + tiny noise
        while len(valid_samples) < nwalkers:
            valid_samples.append(p0 + 0.001 * scales * np.random.randn(ndim))
    
    return np.array(valid_samples[:nwalkers])


def run_mcmc(nwalkers=32, nsteps_burn=2000, nsteps_prod=10000):
    print("="*70)
    print("CCBH MCMC ANALYSIS")
    print("="*70)
    print()
    
    emulator = CCBHEmulator()
    print()
    
    ndim = 7
    param_names = ['H0', 'Omega_m', 'k1', 'k2', 'k3', 'a1', 'a2']
    
    # Best-fit from grid search
    p0 = np.array([68.0, 0.32, 0.0, 1.5, 3.0, 0.03, 0.08])
    scales = np.array([2.0, 0.03, 0.1, 0.3, 0.5, 0.01, 0.02])
    
    # Initialize walkers smartly
    pos = initialize_walkers_smart(emulator, nwalkers, p0, scales)
    print(f"Initialized {len(pos)} walkers")
    print()
    
    # Initialize sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, emulator.log_probability)
    
    # Burn-in
    print("="*70)
    print("BURN-IN PHASE")
    print("="*70)
    state = sampler.run_mcmc(pos, nsteps_burn, progress=True)
    print(f"Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    
    sampler.reset()
    
    # Production
    print()
    print("="*70)
    print("PRODUCTION PHASE")
    print("="*70)
    sampler.run_mcmc(state, nsteps_prod, progress=True)
    
    # Analysis
    samples = sampler.get_chain(flat=True)
    log_prob = sampler.get_log_prob(flat=True)
    
    print()
    print(f"Final acceptance: {np.mean(sampler.acceptance_fraction):.3f}")
    
    # Save
    np.save('ccbh_samples.npy', samples)
    np.save('ccbh_logprob.npy', log_prob)
    
    # Best-fit
    best_idx = np.argmax(log_prob)
    best_params = samples[best_idx]
    best_chi2 = -2 * log_prob[best_idx]
    
    print()
    print("="*70)
    print("BEST-FIT")
    print("="*70)
    for name, val in zip(param_names, best_params):
        print(f"{name:10s} = {val:.6f}")
    print(f"chi2_total = {best_chi2:.2f}")
    
    # Posteriors
    print()
    print("="*70)
    print("POSTERIORS")
    print("="*70)
    for i, name in enumerate(param_names):
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
        print(f"{name:10s} = {q50:.6f} +{q84-q50:.6f} -{q50-q16:.6f}")
    
    # Corner plot
    fig = corner.corner(samples, labels=param_names, truths=best_params,
                       quantiles=[0.16, 0.5, 0.84], show_titles=True)
    plt.savefig('ccbh_corner.png', dpi=150, bbox_inches='tight')
    print("\nSaved corner plot")
    
    return sampler, samples, log_prob


if __name__ == "__main__":
    sampler, samples, log_prob = run_mcmc(
        nwalkers=32,
        nsteps_burn=1000,  # Shorter for testing
        nsteps_prod=5000
    )
