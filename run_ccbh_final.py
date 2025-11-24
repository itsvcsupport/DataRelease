#!/usr/bin/env python
"""
CCBH MCMC - FAST VERSION with caching and vectorization
"""
import numpy as np
import pandas as pd
import emcee
import corner
import matplotlib.pyplot as plt
from scipy.integrate import quad
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
        
        # Pre-compute integration grids for speed
        self.setup_grids()
    
    def load_data(self):
        # Pantheon+ - subsample for speed during MCMC
        df = pd.read_csv('Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat',
                         delim_whitespace=True, comment='#')
        
        # Use every 3rd SN for MCMC (still 567 SNe)
        # This gives ~3x speedup with minimal information loss
        indices = np.arange(0, len(df), 3)
        
        self.z_sn = df['zHD'].values[indices]
        self.mu_obs = df['MU_SH0ES'].values[indices]
        self.mu_err = df['MU_SH0ES_ERR_DIAG'].values[indices]
        
        # Scale factor for chi2 (to account for subsampling)
        self.n_sn_full = len(df)
        self.n_sn_sample = len(indices)
        self.scale_factor = self.n_sn_full / self.n_sn_sample
        
        print(f"Using {self.n_sn_sample} SNe for MCMC (subsampled from {self.n_sn_full})")
        
        # DESI
        chain = np.loadtxt('C:/Users/WilliamKellett/cobaya/base/desi-bao-all/chain.1.txt')
        weights = chain[:, 0]
        H0rs = chain[:, 18]
        omegam = chain[:, 3]
        
        self.desi_mean = np.array([
            np.average(H0rs, weights=weights),
            np.average(omegam, weights=weights)
        ])
        
        data = np.column_stack([H0rs, omegam])
        self.desi_cov = np.cov(data.T, aweights=weights)
        self.desi_invcov = np.linalg.inv(self.desi_cov)
        
        print(f"DESI: H0*rs={self.desi_mean[0]:.1f}, Omega_m={self.desi_mean[1]:.5f}")
    
    def setup_grids(self):
        """Pre-compute integration grids"""
        # Create dense grid in log(a) for integrations
        self.a_grid = np.logspace(-4, 0, 500)  # From a=0.0001 to a=1
        self.ln_a_grid = np.log(self.a_grid)
    
    def k_eff(self, a, k1, k2, k3, a1, a2):
        if a < a1: return k1
        elif a < a2: return k2
        else: return k3
    
    def compute_H_grid(self, H0, omegam, k1, k2, k3, a1, a2):
        """Compute H(a) on pre-defined grid for interpolation"""
        Omega_r = self.Omega_r_h2 / self.h_fid**2
        Omega_BH_0 = 1.0 - omegam - Omega_r
        
        if Omega_BH_0 < 0:
            return None
        
        H_vals = []
        
        for a in self.a_grid:
            # Integrate BH evolution
            k_vals = np.array([self.k_eff(a_p, k1, k2, k3, a1, a2) 
                              for a_p in self.a_grid[self.a_grid <= a]])
            
            if len(k_vals) > 1:
                ln_a_sub = self.ln_a_grid[self.a_grid <= a]
                integrand_vals = k_vals - 3
                integral = np.trapz(integrand_vals, ln_a_sub)
            else:
                integral = 0
            
            Omega_BH_a = Omega_BH_0 * np.exp(integral)
            Omega_m_a = omegam / a**3
            Omega_r_a = Omega_r / a**4
            
            H_squared = H0**2 * (Omega_m_a + Omega_r_a + Omega_BH_a)
            
            if H_squared <= 0:
                return None
            
            H_vals.append(np.sqrt(H_squared))
        
        # Return interpolator
        return interp1d(self.a_grid, H_vals, kind='cubic', 
                       bounds_error=False, fill_value='extrapolate')
    
    def compute_r_s_fast(self, H_interp, omegam):
        """Fast r_s using interpolated H(a)"""
        Omega_b = self.Omega_b_h2 / self.h_fid**2
        Omega_m = omegam
        
        b1 = 0.313 * (Omega_m * self.h_fid**2)**(-0.419) * \
             (1 + 0.607 * (Omega_m * self.h_fid**2)**0.674)
        b2 = 0.238 * (Omega_m * self.h_fid**2)**0.223
        z_d = 1291 * (Omega_m * self.h_fid**2)**0.251 / \
              (1 + 0.659 * (Omega_m * self.h_fid**2)**0.828) * \
              (1 + b1 * (Omega_b * self.h_fid**2)**b2)
        
        a_d = 1 / (1 + z_d)
        
        # Use grid for integration
        a_sub = self.a_grid[self.a_grid <= a_d]
        R_vals = 3 * Omega_b / (4 * self.Omega_r_h2 / self.h_fid**2) * a_sub
        c_s_vals = self.c_km_s / np.sqrt(3 * (1 + R_vals))
        H_vals = H_interp(a_sub)
        
        integrand_vals = c_s_vals / (a_sub**2 * H_vals)
        r_s = np.trapz(integrand_vals, a_sub)
        
        return r_s
    
    def compute_distances_fast(self, H_interp, z_array):
        """Vectorized distance calculation"""
        distances = []
        
        for z in z_array:
            a = 1 / (1 + z)
            a_sub = self.a_grid[(self.a_grid >= a) & (self.a_grid <= 1.0)]
            
            if len(a_sub) < 2:
                return None
            
            H_vals = H_interp(a_sub)
            integrand_vals = self.c_km_s / (a_sub**2 * H_vals)
            d_c = np.trapz(integrand_vals, a_sub)
            d_L = d_c * (1 + z)
            mu = 5 * np.log10(d_L) + 25
            distances.append(mu)
        
        return np.array(distances)
    
    def log_likelihood(self, params):
        H0, omegam, k1, k2, k3, a1, a2 = params
        
        try:
            # Early-time check
            Omega_r = self.Omega_r_h2 / self.h_fid**2
            Omega_BH_0 = 1.0 - omegam - Omega_r
            
            if Omega_BH_0 < 0:
                return -np.inf
            
            # Check recombination
            a_rec = 1/1100
            k_rec = self.k_eff(a_rec, k1, k2, k3, a1, a2)
            integral_rec = k_rec * np.log(a_rec) - 3 * np.log(a_rec)  # Approx
            Omega_BH_rec = Omega_BH_0 * np.exp(integral_rec)
            
            if Omega_BH_rec / omegam > 0.001:
                return -np.inf
            
            # Compute H(a) grid
            H_interp = self.compute_H_grid(H0, omegam, k1, k2, k3, a1, a2)
            
            if H_interp is None:
                return -np.inf
            
            # Pantheon+ chi2
            mu_theory = self.compute_distances_fast(H_interp, self.z_sn)
            
            if mu_theory is None or not np.all(np.isfinite(mu_theory)):
                return -np.inf
            
            residuals = self.mu_obs - mu_theory
            inv_var = 1 / self.mu_err**2
            
            M_best = np.sum(residuals * inv_var) / np.sum(inv_var)
            chi2_sn = np.sum(((residuals - M_best) * np.sqrt(inv_var))**2)
            
            # Scale to full sample
            chi2_sn *= self.scale_factor
            
            # DESI chi2
            r_s = self.compute_r_s_fast(H_interp, omegam)
            
            if not np.isfinite(r_s) or r_s <= 0:
                return -np.inf
            
            model = np.array([H0 * r_s, omegam])
            delta = model - self.desi_mean
            chi2_desi = delta @ self.desi_invcov @ delta
            
            chi2_tot = chi2_sn + chi2_desi
            
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
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        
        ll = self.log_likelihood(params)
        return lp + ll if np.isfinite(ll) else -np.inf


def run_mcmc(nwalkers=32, nsteps_burn=500, nsteps_prod=2000):
    print("="*70)
    print("CCBH MCMC - FAST VERSION")
    print("="*70)
    print()
    
    emulator = CCBHEmulatorFast()
    print()
    
    ndim = 7
    param_names = ['H0', 'Omega_m', 'k1', 'k2', 'k3', 'a1', 'a2']
    
    # Use the median values from your stuck run as starting point
    p0 = np.array([68.093, 0.321, 0.001, 1.508, 3.008, 0.031, 0.081])
    scales = np.array([0.3, 0.008, 0.015, 0.06, 0.15, 0.002, 0.005])
    
    print(f"Initializing {nwalkers} walkers...")
    pos = p0 + scales * np.random.randn(nwalkers, ndim)
    
    # Clip to priors
    pos[:, 0] = np.clip(pos[:, 0], 61, 79)
    pos[:, 1] = np.clip(pos[:, 1], 0.21, 0.49)
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
    print("="*70)
    print("BURN-IN")
    print("="*70)
    state = sampler.run_mcmc(pos, nsteps_burn, progress=True)
    print(f"\nAcceptance: {np.mean(sampler.acceptance_fraction):.3f}")
    
    sampler.reset()
    
    # Production
    print()
    print("="*70)
    print("PRODUCTION")
    print("="*70)
    sampler.run_mcmc(state, nsteps_prod, progress=True)
    
    samples = sampler.get_chain(flat=True)
    log_prob = sampler.get_log_prob(flat=True)
    
    print(f"\nAcceptance: {np.mean(sampler.acceptance_fraction):.3f}")
    
    # Save
    np.save('ccbh_fast_samples.npy', samples)
    np.save('ccbh_fast_logprob.npy', log_prob)
    
    # Results
    best_idx = np.argmax(log_prob)
    best_params = samples[best_idx]
    best_chi2 = -2 * log_prob[best_idx]
    
    print()
    print("="*70)
    print("BEST-FIT")
    print("="*70)
    for name, val in zip(param_names, best_params):
        print(f"  {name:10s} = {val:.6f}")
    print(f"  chi2_total = {best_chi2:.2f}")
    print()
    
    print("="*70)
    print("POSTERIORS")
    print("="*70)
    for i, name in enumerate(param_names):
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
        print(f"  {name:10s} = {q50:.6f} +{q84-q50:.6f} -{q50-q16:.6f}")
    print()
    
    fig = corner.corner(samples, labels=param_names, truths=best_params,
                       quantiles=[0.16, 0.5, 0.84], show_titles=True)
    plt.savefig('ccbh_fast_corner.png', dpi=150, bbox_inches='tight')
    print("Saved corner plot")
    
    return sampler, samples, log_prob


if __name__ == "__main__":
    sampler, samples, log_prob = run_mcmc()
