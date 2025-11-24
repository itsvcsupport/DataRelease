#!/usr/bin/env python
"""
Full MCMC sampling for CCBH model using emcee - ROBUST VERSION
"""
import numpy as np
import pandas as pd
import emcee
import corner
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import stats
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================================
# Load your existing emulator code (FIXED VERSION)
# ============================================================================

class CCBHEmulator:
    """Your existing CCBH background emulator with robustness improvements"""
    
    def __init__(self):
        self.c_km_s = 299792.458
        self.setup_cosmology()
        self.load_data()
        
    def setup_cosmology(self):
        """Physical constants"""
        self.h_fid = 0.6766
        self.Omega_b_h2 = 0.02242
        self.Omega_r_h2 = 4.18e-5
        
    def load_data(self):
        """Load Pantheon+ and DESI"""
        # Pantheon+
        df = pd.read_csv('Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat',
                         delim_whitespace=True, comment='#')
        self.z_sn = df['zHD'].values
        self.mu_obs = df['MU_SH0ES'].values
        self.mu_err = df['MU_SH0ES_ERR_DIAG'].values
        self.n_sn = len(self.z_sn)
        
        # DESI
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
        
        print(f"Loaded {self.n_sn} SNe")
        print(f"DESI mean: H0*rs={self.desi_mean[0]:.1f}, Omega_m={self.desi_mean[1]:.4f}")
        
    def k_eff(self, a, k1, k2, k3, a1, a2):
        """3-phase coupling"""
        if a < a1:
            return k1
        elif a < a2:
            return k2
        else:
            return k3
            
    def H_evolution(self, a, H0, omegam, k1, k2, k3, a1, a2):
        """Hubble parameter H(a) - ROBUST VERSION"""
        try:
            Omega_r = self.Omega_r_h2 / self.h_fid**2
            Omega_BH_0 = 1.0 - omegam - Omega_r
            
            if Omega_BH_0 < 0:
                return np.nan
            
            # Integrate BH density evolution
            def integrand(ln_a_prime):
                a_prime = np.exp(ln_a_prime)
                k = self.k_eff(a_prime, k1, k2, k3, a1, a2)
                return k - 3
                
            ln_a = np.log(a)
            integral, _ = quad(integrand, 0, ln_a, limit=50, epsabs=1e-8, epsrel=1e-8)
            
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
        """Sound horizon at drag epoch - ROBUST VERSION"""
        try:
            # Drag redshift (Eisenstein & Hu 1998)
            Omega_b = self.Omega_b_h2 / self.h_fid**2
            Omega_m = omegam
            
            b1 = 0.313 * (Omega_m * self.h_fid**2)**(-0.419) * \
                 (1 + 0.607 * (Omega_m * self.h_fid**2)**0.674)
            b2 = 0.238 * (Omega_m * self.h_fid**2)**0.223
            z_d = 1291 * (Omega_m * self.h_fid**2)**0.251 / \
                  (1 + 0.659 * (Omega_m * self.h_fid**2)**0.828) * \
                  (1 + b1 * (Omega_b * self.h_fid**2)**b2)
            
            a_d = 1 / (1 + z_d)
            
            # Integrate sound horizon
            def integrand(a_prime):
                R = 3 * Omega_b / (4 * self.Omega_r_h2 / self.h_fid**2) * a_prime
                c_s = self.c_km_s / np.sqrt(3 * (1 + R))
                H_a = self.H_evolution(a_prime, H0, omegam, k1, k2, k3, a1, a2)
                if not np.isfinite(H_a) or H_a <= 0:
                    raise ValueError("Bad H(a)")
                return c_s / (a_prime**2 * H_a)
                
            r_s, _ = quad(integrand, 0, a_d, limit=50, epsabs=1e-6, epsrel=1e-6)
            
            if not np.isfinite(r_s) or r_s <= 0:
                return np.nan
                
            return r_s
            
        except:
            return np.nan
        
    def compute_mu(self, z, H0, omegam, k1, k2, k3, a1, a2):
        """Distance modulus - ROBUST VERSION"""
        try:
            a = 1 / (1 + z)
            
            def integrand(a_prime):
                H_a = self.H_evolution(a_prime, H0, omegam, k1, k2, k3, a1, a2)
                if not np.isfinite(H_a) or H_a <= 0:
                    raise ValueError("Bad H(a)")
                return self.c_km_s / (a_prime**2 * H_a)
                
            d_c, _ = quad(integrand, a, 1.0, limit=50, epsabs=1e-6, epsrel=1e-6)
            
            if not np.isfinite(d_c) or d_c <= 0:
                return np.nan
                
            d_L = d_c * (1 + z)
            mu = 5 * np.log10(d_L) + 25
            
            if not np.isfinite(mu):
                return np.nan
                
            return mu
            
        except:
            return np.nan
        
    def chi2_sn(self, H0, omegam, k1, k2, k3, a1, a2):
        """Pantheon+ chi-square with analytic M marginalization"""
        mu_theory = np.array([self.compute_mu(z, H0, omegam, k1, k2, k3, a1, a2) 
                              for z in self.z_sn])
        
        if not np.all(np.isfinite(mu_theory)):
            return np.nan
        
        residuals = self.mu_obs - mu_theory
        inv_var = 1 / self.mu_err**2
        
        M_best = np.sum(residuals * inv_var) / np.sum(inv_var)
        chi2 = np.sum(((residuals - M_best) * np.sqrt(inv_var))**2)
        
        return chi2
        
    def chi2_desi(self, H0, omegam, k1, k2, k3, a1, a2):
        """DESI BAO chi-square"""
        r_s = self.compute_r_s(H0, omegam, k1, k2, k3, a1, a2)
        
        if not np.isfinite(r_s):
            return np.nan
            
        model = np.array([H0 * r_s, omegam])
        delta = model - self.desi_mean
        return delta @ self.desi_invcov @ delta
        
    def log_likelihood(self, params):
        """Total log-likelihood"""
        H0, omegam, k1, k2, k3, a1, a2 = params
        
        try:
            # Safety check: early-time BH must be negligible
            a_rec = 1/1100
            Omega_r = self.Omega_r_h2 / self.h_fid**2
            Omega_BH_0 = 1.0 - omegam - Omega_r
            
            if Omega_BH_0 < 0:
                return -np.inf
            
            def integrand(ln_a_prime):
                a_prime = np.exp(ln_a_prime)
                k = self.k_eff(a_prime, k1, k2, k3, a1, a2)
                return k - 3
                
            integral, _ = quad(integrand, 0, np.log(a_rec), limit=50)
            Omega_BH_rec = Omega_BH_0 * np.exp(integral)
            
            if Omega_BH_rec / omegam > 0.001:  # > 0.1% of matter at recombination
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
        """Flat priors"""
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
        """Total log-probability for emcee"""
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


# ============================================================================
# Run MCMC
# ============================================================================

if __name__ == "__main__":
    
    print("Initializing CCBH emulator...")
    emulator = CCBHEmulator()
    
    # MCMC setup
    ndim = 7
    nwalkers = 32
    nsteps_burn = 2000
    nsteps_prod = 10000
    
    param_names = ['H0', 'Omega_m', 'k1', 'k2', 'k3', 'a1', 'a2']
    
    # Initial positions (near your best-fit)
    p0 = np.array([68.0, 0.32, 0.0, 1.5, 3.0, 0.03, 0.08])
    scales = np.array([1.0, 0.02, 0.05, 0.2, 0.3, 0.005, 0.01])
    
    print("\nInitializing walker positions...")
    pos = []
    for i in range(nwalkers):
        attempts = 0
        while attempts < 1000:
            test_pos = p0 + scales * np.random.randn(ndim)
            logp = emulator.log_probability(test_pos)
            if np.isfinite(logp):
                pos.append(test_pos)
                break
            attempts += 1
        
        if attempts >= 1000:
            print(f"Warning: Walker {i} fell back to best-fit")
            pos.append(p0 + 0.01 * scales * np.random.randn(ndim))
    
    pos = np.array(pos)
    print(f"Initialized {len(pos)} walkers")
    
    print(f"\nRunning MCMC: {nwalkers} walkers × ({nsteps_burn} burn-in + {nsteps_prod} production) steps")
    
    # Initialize sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, emulator.log_probability)
    
    # Burn-in
    print("\n--- Burn-in phase ---")
    state = sampler.run_mcmc(pos, nsteps_burn, progress=True)
    sampler.reset()
    
    # Production
    print("\n--- Production phase ---")
    sampler.run_mcmc(state, nsteps_prod, progress=True)
    
    # Results
    samples = sampler.get_chain(flat=True)
    log_prob = sampler.get_log_prob(flat=True)
    
    print(f"\nMCMC complete!")
    print(f"Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        print(f"Autocorrelation time: {tau}")
    except:
        print("Could not compute autocorrelation time")
    
    # Save chains
    np.save('ccbh_samples.npy', samples)
    np.save('ccbh_logprob.npy', log_prob)
    
    # Best-fit
    best_idx = np.argmax(log_prob)
    best_params = samples[best_idx]
    best_chi2 = -2 * log_prob[best_idx]
    
    print("\n" + "="*60)
    print("BEST-FIT PARAMETERS (from MCMC)")
    print("="*60)
    for name, val in zip(param_names, best_params):
        print(f"{name:10s} = {val:.4f}")
    print(f"chi2_total = {best_chi2:.2f}")
    
    # Posterior statistics
    print("\n" + "="*60)
    print("POSTERIOR STATISTICS (median ± 1σ)")
    print("="*60)
    for i, name in enumerate(param_names):
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
        err_minus = q50 - q16
        err_plus = q84 - q50
        print(f"{name:10s} = {q50:.4f} +{err_plus:.4f} -{err_minus:.4f}")
    
    # Corner plot
    print("\nGenerating corner plot...")
    fig = corner.corner(samples, labels=param_names, 
                       truths=best_params,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True)
    plt.savefig('ccbh_corner.png', dpi=150, bbox_inches='tight')
    print("Saved: ccbh_corner.png")
    
    print("\nAll done!")
