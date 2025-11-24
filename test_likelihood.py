import numpy as np
import pandas as pd
from scipy.integrate import quad

class CCBHEmulator:
    def __init__(self):
        self.c_km_s = 299792.458
        self.h_fid = 0.6766
        self.Omega_b_h2 = 0.02242
        self.Omega_r_h2 = 4.18e-5
        self.load_data()
    
    def load_data(self):
        # Pantheon+
        df = pd.read_csv('Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat',
                         delim_whitespace=True, comment='#')
        self.z_sn = df['zHD'].values
        self.mu_obs = df['MU_SH0ES'].values
        self.mu_err = df['MU_SH0ES_ERR_DIAG'].values
        
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
    
    def k_eff(self, a, k1, k2, k3, a1, a2):
        if a < a1: return k1
        elif a < a2: return k2
        else: return k3
    
    def test_params(self, H0, omegam, k1, k2, k3, a1, a2):
        """Test if parameters give valid likelihood"""
        print(f"\nTesting: H0={H0:.1f}, Omega_m={omegam:.3f}, k1={k1:.2f}, k2={k2:.2f}, k3={k3:.2f}, a1={a1:.3f}, a2={a2:.3f}")
        
        # Check basic physics
        Omega_r = self.Omega_r_h2 / self.h_fid**2
        Omega_BH_0 = 1.0 - omegam - Omega_r
        print(f"  Omega_BH(today) = {Omega_BH_0:.4f}")
        
        if Omega_BH_0 < 0:
            print("  ✗ FAIL: Negative Omega_BH")
            return False
        
        # Test H(a) at a few points
        print("  Testing H(a)...")
        for a_test in [0.01, 0.1, 0.5, 1.0]:
            try:
                def integrand(ln_a_prime):
                    a_prime = np.exp(ln_a_prime)
                    k = self.k_eff(a_prime, k1, k2, k3, a1, a2)
                    return k - 3
                
                integral, _ = quad(integrand, 0, np.log(a_test), limit=50)
                Omega_BH_a = Omega_BH_0 * np.exp(integral)
                Omega_m_a = omegam / a_test**3
                Omega_r_a = Omega_r / a_test**4
                H_squared = H0**2 * (Omega_m_a + Omega_r_a + Omega_BH_a)
                
                if H_squared <= 0:
                    print(f"    ✗ a={a_test:.2f}: H^2 = {H_squared:.2e} (negative!)")
                    return False
                
                H_val = np.sqrt(H_squared)
                print(f"    ✓ a={a_test:.2f}: H = {H_val:.1f} km/s/Mpc")
                
            except Exception as e:
                print(f"    ✗ a={a_test:.2f}: Integration failed - {e}")
                return False
        
        # Test r_s calculation
        print("  Testing r_s...")
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
            print(f"    Drag epoch: z_d = {z_d:.1f}, a_d = {a_d:.5f}")
            
            # Try simplified r_s calculation
            def integrand_rs(a_prime):
                R = 3 * Omega_b / (4 * self.Omega_r_h2 / self.h_fid**2) * a_prime
                c_s = self.c_km_s / np.sqrt(3 * (1 + R))
                
                # H(a) calculation
                def integrand_H(ln_ap):
                    ap = np.exp(ln_ap)
                    k = self.k_eff(ap, k1, k2, k3, a1, a2)
                    return k - 3
                
                integral_H, _ = quad(integrand_H, 0, np.log(a_prime), limit=50)
                Omega_BH_ap = Omega_BH_0 * np.exp(integral_H)
                Omega_m_ap = omegam / a_prime**3
                Omega_r_ap = Omega_r / a_prime**4
                H_ap = H0 * np.sqrt(Omega_m_ap + Omega_r_ap + Omega_BH_ap)
                
                return c_s / (a_prime**2 * H_ap)
            
            r_s, err = quad(integrand_rs, 0, a_d, limit=50)
            print(f"    ✓ r_s = {r_s:.2f} Mpc")
            
        except Exception as e:
            print(f"    ✗ r_s calculation failed: {e}")
            return False
        
        # Test a single SN distance
        print("  Testing SN distance at z=0.5...")
        try:
            z_test = 0.5
            a_test = 1 / (1 + z_test)
            
            def integrand_d(a_prime):
                def integrand_H(ln_ap):
                    ap = np.exp(ln_ap)
                    k = self.k_eff(ap, k1, k2, k3, a1, a2)
                    return k - 3
                
                integral_H, _ = quad(integrand_H, 0, np.log(a_prime), limit=50)
                Omega_BH_ap = Omega_BH_0 * np.exp(integral_H)
                Omega_m_ap = omegam / a_prime**3
                Omega_r_ap = Omega_r / a_prime**4
                H_ap = H0 * np.sqrt(Omega_m_ap + Omega_r_ap + Omega_BH_ap)
                
                return self.c_km_s / (a_prime**2 * H_ap)
            
            d_c, err = quad(integrand_d, a_test, 1.0, limit=50)
            d_L = d_c * (1 + z_test)
            mu = 5 * np.log10(d_L) + 25
            
            print(f"    ✓ mu(z={z_test}) = {mu:.2f}")
            
        except Exception as e:
            print(f"    ✗ Distance calculation failed: {e}")
            return False
        
        print("  ✓✓ All tests passed!")
        return True

# Run diagnostics
print("="*70)
print("CCBH LIKELIHOOD DIAGNOSTICS")
print("="*70)

emulator = CCBHEmulator()

# Test your grid-search best-fit
print("\n" + "="*70)
print("TEST 1: Grid-search best-fit")
print("="*70)
success1 = emulator.test_params(68.0, 0.32, 0.0, 1.5, 3.0, 0.03, 0.08)

# Test ΛCDM-like
print("\n" + "="*70)
print("TEST 2: ΛCDM-like (k1=k2=k3=3)")
print("="*70)
success2 = emulator.test_params(70.0, 0.30, 3.0, 3.0, 3.0, 0.03, 0.08)

# Test with small perturbations
print("\n" + "="*70)
print("TEST 3: Small perturbation from best-fit")
print("="*70)
success3 = emulator.test_params(68.5, 0.31, 0.05, 1.4, 2.8, 0.032, 0.085)

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Test 1 (best-fit):     {'PASS' if success1 else 'FAIL'}")
print(f"Test 2 (ΛCDM-like):    {'PASS' if success2 else 'FAIL'}")
print(f"Test 3 (perturbed):    {'PASS' if success3 else 'FAIL'}")

if not success1:
    print("\n⚠ WARNING: Your best-fit parameters are failing!")
    print("  This suggests a bug in the likelihood calculation.")
    print("  Compare carefully with your original grid-search code.")
