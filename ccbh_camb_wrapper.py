# ccbh_camb_wrapper.py
import numpy as np
import camb
from cobaya.theory import Theory
from ccbh_theory import E_of_z, omega_ccbh0


def approximate_w0_wa(H0, ombh2, omch2,
                      k1, k2, k3, z_t1, z_t2):
    """
    Crude mapping from your CCBH background to an effective (w0, wa)
    for CAMB's PPF dark energy.

    Idea:
      - Use Omega_BH0 as "dark energy fraction" today.
      - Assume DE negligible at high z (so w(z) ~ 0 early),
        and close to -1 today (for acceleration).
    """
    # You can make this fancier later. For now:
    Omega_BH0 = omega_ccbh0(H0, ombh2, omch2,
                            k1, k2, k3, z_t1, z_t2)
    # Start with almost-ΛCDM:
    w0 = -1.0

    # If BH fraction is large, allow a bit of evolution:
    # if Omega_BH0 > 0.7, let wa tilt slightly positive.
    wa = 0.2 * (Omega_BH0 - 0.7)

    # Clamp wa a bit so it doesn't go wild:
    wa = max(min(wa, 0.5), -0.5)

    return w0, wa


class CCBH_Cosmology(Theory):
    """
    Cobaya theory wrapper: uses CAMB with an effective DE (w0, wa)
    approximating the CCBH background.
    """

    def initialize(self):
        # create a CAMB params object; we'll reuse it
        self.pars = camb.CAMBparams()
        # Some sane accuracy defaults (tweak later if needed)
        self.pars.set_accuracy(AccuracyBoost=1.5,
                               lAccuracyBoost=1.2,
                               lSampleBoost=1)
    

    def get_requirements(self):
        # Tell Cobaya we provide C_l's and derived params
        return {"Cl": {"types": ["tt", "te", "ee", "pp"], "lmax": 2500},
                "derived": None}
    def get_can_provide_params(self):
        """
        Tell Cobaya which derived parameters this theory outputs.
        """
        return ["Omega_ccbh0", "sigma8"]

    def calculate(self, state, params):
        # --- 1. unpack standard parameters ---
        H0     = params["H0"]
        ombh2  = params["ombh2"]
        omch2  = params["omch2"]
        ns     = params["ns"]
        As     = params["As"]
        tau    = params["tau"]

        # --- 2. unpack CCBH parameters ---
        k1   = params["k1"]
        k2   = params["k2"]
        k3   = params["k3"]
        z_t1 = params["z_t1"]
        z_t2 = params["z_t2"]

        # --- 3. baseline flat cosmology, no explicit Λ ---
        self.pars.set_cosmology(
            H0=H0,
            ombh2=ombh2,
            omch2=omch2,
            omk=0.0,
            mnu=0.0,
            num_massive_neutrinos=0,
            Omega_lambda=0.0  # CCBH will play the role of DE
        )
        self.pars.InitPower.set_params(ns=ns, As=As)
        self.pars.set_for_lmax(2500, lens_potential_accuracy=1)
        self.pars.Reion.set_tau(tau)

        # --- 4. approximate CCBH as (w0, wa) ---
        w0, wa = approximate_w0_wa(H0, ombh2, omch2, k1, k2, k3, z_t1, z_t2)
        self.pars.set_dark_energy(w=w0, wa=wa, dark_energy_model="ppf")

        # --- 5. run CAMB ---
        results = camb.get_results(self.pars)

        cl_tot = results.get_cmb_power_spectra(self.pars, CMB_unit="muK")["total"]
        # cl_tot shape: (lmax+1, 4): TT, EE, BB, TE
        cl_lens = results.get_lens_potential_cls(lmax=2500)

        state["Cl"] = {
            "tt": cl_tot[:, 0],
            "ee": cl_tot[:, 1],
            "bb": cl_tot[:, 2],
            "te": cl_tot[:, 3],
            "pp": cl_lens[:, 0],
        }

        # --- 6. derived parameters ---
        # reuse your Omega_BH0, and CAMB's sigma8
        Omega_BH0 = omega_ccbh0(H0, ombh2, omch2,
                                k1, k2, k3, z_t1, z_t2)
        state["derived"] = {
            "Omega_ccbh0": Omega_BH0,
            "sigma8": results.get_sigma8(),
        }
