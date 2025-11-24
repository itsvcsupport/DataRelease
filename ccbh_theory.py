# ccbh_theory.py
import numpy as np
from cobaya.theory import Theory


class CCBHCosmology(Theory):
    """
    Background-only CCBH cosmology for SN+BAO:
      - Flat universe: Omega_k = 0
      - Radiation from T_cmb, N_eff
      - Matter from ombh2, omch2 (neutrinos treated as relativistic via N_eff)
      - Dark energy = cosmologically coupled BH fluid with rho_BH(a) set by k(a)
    """

    # Input and derived parameters exposed to Cobaya
    params = {
        # Standard background
        "H0": {
            "prior": {"min": 40.0, "max": 90.0},
            "ref": {"dist": "norm", "loc": 70.0, "scale": 5.0},
            "proposal": 1.0,
            "latex": r"H_0",
        },
        "ombh2": {
            "prior": {"min": 0.01, "max": 0.04},
            "ref": {"dist": "norm", "loc": 0.0224, "scale": 0.0005},
            "proposal": 0.0003,
            "latex": r"\Omega_b h^2",
        },
        "omch2": {
            "prior": {"min": 0.05, "max": 0.3},
            "ref": {"dist": "norm", "loc": 0.12, "scale": 0.005},
            "proposal": 0.001,
            "latex": r"\Omega_c h^2",
        },
        # CMB temperature and Neff fixed by default (can be freed if you want)
        "T_cmb": {
            "value": 2.7255,
            "latex": r"T_{\rm CMB}",
        },
        "N_eff": {
            "value": 3.046,
            "latex": r"N_{\rm eff}",
        },

        # CCBH coupling exponents: k(a) piecewise
        "k1": {
            "prior": {"min": -3.0, "max": 6.0},
            "ref": {"dist": "norm", "loc": 0.0, "scale": 1.0},
            "proposal": 0.1,
            "latex": r"k_1",
        },
        "k2": {
            "prior": {"min": -3.0, "max": 6.0},
            "ref": {"dist": "norm", "loc": 3.0, "scale": 1.0},
            "proposal": 0.1,
            "latex": r"k_2",
        },
        "k3": {
            "prior": {"min": -3.0, "max": 6.0},
            "ref": {"dist": "norm", "loc": 3.0, "scale": 1.0},
            "proposal": 0.1,
            "latex": r"k_3",
        },

        # Redshift of transitions between the three k(a) phases
        # We enforce z_t1 > z_t2 in calculate()
        "z_t1": {
            "prior": {"min": 1.0, "max": 10.0},
            "ref": {"dist": "norm", "loc": 4.0, "scale": 1.0},
            "proposal": 0.1,
            "latex": r"z_{t1}",
        },
        "z_t2": {
            "prior": {"min": 0.1, "max": 4.0},
            "ref": {"dist": "norm", "loc": 1.0, "scale": 0.5},
            "proposal": 0.1,
            "latex": r"z_{t2}",
        },

        # Derived quantities
      # Derived quantities
        "Omega_m": {"derived": True, "latex": r"\Omega_m"},
        "Omega_bh0": {"derived": True, "latex": r"\Omega_{\rm BH,0}"},
        # DESI BAO expects this name:
        "rdrag": {"derived": True, "latex": r"r_{\rm drag}"},
    }

    def initialize(self):
        # Constants and grids
        self.c = 299792.458  # km/s
        # z grid for distance integrals (covers Pantheon+ and DESI BAO nicely)
        self._z_max = 5.0
        self._nz = 4000

        # Place-holders for background tables
        self._z_grid = None
        self._H_grid = None
        self._chi_grid = None

    def initialize_with_provider(self, provider):
        self.provider = provider

    # ------------------------------------------------------------------
    # Cobaya interface: what this theory can provide
    # ------------------------------------------------------------------
    def get_can_provide(self):
        # Background functions requested by SN and BAO likelihoods
        return [
            "Hubble",
            "comoving_radial_distance",
            "angular_diameter_distance",
        ]

    def get_can_provide_params(self):
        # Expose rdrag for DESI BAO
        return ["Omega_m", "Omega_bh0", "rdrag"]

    # We don't depend on any other theory, so no get_requirements or must_provide

    # ------------------------------------------------------------------
    # Background helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _omega_gamma_from_Tcmb(T_cmb, h):
        """
        Photon density parameter today for given T_cmb [K] and h.
        Standard ~ 2.47e-5 / h^2 for T=2.7255 K.
        """
        T_ratio = T_cmb / 2.7255
        return 2.469e-5 * T_ratio**4 / h**2

    def _build_background_tables(self, H0, Omega_m, Omega_r, Omega_bh0,
                                 k1, k2, k3, z_t1, z_t2):
        # enforce z_t1 > z_t2; if not, swap
        if z_t1 <= z_t2:
            z_t1, z_t2 = z_t2, z_t1

        a_t1 = 1.0 / (1.0 + z_t1)
        a_t2 = 1.0 / (1.0 + z_t2)

        # z grid for distances
        z = np.linspace(0.0, self._z_max, self._nz)
        a = 1.0 / (1.0 + z)

        # piecewise BH density evolution using analytic integral of dln rho / dln a = k(a) - 3
        # integrals anchored at a=1, rho_BH(1) = Omega_bh0 * rho_crit,0
        ln_a = np.log(a + 1e-40)
        ln_a_t1 = np.log(a_t1)
        ln_a_t2 = np.log(a_t2)

        # start with zeros, then fill three regions
        ln_rho_BH_over_rho0 = np.zeros_like(a)

        # region 3: a >= a_t2, k = k3
        mask3 = a >= a_t2
        ln_rho_BH_over_rho0[mask3] = (k3 - 3.0) * ln_a[mask3]

        # region 2: a_t1 <= a < a_t2, k = k2
        mask2 = (a >= a_t1) & (a < a_t2)
        ln_rho_BH_over_rho0[mask2] = (
            (k3 - 3.0) * ln_a_t2
            + (k2 - 3.0) * (ln_a[mask2] - ln_a_t2)
        )

        # region 1: a < a_t1, k = k1
        mask1 = a < a_t1
        ln_rho_BH_over_rho0[mask1] = (
            (k3 - 3.0) * ln_a_t2
            + (k2 - 3.0) * (ln_a_t1 - ln_a_t2)
            + (k1 - 3.0) * (ln_a[mask1] - ln_a_t1)
        )

        # convert to Omega_BH(a) = Omega_bh0 * exp(ln_rho_BH_over_rho0)
        Omega_BH_a = Omega_bh0 * np.exp(ln_rho_BH_over_rho0)

        # dimensionless expansion history E(a) = H(a)/H0
        E2 = Omega_r * a**-4 + Omega_m * a**-3 + Omega_BH_a
        # keep it safe
        E = np.sqrt(np.maximum(E2, 1e-30))

        H = H0 * E  # km/s/Mpc

        # comoving radial distance chi(z) = ∫0^z c/H(z') dz'
        dz = z[1] - z[0]
        invH = 1.0 / H
        chi = np.cumsum(invH) * dz * self.c
        chi[0] = 0.0

        self._z_grid = z
        self._H_grid = H
        self._chi_grid = chi

        return a_t1, a_t2  # can be useful if you want to debug

    def _H_of_z(self, z):
        if self._z_grid is None:
            raise RuntimeError("Background tables not built yet.")
        z = np.atleast_1d(z)
        H = np.interp(z, self._z_grid, self._H_grid)
        return H

    def _chi_of_z(self, z):
        if self._z_grid is None:
            raise RuntimeError("Background tables not built yet.")
        z = np.atleast_1d(z)
        chi = np.interp(z, self._z_grid, self._chi_grid)
        return chi

    # ------------------------------------------------------------------
    # Sound horizon at drag epoch
    # ------------------------------------------------------------------
    def _z_drag_eisenstein_hu(self, omh2, obh2, T_cmb):
        """
        Eisenstein & Hu-style fitting formula for the drag redshift z_d.

        omh2  = Omega_m h^2
        obh2  = Omega_b h^2
        T_cmb in K
        """
        theta = T_cmb / 2.7
        # matter-radiation equality redshift (approx)
        z_eq = 2.5e4 * omh2 * theta**-4

        # fitting coefficients b1,b2 (Eisenstein & Hu 1998)
        b1 = 0.313 * omh2**-0.419 * (1.0 + 0.607 * omh2**0.674)
        b2 = 0.238 * omh2**0.223

        z_d = 1291.0 * omh2**0.251 / (1.0 + 0.659 * omh2**0.828)
        z_d *= 1.0 + b1 * obh2**b2

        # You could clamp to be > z_eq if desired, but not strictly necessary
        return z_d

    def _rs_drag(self, H0, Omega_m, Omega_r, Omega_b, a_max, n=800):
        """
        Compute sound horizon at drag epoch via direct integral:

        r_s(a_d) = c/(sqrt(3) H0) * ∫_0^{a_d} da / [a^2 E(a) sqrt(1+R(a))]
        R(a) = 3 Omega_b/(4 Omega_gamma) * a
        """
        # find Omega_gamma from Omega_r and N_eff assumption:
        # Omega_r = Omega_gamma * (1 + 0.2271 * N_eff)
        # => Omega_gamma = Omega_r / (1 + 0.2271 N_eff)
        N_eff = self._N_eff
        Omega_gamma = Omega_r / (1.0 + 0.2271 * N_eff)
        # baryon loading
        R_prefac = 3.0 * Omega_b / (4.0 * Omega_gamma)

        a = np.linspace(1e-6, a_max, n)
        # We approximate DE/BH negligible at early times, so E(a) ~ sqrt(Om a^-3 + Or a^-4)
        E2 = Omega_r * a**-4 + Omega_m * a**-3
        E = np.sqrt(np.maximum(E2, 1e-30))
        R = R_prefac * a
        integrand = 1.0 / (a**2 * E * np.sqrt(1.0 + R))

        # simple trapezoidal integral
        rs = (self.c / H0) / np.sqrt(3.0) * np.trapz(integrand, a)
        return rs

    # ------------------------------------------------------------------
    # Main Cobaya entry: calculate()
    # ------------------------------------------------------------------
    def calculate(self, state, want_derived=True, **params_values_dict):
        # unpack parameters
        H0 = params_values_dict["H0"]
        h = H0 / 100.0
        ombh2 = params_values_dict["ombh2"]
        omch2 = params_values_dict["omch2"]
        T_cmb = params_values_dict.get("T_cmb", 2.7255)
        N_eff = params_values_dict.get("N_eff", 3.046)
        self._N_eff = N_eff  # cache for rs

        k1 = params_values_dict["k1"]
        k2 = params_values_dict["k2"]
        k3 = params_values_dict["k3"]
        z_t1 = params_values_dict["z_t1"]
        z_t2 = params_values_dict["z_t2"]

        # Standard matter densities today
        Omega_b = ombh2 / h**2
        Omega_c = omch2 / h**2
        Omega_m = Omega_b + Omega_c

        # Radiation (photons + massless neutrinos)
        Omega_gamma = self._omega_gamma_from_Tcmb(T_cmb, h)
        Omega_nu_rel = Omega_gamma * 0.2271 * N_eff
        Omega_r = Omega_gamma + Omega_nu_rel

        # Flatness: Omega_BH0 = 1 - Omega_m - Omega_r
        Omega_bh0 = 1.0 - Omega_m - Omega_r

        # Build background tables H(z), chi(z)
        self._build_background_tables(
            H0, Omega_m, Omega_r, Omega_bh0, k1, k2, k3, z_t1, z_t2
        )

        if want_derived:
            # Drag scale
            omh2 = Omega_m * h**2
            z_d = self._z_drag_eisenstein_hu(omh2, ombh2, T_cmb)
            a_d = 1.0 / (1.0 + z_d)
            rs_drag = self._rs_drag(H0, Omega_m, Omega_r, Omega_b, a_d)

            state["derived"] = {
                "Omega_m": Omega_m,
                "Omega_bh0": Omega_bh0,
                "rdrag": rs_drag,
            }

    # ------------------------------------------------------------------
    # get_* methods for SN and BAO likelihoods
    # ------------------------------------------------------------------
    def get_Hubble(self, z, units="km/s/Mpc"):
        """
        Return H(z) in km/s/Mpc or 1/Mpc
        """
        H = self._H_of_z(z)
        if units == "km/s/Mpc":
            return H
        elif units == "1/Mpc":
            return H / self.c
        else:
            raise ValueError(f"Unsupported units for Hubble: {units}")

    def get_comoving_radial_distance(self, z):
        """
        Comoving radial distance chi(z) in Mpc.
        """
        chi = self._chi_of_z(z)
        return chi

    def get_angular_diameter_distance(self, z):
        """
        Angular diameter distance D_A(z) = chi(z)/(1+z) in Mpc.
        """
        z = np.atleast_1d(z)
        chi = self._chi_of_z(z)
        dA = chi / (1.0 + z)
        return dA


# ----------------------------------------------------------------------
# Convenience helpers for use outside Cobaya
# ----------------------------------------------------------------------
def make_ccbh_background(H0, ombh2, omch2,
                         k1, k2, k3, z_t1, z_t2,
                         T_cmb=2.7255, N_eff=3.046):
    """
    Build a CCBHCosmology instance and precompute background tables
    for the given parameter set, so we can query H(z), chi(z), etc.
    """
    cosmo = CCBHCosmology()
    cosmo.initialize()
    state = {}

    cosmo.calculate(
        state,
        want_derived=True,
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        T_cmb=T_cmb,
        N_eff=N_eff,
        k1=k1,
        k2=k2,
        k3=k3,
        z_t1=z_t1,
        z_t2=z_t2,
    )
    return cosmo, state


def H_of_z(z, H0, ombh2, omch2,
           k1, k2, k3, z_t1, z_t2,
           T_cmb=2.7255, N_eff=3.046):
    """
    Convenience wrapper: return H(z) [km/s/Mpc] for the CCBH model.
    """
    cosmo, _ = make_ccbh_background(
        H0, ombh2, omch2, k1, k2, k3, z_t1, z_t2, T_cmb, N_eff
    )
    return cosmo.get_Hubble(z, units="km/s/Mpc")


def E_of_z(z, H0, ombh2, omch2,
           k1, k2, k3, z_t1, z_t2,
           T_cmb=2.7255, N_eff=3.046):
    """
    Return dimensionless expansion rate E(z) = H(z)/H0.
    """
    Hz = H_of_z(z, H0, ombh2, omch2,
                k1, k2, k3, z_t1, z_t2,
                T_cmb=T_cmb, N_eff=N_eff)
    return np.asarray(Hz) / H0


def omega_ccbh0(H0, ombh2, omch2,
                k1, k2, k3, z_t1, z_t2,
                T_cmb=2.7255, N_eff=3.046):
    """
    Return Omega_BH,0 (present-day CCBH density fraction).
    This just reads the 'Omega_bh0' derived quantity you already compute.
    """
    _, state = make_ccbh_background(
        H0, ombh2, omch2, k1, k2, k3, z_t1, z_t2, T_cmb, N_eff
    )
    return state["derived"]["Omega_bh0"]
