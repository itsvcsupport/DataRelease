import os
import numpy as np

c = 299792.458  # km/s


class Planck18DistancePriors:
    def __init__(self, filename: str):
        self.load_file(filename)

    def load_file(self, filename: str):
        # Resolve full path and show where we're reading from
        path = os.path.abspath(filename)
        print(f"[Planck18DistancePriors] Reading priors from: {path}")

        with open(path, "r") as f:
            raw = f.readlines()

        # Show raw lines for debugging
        print("[Planck18DistancePriors] Raw non-empty lines (including comments):")
        for i, l in enumerate(raw):
            if l.strip():
                print(f"  {i}: {repr(l)}")

        # Filter out blank lines and comment lines (even if they have leading spaces)
        lines = [l.strip() for l in raw if l.strip() and not l.lstrip().startswith("#")]

        if len(lines) < 6:
            raise RuntimeError(
                f"Expected at least 6 numeric lines in {path}, got {len(lines)}. "
                f"Filtered lines = {lines}"
            )

        # First line = mean vector
        self.mean = np.array([float(x) for x in lines[0].split()])

        # Second line = sigmas
        self.sigma = np.array([float(x) for x in lines[1].split()])

        # Next 4 lines = correlation matrix
        corr = np.array([[float(y) for y in lines[i].split()] for i in range(2, 6)])

        # Build covariance matrix
        self.cov = corr * np.outer(self.sigma, self.sigma)
        self.icov = np.linalg.inv(self.cov)

    # -------------------------------------------------------
    # Background helpers
    # -------------------------------------------------------

    def _omega_r_and_omega_gamma_h2(self, params: dict):
        """
        Compute present-day radiation density Ω_r and Ω_γ h^2
        using eq. (6) and standard photon density formula.
        """
        H0 = params["H0"]
        h = H0 / 100.0
        Omega_m = params["Omega_m"]
        Tcmb = params.get("T_cmb", 2.7255)

        # Photon density parameter: Ω_γ h^2 ≈ 2.469e-5 (Tcmb/2.7255)^4
        Omega_gamma_h2 = 2.469e-5 * (Tcmb / 2.7255) ** 4

        # Matter density today: Ω_m h^2
        Omega_m_h2 = Omega_m * h ** 2

        # z_eq from eq. (6): z_eq = 2.5×10^4 Ω_m h^2 (Tcmb/2.7K)^(-4)
        z_eq = 2.5e4 * Omega_m_h2 * (Tcmb / 2.7) ** (-4.0)

        # Ω_r = Ω_m / (1 + z_eq)
        Omega_r = Omega_m / (1.0 + z_eq)

        return Omega_r, Omega_gamma_h2

    def E(self, z: float, params: dict) -> float:
        """
        Dimensionless Hubble rate E(z) = H(z)/H0, flat ΛCDM with radiation,
        following eq. (5) in Chen+2018.
        """
        Omega_m = params["Omega_m"]
        Omega_r, _ = self._omega_r_and_omega_gamma_h2(params)

        # Flat universe: Ω_k = 0, Ω_de = 1 - Ω_m - Ω_r
        Omega_k = 0.0
        Omega_de = 1.0 - Omega_m - Omega_r

        a = 1.0 / (1.0 + z)
        Ez2 = (
            Omega_r * a ** (-4) +
            Omega_m * a ** (-3) +
            Omega_k * a ** (-2) +
            Omega_de
        )
        return np.sqrt(Ez2)

    def comoving_distance(self, z: float, params: dict) -> float:
        """
        Comoving distance r(z) = c/H0 ∫_0^z dz'/E(z').
        """
        z_grid = np.linspace(0.0, z, 2000)
        ez = np.array([self.E(zz, params) for zz in z_grid])
        integral = np.trapezoid(1.0 / ez, z_grid)
        return (c / params["H0"]) * integral  # Mpc

    def sound_horizon_z(self, z: float, params: dict) -> float:
        """
        Comoving sound horizon r_s(z) in Mpc using eq. (3) of Chen+2018.
        """
        H0 = params["H0"]
        Omega_b_h2 = params["Omega_b_h2"]
        Omega_r, Omega_gamma_h2 = self._omega_r_and_omega_gamma_h2(params)

        def E_of_a(a):
            z_local = 1.0 / a - 1.0
            return self.E(z_local, params)

        def integrand(a):
            # Baryon-loading term R_b(a) = 3 Ω_b h^2 / (4 Ω_γ h^2 a)
            R_b = 3.0 * Omega_b_h2 / (4.0 * Omega_gamma_h2 * a)
            return 1.0 / (a ** 2 * E_of_a(a) * np.sqrt(3.0 * (1.0 + R_b)))

        a_star = 1.0 / (1.0 + z)
        a_grid = np.linspace(1e-8, a_star, 5000)
        integrand_vals = np.array([integrand(a) for a in a_grid])
        integral = np.trapezoid(integrand_vals, a_grid)

        return (c / H0) * integral  # Mpc

    def z_star(self, params: dict) -> float:
        """
        Hu & Sugiyama decoupling redshift z* (eqs. 8–10 in Chen+2018).
        """
        obh2 = params["Omega_b_h2"]
        omh2 = params["Omega_m"] * (params["H0"] / 100.0) ** 2

        g1 = 0.0783 * obh2 ** (-0.238) / (1.0 + 39.5 * obh2 ** 0.763)
        g2 = 0.560 / (1.0 + 21.1 * obh2 ** 1.81)

        return 1048.0 * (1.0 + 0.00124 * obh2 ** (-0.738)) * (1.0 + g1 * omh2 ** g2)

    def R_and_la(self, params: dict):
        """
        Compute the shift parameter R and acoustic scale l_A.
        """
        zstar = self.z_star(params)
        rstar = self.comoving_distance(zstar, params)
        r_s = self.sound_horizon_z(zstar, params)

        # Shift parameter R (eq. 2)
        R = np.sqrt(params["Omega_m"]) * (params["H0"] / 100.0) * rstar / (c / 100.0)

        # Acoustic scale l_A (eq. 1): l_A = (1+z*) π D_A(z*) / r_s(z*)
        # D_A = r(z*) / (1+z*)
        la = (1.0 + zstar) * np.pi * rstar / r_s

        return R, la

    # -------------------------------------------------------
    # Likelihood functions
    # -------------------------------------------------------

    def p_model(self, params: dict) -> np.ndarray:
        R, la = self.R_and_la(params)
        return np.array(
            [
                R,
                la,
                params["Omega_b_h2"],
                params["n_s"],
            ]
        )

    def chi2(self, params: dict) -> float:
        p = self.p_model(params)
        d = p - self.mean
        return float(d.T @ self.icov @ d)

    def loglike(self, params: dict) -> float:
        return -0.5 * self.chi2(params)


# -------------------------------------------------------
# Usage example
# -------------------------------------------------------

if __name__ == "__main__":
    priors = Planck18DistancePriors("planck2018_distance_priors.txt")

    # Planck-ish ΛCDM reference parameters (base TT,TE,EE+lowE)
    params_LCDM = {
        "H0": 67.4,
        "Omega_m": 0.315,
        "Omega_b_h2": 0.02236,
        "n_s": 0.9649,
        "T_cmb": 2.7255,
    }

    print("Planck 2018 chi2 (Planck-ish ΛCDM):", priors.chi2(params_LCDM))
