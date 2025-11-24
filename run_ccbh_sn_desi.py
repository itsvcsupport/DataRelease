# run_ccbh_sn_desi.py
import os
import argparse
from cobaya.run import run
from ccbh_theory import CCBHCosmology  # our external theory class


def build_info(test_mode="joint"):
    """
    Build Cobaya info dictionary for CCBH + Pantheon+ + DESI BAO (+ Planck).

    test_mode:
      - "sn"    : Pantheon+ only
      - "desi"  : DESI BAO only
      - "joint" : Pantheon+ + DESI
      - "full"  : Pantheon+ + DESI + Planck 2018 (CMB)
    """

    # ----------------------------------------
    # 1. Likelihood block
    # ----------------------------------------
    if test_mode == "sn":
        likelihood = {
            "sn.pantheonplus": None,
        }
        output_prefix = "chains/ccbh_sn_only"

    elif test_mode == "desi":
        likelihood = {
            "bao.desi_2024_bao_all": None,
        }
        output_prefix = "chains/ccbh_desi_only"

    elif test_mode == "joint":
        likelihood = {
            "sn.pantheonplus": None,
            "bao.desi_2024_bao_all": None,
        }
        output_prefix = "chains/ccbh_sn_desi_tuned"

    elif test_mode == "full":
        # SN + DESI + Planck 2018 CMB
        likelihood = {
            # SNe Ia
            "sn.pantheonplus": None,

            # DESI BAO Y1 (your existing setup)
            "bao.desi_2024_bao_all": None,

            # Planck 2018 Temperature + Polarization
            # These names assume you have the official Cobaya Planck 2018 data package installed.
            # You can adjust if your local names differ (see Cobaya docs).
            "planck_2018_lowl.TT": None,
            "planck_2018_lowl.EE": None,
            "planck_2018_highl_plik.TTTEEE": None,
        }
        output_prefix = "chains/ccbh_sn_desi_planck"

    else:
        raise ValueError(f"Unknown test_mode='{test_mode}'")

    # ----------------------------------------
    # 2. Info dict
    # ----------------------------------------
    info = {
        # Our external theory that wraps CLASS/CCBH
        "theory": {
            "CCBH": {
                "external": CCBHCosmology
            }
        },

        "likelihood": likelihood,

        "params": {
            # Primary cosmological + CCBH params
            "H0": {
                "prior": {"min": 40.0, "max": 90.0},
                "ref": {"dist": "norm", "loc": 70.0, "scale": 2.0},
                "proposal": 0.3,
            },
            "ombh2": {
                "prior": {"min": 0.01, "max": 0.04},
                "ref": {"dist": "norm", "loc": 0.0224, "scale": 0.0003},
                "proposal": 0.0001,
            },
            "omch2": {
                "prior": {"min": 0.05, "max": 0.3},
                "ref": {"dist": "norm", "loc": 0.12, "scale": 0.003},
                "proposal": 0.0008,
            },

            # CCBH logistic couplings + transition redshifts
            "k1": {
                "prior": {"min": -3.0, "max": 6.0},
                "ref": {"dist": "norm", "loc": 0.0, "scale": 0.5},
                "proposal": 0.15,
            },
            "k2": {
                "prior": {"min": -3.0, "max": 6.0},
                "ref": {"dist": "norm", "loc": 3.0, "scale": 0.5},
                "proposal": 0.15,
            },
            "k3": {
                "prior": {"min": -3.0, "max": 6.0},
                "ref": {"dist": "norm", "loc": 3.0, "scale": 0.5},
                "proposal": 0.15,
            },
            "z_t1": {
                "prior": {"min": 1.0, "max": 10.0},
                "ref": {"dist": "norm", "loc": 4.0, "scale": 0.6},
                "proposal": 0.2,
            },
            "z_t2": {
                "prior": {"min": 0.01, "max": 4.0},
                "ref": {"dist": "norm", "loc": 1.0, "scale": 0.4},
                "proposal": 0.15,
            },

            # Microphysics
            "T_cmb": {"value": 2.7255},
            "N_eff": {"value": 3.046},

            # Planck-sensitive parameters (if you want to vary them later):
            # For now, keep tau fixed or broad; you can turn it into a free param.
            "tau": {"value": 0.054},

            # Derived parameters (computed in your CCBH theory)
            "Omega_m": {"derived": True},
            "Omega_bh0": {"derived": True},
        },

        "sampler": {
            "mcmc": {
                "max_tries": 1000,
                "Rminus1_stop": 0.05,
                "Rminus1_cl_stop": 0.4,
                "drag": False,
                "proposal_scale": 0.5,
                "covmat": "auto",
            }
        },

        "output": output_prefix,
    }

    return info




def chi2_breakdown_from_chain(root):
    """
    Helper: read the output chains with GetDist and print χ²_SN and χ²_DESI.

    We don't rely on hard-coded parameter names; instead we look for
    chi2__* parameters whose names contain the right likelihood labels.
    """
    try:
        from getdist.mcsamples import loadMCSamples
    except ImportError:
        print("GetDist not found; install it (pip install getdist) for χ² helper.")
        return

    # Load Cobaya chains
    samples = loadMCSamples(root)

    # Get the parameter names from GetDist's ParamNames object
    param_names_obj = samples.getParamNames()
    names = [p.name for p in param_names_obj.names]

    def find_chi2(substr_options):
        """
        Find the first chi2__* parameter whose name contains any of the
        substrings in substr_options.
        """
        for name in names:
            if not name.startswith("chi2"):
                continue
            for s in substr_options:
                if s in name:
                    idx = names.index(name)
                    return name, samples.mean(idx)
        return None, None

    # Try to locate Pantheon+ and DESI chi2 parameters
    sn_name, chi2_sn = find_chi2(["pantheon", "sn.pantheonplus"])
    desi_name, chi2_desi = find_chi2(["desi_2024_bao_all", "bao.desi_2024_bao_all", "desi_2024"])

    print("\n=== χ² breakdown (chain means) ===")

    if chi2_sn is not None:
        print(f"{sn_name}: {chi2_sn:.2f}")
    else:
        print("Could not find a chi2__ parameter for Pantheon+ in the chain.")

    if chi2_desi is not None:
        print(f"{desi_name}: {chi2_desi:.2f}")
    else:
        print("Could not find a chi2__ parameter for DESI BAO in the chain.")


def main():
    parser = argparse.ArgumentParser(
        description="Run CCBH cosmology against Pantheon+ and DESI BAO."
    )
    parser.add_argument(
    "--test",
    choices=["sn", "desi", "joint", "full"],
    default="joint",
    help=(
        "Which test to run: "
        "'sn' (Pantheon+ only), "
        "'desi' (DESI BAO only), "
        "'joint' (SN+DESI), or "
        "'full' (SN+DESI+Planck 2018)."
    ),
)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing chain instead of starting a new one.",
    )
    args = parser.parse_args()

    info = build_info(test_mode=args.test)

    # Use Cobaya's resume flag so you don't hit the 'delete output' error
    updated_info, sampler = run(info, resume=args.resume)
    root = updated_info.get("output", "chains/ccbh_sn_desi_tuned")

    # For SN-only and joint we care about Pantheon+; for DESI-only and joint, DESI
    chi2_breakdown_from_chain(root)


if __name__ == "__main__":
    main()
