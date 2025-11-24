# ccbh_quick_eval.py
import argparse
import yaml
from cobaya.model import get_model
import numpy as np
import matplotlib.pyplot as plt
import ccbh_theory as c

def main():
    parser = argparse.ArgumentParser(description="Quick eval of CCBH for given k1,k2,k3")
    # standard params with defaults
    parser.add_argument("--H0",    type=float, default=70.0)
    parser.add_argument("--ombh2", type=float, default=0.0224)
    parser.add_argument("--omch2", type=float, default=0.12)
    parser.add_argument("--ns",    type=float, default=0.965)
    parser.add_argument("--As",    type=float, default=2.1e-9)
    parser.add_argument("--tau",   type=float, default=0.054)

    # CCBH params
    parser.add_argument("--k1",  type=float, required=True)
    parser.add_argument("--k2",  type=float, required=True)
    parser.add_argument("--k3",  type=float, required=True)
    parser.add_argument("--z_t1", type=float, default=4.0)
    parser.add_argument("--z_t2", type=float, default=1.0)

    parser.add_argument("--no-SN",     action="store_true")
    parser.add_argument("--no-BAO",    action="store_true")
    parser.add_argument("--no-Planck", action="store_true")
    parser.add_argument("--config",    type=str, default="ccbh_planck_sn_desi.yaml")
    args = parser.parse_args()

    # load base config
    with open(args.config, "r") as f:
        info = yaml.safe_load(f)

    # toggle likelihoods
    if args.no_SN:
        info["likelihood"].pop("sn.pantheonplus", None)
    if args.no_BAO:
        info["likelihood"].pop("bao.desi_dr2", None)
    if args.no_Planck:
        for key in list(info["likelihood"].keys()):
            if key.startswith("planck_2018"):
                info["likelihood"].pop(key)

    # Build a parameter point dict
    param_point = {
        "H0": args.H0,
        "ombh2": args.ombh2,
        "omch2": args.omch2,
        "ns": args.ns,
        "As": args.As,
        "tau": args.tau,
        "k1": args.k1,
        "k2": args.k2,
        "k3": args.k3,
        "z_t1": args.z_t1,
        "z_t2": args.z_t2,
    }

    # Build Cobaya model and evaluate posterior
    model = get_model(info)
    loglikes, derived = model.logposterior(param_point, return_derived=True)
    total_loglike = float(np.sum(loglikes))

    print("Loglikes per dataset:", loglikes)
    print("Total loglike:", total_loglike)
    print("Derived:", derived)

    # TT spectrum
    cl = model.provider.get_Cl(ell_factor=False)
    ell = np.arange(len(cl["tt"]))
    plt.figure()
    plt.semilogx(ell[2:], cl["tt"][2:], label="TT CCBH")
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$C_\ell^{TT}$ [$\mu K^2$]")
    plt.legend()
    out_tt = f"chains/quick_TT_k1_{args.k1}_k2_{args.k2}_k3_{args.k3}.png"
    plt.savefig(out_tt, bbox_inches="tight")
    print("Saved TT plot to", out_tt)

    # H(z) using your background
    z = np.linspace(0, 4, 200)
    Hz = [c.H_of_z(zi, args.H0, args.ombh2, args.omch2,
                   args.k1, args.k2, args.k3,
                   args.z_t1, args.z_t2) for zi in z]
    plt.figure()
    plt.plot(z, Hz, label="H(z) CCBH")
    plt.xlabel("z")
    plt.ylabel("H(z) [km/s/Mpc]")
    plt.legend()
    out_hz = f"chains/quick_Hz_k1_{args.k1}_k2_{args.k2}_k3_{args.k3}.png"
    plt.savefig(out_hz, bbox_inches="tight")
    print("Saved H(z) plot to", out_hz)

if __name__ == "__main__":
    main()
