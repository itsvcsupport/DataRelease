# run_ccbh_cobaya.py
import argparse
import yaml
from cobaya.run import run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-SN",  action="store_true", help="Disable Pantheon+ likelihood")
    parser.add_argument("--no-BAO", action="store_true", help="Disable DESI BAO likelihood")
    parser.add_argument("--tag",    type=str, default="ccbh_run", help="Chain output tag")
    parser.add_argument("--config", type=str, default="ccbh_planck_sn_desi.yaml",
                        help="Base Cobaya YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        info = yaml.safe_load(f)

    # Toggle SN / BAO
    if args.no_SN:
        info["likelihood"].pop("sn.pantheonplus", None)
    if args.no_BAO:
        info["likelihood"].pop("bao.desi_dr2", None)

    # Output root (chains will be chains/<tag>_*.txt)
    info["output"] = f"chains/{args.tag}"

    print(f"Running Cobaya with output root: {info['output']}")
    updated_info, sampler = run(info)

    print("\nRun finished.")
    if "bestfit" in updated_info:
        print("Best-fit parameters:")
        for p, v in updated_info["bestfit"].items():
            if p.startswith("__"):
                continue
            print(f"  {p}: {v}")

if __name__ == "__main__":
    main()
