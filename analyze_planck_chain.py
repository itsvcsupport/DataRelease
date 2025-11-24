# analyze_planck_chain.py
import numpy as np

def load_chain(root):
    # minimal loader: assumes single chain file root_1.txt
    fname = f"chains/{root}_1.txt"
    data = np.loadtxt(fname)
    # Cobaya stores samples with many columns; last column is logpost, first N are params
    return data

def main():
    root = "planck_camb_clean"
    data = load_chain(root)

    # You can read param names from the .yaml Cobaya writes:
    import yaml
    with open(f"chains/{root}.yaml", "r") as f:
        info = yaml.safe_load(f)
    param_names = info["params"].keys()

    print("Parameters in this run:")
    print(list(param_names))

    # crude summary: just print mean and std of first few cosmological params
    # Cobaya's order in the chain is the same as params order in yaml
    cols = list(param_names)
    # pick cosmology ones
    keep = ["H0", "ombh2", "omch2", "ns", "As", "tau"]
    for p in keep:
        i = cols.index(p)
        mean = np.mean(data[:, i])
        std  = np.std(data[:, i])
        print(f"{p}: {mean:.3f} ± {std:.3f}")

if __name__ == "__main__":
    main()
