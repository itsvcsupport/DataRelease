import numpy as np
import os

# Point to your actual data files
PANTHEON_PATH = "Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat"
DESI_CHAIN_PATH = "C:/Users/WilliamKellett/cobaya/base/desi-bao-all/chain.1.txt"

print("Testing Pantheon+ data loading...")
try:
    pantheon = np.loadtxt(PANTHEON_PATH, skiprows=1)
    print(f"✓ Pantheon+ loaded: {pantheon.shape[0]} SNe")
except Exception as e:
    print(f"✗ Failed to load Pantheon+: {e}")

print("\nTesting DESI chain loading...")
try:
    desi = np.loadtxt(DESI_CHAIN_PATH)
    print(f"✓ DESI chain loaded: {desi.shape}")
except Exception as e:
    print(f"✗ Failed to load DESI: {e}")

print("\nIf both loaded successfully, your data is ready for MCMC!")
