import numpy as np

# Load DESI chain
chain_path = 'C:/Users/WilliamKellett/cobaya/base/desi-bao-all/chain.1.txt'
chain = np.loadtxt(chain_path)

print("DESI chain shape:", chain.shape)
print("\nFirst 5 rows:")
print(chain[:5])
print("\nColumn statistics:")
for i in range(min(10, chain.shape[1])):
    print(f"  Column {i}: mean={chain[:, i].mean():.2f}, std={chain[:, i].std():.2f}, min={chain[:, i].min():.2f}, max={chain[:, i].max():.2f}")

# Find the right columns
print("\n" + "="*70)
print("Looking for H0*r_s (should be ~10,000-10,500)...")
for i in range(chain.shape[1]):
    mean = chain[:, i].mean()
    if 9000 < mean < 11000:
        print(f"  ✓ Column {i}: mean = {mean:.1f} - THIS IS H0*r_s!")

print("\nLooking for Omega_m (should be ~0.27-0.32)...")
for i in range(chain.shape[1]):
    mean = chain[:, i].mean()
    if 0.2 < mean < 0.4:
        print(f"  ✓ Column {i}: mean = {mean:.4f} - THIS IS Omega_m!")
