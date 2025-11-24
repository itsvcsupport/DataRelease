from getdist.mcsamples import loadMCSamples
samples_joint = loadMCSamples("chains/ccbh_sn_desi_tuned")
for name in ["H0","Omega_m","k1","k2","k3","z_t1","z_t2"]:
    print(name, samples_joint.mean(name), samples_joint.std(name))