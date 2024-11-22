from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

nfiles = 72

files = [f"/lustre/orion/stf006/scratch/imn/portUrb/build/supercell_1000m_{i:08}.nc" for i in range(nfiles)]
sfc_theta_min  = [0. for i in range(nfiles)]
cold_pool_frac = [0. for i in range(nfiles)]
for i in range(nfiles) :
  nc = Dataset(files[i],"r")
  sfc_theta = np.array(nc["theta_pert"][0,:,:])
  sfc_theta_min [i] = np.min(sfc_theta)
  cold_pool_frac[i] = np.sum(np.where(sfc_theta <= -2,True,False)) / sfc_theta.size

plt.plot(sfc_theta_min)
plt.show()
plt.close()

plt.plot(cold_pool_frac)
plt.show()
plt.close()

