from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

def get_ind(arr,val) :
    return np.argmin(np.abs(arr-val))

nfiles = 72

files = [f"/lustre/storm/nwp501/scratch/imn/portUrb/build/supercell_1000m_{i:08}.nc" for i in range(nfiles)]
z = np.array(Dataset(files[0],"r")["z"])
sfc_theta_min  = [0. for i in range(nfiles)]
cold_pool_frac = [0. for i in range(nfiles)]
precip_accum   = [0. for i in range(nfiles)]
min_w          = [0. for i in range(nfiles)]
max_w          = [0. for i in range(nfiles)]
for i in range(nfiles) :
  nc = Dataset(files[i],"r")
  sfc_theta = np.array(nc["theta_pert"][0,:,:])
  w         = np.array(nc["wvel"])
  rho_d     = np.array(nc["density_dry"])
  sfc_theta_min [i] = np.min(sfc_theta)
  cold_pool_frac[i] = np.sum(np.where(sfc_theta <= -2,True,False)) / sfc_theta.size
  precip_accum  [i] = np.mean(np.array(nc["micro_rainnc"]) + np.array(nc["micro_snownc"]) + np.array(nc["micro_graupelnc"]))
  min_w         [i] = np.min(w[:get_ind(z,3500),:,:])
  max_w         [i] = np.max(w)

plt.plot(sfc_theta_min)
plt.title("sfc_theta_min")
plt.show()
plt.close()

plt.plot(cold_pool_frac)
plt.title("cold_pool_frac")
plt.show()
plt.close()

plt.plot(precip_accum)
plt.title("precip_accum")
plt.show()
plt.close()

plt.plot(min_w)
plt.title("min_w")
plt.show()
plt.close()

plt.plot(max_w)
plt.title("max_w")
plt.show()
plt.close()


nc = Dataset(files[-1],"r")
rho_d = np.array(nc["density_dry"])
rho_v = np.array(nc["water_vapor"])
rho_c = np.array(nc["cloud_water"])
rho_r = np.array(nc["rain_water" ])
rho_i = np.array(nc["cloud_ice"  ])
rho_s = np.array(nc["snow"       ])
rho_g = np.array(nc["graupel"    ])

qv = np.array(nc["water_vapor"]) / rho_d
qc = np.array(nc["cloud_water"]) / rho_d
qr = np.array(nc["rain_water" ]) / rho_d
qi = np.array(nc["cloud_ice"  ]) / rho_d
qs = np.array(nc["snow"       ]) / rho_d
qg = np.array(nc["graupel"    ]) / rho_d

# plt.plot(qv,label="qv")
plt.plot(np.mean(qc,axis=(1,2))*1000,z,label="qc")
plt.plot(np.mean(qr,axis=(1,2))*1000,z,label="qr")
plt.plot(np.mean(qi,axis=(1,2))*1000,z,label="qi")
plt.plot(np.mean(qs,axis=(1,2))*1000,z,label="qs")
plt.plot(np.mean(qg,axis=(1,2))*1000,z,label="qg")
plt.xlim(0,1)
plt.legend()
plt.show()


