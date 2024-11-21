from netCDF4 import Dataset
import numpy as np

nc = Dataset("/lustre/storm/nwp501/scratch/imn/portUrb/build/supercell_1000m_00000001.nc","r")
sfc_theta_pert = np.array(nc["theta_pert"][0,:,:])
print(np.min(sfc_theta_pert))

cold_pool = np.where( sfc_theta_pert <= -2 , True , False );
print( np.sum(cold_pool) / sfc_theta_pert.size );


