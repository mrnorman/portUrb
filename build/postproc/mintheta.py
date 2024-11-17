from netCDF4 import Dataset
import numpy as np

nc = Dataset("/lustre/storm/nwp501/scratch/imn/portUrb/build/supercell_2000m_00000001.nc","r")
print(np.min(np.array(nc["theta_pert"][0,:,:])))

