from netCDF4 import Dataset
import numpy as np

nc0 = Dataset("/lustre/orion/stf006/scratch/imn/portUrb/build/baseline.nc","r")
nc  = Dataset("/lustre/orion/stf006/scratch/imn/portUrb/build/supercell_2000m_00000001.nc","r")
for var in nc.variables.keys() :
  m0 = np.mean(np.abs(np.array(nc0[var])))
  m  = np.mean(np.abs(np.array(nc [var])))
  if (m0 > 0) :
    print( f"{var:20}:  {m-m0:20.3e}  {(m-m0)/m0:20.3e}" )

