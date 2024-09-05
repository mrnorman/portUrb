from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray

winds=[5,7,9,11,13,15,17,19,21,23]
# winds=[5,7]

for wind in winds :
  nc1 = Dataset(f"turbulent_fixed-yaw-upstream_wind-{wind}.000000_fixed-_precursor_00000010.nc","r")
  nc2 = Dataset(f"turbulent_fixed-yaw-upstream_wind-{wind}.000000_floating-_precursor_00000010.nc","r")
  for vname in nc1.variables.keys() :
    v1 = np.array(nc1[vname])
    v2 = np.array(nc2[vname])
    print(f"Wind [{wind}], Var[{vname:20}], MAE: {np.sum(np.abs(v2-v1))/v1.size:12.5}")
  print("\n")

