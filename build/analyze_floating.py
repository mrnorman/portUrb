from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray

t1 = 2
t2 = 3
times = [str(i).zfill(7) for i in range(t1,t2+1)]

winds = [2,5,8,11,14,17,20,23,26]

prefixes_float = [ "turbulent_wind-2.000000_floating-_"  ,\
                   "turbulent_wind-5.000000_floating-_"  ,\
                   "turbulent_wind-8.000000_floating-_"  ,\
                   "turbulent_wind-11.000000_floating-_" ,\
                   "turbulent_wind-14.000000_floating-_" ,\
                   "turbulent_wind-17.000000_floating-_" ,\
                   "turbulent_wind-20.000000_floating-_" ,\
                   "turbulent_wind-23.000000_floating-_" ,\
                   "turbulent_wind-26.000000_floating-_" ]

prefixes_fixed = [ "turbulent_wind-2.000000_fixed-_"  ,\
                   "turbulent_wind-5.000000_fixed-_"  ,\
                   "turbulent_wind-8.000000_fixed-_"  ,\
                   "turbulent_wind-11.000000_fixed-_" ,\
                   "turbulent_wind-14.000000_fixed-_" ,\
                   "turbulent_wind-17.000000_fixed-_" ,\
                   "turbulent_wind-20.000000_fixed-_" ,\
                   "turbulent_wind-23.000000_fixed-_" ,\
                   "turbulent_wind-26.000000_fixed-_" ]

nc_fixed = [xarray.open_mfdataset(prefix+"0*.nc",concat_dim="num_time_steps",combine="nested") for prefix in prefixes_fixed]
nc_float = [xarray.open_mfdataset(prefix+"0*.nc",concat_dim="num_time_steps",combine="nested") for prefix in prefixes_float]

for i in range(9) :
  plt.plot(np.array(nc_fixed[i]["power_trace_turb_0"]),label="fixed")
  plt.plot(np.array(nc_float[i]["power_trace_turb_0"]),label="float")
  plt.legend()
  plt.show()
  plt.close()
