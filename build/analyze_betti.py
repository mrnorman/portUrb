from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray

def get_ind(arr,val) :
    return np.argmin(np.abs(arr-val))

t1 = 1
t2 = 1
times = [str(i).zfill(7) for i in range(t1,t2+1)]

winds = [2,5,8,11,14,17,20,23,26]

prefixes_float = [ "dt_0.05/turbulent_wind-2.000000_floating-_"  ,\
                   "dt_0.05/turbulent_wind-5.000000_floating-_"  ,\
                   "dt_0.05/turbulent_wind-8.000000_floating-_"  ,\
                   "dt_0.05/turbulent_wind-11.000000_floating-_" ,\
                   "dt_0.05/turbulent_wind-14.000000_floating-_" ,\
                   "dt_0.05/turbulent_wind-17.000000_floating-_" ,\
                   "dt_0.05/turbulent_wind-20.000000_floating-_" ,\
                   "dt_0.05/turbulent_wind-23.000000_floating-_" ,\
                   "dt_0.05/turbulent_wind-26.000000_floating-_" ]

nc_float = [Dataset(prefix+"00000010.nc","r") for prefix in prefixes_float]

prefixes_float2 = [ "turbulent_wind-2.000000_floating-_"  ,\
                    "turbulent_wind-5.000000_floating-_"  ,\
                    "turbulent_wind-8.000000_floating-_"  ,\
                    "turbulent_wind-11.000000_floating-_" ,\
                    "turbulent_wind-14.000000_floating-_" ,\
                    "turbulent_wind-17.000000_floating-_" ,\
                    "turbulent_wind-20.000000_floating-_" ,\
                    "turbulent_wind-23.000000_floating-_" ,\
                    "turbulent_wind-26.000000_floating-_" ]

nc_float2 = [Dataset(prefix+"00000001.nc","r") for prefix in prefixes_float2]

x = np.array(nc_float[0]["x"])
y = np.array(nc_float[0]["y"])
z = np.array(nc_float[0]["z"])
dx = x[1]-x[0]
dy = y[1]-y[0]
dz = z[1]-z[0]
xlen = x[-1]+dx/2
ylen = y[-1]+dy/2
zlen = z[-1]+dz/2
ihub = get_ind(x,xlen/2)
jhub = get_ind(y,ylen/2)
khub = get_ind(z,90)

for i in range(9) :
  print(winds[i])
  arr = np.array(nc_float2[i]["betti_trace_turb_0"][:])
  mn = np.min(arr)
  mx = np.max(arr)
  hist_betti,bin_edges2 = np.histogram(arr,bins=np.arange(mn,mx,(mx-mn)/100),density=True)
  plt.stairs(hist_betti           ,bin_edges2,fill=True)
  plt.xlabel("Wind speed (m/s)")
  plt.show()
  plt.close()

  # arr = np.array(nc_float[i]["mag195_trace_turb_0"][:])
  # mn = np.min(arr)
  # mx = np.max(arr)
  # hist_betti,bin_edges2 = np.histogram(arr,bins=np.arange(mn,mx,(mx-mn)/100),density=True)
  # plt.stairs(hist_betti           ,bin_edges2,fill=True)
  # plt.xlabel("Wind speed (m/s)")
  # plt.show()
  # plt.close()

  # u = np.array(nc_float[i]["u_samp_trace_turb_0"][:])
  # v = np.array(nc_float[i]["v_samp_trace_turb_0"][:])
  # arr = np.sqrt(u*u + v*v)
  # mn = np.min(arr)
  # mx = np.max(arr)
  # hist_betti,bin_edges2 = np.histogram(arr,bins=np.arange(mn,mx,(mx-mn)/100),density=True)
  # plt.stairs(hist_betti           ,bin_edges2,fill=True)
  # plt.xlabel("Wind speed (m/s)")
  # plt.show()
  # plt.close()


#	double power_trace_turb_0(num_time_steps) ;
#	double yaw_trace_turb_0(num_time_steps) ;
#	double u_samp_trace_turb_0(num_time_steps) ;
#	double v_samp_trace_turb_0(num_time_steps) ;
#	double mag195_trace_turb_0(num_time_steps) ;
#	double betti_trace_turb_0(num_time_steps) ;
#	double cp_trace_turb_0(num_time_steps) ;
#	double ct_trace_turb_0(num_time_steps) ;

