from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray

def get_ind(arr,val) :
    return np.argmin(np.abs(arr-val))

t1 = 2
t2 = 4
times = [str(i).zfill(7) for i in range(t1,t2+1)]

winds = [i for i in range(5,24,2)]

N = len(winds)

# prefixes_fixed = [ "turbulent_fixed-yaw-upstream_wind-3.000000_fixed-_"  ,\
#                    "turbulent_fixed-yaw-upstream_wind-4.000000_fixed-_"  ,\
#                    "turbulent_fixed-yaw-upstream_wind-5.000000_fixed-_"  ,\
#                    "turbulent_fixed-yaw-upstream_wind-6.000000_fixed-_"  ,\
#                    "turbulent_fixed-yaw-upstream_wind-7.000000_fixed-_"  ,\
#                    "turbulent_fixed-yaw-upstream_wind-8.000000_fixed-_"  ,\
#                    "turbulent_fixed-yaw-upstream_wind-9.000000_fixed-_"  ,\
#                    "turbulent_fixed-yaw-upstream_wind-10.000000_fixed-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-11.000000_fixed-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-12.000000_fixed-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-13.000000_fixed-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-14.000000_fixed-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-15.000000_fixed-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-16.000000_fixed-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-17.000000_fixed-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-18.000000_fixed-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-19.000000_fixed-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-20.000000_fixed-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-21.000000_fixed-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-22.000000_fixed-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-23.000000_fixed-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-24.000000_fixed-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-25.000000_fixed-_" ]
# 
# prefixes_float = [ "turbulent_fixed-yaw-upstream_wind-3.000000_floating-_"  ,\
#                    "turbulent_fixed-yaw-upstream_wind-4.000000_floating-_"  ,\
#                    "turbulent_fixed-yaw-upstream_wind-5.000000_floating-_"  ,\
#                    "turbulent_fixed-yaw-upstream_wind-6.000000_floating-_"  ,\
#                    "turbulent_fixed-yaw-upstream_wind-7.000000_floating-_"  ,\
#                    "turbulent_fixed-yaw-upstream_wind-8.000000_floating-_"  ,\
#                    "turbulent_fixed-yaw-upstream_wind-9.000000_floating-_"  ,\
#                    "turbulent_fixed-yaw-upstream_wind-10.000000_floating-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-11.000000_floating-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-12.000000_floating-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-13.000000_floating-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-14.000000_floating-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-15.000000_floating-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-16.000000_floating-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-17.000000_floating-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-18.000000_floating-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-19.000000_floating-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-20.000000_floating-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-21.000000_floating-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-22.000000_floating-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-23.000000_floating-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-24.000000_floating-_" ,\
#                    "turbulent_fixed-yaw-upstream_wind-25.000000_floating-_" ]

prefixes_fixed = [ "turbulent_fixed-yaw-upstream_wind-5.000000_fixed-_"  ,\
                   "turbulent_fixed-yaw-upstream_wind-7.000000_fixed-_"  ,\
                   "turbulent_fixed-yaw-upstream_wind-9.000000_fixed-_"  ,\
                   "turbulent_fixed-yaw-upstream_wind-11.000000_fixed-_" ,\
                   "turbulent_fixed-yaw-upstream_wind-13.000000_fixed-_" ,\
                   "turbulent_fixed-yaw-upstream_wind-15.000000_fixed-_" ,\
                   "turbulent_fixed-yaw-upstream_wind-17.000000_fixed-_" ,\
                   "turbulent_fixed-yaw-upstream_wind-19.000000_fixed-_" ,\
                   "turbulent_fixed-yaw-upstream_wind-21.000000_fixed-_" ,\
                   "turbulent_fixed-yaw-upstream_wind-23.000000_fixed-_" ]

prefixes_float = [ "turbulent_fixed-yaw-upstream_wind-5.000000_floating-_"  ,\
                   "turbulent_fixed-yaw-upstream_wind-7.000000_floating-_"  ,\
                   "turbulent_fixed-yaw-upstream_wind-9.000000_floating-_"  ,\
                   "turbulent_fixed-yaw-upstream_wind-11.000000_floating-_" ,\
                   "turbulent_fixed-yaw-upstream_wind-13.000000_floating-_" ,\
                   "turbulent_fixed-yaw-upstream_wind-15.000000_floating-_" ,\
                   "turbulent_fixed-yaw-upstream_wind-17.000000_floating-_" ,\
                   "turbulent_fixed-yaw-upstream_wind-19.000000_floating-_" ,\
                   "turbulent_fixed-yaw-upstream_wind-21.000000_floating-_" ,\
                   "turbulent_fixed-yaw-upstream_wind-23.000000_floating-_" ]

nc_fixed = [xarray.open_mfdataset([prefix+"000000"+str(i).zfill(2)+".nc" for i in range(12,24)],concat_dim="num_time_steps",combine="nested") for prefix in prefixes_fixed]
nc_float = [xarray.open_mfdataset([prefix+"000000"+str(i).zfill(2)+".nc" for i in range(12,24)],concat_dim="num_time_steps",combine="nested") for prefix in prefixes_float]

x = np.array(nc_fixed[0]["x"])
y = np.array(nc_fixed[0]["y"])
z = np.array(nc_fixed[0]["z"])
dx = x[1]-x[0]
dy = y[1]-y[0]
dz = z[1]-z[0]
xlen = x[-1]+dx/2
ylen = y[-1]+dy/2
zlen = z[-1]+dz/2
ihub = get_ind(x,xlen/2)
jhub = get_ind(y,ylen/2)
khub = get_ind(z,90)

wind_mean_fixed = np.array([0. for i in range(N)])
wind_mean_float = np.array([0. for i in range(N)])
wind_std_fixed = np.array([0. for i in range(N)])
wind_std_float = np.array([0. for i in range(N)])
for i in range(N) :
  print(winds[i])
  var_fixed = np.array(nc_fixed[i]["u_samp_trace_turb_1"])
  var_float = np.array(nc_float[i]["u_samp_trace_turb_1"])
  wind_mean_fixed[i] = np.mean(var_fixed)
  wind_mean_float[i] = np.mean(var_float)
  wind_std_fixed[i] = np.std(var_fixed)
  wind_std_float[i] = np.std(var_float)
  print("Uvel sample disk Fixed:")
  print("  Mean:   ",np.mean(var_fixed))
  print("  stddev: ",np.std (var_fixed))
  print("Uvel sample disk Floating:")
  print("  Mean:   ",np.mean(var_float))
  print("  stddev: ",np.std (var_float))
  mn = min(np.min(var_fixed),np.min(var_float))
  mx = max(np.max(var_fixed),np.max(var_float))
  # if (mx-mn > 0.01) :
  #   hist_float,bin_edges  = np.histogram(var_float,bins=np.arange(mn,mx,(mx-mn)/1000),density=True)
  #   hist_fixed,bin_edges  = np.histogram(var_fixed,bins=np.arange(mn,mx,(mx-mn)/1000),density=True)
  #   f, (ax1,ax2,ax3) = plt.subplots(3, 1)
  #   ax1.stairs(hist_fixed           ,bin_edges ,fill=True)
  #   ax2.stairs(hist_float           ,bin_edges ,fill=True)
  #   ax3.stairs(hist_float-hist_fixed,bin_edges ,fill=True)
  #   ax1.set_ylabel("Fixed"           ,wrap=True)
  #   ax2.set_ylabel("Floating"        ,wrap=True)
  #   ax3.set_ylabel("Floating - fixed",wrap=True)
  #   ax3.set_xlabel("Power Generation (MW)")
  #   # plt.savefig("downstream_histogram_8D_wind="+str(winds[i])+".png",dpi=600)
  #   plt.show()
  #   plt.close()

  # var_fixed = np.array(nc_fixed[i]["betti_trace_turb_0"])
  # var_float = np.array(nc_float[i]["betti_trace_turb_0"])
  # print("Betti disk Fixed:")
  # print("  Mean:   ",np.mean(var_fixed))
  # print("  stddev: ",np.std (var_fixed))
  # print("Betti disk Floating:")
  # print("  Mean:   ",np.mean(var_float))
  # print("  stddev: ",np.std (var_float))
  # mn = min(np.min(var_fixed),np.min(var_float))
  # mx = max(np.max(var_fixed),np.max(var_float))
  # if (mx-mn > 0.01) :
  #   hist_float,bin_edges  = np.histogram(var_float,bins=np.arange(mn,mx,(mx-mn)/1000),density=True)
  #   hist_fixed,bin_edges  = np.histogram(var_fixed,bins=np.arange(mn,mx,(mx-mn)/1000),density=True)
  #   f, (ax1,ax2,ax3) = plt.subplots(3, 1)
  #   ax1.stairs(hist_fixed           ,bin_edges ,fill=True)
  #   ax2.stairs(hist_float           ,bin_edges ,fill=True)
  #   ax3.stairs(hist_float-hist_fixed,bin_edges ,fill=True)
  #   ax1.set_ylabel("Fixed"           ,wrap=True)
  #   ax2.set_ylabel("Floating"        ,wrap=True)
  #   ax3.set_ylabel("Floating - fixed",wrap=True)
  #   ax3.set_xlabel("Power Generation (MW)")
  #   # plt.savefig("downstream_histogram_8D_wind="+str(winds[i])+".png",dpi=600)
  #   plt.show()
  #   plt.close()

print(wind_mean_float-wind_mean_fixed)
print(wind_std_float-wind_std_fixed)

# plt.plot(winds,wind_mean_float-wind_mean_fixed,label="wind_mean (float-fixed)")
# plt.legend()
# plt.xlabel("Average Hub Height Wind Speed (m/s)")
# plt.grid()
# plt.show()
# plt.close()
# 
# 
# plt.plot(winds,wind_std_float-wind_std_fixed,label="wind_std (float-fixed)")
# plt.legend()
# plt.xlabel("Average Hub Height Wind Speed (m/s)")
# plt.grid()
# plt.show()
# plt.close()
