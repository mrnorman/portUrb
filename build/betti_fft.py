from netCDF4 import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray

def get_ind(arr,val) :
    return np.argmin(np.abs(arr-val))

winds = [i for i in range(5,24,2)]

N = len(winds)

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

nc_fixed = [xarray.open_mfdataset([prefix+"000000"+str(i).zfill(2)+".nc" for i in range(8,20)],concat_dim="num_time_steps",combine="nested") for prefix in prefixes_fixed]
nc_float = [xarray.open_mfdataset([prefix+"000000"+str(i).zfill(2)+".nc" for i in range(8,20)],concat_dim="num_time_steps",combine="nested") for prefix in prefixes_float]

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


fig = plt.figure(figsize=(8,6))
ax = plt.gca()
for i in range(N) :
  print(winds[i])
  var_float = np.array(nc_float[i]["betti_trace_turb_0"])
  dt = 12*1800/len(var_float)
  # fft_float = np.abs( np.fft.rfft(np.concatenate((var_float,var_float[::-1]))) )**2
  # freq      = np.fft.rfftfreq(len(var_float)*2,d=dt)
  fft_float = np.abs( np.fft.rfft(var_float) )**2
  freq      = np.fft.rfftfreq(len(var_float),d=dt)
  scale = 1/freq[1:]
  window = 30
  fft_float = np.convolve(fft_float[1:], np.ones(window)/window, mode='valid')
  scale     = np.convolve(scale        , np.ones(window)/window, mode='valid')
  ax.loglog(scale,fft_float,label=f"wind = {winds[i]} m/s")

ax.vlines(3.4,1.e-1,4.e8,linestyle="--",color="red"  ,label="dt = 3.4 s")
ax.vlines(9.2,1.e-1,4.e8,linestyle="--",color="blue" ,label="dt = 9.2 s")
ax.vlines(64 ,1.e-1,4.e8,linestyle="--",color="green",label="dt = 64  s")
ax.loglog(scale[get_ind(scale,1.5):],100*scale[get_ind(scale,1.5):]**2,color="black",linestyle="--",label="scale^2")
ax.set_ylim(bottom=1.e-6)
ax.set_xlabel("Temporal scale (seconds)")
ax.set_ylabel("Floating motion spectral power")
ax.legend(ncol=3,loc="lower right")
plt.savefig("betti_fft.png",dpi=600)
# plt.show()
plt.close()


