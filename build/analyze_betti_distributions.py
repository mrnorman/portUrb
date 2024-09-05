from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray

def get_ind(arr,val) :
    return np.argmin(np.abs(arr-val))

winds = [i for i in range(5,24,2)]

N = len(winds)

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

nc_float = [xarray.open_mfdataset([prefix+"000000"+str(i).zfill(2)+".nc" for i in range(8,21)],concat_dim="num_time_steps",combine="nested") for prefix in prefixes_float]

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

fig,axestmp = plt.subplots(2,5,figsize=(10,6),sharex=True)
axes = np.reshape(np.array(axestmp),10)
std = [0. for i in range(len(winds))]
print("Wind".center(5),"  ","Mean".center(12),"  ","Std Dev".center(12),"  ","Min".center(12),"  ","Max".center(12),"  ","Std dev / Wind".center(14),"  ","Disk TI".center(12))
for i in range(N) :
  var_float = np.array(nc_float[i]["betti_trace_turb_0"])
  mn = np.min(var_float)
  mx = np.max(var_float)
  std[i] = np.std(var_float)
  udisk = np.array(nc_float[i]["u_samp_trace_turb_0"])
  udisk_ti   = np.std(udisk)/winds[i]
  print(f"{winds[i]:5}  {np.mean(var_float):12.3e}   {std[i]:12.3e}   {mn:12.3e}   {mx:12.3e}   {std[i]/winds[i]:14.3e}   {udisk_ti:12.3e}")
  hist_float,bin_edges  = np.histogram(var_float,bins=np.arange(mn,mx,(mx-mn)/1000),density=True)
  axes[i].stairs(hist_float,bin_edges,fill=True)
  axes[i].set_title(f"Wind: {winds[i]}")
  axes[i].margins(x=0)

plt.tight_layout()
plt.savefig("betti_distributions.png",dpi=600)
plt.close()

fig = plt.figure(figsize=(10,6))
ax = fig.gca()
ax.plot(winds,std)
ax.margins(x=0)
ax.set_xlabel("Mean hub height wind speed (m/s)")
ax.set_ylabel("Standard Deviation (m/s)")
plt.tight_layout()
plt.savefig("betti_stddev.png",dpi=600)

