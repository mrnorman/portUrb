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

prefixes_fixed = [ "turbulent_wind-2.000000_fixed-_"  ,\
                   "turbulent_wind-5.000000_fixed-_"  ,\
                   "turbulent_wind-8.000000_fixed-_"  ,\
                   "turbulent_wind-11.000000_fixed-_" ,\
                   "turbulent_wind-14.000000_fixed-_" ,\
                   "turbulent_wind-17.000000_fixed-_" ,\
                   "turbulent_wind-20.000000_fixed-_" ,\
                   "turbulent_wind-23.000000_fixed-_" ,\
                   "turbulent_wind-26.000000_fixed-_" ]

prefixes_float = [ "turbulent_wind-2.000000_floating-_"  ,\
                   "turbulent_wind-5.000000_floating-_"  ,\
                   "turbulent_wind-8.000000_floating-_"  ,\
                   "turbulent_wind-11.000000_floating-_" ,\
                   "turbulent_wind-14.000000_floating-_" ,\
                   "turbulent_wind-17.000000_floating-_" ,\
                   "turbulent_wind-20.000000_floating-_" ,\
                   "turbulent_wind-23.000000_floating-_" ,\
                   "turbulent_wind-26.000000_floating-_" ]

nc_fixed = [xarray.open_mfdataset([prefix+"000000"+str(i).zfill(2)+".nc" for i in range(2,4)],concat_dim="num_time_steps",combine="nested") for prefix in prefixes_fixed]
nc_float = [xarray.open_mfdataset([prefix+"000000"+str(i).zfill(2)+".nc" for i in range(2,4)],concat_dim="num_time_steps",combine="nested") for prefix in prefixes_float]

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

for i in range(9) :
  print(winds[i])
  ufixed = np.array(nc_fixed[i]["u_samp_trace_turb_1"][:])
  vfixed = np.array(nc_fixed[i]["v_samp_trace_turb_1"][:])
  ufloat = np.array(nc_float[i]["u_samp_trace_turb_1"][:])
  vfloat = np.array(nc_float[i]["v_samp_trace_turb_1"][:])
  mag_fixed = np.sqrt(ufixed*ufixed+vfixed*vfixed)
  mag_float = np.sqrt(ufloat*ufloat+vfloat*vfloat)
  betti_pert = np.array(nc_float[i]["betti_trace_turb_0"][:])
  print("Fixed:")
  print("  Mean:   ",np.mean(mag_fixed))
  print("  stddev: ",np.std (mag_fixed))
  print("Floating:")
  print("  Mean:   ",np.mean(mag_float))
  print("  stddev: ",np.std (mag_float))
  hist_float,bin_edges  = np.histogram(mag_float ,bins=np.arange(0,30,0.05)    ,density=True)
  hist_fixed,bin_edges  = np.histogram(mag_fixed ,bins=np.arange(0,30,0.05)    ,density=True)
  hist_betti,bin_edges2 = np.histogram(betti_pert,bins=np.arange(np.min(betti_pert),np.max(betti_pert),0.01),density=True)
  f, (ax1,ax2,ax3,ax4) = plt.subplots(4, 1)
  ax1.stairs(hist_fixed           ,bin_edges ,fill=True)
  ax2.stairs(hist_float           ,bin_edges ,fill=True)
  ax3.stairs(hist_float-hist_fixed,bin_edges ,fill=True)
  ax4.stairs(hist_betti           ,bin_edges2,fill=True)
  ax1.set_ylabel("Fixed"           ,wrap=True)
  ax2.set_ylabel("Floating"        ,wrap=True)
  ax3.set_ylabel("Floating - fixed",wrap=True)
  ax4.set_ylabel("Floating Perturbations",wrap=True)
  ax4.set_xlabel("Wind speed (m/s)")
  plt.savefig("downstream_histogram_8D_wind="+str(winds[i])+".png",dpi=600)
  plt.close()

# xind = get_ind(x,xlen/3+128)
# for i in range(9) :
#   ufixed = np.array(nc_fixed[i]["avg_u"][khub,:,xind])
#   vfixed = np.array(nc_fixed[i]["avg_v"][khub,:,xind])
#   ufloat = np.array(nc_float[i]["avg_u"][khub,:,xind])
#   vfloat = np.array(nc_float[i]["avg_v"][khub,:,xind])
#   magfixed = np.sqrt(ufixed*ufixed+vfixed*vfixed)
#   magfloat = np.sqrt(ufloat*ufloat+vfloat*vfloat)
#   plt.plot(magfixed,label="fixed")
#   plt.plot(magfloat,label="float")
#   plt.title(str(winds[i])+" , 1 diam")
#   plt.legend()
#   plt.show()
#   plt.close()
# 
# xind = get_ind(x,xlen/3+128*2)
# for i in range(9) :
#   ufixed = np.array(nc_fixed[i]["avg_u"][khub,:,xind])
#   vfixed = np.array(nc_fixed[i]["avg_v"][khub,:,xind])
#   ufloat = np.array(nc_float[i]["avg_u"][khub,:,xind])
#   vfloat = np.array(nc_float[i]["avg_v"][khub,:,xind])
#   magfixed = np.sqrt(ufixed*ufixed+vfixed*vfixed)
#   magfloat = np.sqrt(ufloat*ufloat+vfloat*vfloat)
#   plt.plot(magfixed,label="fixed")
#   plt.plot(magfloat,label="float")
#   plt.title(str(winds[i])+" , 2 diam")
#   plt.legend()
#   plt.show()
#   plt.close()
# 
# xind = get_ind(x,xlen/3+128*4)
# for i in range(9) :
#   ufixed = np.array(nc_fixed[i]["avg_u"][khub,:,xind])
#   vfixed = np.array(nc_fixed[i]["avg_v"][khub,:,xind])
#   ufloat = np.array(nc_float[i]["avg_u"][khub,:,xind])
#   vfloat = np.array(nc_float[i]["avg_v"][khub,:,xind])
#   magfixed = np.sqrt(ufixed*ufixed+vfixed*vfixed)
#   magfloat = np.sqrt(ufloat*ufloat+vfloat*vfloat)
#   plt.plot(magfixed,label="fixed")
#   plt.plot(magfloat,label="float")
#   plt.title(str(winds[i])+" , 4 diam")
#   plt.legend()
#   plt.show()
#   plt.close()

# xind = get_ind(x,xlen/3+128*8)
# for i in range(9) :
#   ufixed = np.array(nc_fixed[i]["avg_u"][khub,:,xind])
#   vfixed = np.array(nc_fixed[i]["avg_v"][khub,:,xind])
#   ufloat = np.array(nc_float[i]["avg_u"][khub,:,xind])
#   vfloat = np.array(nc_float[i]["avg_v"][khub,:,xind])
#   magfixed = np.sqrt(ufixed*ufixed+vfixed*vfixed)
#   magfloat = np.sqrt(ufloat*ufloat+vfloat*vfloat)
#   plt.plot(magfixed,label="fixed")
#   plt.plot(magfloat,label="float")
#   plt.title(str(winds[i])+" , 8 diam")
#   plt.legend()
#   plt.show()
#   plt.close()

