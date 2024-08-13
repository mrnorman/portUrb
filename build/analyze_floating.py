from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray

def get_ind(arr,val) :
    return np.argmin(np.abs(arr-val))

t1 = 2
t2 = 2
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

nc_fixed = [Dataset(prefix+"00000004.nc","r") for prefix in prefixes_fixed]
nc_float = [Dataset(prefix+"00000004.nc","r") for prefix in prefixes_float]

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

xind = get_ind(x,xlen/3+128*8)
for i in range(9) :
  ufixed = np.array(nc_fixed[i]["avg_u"][khub,:,xind])
  vfixed = np.array(nc_fixed[i]["avg_v"][khub,:,xind])
  ufloat = np.array(nc_float[i]["avg_u"][khub,:,xind])
  vfloat = np.array(nc_float[i]["avg_v"][khub,:,xind])
  magfixed = np.sqrt(ufixed*ufixed+vfixed*vfixed)
  magfloat = np.sqrt(ufloat*ufloat+vfloat*vfloat)
  plt.plot(magfixed,label="fixed")
  plt.plot(magfloat,label="float")
  plt.title(str(winds[i])+" , 8 diam")
  plt.legend()
  plt.show()
  plt.close()
