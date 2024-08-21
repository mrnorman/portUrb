from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray

nc = Dataset("building_precursor_0"+str(4).zfill(7)+".nc","r")
x = np.array(nc["x"])
y = np.array(nc["y"])
z = np.array(nc["z"])
dx = x[1]-x[0]
dy = y[1]-y[0]
dz = z[1]-z[0]
xlen = x[-1]+dx/2
ylen = y[-1]+dy/2
zlen = z[-1]+dz/2
u = np.array(nc["avg_u"])
v = np.array(nc["avg_v"])
w = np.array(nc["avg_w"])
counter = 1

for i in range(5,16) :
  nc = Dataset("building_precursor_0"+str(i).zfill(7)+".nc","r")
  u += np.array(nc["avg_u"])
  v += np.array(nc["avg_v"])
  w += np.array(nc["avg_w"])
  counter += 1

u /= counter
v /= counter
w /= counter

u_H = 3.226
H   = 0.2

# plt.plot(np.mean(u/u_H,axis=(1,2)),z/H)
# plt.grid()
# plt.show()
