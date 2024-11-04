from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray

def get_ind(arr,val) :
  return np.argmin(np.abs(arr-val))

t1 = 10
t2 = 49
prefix = "/lustre/storm/nwp501/scratch/imn/cubes_periodic/cubes_periodic_z0-1.000e-03_"
fnames = [f"{prefix}{i:08}.nc" for i in range(t1,t2+1)]

nc   = xarray.open_dataset(fnames[0])
x    = np.array(nc["x"])
y    = np.array(nc["y"])
z    = np.array(nc["z"])
nz   = len(z)
dx   = x[1]-x[0]
dy   = y[1]-y[0]
xlen = x[-1]+dx/2
ylen = y[-1]+dy/2
p0_x = [get_ind(x,2*xlen/8),get_ind(x,6*xlen/8),get_ind(x,2*xlen/8),get_ind(x,6*xlen/8)]
p0_y = [get_ind(y,6*ylen/8),get_ind(y,4*ylen/8),get_ind(y,2*ylen/8),get_ind(y,0*ylen/8)]
p1_x = [get_ind(x,4*ylen/8),get_ind(x,0*ylen/8),get_ind(x,4*ylen/8),get_ind(x,0*ylen/8)]
p1_y = [get_ind(y,6*ylen/8),get_ind(y,4*ylen/8),get_ind(y,2*ylen/8),get_ind(y,0*ylen/8)]
p2_x = [get_ind(x,2*xlen/8),get_ind(x,6*xlen/8),get_ind(x,2*xlen/8),get_ind(x,6*xlen/8)]
p2_y = [get_ind(y,4*ylen/8),get_ind(y,2*ylen/8),get_ind(y,0*ylen/8),get_ind(y,6*ylen/8)]
p3_x = [get_ind(x,4*xlen/8),get_ind(x,0*xlen/8),get_ind(x,4*xlen/8),get_ind(x,0*xlen/8)]
p3_y = [get_ind(y,4*ylen/8),get_ind(y,2*ylen/8),get_ind(y,0*ylen/8),get_ind(y,6*ylen/8)]
u0 = np.array([0. for k in range(nz)])
u1 = np.array([0. for k in range(nz)])
u2 = np.array([0. for k in range(nz)])
u3 = np.array([0. for k in range(nz)])
for i in range(len(fnames)) :
  nc = xarray.open_dataset(fnames[i])
  for j in range(len(p0_x)) :
    u0 += np.array(nc["avg_u"][:,p0_y[j],p0_x[j]])
    u1 += np.array(nc["avg_u"][:,p1_y[j],p1_x[j]])
    u2 += np.array(nc["avg_u"][:,p2_y[j],p2_x[j]])
    u3 += np.array(nc["avg_u"][:,p3_y[j],p3_x[j]])
u0 /= len(fnames)*len(p0_x)
u1 /= len(fnames)*len(p0_x)
u2 /= len(fnames)*len(p0_x)
u3 /= len(fnames)*len(p0_x)

plt.plot(u0/10,z/.02,label=r"${P_0}$")
plt.plot(u1/10,z/.02,label=r"${P_1}$")
plt.plot(u2/10,z/.02,label=r"${P_2}$")
plt.plot(u3/10,z/.02,label=r"${P_3}$")
plt.xlim(-0.2,0.8)
plt.ylim(0,3)
plt.grid()
plt.legend()
plt.show()
plt.close()

