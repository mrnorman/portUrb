from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray

def get_ind(arr,val) :
  return np.argmin(np.abs(arr-val))

t1 = 10
t2 = 24
prefixes = ["/lustre/storm/nwp501/scratch/imn/portUrb/build/cubes_periodic_z0-1.000e-05_",\
            "/lustre/storm/nwp501/scratch/imn/portUrb/build/cubes_periodic_z0-1.000e-06_",\
            "/lustre/storm/nwp501/scratch/imn/portUrb/build/cubes_periodic_z0-1.000e-07_"]
fnames = [ [f"{prefix}{i:08}.nc" for i in range(t1,t2+1)] for prefix in prefixes]

u0 = [0. for i in range(len(prefixes))]
u1 = [0. for i in range(len(prefixes))]
u2 = [0. for i in range(len(prefixes))]
u3 = [0. for i in range(len(prefixes))]
for k in range(len(prefixes)) :
  nc   = xarray.open_dataset(fnames[k][0])
  x    = np.array(nc["x"])
  y    = np.array(nc["y"])
  z    = np.array(nc["z"])
  nx   = len(x)
  ny   = len(y)
  nz   = len(z)
  dx   = x[1]-x[0]
  dy   = y[1]-y[0]
  dz   = z[1]-z[0]
  xlen = x[-1]+dx/2
  ylen = y[-1]+dy/2
  zlen = z[-1]+dz/2
  k2   = get_ind(z,.02*3)
  p0_x = [get_ind(x,2*xlen/8),get_ind(x,6*xlen/8),get_ind(x,2*xlen/8),get_ind(x,6*xlen/8)];
  p0_y = [get_ind(y,6*ylen/8),get_ind(y,4*ylen/8),get_ind(y,2*ylen/8),get_ind(y,0*ylen/8)];
  p1_x = [get_ind(x,4*ylen/8),get_ind(x,0*ylen/8),get_ind(x,4*ylen/8),get_ind(x,0*ylen/8)];
  p1_y = [get_ind(y,6*ylen/8),get_ind(y,4*ylen/8),get_ind(y,2*ylen/8),get_ind(y,0*ylen/8)];
  p2_x = [get_ind(x,2*xlen/8),get_ind(x,6*xlen/8),get_ind(x,2*xlen/8),get_ind(x,6*xlen/8)];
  p2_y = [get_ind(y,4*ylen/8),get_ind(y,2*ylen/8),get_ind(y,0*ylen/8),get_ind(y,6*ylen/8)];
  p3_x = [get_ind(x,4*xlen/8),get_ind(x,0*xlen/8),get_ind(x,4*xlen/8),get_ind(x,0*xlen/8)];
  p3_y = [get_ind(y,4*ylen/8),get_ind(y,2*ylen/8),get_ind(y,0*ylen/8),get_ind(y,6*ylen/8)];

  for i in range(len(fnames[k])) :
    nc = xarray.open_dataset(fnames[k][i])
    for j in range(len(p0_x)) :
      u = np.array(nc["avg_u"][:,p0_y[j],p0_x[j]])
      u0[k] += u
      u = np.array(nc["avg_u"][:,p1_y[j],p1_x[j]])
      u1[k] += u
      u = np.array(nc["avg_u"][:,p2_y[j],p2_x[j]])
      u2[k] += u
      u = np.array(nc["avg_u"][:,p3_y[j],p3_x[j]])
      u3[k] += u
  u0[k] /= len(fnames[k])*len(p0_x)
  u1[k] /= len(fnames[k])*len(p0_x)
  u2[k] /= len(fnames[k])*len(p0_x)
  u3[k] /= len(fnames[k])*len(p0_x)

for k in range(len(prefixes)) :
  plt.plot(u0[k]/10,z/.02,label=f"${k}$")
plt.xlim(-0.2,0.8)
plt.ylim(0,3)
plt.grid()
plt.legend()
plt.show()
plt.close()

for k in range(len(prefixes)) :
  plt.plot(u1[k]/10,z/.02,label=f"${k}$")
plt.xlim(-0.2,0.8)
plt.ylim(0,3)
plt.grid()
plt.legend()
plt.show()
plt.close()

for k in range(len(prefixes)) :
  plt.plot(u2[k]/10,z/.02,label=f"${k}$")
plt.xlim(-0.2,0.8)
plt.ylim(0,3)
plt.grid()
plt.legend()
plt.show()
plt.close()

for k in range(len(prefixes)) :
  plt.plot(u3[k]/10,z/.02,label=f"${k}$")
plt.xlim(-0.2,0.8)
plt.ylim(0,3)
plt.grid()
plt.legend()
plt.show()
plt.close()

