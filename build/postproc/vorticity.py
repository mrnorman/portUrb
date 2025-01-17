from netCDF4 import Dataset
import numpy as np
import sys


prefix = f"awaken_simplest"
times = [i for i in range(10,11)]
files = [f"{prefix}_{i:08d}.nc" for i in range(times[-1]+1)]
nc = Dataset(files[0],"r")
x = np.array(nc["x"][:])
y = np.array(nc["y"][:])
z = np.array(nc["z"][:])
nx = len(x)
ny = len(y)
nz = len(z)
dx = x[1]-x[0]
dy = y[1]-y[0]
dz = z[1]-z[0]
xlen = x[-1]+dx/2
ylen = y[-1]+dy/2
zlen = z[-1]+dz/2
def d_dx(q) :
  return (q[1:nz-1,1:ny-1,2:nx  ]-q[1:nz-1,1:ny-1,0:nx-2])/(2*dx)
def d_dy(q) :
  return (q[1:nz-1,2:ny  ,1:nx-1]-q[1:nz-1,0:ny-2,1:nx-1])/(2*dy)
def d_dz(q) :
  return (q[2:nz  ,1:ny-1,1:nx-1]-q[0:nz-2,1:ny-1,1:nx-1])/(2*dz)
for i in times :
  print(i)
  nc = Dataset(files[i],"r")
  u = np.array(nc["uvel"][:,:,:])
  v = np.array(nc["vvel"][:,:,:])
  w = np.array(nc["wvel"][:,:,:])
  vort_x = d_dy(w) - d_dz(v)
  vort_y = d_dz(u) - d_dx(w)
  vort_z = d_dx(v) - d_dy(u)
  vortmag = np.sqrt(vort_x*vort_x + vort_y*vort_y + vort_z*vort_z)
  ncout = Dataset(f"vorticity_{i:08d}.nc","w")
  ncout.createDimension("x",nx)
  ncout.createDimension("y",ny)
  ncout.createDimension("z",nz)
  ncout.createVariable("x",'f4',("x"))[:] = nc.variables["x"][:]
  ncout.createVariable("y",'f4',("y"))[:] = nc.variables["y"][:]
  ncout.createVariable("z",'f4',("z"))[:] = nc.variables["z"][:]
  var = ncout.createVariable("vort_z",'f4',("z","y","x"))
  var[:,:,:] = 0
  var[1:nz-1,1:ny-1,1:nx-1] = np.abs(vort_z[:,:,:])
  var = ncout.createVariable("vortmag",'f4',("z","y","x"))
  var[:,:,:] = 0
  var[1:nz-1,1:ny-1,1:nx-1] = vortmag[:,:,:]
  nc.close()
  ncout.close()

