from netCDF4 import Dataset
import numpy as np
import sys


fname = sys.argv[1]
fprefix = fname.split(".")[0]
print(f"Prefix: {fprefix}")
nc = Dataset(fname,"r")
x = np.array(nc["x"][:])
y = np.array(nc["y"][:])
z = np.array(nc["z"][:])
nx = len(x)
ny = len(y)
nz = len(z)
dx = x[1]-x[0]
dy = y[1]-y[0]
dz = z[1]-z[0]
u = np.array(nc["uvel"][:,:,:])
v = np.array(nc["vvel"][:,:,:])
w = np.array(nc["wvel"][:,:,:])
def d_dx(q) :
  return (q[1:nz-1,1:ny-1,2:nx  ]-q[1:nz-1,1:ny-1,0:nx-2])/dx
def d_dy(q) :
  return (q[1:nz-1,2:ny  ,1:nx-1]-q[1:nz-1,0:ny-2,1:nx-1])/dy
def d_dz(q) :
  return (q[2:nz  ,1:ny-1,1:nx-1]-q[0:nz-2,1:ny-1,1:nx-1])/dz
vort_x = d_dy(w) - d_dz(v)
vort_y = d_dz(u) - d_dx(w)
vort_z = d_dx(v) - d_dy(u)
vortmag = np.sqrt(vort_x*vort_x + vort_y*vort_y + vort_z*vort_z)
nc_new = Dataset(f"{fprefix}_vortmag.nc","w")
nc_new.createDimension("x",nc.dimensions["x"].size)
nc_new.createDimension("y",nc.dimensions["y"].size)
nc_new.createDimension("z",nc.dimensions["z"].size)
nc_new.createVariable("x",nc.variables["x"].datatype,nc.variables["x"].dimensions)[:] = nc.variables["x"][:]
nc_new.createVariable("y",nc.variables["y"].datatype,nc.variables["y"].dimensions)[:] = nc.variables["y"][:]
nc_new.createVariable("z",nc.variables["z"].datatype,nc.variables["z"].dimensions)[:] = nc.variables["z"][:]
dims  = nc["uvel"].dimensions
dtype = nc["uvel"].datatype
# Vorticity magnitude
nc_new.createVariable("vortmag",dtype,dims)[:,:,:] = 0
nc_new["vortmag"][1:nz-1,1:ny-1,1:nx-1] = vortmag
# Enstrophy
nc_new.createVariable("enstrophy",dtype,dims)[:,:,:] = 0
nc_new["enstrophy"][1:nz-1,1:ny-1,1:nx-1] = vortmag*vortmag
# Velocity magnitude
nc_new.createVariable("windmag",dtype,dims)[:,:,:] = np.sqrt(u*u+v*v+w*w)
# Helicity
nc_new.createVariable("helicity",dtype,dims)[:,:,:] = 0
nc_new["helicity"][1:nz-1,1:ny-1,1:nx-1] = vort_x*u[1:nz-1,1:ny-1,1:nx-1] + \
                                           vort_y*v[1:nz-1,1:ny-1,1:nx-1] + \
                                           vort_z*w[1:nz-1,1:ny-1,1:nx-1]
nc_new.close()

