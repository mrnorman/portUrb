from netCDF4 import Dataset
import numpy as np
import sys

x1 = 260
x2 = 615
y1 = 103
y2 = 405
z1 = 1
z2 = 225

prefix = f"vortmag_dump"
times = [i for i in range(0,200)]
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
i1 = int((x1-dx/2)/dx)
i2 = int((x2+dx/2)/dx)
j1 = int((y1-dy/2)/dy)
j2 = int((y2+dy/2)/dy)
k1 = int((z1-dz/2)/dz)
k2 = int((z2+dz/2)/dz)
print(i1,i2,j1,j2,k1,k2)
nxl = i2-i1+1
nyl = j2-j1+1
nzl = k2-k1+1
for l in times :
  print(l)
  nc = Dataset(files[l],"r")
  ncout = Dataset(f"{prefix}_reduced_{l:08d}.nc","w")
  ncout.createDimension("x",nxl)
  ncout.createDimension("y",nyl)
  ncout.createDimension("z",nzl)
  ncout.createVariable("x",'f4',("x"))[:] = nc.variables["x"][i1:i2+1]
  ncout.createVariable("y",'f4',("y"))[:] = nc.variables["y"][j1:j2+1]
  ncout.createVariable("z",'f4',("z"))[:] = nc.variables["z"][k1:k2+1]
  ncout.createVariable("vortmag",'f4',("z","y","x"))[:,:,:] = nc["vortmag"][k1:k2+1,j1:j2+1,i1:i2+1]
  nc.close()
  ncout.close()

