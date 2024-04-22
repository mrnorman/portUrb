
from netCDF4 import Dataset
import numpy as np

v = np.loadtxt("velo.dat" )
t = np.loadtxt("times.dat")
nc = Dataset("platform_motions.nc","w")
nc.createDimension("time",len(t))
nc.createVariable("time",'f4',("time",))[:] = t
nc.createVariable("pert",'f4',("time",))[:] = v-12
nc.close()

