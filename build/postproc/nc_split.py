from netCDF4 import Dataset
import sys

fname = sys.argv[1]
fprefix = fname.split('.')[0]
nc = Dataset(fname,"r")
for var in nc.variables :
  if not var in ["x","y","z","etime"] :
    if (nc[var].dimensions == ("z","y","x")) :
      print(nc[var].name)
      nc_new = Dataset(f"{fprefix}_{var}.nc","w")
      nc_new.createDimension("x",nc.dimensions["x"].size)
      nc_new.createDimension("y",nc.dimensions["y"].size)
      nc_new.createDimension("z",nc.dimensions["z"].size)
      var_new_x      = nc_new.createVariable("x",nc.variables["x"].datatype,nc.variables["x"].dimensions)
      var_new_y      = nc_new.createVariable("y",nc.variables["y"].datatype,nc.variables["y"].dimensions)
      var_new_z      = nc_new.createVariable("z",nc.variables["z"].datatype,nc.variables["z"].dimensions)
      var_new_x[:]   = nc.variables["x"][:]
      var_new_y[:]   = nc.variables["y"][:]
      var_new_z[:]   = nc.variables["z"][:]
      var_new        = nc_new.createVariable(nc[var].name,nc[var].datatype,nc[var].dimensions)
      var_new[:,:,:] = nc.variables[var][:,:,:]
      nc_new.close()
