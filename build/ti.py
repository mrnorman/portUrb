from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np

def get_ind(arr,val) :
    return np.argmin(np.abs(arr-val))

def turbulent_intensity(fname) :
    nc    = Dataset(fname,"r")
    nz    = nc.dimensions["z"].size
    u     = nc.variables["uvel"][:,:,:]
    v     = nc.variables["vvel"][:,:,:]
    z     = nc.variables["z"][:]
    k1    = get_ind(z,89-127/2)
    k2    = get_ind(z,89+127/2)
    mag   = np.sqrt(u*u + v*v)
    tot   = 0
    count = 0
    for k in range(k1,k2+1) :
        tot  += np.std(mag [k,:,:])/np.mean(mag [k,:,:])
        count = count + 1
    return tot / count

def shear_exponent(fname) :
    nc    = Dataset(fname,"r")
    nz    = nc.dimensions["z"].size
    z     = nc.variables["z"][:]
    k1    = get_ind(z,89-127/2)
    k2    = get_ind(z,89+127/2)
    u     = nc.variables["avg_u"][k1,:,:]
    v     = nc.variables["avg_v"][k1,:,:]
    mag1  = np.mean(np.sqrt(u*u + v*v))
    u     = nc.variables["avg_u"][k2,:,:]
    v     = nc.variables["avg_v"][k2,:,:]
    mag2  = np.mean(np.sqrt(u*u + v*v))
    return np.log(mag2/mag1)/np.log(z[k2]/z[k1])

iend = 24

for i in range(1,iend+1) :
    print(turbulent_intensity( "z0_0.3/validation_precursor_000000"+str(i).zfill(2)+".nc"),end=" , ")
    print(turbulent_intensity( "z0_0.1/validation_precursor_000000"+str(i).zfill(2)+".nc"),end=" , ")
    print(turbulent_intensity("z0_0.07/validation_precursor_000000"+str(i).zfill(2)+".nc"),end=" , ")
    print(turbulent_intensity("z0_0.05/validation_precursor_000000"+str(i).zfill(2)+".nc")          )

print("")

for i in range(1,iend+1) :
    print(shear_exponent( "z0_0.3/validation_precursor_000000"+str(i).zfill(2)+".nc"),end=" , ")
    print(shear_exponent( "z0_0.1/validation_precursor_000000"+str(i).zfill(2)+".nc"),end=" , ")
    print(shear_exponent("z0_0.07/validation_precursor_000000"+str(i).zfill(2)+".nc"),end=" , ")
    print(shear_exponent("z0_0.05/validation_precursor_000000"+str(i).zfill(2)+".nc")          )


