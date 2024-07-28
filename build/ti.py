from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np

iend = 10

for i in range(1,iend) :
    nc  = Dataset("z0_0.3/validation_precursor_000000"+str(i).zfill(2)+".nc","r")
    nz = nc.dimensions["z"].size
    u = nc.variables["uvel"][:,:,:]
    v = nc.variables["vvel"][:,:,:]
    z = nc.variables["z"][:]
    mag = np.sqrt(u*u + v*v)
    tot  = 0
    count = 0
    for k in range(2,16) :
        tot  += np.std(mag [k,:,:])/np.mean(mag [k,:,:])
        count = count + 1
    ti = tot / count
    print(ti,end=" , ")

    nc  = Dataset("z0_0.1/validation_precursor_000000"+str(i).zfill(2)+".nc","r")
    nz = nc.dimensions["z"].size
    u = nc.variables["uvel"][:,:,:]
    v = nc.variables["vvel"][:,:,:]
    z = nc.variables["z"][:]
    mag = np.sqrt(u*u + v*v)
    tot  = 0
    count = 0
    for k in range(2,16) :
        tot  += np.std(mag [k,:,:])/np.mean(mag [k,:,:])
        count = count + 1
    ti = tot / count
    print(ti,end=" , ")

    nc  = Dataset("z0_0.07/validation_precursor_000000"+str(i).zfill(2)+".nc","r")
    nz = nc.dimensions["z"].size
    u = nc.variables["uvel"][:,:,:]
    v = nc.variables["vvel"][:,:,:]
    z = nc.variables["z"][:]
    mag = np.sqrt(u*u + v*v)
    tot  = 0
    count = 0
    for k in range(2,16) :
        tot  += np.std(mag [k,:,:])/np.mean(mag [k,:,:])
        count = count + 1
    ti = tot / count
    print(ti,end=" , ")

    nc  = Dataset("validation_precursor_000000"+str(i).zfill(2)+".nc","r")
    nz = nc.dimensions["z"].size
    u = nc.variables["uvel"][:,:,:]
    v = nc.variables["vvel"][:,:,:]
    z = nc.variables["z"][:]
    mag = np.sqrt(u*u + v*v)
    tot  = 0
    count = 0
    for k in range(2,16) :
        tot  += np.std(mag [k,:,:])/np.mean(mag [k,:,:])
        count = count + 1
    ti = tot / count
    print(ti)

print("")

for i in range(1,iend) :
    nc  = Dataset("z0_0.3/validation_precursor_000000"+str(i).zfill(2)+".nc","r")
    nz = nc.dimensions["z"].size
    z = nc.variables["z"][:]
    u = nc.variables["avg_u"][2,:,:]
    v = nc.variables["avg_v"][2,:,:]
    mag1 = np.mean(np.sqrt(u*u + v*v))
    u = nc.variables["avg_u"][15,:,:]
    v = nc.variables["avg_v"][15,:,:]
    mag2 = np.mean(np.sqrt(u*u + v*v))
    z1 = z[ 2]
    z2 = z[15]
    p = np.log(mag2/mag1)/np.log(z2/z1)
    print(p,end=" , ")

    nc  = Dataset("z0_0.1/validation_precursor_000000"+str(i).zfill(2)+".nc","r")
    nz = nc.dimensions["z"].size
    z = nc.variables["z"][:]
    u = nc.variables["avg_u"][2,:,:]
    v = nc.variables["avg_v"][2,:,:]
    mag1 = np.mean(np.sqrt(u*u + v*v))
    u = nc.variables["avg_u"][15,:,:]
    v = nc.variables["avg_v"][15,:,:]
    mag2 = np.mean(np.sqrt(u*u + v*v))
    z1 = z[ 2]
    z2 = z[15]
    p = np.log(mag2/mag1)/np.log(z2/z1)
    print(p,end=" , ")

    nc  = Dataset("z0_0.07/validation_precursor_000000"+str(i).zfill(2)+".nc","r")
    nz = nc.dimensions["z"].size
    z = nc.variables["z"][:]
    u = nc.variables["avg_u"][2,:,:]
    v = nc.variables["avg_v"][2,:,:]
    mag1 = np.mean(np.sqrt(u*u + v*v))
    u = nc.variables["avg_u"][15,:,:]
    v = nc.variables["avg_v"][15,:,:]
    mag2 = np.mean(np.sqrt(u*u + v*v))
    z1 = z[ 2]
    z2 = z[15]
    p = np.log(mag2/mag1)/np.log(z2/z1)
    print(p,end=" , ")

    nc  = Dataset("validation_precursor_000000"+str(i).zfill(2)+".nc","r")
    nz = nc.dimensions["z"].size
    z = nc.variables["z"][:]
    u = nc.variables["avg_u"][2,:,:]
    v = nc.variables["avg_v"][2,:,:]
    mag1 = np.mean(np.sqrt(u*u + v*v))
    u = nc.variables["avg_u"][15,:,:]
    v = nc.variables["avg_v"][15,:,:]
    mag2 = np.mean(np.sqrt(u*u + v*v))
    z1 = z[ 2]
    z2 = z[15]
    p = np.log(mag2/mag1)/np.log(z2/z1)
    print(p)


