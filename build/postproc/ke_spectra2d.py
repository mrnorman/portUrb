from netCDF4 import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def spectra(T,nbins,dx = 1) :
  spd = np.abs( np.fft.rfft2(T[0,:,:]) )**2
  spd = 0
  for k in range(T.shape[0]) :
    spd = spd + np.abs( np.fft.rfft2(T[k,:,:]) )**2
  freq2d = np.sqrt(np.outer(np.fft.rfftfreq(len(T[k,0,:])),np.fft.rfftfreq(len(T[k,:,0]))))
  spd /= (T.shape[0] * T.shape[1])
  spd = spd.reshape(spd.shape[0]*spd.shape[1])
  freq2d = freq2d.reshape(freq2d.shape[0]*freq2d.shape[1])
  indices = np.argsort(freq2d)
  freq2d = freq2d[indices[:]]
  spd    = spd   [indices[:]]

  num_unique = len(set(freq2d))
  freq2d_unique = np.array([0. for i in range(num_unique)])
  spd_unique    = np.array([0. for i in range(num_unique)])

  iglob = 0;
  for i in range(num_unique) :
    indices = np.where( freq2d == freq2d[iglob] )[0];
    freq2d_unique[i] = freq2d[iglob]
    spd_unique   [i] = np.mean(spd[indices])
    iglob += len(indices)

  if (nbins == -1) :
    return freq2d_unique*2*2*np.pi/(2*dx) , spd_unique

  freq_bins = np.array([0. for i in range(nbins)])
  spd_bins  = np.array([0. for i in range(nbins)])
  binsize = len(freq2d_unique)/nbins
  for i in range(nbins) :
    i1 = int(round(i*binsize))
    i2 = int(round((i+1)*binsize))
    freq_bins[i] = np.mean(freq2d_unique[i1:i2])
    spd_bins[i]  = np.mean(spd_unique   [i1:i2])

  return freq_bins*2*2*np.pi/(2*dx) , spd_bins



nbins = 1000
dx = 5

nc = Dataset("test_00000008.nc", "r")
nz = nc.dimensions["z"].size
u = nc.variables["uvel"][int(30./700.*nz):int(150./700.*nz)+1,:,:,0]
v = nc.variables["vvel"][int(30./700.*nz):int(150./700.*nz)+1,:,:,0]
w = nc.variables["wvel"][int(30./700.*nz):int(150./700.*nz)+1,:,:,0]
ke = (u*u+v*v+w*w)/2
freq,spd = spectra(ke - np.mean(ke),nbins,dx)
plt.loglog( freq , spd )

# nc = Dataset("test_00000003.nc", "r")
# nz = nc.dimensions["z"].size
# u = nc.variables["uvel"][int(80./700.*nz):int(80./700.*nz)+1,:,:,0]
# v = nc.variables["vvel"][int(80./700.*nz):int(80./700.*nz)+1,:,:,0]
# w = nc.variables["wvel"][int(80./700.*nz):int(80./700.*nz)+1,:,:,0]
# ke = (u*u+v*v+w*w)/2
# freq,spd = spectra(ke - np.mean(ke),nbins,dx)
# plt.loglog( freq , spd )
# 
# nc = Dataset("test_00000006.nc", "r")
# nz = nc.dimensions["z"].size
# u = nc.variables["uvel"][int(80./700.*nz):int(80./700.*nz)+1,:,:,0]
# v = nc.variables["vvel"][int(80./700.*nz):int(80./700.*nz)+1,:,:,0]
# w = nc.variables["wvel"][int(80./700.*nz):int(80./700.*nz)+1,:,:,0]
# ke = (u*u+v*v+w*w)/2
# freq,spd = spectra(ke - np.mean(ke),nbins,dx)
# plt.loglog( freq , spd )
# 
# nc = Dataset("test_00000008.nc", "r")
# nz = nc.dimensions["z"].size
# u = nc.variables["uvel"][int(80./700.*nz):int(80./700.*nz)+1,:,:,0]
# v = nc.variables["vvel"][int(80./700.*nz):int(80./700.*nz)+1,:,:,0]
# w = nc.variables["wvel"][int(80./700.*nz):int(80./700.*nz)+1,:,:,0]
# ke = (u*u+v*v+w*w)/2
# freq,spd = spectra(ke - np.mean(ke),nbins,dx)
# plt.loglog( freq , spd )

# ind1 = next(x for x, val in enumerate(freq) if val > 0.02)
# ind1 = len(freq)
# plt.loglog( freq[:ind1] , 1.5e0 * freq[:ind1]**(-3.   ) )
# plt.loglog( freq[ind1:] , 5.0e2 * freq[ind1:]**(-5./3.) )
plt.loglog( freq[:] , 1.e4 * freq[:]**(-5./3.) )
# plt.legend(["0.5hr","1.0hr","1.5hr","2hr","$k^{-5/3}$"],loc='lower left')
plt.legend(["2hr","$k^{-5/3}$"],loc='lower left')
plt.xlabel("Spatial wavenumber")
plt.ylabel("Spectral power of kinetic energy")
plt.gcf().set_size_inches(8,4)
wn_dx2  = 2*np.pi/( 2*dx);
wn_dx4  = 2*np.pi/( 4*dx);
wn_dx8  = 2*np.pi/( 8*dx);
wn_dx16 = 2*np.pi/(16*dx);
plt.vlines([wn_dx2,wn_dx4,wn_dx8,wn_dx16],1.e1,1.e8,linestyles="dashed",colors="black")
plt.text(wn_dx2 *0.9,2.e8,"$2\Delta x$" ,size=14)
plt.text(wn_dx4 *0.9,2.e8,"$4\Delta x$" ,size=14)
plt.text(wn_dx8 *0.9,2.e8,"$8\Delta x$" ,size=14)
plt.text(wn_dx16*0.9,2.e8,"$16\Delta x$",size=14)
# plt.savefig("ke_spectra.eps")
plt.show()
plt.close()
