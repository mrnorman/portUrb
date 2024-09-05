from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray


def spectra(T,nbins,dx = 1) :
  spd = np.abs( np.fft.rfft2(T[0,:,:]) )**2
  spd = 0
  for k in range(T.shape[0]) :
    spd = spd + np.abs( np.fft.rfft2(T[k,:,:]) )**2
  freq2d = np.sqrt(np.outer(np.fft.rfftfreq(len(T[k,0,:])),np.fft.rfftfreq(len(T[k,:,0]))))
  spd /= T.shape[0]
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
    spd_unique   [i] = np.max(spd[indices])
    iglob += len(indices)

  if (nbins == -1) :
    return freq2d_unique*2*2*np.pi/(2*dx) , spd_unique

  freq_bins = np.array([0. for i in range(nbins)])
  spd_bins  = np.array([0. for i in range(nbins)])
  binsize = len(freq2d_unique)/nbins
  for i in range(nbins) :
    i1 = int(round(i*binsize))
    i2 = int(round((i+1)*binsize))
    freq_bins[i] = np.max(freq2d_unique[i1:i2])
    spd_bins[i]  = np.max(spd_unique   [i1:i2])

  return freq_bins*2*2*np.pi/(2*dx) , spd_bins


def get_ind(arr,val) :
    return np.argmin(np.abs(arr-val))


nc = Dataset("ABL_neutral_00000016.nc","r")
x = np.array(nc["x"])/1000
y = np.array(nc["y"])/1000
z = np.array(nc["z"])/1000
dx = x[1]-x[0]
dy = y[1]-y[0]
dz = z[1]-z[0]
xlen = x[-1]+dx/2
ylen = y[-1]+dy/2
zlen = z[-1]+dz/2
uvel  = np.array(nc["uvel"][:,:,:])
vvel  = np.array(nc["vvel"][:,:,:])
wvel  = np.array(nc["wvel"][:,:,:])
mag   = np.sqrt(uvel*uvel+vvel*vvel+wvel*wvel)


umin  = np.min (mag,axis=(1,2))
umax  = np.max (mag,axis=(1,2))
umean = np.mean(mag,axis=(1,2))
ustd  = np.std (mag,axis=(1,2))
ustdm = umean-ustd
ustdp = umean+ustd
roughness = 0.1
uref = 10
href = 800
u_mo  = uref*np.log((z*1000+roughness)/roughness)/np.log((href+roughness)/roughness);
z2 = get_ind(z,0.75)
fig = plt.figure(figsize=(6,4))
ax = fig.gca()
ax.fill_betweenx(z[:z2],umin [:z2],umax [:z2],color="lightskyblue",label="[min,max]")
ax.fill_betweenx(z[:z2],ustdm[:z2],ustdp[:z2],color="deepskyblue",label="mean+[-stddev,stddev]")
ax.plot         (umean[:z2],z[:z2],color="black",label="mean")
ax.plot         (u_mo [:z2],z[:z2],color="black",linestyle="--",label=r"Log law")
ax.set_xlabel("u-velocity (m/s)")
ax.set_ylabel("z-location (km)")
ax.set_yscale("log")
ax.legend(loc="upper left")
ax.set_xlim(left=0)
ax.margins(x=0)
plt.tight_layout()
plt.savefig("ABL_neutral_uvel_height.png",dpi=600)
plt.close()


umean = np.mean(uvel,axis=(1,2))
vmean = np.mean(vvel,axis=(1,2))
wmean = np.mean(wvel,axis=(1,2))
umean3d = uvel.copy()
vmean3d = vvel.copy()
wmean3d = wvel.copy()
umean3d[:,:,:] = umean[:,np.newaxis,np.newaxis]
vmean3d[:,:,:] = vmean[:,np.newaxis,np.newaxis]
wmean3d[:,:,:] = wmean[:,np.newaxis,np.newaxis]
up = uvel - umean3d
vp = vvel - vmean3d
wp = wvel - wmean3d
up_wp = np.abs(up*wp)
up_wp_min  = np.min (up_wp,axis=(1,2))
up_wp_max  = np.max (up_wp,axis=(1,2))
up_wp_mean = np.mean(up_wp,axis=(1,2))
up_wp_std  = np.std (up_wp,axis=(1,2))
up_wp_stdm = up_wp_mean - up_wp_std
up_wp_stdp = up_wp_mean + up_wp_std

z2 = get_ind(z,0.75)
fig = plt.figure(figsize=(6,4))
ax = fig.gca()
ax.fill_betweenx(z[:z2],up_wp_min [:z2],up_wp_max [:z2],color="lightskyblue",label="[min,max]")
ax.fill_betweenx(z[:z2],up_wp_stdm[:z2],up_wp_stdp[:z2],color="deepskyblue",label="mean+[-stddev,stddev]")
ax.plot         (up_wp_mean[:z2],z[:z2],color="black",label="mean")
ax.set_xlabel("u-velocity (m/s)")
ax.set_ylabel("z-location (km)")
ax.legend()
ax.margins(x=0)
plt.tight_layout()
plt.savefig("ABL_neutral_up_wp_height.png",dpi=600)
plt.close()

tke = (up*up + vp*vp + wp*wp)/2 + np.array(nc.variables["TKE"][:,:,:])
tke_min  = np.min (tke,axis=(1,2))
tke_max  = np.max (tke,axis=(1,2))
tke_mean = np.mean(tke,axis=(1,2))
tke_std  = np.std (tke,axis=(1,2))
tke_stdm = tke_mean - tke_std
tke_stdp = tke_mean + tke_std

z2 = get_ind(z,0.75)
fig = plt.figure(figsize=(6,4))
ax = fig.gca()
ax.fill_betweenx(z[:z2],tke_min [:z2],tke_max [:z2],color="lightskyblue",label="[min,max]")
ax.fill_betweenx(z[:z2],tke_stdm[:z2],tke_stdp[:z2],color="deepskyblue",label="mean+[-stddev,stddev]")
ax.plot         (tke_mean[:z2],z[:z2],color="black",label="mean")
ax.set_xlabel("u-velocity (m/s)")
ax.set_ylabel("z-location (km)")
plt.legend()
ax.margins(x=0)
plt.tight_layout()
plt.savefig("ABL_neutral_tke_height.png",dpi=600)
plt.close()


dx = 10
freq,spd1 = spectra(mag[get_ind(z,0.1):get_ind(z,0.2)+1,:,:],nbins=2500,dx=10)
fig = plt.figure(figsize=(6,4))
ax = fig.gca()
ax.plot(freq,spd1,label="Wind Speed spectra")
ax.plot(freq,1.e3*freq**(-5/3),label=r"$f^{-5/3}$")
ax.vlines(2*np.pi/(2 *dx),1.e-2,1.e7,linestyle="--",color="red")
ax.vlines(2*np.pi/(4 *dx),1.e-2,1.e7,linestyle="--",color="red")
ax.vlines(2*np.pi/(8 *dx),1.e-2,1.e7,linestyle="--",color="red")
ax.vlines(2*np.pi/(16*dx),1.e-2,1.e7,linestyle="--",color="red")
ax.text(0.9*2*np.pi/(2 *dx),2.e7,"$2  \Delta x$")
ax.text(0.9*2*np.pi/(4 *dx),2.e7,"$4  \Delta x$")
ax.text(0.9*2*np.pi/(8 *dx),2.e7,"$8  \Delta x$")
ax.text(0.9*2*np.pi/(16*dx),2.e7,"$16 \Delta x$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Frequency")
ax.set_ylabel("Spectral Power")
ax.legend(loc='lower left')
ax.set_xlim(left=0.0045)
ax.set_ylim(top=1.e9)
ax.margins(x=0)
plt.tight_layout()
plt.savefig("ABL_neutral_spectra.png",dpi=600)
# plt.show()
plt.close()


fig = plt.figure(figsize=(6,6))
ax = fig.gca()
X,Y = np.meshgrid(x,y)
print(z[get_ind(z,.0786)])
mn  = np.mean(mag[get_ind(z,.0786),:,:])
std = np.std (mag[get_ind(z,.0786),:,:])
CS = ax.contourf(X,Y,mag[get_ind(z,.0786),:,:],levels=np.arange(mn-2*std,mn+2*std,4*std/100),cmap="turbo",extend="both")
ax.axis('scaled')
ax.set_xlabel("x-location (km)")
ax.set_ylabel("y-location (km)")
ax.margins(x=0)
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("bottom", size="4%", pad=0.5)
plt.colorbar(CS,orientation="horizontal",cax=cax)
plt.tight_layout()
plt.savefig("ABL_neutral_contour_xy.png",dpi=600)

