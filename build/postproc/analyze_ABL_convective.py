from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cmap import Colormap
import xarray

workdir = "/lustre/storm/nwp501/scratch/imn/ABL_convective"

def spectra(T,dx = 1) :
  spd = np.abs( np.fft.rfft(T[0,0,:]) )**2
  spd = 0
  for k in range(T.shape[0]) :
    for j in range(T.shape[1]) :
      spd += np.abs( np.fft.rfft(T[k,j,:]) )**2
      spd += np.abs( np.fft.rfft(T[k,:,j]) )**2
  freq = np.fft.rfftfreq(len(T[k,0,:]))
  spd /= T.shape[0]*T.shape[1]*2
  return freq[1:]*2*2*np.pi/(2*dx) , spd[1:]


def get_ind(arr,val) :
    return np.argmin(np.abs(arr-val))


nc = Dataset(f"{workdir}/ABL_convective_00000006.nc","r")
x = np.array(nc["x"])/1000
y = np.array(nc["y"])/1000
z = np.array(nc["z"])/1000
nx = len(x)
ny = len(y)
nz = len(z)
dx = x[1]-x[0]
dy = y[1]-y[0]
dz = z[1]-z[0]
xlen = x[-1]+dx/2
ylen = y[-1]+dy/2
zlen = z[-1]+dz/2
hs   = 5
uvel  = np.array(nc["uvel"])
vvel  = np.array(nc["vvel"])
wvel  = np.array(nc["wvel"])
theta = np.array(nc["theta_pert"]) + np.array(nc["hy_theta_cells"])[hs:hs+nz,np.newaxis,np.newaxis]
mag   = np.sqrt(uvel*uvel+vvel*vvel)


fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(12,10))
X,Y = np.meshgrid(x,y)
zind = get_ind(z,.078)
mn  = np.mean(theta[zind,:,:])
std = np.std (theta[zind,:,:])
CS1 = ax1.contourf(X,Y,theta[zind,:,:],levels=np.arange(310,315,5/200),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
ax1.axis('scaled')
ax1.set_xlabel("x-location (km)")
ax1.set_ylabel("y-location (km)")
ax1.margins(x=0)
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("bottom", size="4%", pad=0.5)
cbar1 = plt.colorbar(CS1,orientation="horizontal",cax=cax1)
cbar1.ax.tick_params(labelrotation=30)

mn  = np.mean(wvel[zind,:,:])
std = np.std (wvel[zind,:,:])
CS2 = ax2.contourf(X,Y,wvel[zind,:,:],levels=np.arange(-6,6,12/200),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
ax2.axis('scaled')
ax2.set_xlabel("x-location (km)")
ax2.set_ylabel("y-location (km)")
ax2.margins(x=0)
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("bottom", size="4%", pad=0.5)
cbar2 = plt.colorbar(CS2,orientation="horizontal",cax=cax2)
cbar2.ax.tick_params(labelrotation=30)

X,Z = np.meshgrid(x,z)
yind = get_ind(y,ylen/2)
t1 = 310
t2 = 315
CS3 = ax3.contourf(X,Z,theta[:,yind,:],levels=np.arange(t1,t2,(t2-t1)/200),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
ax3.axis('scaled')
ax3.set_xlabel("x-location (km)")
ax3.set_ylabel("z-location (km)")
ax3.margins(x=0)
divider = make_axes_locatable(ax3)
cax3 = divider.append_axes("bottom", size="4%", pad=0.5)
cbar3 = plt.colorbar(CS3,orientation="horizontal",cax=cax3)
cbar3.ax.tick_params(labelrotation=30)

mn  = np.mean(wvel[:,yind,:])
std = np.std (wvel[:,yind,:])
CS4 = ax4.contourf(X,Z,wvel[:,yind,:],levels=np.arange(-6,6,12/200),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
ax4.axis('scaled')
ax4.set_xlabel("x-location (km)")
ax4.set_ylabel("z-location (km)")
ax4.margins(x=0)
divider = make_axes_locatable(ax4)
cax4 = divider.append_axes("bottom", size="4%", pad=0.5)
cbar4 = plt.colorbar(CS4,orientation="horizontal",cax=cax4)
cbar4.ax.tick_params(labelrotation=30)
plt.tight_layout()
plt.savefig("ABL_convective_contourf.png",dpi=600)
plt.show()
plt.close()



dx = 10
freq,spd1 = spectra(mag  [get_ind(z,0.1):get_ind(z,0.2)+1,:,:],dx=10)
freq,spd2 = spectra(wvel [get_ind(z,0.1):get_ind(z,0.2)+1,:,:],dx=10)
freq,spd3 = spectra(theta[get_ind(z,0.1):get_ind(z,0.2)+1,:,:],dx=10)
spd1 = spd1/np.mean(spd1)
spd2 = spd2/np.mean(spd2)
spd3 = spd3/np.mean(spd3)
freq = freq
fig = plt.figure(figsize=(6,4))
ax = fig.gca()
ax.plot(freq,spd1,label="Horizontal Wind Speed spectra",color="black")
ax.plot(freq,spd2,label="Vertical Velocity spectra"    ,color="blue" )
ax.plot(freq,spd3,label="Potential Temperature spectra",color="lightgreen")
ax.plot(freq,1.e-2*freq**(-5/3),label=r"$f^{-5/3}$"    ,color="magenta"  )
ax.vlines(2*np.pi/(2 *dx),2.e-4,1.e1,linestyle="--",color="red")
ax.vlines(2*np.pi/(4 *dx),2.e-4,1.e1,linestyle="--",color="red")
ax.vlines(2*np.pi/(8 *dx),2.e-4,1.e1,linestyle="--",color="red")
ax.vlines(2*np.pi/(16*dx),2.e-4,1.e1,linestyle="--",color="red")
ax.text(0.9*2*np.pi/(2 *dx),2.e1,"$2  \Delta x$")
ax.text(0.9*2*np.pi/(4 *dx),2.e1,"$4  \Delta x$")
ax.text(0.9*2*np.pi/(8 *dx),2.e1,"$8  \Delta x$")
ax.text(0.9*2*np.pi/(16*dx),2.e1,"$16 \Delta x$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Frequency")
ax.set_ylabel("Spectral Power")
ax.legend(loc='lower left')
# ax.set_xlim(left=0.0045)
ax.set_ylim(top=1.e3)
ax.margins(x=0)
plt.tight_layout()
plt.savefig("ABL_convective_spectra.png",dpi=600)
plt.show()
plt.close()
