from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import colormaps as cmaps

plt.rcParams.update({'font.size': 10})


def spectra(T,binfact,maxbinsize,dx = 1) :
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
    spd_unique   [i] = np.max(spd[indices])
    iglob += len(indices)

  freq_bins = np.array([0. for i in range(len(freq2d_unique))])
  spd_bins  = np.array([0. for i in range(len(freq2d_unique))])
  binsize = 1
  ind = 1
  count = 0
  while (ind < len(freq2d_unique)) :
    inc = min(maxbinsize,int(binsize))
    i2  = min(len(freq2d_unique),ind + inc)
    freq_bins[count] = np.mean(freq2d_unique[ind:(ind+inc)])
    spd_bins [count] = np.mean(spd_unique   [ind:(ind+inc)])
    binsize *= binfact
    ind += inc
    count = count + 1
  freq_bins[count-2] = np.mean(freq_bins[count-2:count])
  spd_bins [count-2] = np.mean(spd_bins [count-2:count])
  count -= 1

  return freq_bins[:count]*2*2*np.pi/(2*dx) , spd_bins[:count]


def mams_plot(z,v,t,f,lgy=False) :
  un = np.min (v,axis=(1,2))
  ua = np.mean(v,axis=(1,2))
  ux = np.max (v,axis=(1,2))
  us = np.std (v,axis=(1,2))
  plt.fill_betweenx(z,un,ux,color="lightblue")
  plt.fill_betweenx(z,ua-us,ua+us,color="deepskyblue")
  plt.plot(ua,z,color="black")
  plt.margins(x=0)
  plt.xlabel(t)
  plt.ylabel("z-location (km)")
  if (lgy) :
    plt.yscale("log")
  plt.savefig(f,dpi=300,bbox_inches="tight")
  plt.show()
  plt.close()


nc = Dataset("abl_neutral_10m_00000040.nc","r")
nz = nc.dimensions["z"].size
ny = nc.dimensions["y"].size
nx = nc.dimensions["x"].size
x = nc.variables["x"][:]/1000.
y = nc.variables["y"][:]/1000.
z = nc.variables["z"][:]/1000.
mycmap = cmaps.vik
num_levels = 256

yind = int(ny/2)
z2   = int(750./1200.*nz)
u = nc.variables["uvel"][:z2,yind,:,0]
v = nc.variables["vvel"][:z2,yind,:,0]
w = nc.variables["wvel"][:z2,yind,:,0]
mag = np.sqrt(u*u+v*v+w*w)
X,Z = np.meshgrid(x,z[:z2])
CS = plt.contourf(X,Z,mag,levels=num_levels,cmap=mycmap,extend="both")
plt.axis('scaled')
plt.margins(x=0)
plt.xlabel("x-location (km)")
plt.ylabel("z-location (km)")
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("bottom", size="5%", pad=0.5)
cbar = plt.colorbar(CS,orientation="horizontal",cax=cax)
cbar.ax.tick_params(rotation=25)
plt.savefig("abl_neutral_wind_countour_xz.png",dpi=300,bbox_inches="tight")
plt.show()
plt.close()

zind = int(80./1200.*nz)
u = nc.variables["uvel"][zind,:,:,0]
v = nc.variables["vvel"][zind,:,:,0]
w = nc.variables["wvel"][zind,:,:,0]
mag = np.sqrt(u*u+v*v+w*w)
X,Y = np.meshgrid(x,y)
CS = plt.contourf(X,Y,mag,levels=num_levels,cmap=mycmap,extend="both")
plt.axis('scaled')
plt.margins(x=0)
plt.xlabel("x-location (km)")
plt.ylabel("y-location (km)")
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("bottom", size="5%", pad=0.5)
cbar = plt.colorbar(CS,orientation="horizontal",cax=cax)
cbar.ax.tick_params(rotation=25)
plt.savefig("abl_neutral_wind_countour_xy.png",dpi=300,bbox_inches="tight")
plt.show()
plt.close()


z2 = int(1000./1200.*nz)
u     = nc.variables["uvel"][:z2,:,:,0]
v     = nc.variables["vvel"][:z2,:,:,0]
w     = nc.variables["wvel"][:z2,:,:,0]
tke_u = nc.variables["TKE" ][:z2,:,:,0]
mag = np.sqrt(u*u+v*v+w*w)
up = np.subtract( u , np.mean(u,axis=(1,2))[:,np.newaxis,np.newaxis] )
vp = np.subtract( v , np.mean(v,axis=(1,2))[:,np.newaxis,np.newaxis] )
wp = np.subtract( w , np.mean(w,axis=(1,2))[:,np.newaxis,np.newaxis] )
tke_r = (up*up + vp*vp + wp*wp)/2

mams_plot(z[:z2],u    ,'u-velocity (m/s)','abl_neutral_uvel.png' )
mams_plot(z[:z2],v    ,'v-velocity (m/s)','abl_neutral_vvel.png' )
mams_plot(z[:z2],w    ,'w-velocity (m/s)','abl_neutral_wvel.png' )
mams_plot(z[:z2],mag  ,'Wind Speed (m/s)','abl_neutral_mag.png'  ,True)
mams_plot(z[:z2],tke_r,'Resolved TKE'    ,'abl_neutral_tke_r.png')
mams_plot(z[:z2],tke_u,'Uesolved TKE'    ,'abl_neutral_tke_u.png')

# Kinetic Energy Spectra
dx = 10
z1 = int(100./1200.*nz)
z2 = int(400./1200.*nz)
u  = nc.variables["uvel"][z1:z2,:,:,0]
v  = nc.variables["vvel"][z1:z2,:,:,0]
w  = nc.variables["wvel"][z1:z2,:,:,0]
freq,spd = spectra((u*u+v*v+w*w)/2,1.03,50,dx)
plt.loglog( freq , spd*freq**(5/3) , label="KE")
plt.xlabel(r"Frequency")
plt.ylabel(r"$E(K) \omega^{5/3}$ (Compensated KE Spectra) ")
plt.legend()
plt.savefig("abl_neutral_ke_spectra.png",dpi=300,bbox_inches="tight")
plt.show()
plt.close()


