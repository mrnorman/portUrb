from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray


def get_ind(arr,val) :
    return np.argmin(np.abs(arr-val))


fnames = ["abl_stable_2m_00000033.nc",\
          "abl_stable_2m_00000034.nc",\
          "abl_stable_2m_00000035.nc",\
          "abl_stable_2m_00000036.nc"]

nc = Dataset(fnames[0],"r")
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
avg_uvel  = np.array(nc["avg_u"])
avg_vvel  = np.array(nc["avg_v"])
avg_wvel  = np.array(nc["avg_w"])
avg_theta = np.array(nc["theta_pert"]) + np.array(nc["hy_theta_cells"])[hs:hs+nz,np.newaxis,np.newaxis]
for fname in fnames[1:] :
    nc = Dataset(fname,"r")
    avg_uvel  += np.array(nc["avg_u"])
    avg_vvel  += np.array(nc["avg_v"])
    avg_wvel  += np.array(nc["avg_w"])
    avg_theta += np.array(nc["theta_pert"]) + np.array(nc["hy_theta_cells"])[hs:hs+nz,np.newaxis,np.newaxis]
avg_uvel  /= len(fnames)
avg_vvel  /= len(fnames)
avg_wvel  /= len(fnames)
avg_theta /= len(fnames)
avg_mag   = np.mean(np.sqrt(avg_uvel*avg_uvel+avg_vvel*avg_vvel+avg_wvel*avg_wvel),axis=(1,2))
avg_theta = np.mean(avg_theta,axis=(1,2))

nc = Dataset(fnames[len(fnames)-1],"r")
uvel  = np.array(nc["uvel"])
vvel  = np.array(nc["vvel"])
wvel  = np.array(nc["wvel"])
theta = np.array(nc["theta_pert"]) + np.array(nc["hy_theta_cells"])[hs:hs+nz,np.newaxis,np.newaxis]


fig,((ax1),(ax2)) = plt.subplots(1,2,figsize=(6,4))
ax = fig.gca()
z2 = get_ind(z,0.3)
ax1.plot(avg_mag[:z2],z[:z2],color="black",label="mean")
ax1.set_xlabel("Wind speed (m/s)")
ax1.set_ylabel("z-location (km)")
ax1.set_xlim(left=0,right=10)
ax1.margins(x=0)
ax1.grid()
ax2.plot(avg_theta[:z2],z[:z2],color="black",label="mean")
ax2.set_xlabel("Potential Temperature (K)")
ax2.set_ylabel("z-location (km)")
ax2.set_xlim(left=262,right=268)
ax2.margins(x=0)
ax2.grid()
plt.tight_layout()
plt.savefig("ABL_stable_avg_uvel_theta_height.png",dpi=600)
# plt.show()
plt.close()


fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(6,8),sharey=True)
X,Y = np.meshgrid(x,y)
zind = get_ind(z,.1)
mn  = np.mean(theta[zind,:,:])
std = np.std (theta[zind,:,:])
CS1 = ax1.contourf(X,Y,theta[zind,:,:],levels=np.arange(mn-2*std,mn+2*std,4*std/100),cmap="turbo",extend="both")
ax1.axis('scaled')
ax1.set_xlabel("x-location (km)")
ax1.set_ylabel("y-location (km)")
ax1.margins(x=0)
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("bottom", size="4%", pad=0.5)
cbar1 = plt.colorbar(CS1,orientation="horizontal",cax=cax1)
cbar1.ax.tick_params(labelrotation=40)

mn  = np.mean(wvel[zind,:,:])
std = np.std (wvel[zind,:,:])
CS2 = ax2.contourf(X,Y,wvel[zind,:,:],levels=np.arange(mn-2*std,mn+2*std,4*std/100),cmap="turbo",extend="both")
ax2.axis('scaled')
ax2.set_xlabel("x-location (km)")
ax2.set_ylabel("y-location (km)")
ax2.margins(x=0)
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("bottom", size="4%", pad=0.5)
cbar2 = plt.colorbar(CS2,orientation="horizontal",cax=cax2)
cbar2.ax.tick_params(labelrotation=40)

X,Z = np.meshgrid(x,z)
yind = get_ind(z,0.2)
t1 = 262.75
t2 = 265.25
CS3 = ax3.contourf(X,Z,theta[:,yind,:],levels=np.arange(t1,t2,(t2-t1)/100),cmap="Spectral",extend="both")
ax3.axis('scaled')
ax3.set_xlabel("x-location (km)")
ax3.set_ylabel("y-location (km)")
ax3.margins(x=0)
divider = make_axes_locatable(ax3)
cax3 = divider.append_axes("bottom", size="4%", pad=0.5)
cbar3 = plt.colorbar(CS3,orientation="horizontal",cax=cax3)
cbar3.ax.tick_params(labelrotation=40)

mn  = np.mean(wvel[:,yind,:])
std = np.std (wvel[:,yind,:])
CS4 = ax4.contourf(X,Z,wvel[:,yind,:],levels=np.arange(mn-2*std,mn+2*std,4*std/100),cmap="turbo",extend="both")
ax4.axis('scaled')
ax4.set_xlabel("x-location (km)")
ax4.set_ylabel("y-location (km)")
ax4.margins(x=0)
divider = make_axes_locatable(ax4)
cax4 = divider.append_axes("bottom", size="4%", pad=0.5)
cbar4 = plt.colorbar(CS4,orientation="horizontal",cax=cax4)
cbar4.ax.tick_params(labelrotation=40)
plt.tight_layout()
plt.savefig("ABL_stable_contourf.png",dpi=600)
# plt.show()
plt.close()

