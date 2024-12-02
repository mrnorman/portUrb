
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_ind(arr,val) :
    return np.argmin(np.abs(arr-val))

workdir = "/lustre/orion/stf006/scratch/imn/sphere"
tlist = [2,3,4]
for i in tlist :
  nc = Dataset(f"{workdir}/sphere_hi_0000000{i}.nc","r")
  if i==tlist[0] :
    x_hi = np.array(nc["x"])
    y_hi = np.array(nc["y"])
    z_hi = np.array(nc["z"])
    dx_hi = x_hi[1]-x_hi[0]
    dy_hi = y_hi[1]-y_hi[0]
    dz_hi = z_hi[1]-z_hi[0]
    xlen_hi = x_hi[-1]+dx_hi/2
    ylen_hi = y_hi[-1]+dy_hi/2
    zlen_hi = z_hi[-1]+dz_hi/2
    nx_hi = nc["x"].size
    ny_hi = nc["y"].size
    nz_hi = nc["z"].size
  utmp = np.mean(np.reshape(np.array(nc["avg_u"]),(nz_hi//5,5,ny_hi//5,5,nx_hi//5,5)),axis=(1,3,5))
  vtmp = np.mean(np.reshape(np.array(nc["avg_v"]),(nz_hi//5,5,ny_hi//5,5,nx_hi//5,5)),axis=(1,3,5))
  wtmp = np.mean(np.reshape(np.array(nc["avg_w"]),(nz_hi//5,5,ny_hi//5,5,nx_hi//5,5)),axis=(1,3,5))
  u_hi = utmp  if i==tlist[0]  else  u_hi+utmp
  v_hi = vtmp  if i==tlist[0]  else  v_hi+vtmp
  w_hi = wtmp  if i==tlist[0]  else  w_hi+wtmp
  if (i == tlist[-1]) :
    u_hi /= len(tlist)
    v_hi /= len(tlist)
    w_hi /= len(tlist)

for p in [1,2,3,4,5,6,7] :
  for i in tlist :
    nc = Dataset(f"{workdir}/sphere_lo_{p}_0000000{i}.nc","r")
    if p==1 and i==tlist[0] :
      x_lo = np.array(nc["x"])
      y_lo = np.array(nc["y"])
      z_lo = np.array(nc["z"])
      dx_lo = x_lo[1]-x_lo[0]
      dy_lo = y_lo[1]-y_lo[0]
      dz_lo = z_lo[1]-z_lo[0]
      xlen_lo = x_lo[-1]+dx_lo/2
      ylen_lo = y_lo[-1]+dy_lo/2
      zlen_lo = z_lo[-1]+dz_lo/2
      nx_lo = nc["x"].size
      ny_lo = nc["y"].size
      nz_lo = nc["z"].size
    utmp = np.array(nc["avg_u"])
    vtmp = np.array(nc["avg_v"])
    wtmp = np.array(nc["avg_w"])
    u_lo = utmp  if i==tlist[0]  else  u_lo+utmp
    v_lo = vtmp  if i==tlist[0]  else  v_lo+vtmp
    w_lo = wtmp  if i==tlist[0]  else  w_lo+wtmp
    if (i == tlist[-1]) :
      u_lo /= len(tlist)
      v_lo /= len(tlist)
      w_lo /= len(tlist)
  uerr = np.mean(np.abs(u_hi[:,:,:nx_lo//2]-u_lo[:,:,:nx_lo//2]))
  verr = np.mean(np.abs(v_hi[:,:,:nx_lo//2]-v_lo[:,:,:nx_lo//2]))
  werr = np.mean(np.abs(w_hi[:,:,:nx_lo//2]-w_lo[:,:,:nx_lo//2]))
  print( f"{p}: {(uerr+verr+werr)/3}" )


p = 5
for i in tlist :
  nc = Dataset(f"{workdir}/sphere_lo_{p}_0000000{i}.nc","r")
  if p==1 and i==tlist[0] :
    x_lo = np.array(nc["x"])
    y_lo = np.array(nc["y"])
    z_lo = np.array(nc["z"])
    dx_lo = x_lo[1]-x_lo[0]
    dy_lo = y_lo[1]-y_lo[0]
    dz_lo = z_lo[1]-z_lo[0]
    xlen_lo = x_lo[-1]+dx_lo/2
    ylen_lo = y_lo[-1]+dy_lo/2
    zlen_lo = z_lo[-1]+dz_lo/2
    nx_lo = nc["x"].size
    ny_lo = nc["y"].size
    nz_lo = nc["z"].size
  utmp = np.array(nc["avg_u"])
  vtmp = np.array(nc["avg_v"])
  wtmp = np.array(nc["avg_w"])
  u_lo = utmp  if i==tlist[0]  else  u_lo+utmp
  v_lo = vtmp  if i==tlist[0]  else  v_lo+vtmp
  w_lo = wtmp  if i==tlist[0]  else  w_lo+wtmp
  if (i == tlist[-1]) :
    u_lo /= len(tlist)
    v_lo /= len(tlist)
    w_lo /= len(tlist)


fig = plt.figure(figsize=(6,4))
ax = fig.gca()
X,Y = np.meshgrid(x_lo,y_lo)
mag = np.sqrt(u_hi*u_hi+v_hi*v_hi+w_hi*w_hi)[nz_lo//2-1,:,:]
mn  = np.min(mag)
mx  = np.max(mag)
CS = ax.contourf(X,Y,mag,levels=np.arange(mn,mx,(mx-mn)/100),cmap="turbo",extend="both")
ax.axis('scaled')
ax.set_xlabel("x-location (km)")
ax.set_ylabel("y-location (km)")
ax.margins(x=0)
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("bottom", size="4%", pad=0.5)
plt.colorbar(CS,orientation="horizontal",cax=cax)
plt.margins(x=0)
plt.tight_layout()
plt.savefig("sphere_hi_fine.png",dpi=600)
plt.show()
plt.close()

fig = plt.figure(figsize=(6,4))
ax = fig.gca()
X,Y = np.meshgrid(x_lo,y_lo)
mag = np.sqrt(u_lo*u_lo+v_lo*v_lo+w_lo*w_lo)[nz_lo//2-1,:,:]
CS = ax.contourf(X,Y,mag,levels=np.arange(mn,mx,(mx-mn)/100),cmap="turbo",extend="both")
ax.axis('scaled')
ax.set_xlabel("x-location (km)")
ax.set_ylabel("y-location (km)")
ax.margins(x=0)
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("bottom", size="4%", pad=0.5)
plt.colorbar(CS,orientation="horizontal",cax=cax)
plt.margins(x=0)
plt.tight_layout()
plt.savefig("sphere_lo_coarse.png",dpi=600)
plt.show()
plt.close()

