from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray

def get_ind(arr,val) :
    return np.argmin(np.abs(arr-val))

t1 = 8
t2 = 20
times = [str(i).zfill(7) for i in range(t1,t2+1)]

winds = [i for i in range(5,24,2)]

N = len(winds)

prefixes_fixed = [ "turbulent_fixed-yaw-upstream_wind-5.000000_fixed-_"  ,\
                   "turbulent_fixed-yaw-upstream_wind-7.000000_fixed-_"  ,\
                   "turbulent_fixed-yaw-upstream_wind-9.000000_fixed-_"  ,\
                   "turbulent_fixed-yaw-upstream_wind-11.000000_fixed-_" ,\
                   "turbulent_fixed-yaw-upstream_wind-13.000000_fixed-_" ,\
                   "turbulent_fixed-yaw-upstream_wind-15.000000_fixed-_" ,\
                   "turbulent_fixed-yaw-upstream_wind-17.000000_fixed-_" ,\
                   "turbulent_fixed-yaw-upstream_wind-19.000000_fixed-_" ,\
                   "turbulent_fixed-yaw-upstream_wind-21.000000_fixed-_" ,\
                   "turbulent_fixed-yaw-upstream_wind-23.000000_fixed-_" ,]

prefixes_float = [ "turbulent_fixed-yaw-upstream_wind-5.000000_floating-_"  ,\
                   "turbulent_fixed-yaw-upstream_wind-7.000000_floating-_"  ,\
                   "turbulent_fixed-yaw-upstream_wind-9.000000_floating-_"  ,\
                   "turbulent_fixed-yaw-upstream_wind-11.000000_floating-_" ,\
                   "turbulent_fixed-yaw-upstream_wind-13.000000_floating-_" ,\
                   "turbulent_fixed-yaw-upstream_wind-15.000000_floating-_" ,\
                   "turbulent_fixed-yaw-upstream_wind-17.000000_floating-_" ,\
                   "turbulent_fixed-yaw-upstream_wind-19.000000_floating-_" ,\
                   "turbulent_fixed-yaw-upstream_wind-21.000000_floating-_" ,\
                   "turbulent_fixed-yaw-upstream_wind-23.000000_floating-_" ]

nctmp = Dataset(prefixes_fixed[0]+str(times[0]).zfill(8)+".nc","r")
x = np.array(nctmp["x"])
y = np.array(nctmp["y"])
z = np.array(nctmp["z"])
dx = x[1]-x[0]
dy = y[1]-y[0]
dz = z[1]-z[0]
xlen = x[-1]+dx/2
ylen = y[-1]+dy/2
zlen = z[-1]+dz/2
ihub = get_ind(x,xlen/2)
jhub = get_ind(y,ylen/2)
khub = get_ind(z,90)
X,Y = np.meshgrid(x,y)
z1 = get_ind(z,90-63)
z2 = get_ind(z,90+63)
y1 = get_ind(y,ylen/2-63)
y2 = get_ind(y,ylen/2+63)


fig,axestmp = plt.subplots(5,2,figsize=(8,10),sharex=True)
axes = np.reshape(np.array(axestmp),10)
for i in range(len(winds)):
  print(winds[i])
  var_fixed = np.sqrt(np.array(Dataset(prefixes_fixed[i]+str(times[0]).zfill(8)+".nc","r")["avg_up_up"][z1:z2+1,:,:]))
  var_float = np.sqrt(np.array(Dataset(prefixes_float[i]+str(times[0]).zfill(8)+".nc","r")["avg_up_up"][z1:z2+1,:,:]))
  for time in times[1:] :
    var_fixed += np.sqrt(np.array(Dataset(prefixes_fixed[i]+str(time).zfill(8)+".nc","r")["avg_up_up"][z1:z2+1,:,:]))
    var_float += np.sqrt(np.array(Dataset(prefixes_float[i]+str(time).zfill(8)+".nc","r")["avg_up_up"][z1:z2+1,:,:]))
  var_fixed /= len(times)
  var_float /= len(times)
  var = np.mean(var_float/var_fixed,axis=0)
  diff = 2*np.round(np.std(np.abs(1-var)),5)
  CS = axes[i].contourf(X,Y,var,levels=np.arange(1-diff,1+diff,2*diff/100),extend="both",cmap="bwr")
  axes[i].set_title(f"Winds: {winds[i]}")
  axes[i].axis('scaled')
  axes[i].margins(x=0)
  axes[i].set_xticks([])
  axes[i].set_yticks([])
  divider = make_axes_locatable(axes[i])
  cax = divider.append_axes("bottom", size="8%", pad=0.05)
  cbar = fig.colorbar(CS,orientation="horizontal",cax=cax)
  cbar.ax.tick_params(labelrotation=30)
fig.tight_layout()
plt.savefig("betti_contourxy_zhubavg_up_up_factor.png",dpi=600)
# plt.show()
plt.close()


fig,axestmp = plt.subplots(5,2,figsize=(8,10),sharex=True)
axes = np.reshape(np.array(axestmp),10)
for i in range(len(winds)):
  print(winds[i])
  var_fixed = np.abs(np.array(Dataset(prefixes_fixed[i]+str(times[0]).zfill(8)+".nc","r")["avg_u"][z1:z2+1,:,:]))
  var_float = np.abs(np.array(Dataset(prefixes_float[i]+str(times[0]).zfill(8)+".nc","r")["avg_u"][z1:z2+1,:,:]))
  for time in times[1:] :
    var_fixed += np.abs(np.array(Dataset(prefixes_fixed[i]+str(time).zfill(8)+".nc","r")["avg_u"][z1:z2+1,:,:]))
    var_float += np.abs(np.array(Dataset(prefixes_float[i]+str(time).zfill(8)+".nc","r")["avg_u"][z1:z2+1,:,:]))
  var_fixed /= len(times)
  var_float /= len(times)
  var = np.mean(var_float/var_fixed,axis=0)
  diff = 2*np.round(np.std(np.abs(1-var)),5)
  CS = axes[i].contourf(X,Y,var,levels=np.arange(1-diff,1+diff,2*diff/100),extend="both",cmap="bwr")
  axes[i].set_title(f"Winds: {winds[i]}")
  axes[i].axis('scaled')
  axes[i].margins(x=0)
  axes[i].set_xticks([])
  axes[i].set_yticks([])
  divider = make_axes_locatable(axes[i])
  cax = divider.append_axes("bottom", size="8%", pad=0.05)
  cbar = fig.colorbar(CS,orientation="horizontal",cax=cax)
  cbar.ax.tick_params(labelrotation=30)
fig.tight_layout()
plt.savefig("betti_contourxy_zhubavg_umeanfactor.png",dpi=600)
# plt.show()
plt.close()


x1 = get_ind(x,0.2*xlen)
x2 = get_ind(x,0.2*xlen+10*126)
fig,axestmp = plt.subplots(5,2,figsize=(8,10),sharex=True,sharey=True)
axes = np.reshape(np.array(axestmp),10)
for i in range(len(winds)):
  print(winds[i])
  var_fixed = np.sqrt(np.array(Dataset(prefixes_fixed[i]+str(times[0]).zfill(8)+".nc","r")["avg_up_up"][z1:z2+1,y1:y2+1,:]))
  var_float = np.sqrt(np.array(Dataset(prefixes_float[i]+str(times[0]).zfill(8)+".nc","r")["avg_up_up"][z1:z2+1,y1:y2+1,:]))
  for time in times[1:] :
    var_fixed += np.sqrt(np.array(Dataset(prefixes_fixed[i]+str(time).zfill(8)+".nc","r")["avg_up_up"][z1:z2+1,y1:y2+1,:]))
    var_float += np.sqrt(np.array(Dataset(prefixes_float[i]+str(time).zfill(8)+".nc","r")["avg_up_up"][z1:z2+1,y1:y2+1,:]))
  var_fixed /= len(times)
  var_float /= len(times)
  var = np.mean(var_float/var_fixed,axis=(0,1))
  axes[i].set_title(f"Winds: {winds[i]}")
  axes[i].plot(x,var)
  axes[i].margins(x=0)
  axes[i].set_ylim(0.99,1.01)
  axes[i].text(500,1.007,f"10D Wake Avg: {np.mean(var[x1:x2+1]):10.5e}")
fig.tight_layout()
plt.savefig("betti_linex_zhubavg_yhubavg_up_up_factor.png",dpi=600)
plt.close()


