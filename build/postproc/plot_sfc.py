
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

prefix="city_2m_1e-6_"
t1 = 10
t2 = 41
for t in range(t1,t2) :
  nc = Dataset(f"{prefix}{t:08d}.nc","r")
  u = np.array(nc["avg_up_up"][:,:,:])
  v = np.array(nc["avg_vp_vp"][:,:,:])
  w = np.array(nc["avg_wp_wp"][:,:,:])
  uavg = u if (t==t1) else uavg+u
  vavg = v if (t==t1) else vavg+v
  wavg = w if (t==t1) else wavg+w
uavg /= (t2-t1)
vavg /= (t2-t1)
wavg /= (t2-t1)
mag_slip = np.sqrt(uavg*uavg+vavg*vavg+wavg*wavg)
x = np.array(nc["x"][:])
y = np.array(nc["y"][:])
z = np.array(nc["z"][:])
X,Y = np.meshgrid(x,y)
immersed = np.array(nc["immersed_proportion"][:,:,:])


prefix="city_2m_5e-1_"
for t in range(t1,t2) :
  nc = Dataset(f"{prefix}{t:08d}.nc","r")
  u = np.array(nc["avg_up_up"][:,:,:])
  v = np.array(nc["avg_vp_vp"][:,:,:])
  w = np.array(nc["avg_wp_wp"][:,:,:])
  uavg = u if (t==t1) else uavg+u
  vavg = v if (t==t1) else vavg+v
  wavg = w if (t==t1) else wavg+w
uavg /= (t2-t1)
vavg /= (t2-t1)
wavg /= (t2-t1)
mag_fric = np.sqrt(uavg*uavg+vavg*vavg+wavg*wavg)


for k in range(0,len(z),10) :
  var = mag_slip[k,:,:]-mag_fric[k,:,:]
  print(z[k],np.min(var),np.max(var))
  mn = np.min(var)
  mx = np.max(var)
  mx = max(abs(mn),abs(mx))
  CS = plt.contourf(X,Y,var,levels=np.arange(-mx,mx,2*mx/100),cmap="seismic",extend="both")
  plt.axis('scaled')
  plt.xlabel("x-location (km)")
  plt.ylabel("y-location (km)")
  divider = make_axes_locatable(plt.gca())
  cax = divider.append_axes("bottom", size="5%", pad=0.5)
  cbar = plt.colorbar(CS,orientation="horizontal",cax=cax)
  cbar.ax.tick_params(labelrotation=45)
  plt.margins(x=0)
  plt.tight_layout()
  plt.show()
  plt.close()

