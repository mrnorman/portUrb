from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

nc = Dataset("no_platform_00000001.nc","r")
nz = nc.dimensions["z"].size
ny = nc.dimensions["y"].size
nx = nc.dimensions["x"].size
x = nc.variables["x"][:]/1000.
y = nc.variables["y"][:]/1000.
z = nc.variables["z"][:]/1000.
X,Y = np.meshgrid(x,y)

zind = int(150./1000.*nz)
TKE = ( nc.variables["avg_up_up"][zind,:,:,0] + \
        nc.variables["avg_vp_vp"][zind,:,:,0] + \
        nc.variables["avg_wp_wp"][zind,:,:,0] )

CS = plt.contourf(X,Y,TKE,levels=100,cmap="jet",extend="both")
plt.title("u_moving_platform - u_orig at hub height")
plt.axis('scaled')
plt.margins(x=0)
plt.tight_layout()
plt.xlabel("x-location (km)")
plt.ylabel("y-location (km)")
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("bottom", size="5%", pad=0.5)
plt.colorbar(CS,orientation="horizontal",cax=cax)
plt.show()

plt.close()

up = nc.variables["uvel"][zind,:,:,0]-nc.variables["avg_u"][zind,:,:,0]
vp = nc.variables["vvel"][zind,:,:,0]-nc.variables["avg_v"][zind,:,:,0]
wp = nc.variables["wvel"][zind,:,:,0]-nc.variables["avg_w"][zind,:,:,0]
CS = plt.contourf(X,Y,(up*up+vp*vp+wp*wp)/2,levels=100,cmap="jet",extend="both")
plt.title("u_moving_platform - u_orig at hub height")
plt.axis('scaled')
plt.margins(x=0)
plt.tight_layout()
plt.xlabel("x-location (km)")
plt.ylabel("y-location (km)")
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("bottom", size="5%", pad=0.5)
plt.colorbar(CS,orientation="horizontal",cax=cax)
plt.show()

plt.close()

