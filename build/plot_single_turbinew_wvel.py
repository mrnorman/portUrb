from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


nc = Dataset("test_00000018.nc","r")
nx = nc.dimensions["x"].size
ny = nc.dimensions["y"].size
nz = nc.dimensions["z"].size
y1   = int(3*ny/8)
y2   = int(5*ny/8)
zind = int(150/500*nz)
w = nc.variables ["wvel"][zind,y1:y2,:,0]
x = nc.variables ["x"][:]/1000
y = nc.variables ["y"][y1:y2]/1000
z = nc.variables ["z"][zind]
X,Y = np.meshgrid(x,y)
CS = plt.contourf(X,Y,w,levels=np.arange(-1,1,0.01),cmap="jet",extend="both")
plt.axis('scaled')
plt.margins(x=0)
plt.tight_layout()
plt.ylim(y[0],y[y2-y1-1])
plt.xlabel("x-location (km)")
plt.ylabel("y-location (km)")
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("bottom", size="5%", pad=0.5)
plt.colorbar(CS,orientation="horizontal",cax=cax)
plt.savefig('wvel_turbine.png',dpi=600,bbox_inches='tight')
plt.show()





# double x(x) ;
# double y(y) ;
# double z(z) ;
# double density_dry(z, y, x, ens) ;
# double uvel(z, y, x, ens) ;
# double vvel(z, y, x, ens) ;
# double wvel(z, y, x, ens) ;
# double temperature(z, y, x, ens) ;
# double etime(t) ;
# double file_counter(t) ;
# double water_vapor(z, y, x, ens) ;
# double TKE(z, y, x, ens) ;
# double windmill_prop(z, y, x, ens) ;
# double blade_prop(z, y, x, ens) ;
# double immersed_proportion(z, y, x, ens) ;
# double surface_temp(y, x, ens) ;
# double avg_u(z, y, x, ens) ;
# double avg_v(z, y, x, ens) ;
# double avg_w(z, y, x, ens) ;
# double avg_tke(z, y, x, ens) ;
# double power_trace_turb_0(num_time_steps) ;
# double yaw_trace_turb_0(num_time_steps) ;
# double hy_dens_cells(z_halo, ens) ;
# double hy_theta_cells(z_halo, ens) ;
# double hy_pressure_cells(z_halo, ens) ;
# double theta(z, y, x, ens) ;

