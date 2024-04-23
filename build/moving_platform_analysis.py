
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

nc_orig = Dataset('platform_motion_off_00000008.nc','r')
nc_pert = Dataset('platform_motion_on_00000008.nc','r')
nx = nc_orig.dimensions["x"].size
ny = nc_orig.dimensions["y"].size
nz = nc_orig.dimensions["z"].size
x  = nc_orig.variables["x"][:]/1000
y  = nc_orig.variables["y"][:]/1000
z  = nc_orig.variables["z"][:]/1000
X,Y = np.meshgrid(x,y)

zhub = int(150./500.*nz)
u_orig = nc_orig.variables["avg_u"][zhub,:,:,0]
u_pert = nc_pert.variables["avg_u"][zhub,:,:,0]
CS = plt.contourf(X,Y,u_pert-u_orig,levels=100,cmap="jet",extend="both")
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

xind = int(nx/3)
plt.plot(nc_orig.variables["avg_u"][zhub,:,xind,0],color="black",label="original")
plt.plot(nc_pert.variables["avg_u"][zhub,:,xind,0],color="red"  ,label="moving_platform")
plt.title(f"u-vel y-cross-sections, z=150m (hub height), x={x[xind]}km")
# plt.axis('scaled')
# plt.margins(x=0)
plt.tight_layout()
plt.xlabel("x-location (km)")
plt.ylabel("u-velocity")
plt.legend()
plt.show()

plt.close()

xind = int(2*nx/3)
plt.plot(nc_orig.variables["avg_u"][zhub,:,xind,0],color="black",label="original")
plt.plot(nc_pert.variables["avg_u"][zhub,:,xind,0],color="red"  ,label="moving_platform")
plt.title(f"u-vel y-cross-sections, z=150m (hub height), x={x[xind]}km")
# plt.axis('scaled')
# plt.margins(x=0)
plt.tight_layout()
plt.xlabel("x-location (km)")
plt.ylabel("u-velocity")
plt.legend()
plt.show()

plt.close()

xind = nx-1
plt.plot(nc_orig.variables["avg_u"][zhub,:,xind,0],color="black",label="original")
plt.plot(nc_pert.variables["avg_u"][zhub,:,xind,0],color="red"  ,label="moving_platform")
plt.title(f"u-vel y-cross-sections, z=150m (hub height), x={x[xind]}km")
# plt.axis('scaled')
# plt.margins(x=0)
plt.tight_layout()
plt.xlabel("x-location (km)")
plt.ylabel("u-velocity")
plt.legend()
plt.show()

plt.close()

print("Orig Mean MW: ",np.mean(nc_orig.variables["power_trace_turb_0"][:]+
                               nc_orig.variables["power_trace_turb_1"][:]+
                               nc_orig.variables["power_trace_turb_2"][:]+
                               nc_orig.variables["power_trace_turb_3"][:]+
                               nc_orig.variables["power_trace_turb_4"][:]+
                               nc_orig.variables["power_trace_turb_5"][:]+
                               nc_orig.variables["power_trace_turb_6"][:]+
                               nc_orig.variables["power_trace_turb_7"][:]+
                               nc_orig.variables["power_trace_turb_8"][:]))
print("Pert Mean MW: ",np.mean(nc_pert.variables["power_trace_turb_0"][:]+
                               nc_pert.variables["power_trace_turb_1"][:]+
                               nc_pert.variables["power_trace_turb_2"][:]+
                               nc_pert.variables["power_trace_turb_3"][:]+
                               nc_pert.variables["power_trace_turb_4"][:]+
                               nc_pert.variables["power_trace_turb_5"][:]+
                               nc_pert.variables["power_trace_turb_6"][:]+
                               nc_pert.variables["power_trace_turb_7"][:]+
                               nc_pert.variables["power_trace_turb_8"][:]))

