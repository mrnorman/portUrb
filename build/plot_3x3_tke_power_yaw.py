from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import numpy as np


nc = Dataset("test_00000006.nc","r")
nx = nc.dimensions["x"].size
ny = nc.dimensions["y"].size
nz = nc.dimensions["z"].size
y1   = int(3*ny/8)
y2   = int(5*ny/8)
zind = int(150/500*nz)
tke = nc.variables ["TKE"][zind,:,:,0]
x = nc.variables ["x"][:]/1000
y = nc.variables ["y"][:]/1000
z = nc.variables ["z"][zind]
X,Y = np.meshgrid(x,y)
cmap = plt.cm.jet
CS = plt.contourf(X,Y,tke,levels=np.arange(0.005,2,0.005),cmap=cmap,norm=LogNorm(),extend="both")
plt.axis('scaled')
plt.margins(x=0)
plt.tight_layout()
plt.xlabel("x-location (km)")
plt.ylabel("y-location (km)")
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="3%", pad=0.2)
cbar = plt.colorbar(CS,orientation="vertical",cax=cax)
cbar.ax.tick_params(rotation=45)
plt.savefig('tke_3x3_turbines.png',dpi=600,bbox_inches='tight')
plt.show()

plt.close()

pow0 = nc.variables["power_trace_turb_0"][:]
pow1 = nc.variables["power_trace_turb_1"][:]
pow2 = nc.variables["power_trace_turb_2"][:]
pow3 = nc.variables["power_trace_turb_3"][:]
pow4 = nc.variables["power_trace_turb_4"][:]
pow5 = nc.variables["power_trace_turb_5"][:]
pow6 = nc.variables["power_trace_turb_6"][:]
pow7 = nc.variables["power_trace_turb_7"][:]
pow8 = nc.variables["power_trace_turb_8"][:]
yaw0 = nc.variables["yaw_trace_turb_0"][:]
yaw1 = nc.variables["yaw_trace_turb_1"][:]
yaw2 = nc.variables["yaw_trace_turb_2"][:]
yaw3 = nc.variables["yaw_trace_turb_3"][:]
yaw4 = nc.variables["yaw_trace_turb_4"][:]
yaw5 = nc.variables["yaw_trace_turb_5"][:]
yaw6 = nc.variables["yaw_trace_turb_6"][:]
yaw7 = nc.variables["yaw_trace_turb_7"][:]
yaw8 = nc.variables["yaw_trace_turb_8"][:]

t = [0.005769231*i/3600 for i in range(yaw8.size)]

plt.plot(t,pow0+pow1+pow2+pow3+pow4+pow5+pow6+pow7+pow8,color="black")
plt.xlabel("Model Time (hours)")
plt.ylabel("Total Power Production (MW)")
plt.savefig('power_production.eps',bbox_inches='tight')
plt.show()
plt.close()


plt.plot(t,yaw0,color="black"      ,linewidth=0.5)
plt.plot(t,yaw1,color="red"        ,linewidth=0.5)
plt.plot(t,yaw2,color="green"      ,linewidth=0.5)
plt.plot(t,yaw3,color="blue"       ,linewidth=0.5)
plt.plot(t,yaw4,color="cyan"       ,linewidth=0.5)
plt.plot(t,yaw5,color="purple"     ,linewidth=0.5)
plt.plot(t,yaw6,color="steelblue"  ,linewidth=0.5)
plt.plot(t,yaw7,color="orange"     ,linewidth=0.5)
plt.plot(t,yaw8,color="saddlebrown",linewidth=0.5)
plt.xlabel("Model Time (hours)")
plt.ylabel("Yaw Angle (degrees)")
plt.savefig('yaw_angles.eps',bbox_inches='tight')
plt.show()
plt.close()



