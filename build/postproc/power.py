from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np

# nc = Dataset("test_00000051.nc","r")
# nz = nc.dimensions["z"].size
# ny = nc.dimensions["y"].size
# nx = nc.dimensions["x"].size
# x = nc.variables["x"][:]/1000.
# y = nc.variables["y"][:]/1000.
# z = nc.variables["z"][:]/1000.
# nlevs = 200
# 
# power = nc.variables["power_trace_turb_0"][:] + \
#         nc.variables["power_trace_turb_1"][:] + \
#         nc.variables["power_trace_turb_2"][:] + \
#         nc.variables["power_trace_turb_3"][:] + \
#         nc.variables["power_trace_turb_4"][:] + \
#         nc.variables["power_trace_turb_5"][:] + \
#         nc.variables["power_trace_turb_6"][:] + \
#         nc.variables["power_trace_turb_7"][:] + \
#         nc.variables["power_trace_turb_8"][:]
# 
# plt.plot( power )
# plt.xlabel("Time Step number")
# plt.ylabel("Total power (MW)")
# plt.show()
# plt.close()
# 
# yaw0 = nc.variables["yaw_trace_turb_0"][:]
# yaw1 = nc.variables["yaw_trace_turb_1"][:]
# yaw2 = nc.variables["yaw_trace_turb_2"][:]
# yaw3 = nc.variables["yaw_trace_turb_3"][:]
# yaw4 = nc.variables["yaw_trace_turb_4"][:]
# yaw5 = nc.variables["yaw_trace_turb_5"][:]
# yaw6 = nc.variables["yaw_trace_turb_6"][:]
# yaw7 = nc.variables["yaw_trace_turb_7"][:]
# yaw8 = nc.variables["yaw_trace_turb_8"][:]
# 
# plt.plot( yaw0 , linewidth=0.25 )
# plt.plot( yaw1 , linewidth=0.25 )
# plt.plot( yaw2 , linewidth=0.25 )
# plt.plot( yaw3 , linewidth=0.25 )
# plt.plot( yaw4 , linewidth=0.25 )
# plt.plot( yaw5 , linewidth=0.25 )
# plt.plot( yaw6 , linewidth=0.25 )
# plt.plot( yaw7 , linewidth=0.25 )
# plt.plot( yaw8 , linewidth=0.25 )
# plt.xlabel("Time Step number")
# plt.ylabel("Yaw angle (deg. counter-clockwise from west-facing)")
# plt.legend(["Turbine 1", \
#             "Turbine 2", \
#             "Turbine 3", \
#             "Turbine 4", \
#             "Turbine 5", \
#             "Turbine 6", \
#             "Turbine 7", \
#             "Turbine 8", \
#             "Turbine 9"],loc="lower right")
# plt.show()



# X,Y = np.meshgrid(x,y)
# prop = nc.variables["windmill_prop"][int(150./1000.)-1,:,:,0]
# fig1, ax2 = plt.subplots(layout='constrained')
# CS = ax2.contourf(X,Y,prop,nlevs, cmap="gist_rainbow")
# ax2.set_title('u at z = '+str(int(150))+' m')
# ax2.set_xlabel('$x$-location (km)')
# ax2.set_ylabel('$y$-location (km)')
# ax2.set_aspect('equal')
# cbar = fig1.colorbar(CS,orientation='horizontal')
# cbar.ax.set_ylabel('wind magnitude (m/s)')
# plt.show()
# plt.close()




nc = Dataset("test_00000051.nc","r")
nz = nc.dimensions["z"].size
ny = nc.dimensions["y"].size
nx = nc.dimensions["x"].size
x = nc.variables["x"][:]/1000.
y = nc.variables["y"][:]/1000.
z = nc.variables["z"][:]/1000.
nlevs = 200

power = nc.variables["power_trace_turb_0"][:]

plt.plot( power )
plt.xlabel("Time Step number")
plt.ylabel("Total power (MW)")
plt.show()
plt.close()

yaw0 = nc.variables["yaw_trace_turb_0"][:]

plt.plot( yaw0 , linewidth=0.25 )
plt.xlabel("Time Step number")
plt.ylabel("Yaw angle (deg. counter-clockwise from west-facing)")
plt.legend(["Turbine 1"],loc="lower right")
plt.show()
