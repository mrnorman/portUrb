from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

nc_float = Dataset("floating/no_platform_00000004.nc","r")
nc_fixed = Dataset("fixed/no_platform_00000004.nc","r")
nz = nc_float.dimensions["z"].size
ny = nc_float.dimensions["y"].size
nx = nc_float.dimensions["x"].size
x = nc_float.variables["x"][:]/1000.
y = nc_float.variables["y"][:]/1000.
z = nc_float.variables["z"][:]/1000.
X,Y = np.meshgrid(x,y)

float_power_trace_turb_0 = nc_float.variables["power_trace_turb_0"][:]
fixed_power_trace_turb_0 = nc_fixed.variables["power_trace_turb_0"][:]

plt.plot(float_power_trace_turb_0,label="floating")
plt.plot(fixed_power_trace_turb_0,label="fixed")
plt.legend()
plt.show()

print("Floating: ",np.mean(float_power_trace_turb_0))
print("Fixed   : ",np.mean(fixed_power_trace_turb_0))


plt.close()

