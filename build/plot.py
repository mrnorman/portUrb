from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np

nc = Dataset("test_00000040.nc","r")
nz = nc.dimensions["z"].size
ny = nc.dimensions["y"].size
nx = nc.dimensions["x"].size
x = nc.variables["x"][:]/1000.
y = nc.variables["y"][:]/1000.
z = nc.variables["z"][:]/1000.
u = nc.variables["uvel"][0,nz/5,:,:]
v = nc.variables["vvel"][0,nz/5,:,:]
w = nc.variables["wvel"][0,nz/5,:,:]
mag = np.sqrt( u*u + v*v + w*w )

nlevs = 50

X,Y = np.meshgrid(x,y)
fig1, ax2 = plt.subplots(layout='constrained')
CS = ax2.contourf(X,Y,mag,nlevs, cmap="jet")
ax2.set_title('Wind magnitude at z=nz/5')
ax2.set_xlabel('$x$-location (km)')
ax2.set_ylabel('$y$-location (km)')
cbar = fig1.colorbar(CS)
cbar.ax.set_ylabel('wind magnitude (m/s)')
plt.show()
plt.close()

fig1, ax2 = plt.subplots(layout='constrained')
CS = ax2.contourf(X,Y,u,nlevs, cmap="jet")
ax2.set_title('u-velocity at z=nz/5')
ax2.set_xlabel('$x$-location (km)')
ax2.set_ylabel('$y$-location (km)')
cbar = fig1.colorbar(CS)
cbar.ax.set_ylabel('Velocity (m/s)')
plt.show()
plt.close()

fig1, ax2 = plt.subplots(layout='constrained')
CS = ax2.contourf(X,Y,w,nlevs, cmap="jet")
ax2.set_title('w-velocity at z=nz/5')
ax2.set_xlabel('$x$-location (km)')
ax2.set_ylabel('$y$-location (km)')
cbar = fig1.colorbar(CS)
cbar.ax.set_ylabel('Velocity (m/s)')
plt.show()
plt.close()
