from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np

nc = Dataset("test_00000050.nc","r")
nz = nc.dimensions["z"].size
ny = nc.dimensions["y"].size
nx = nc.dimensions["x"].size
x = nc.variables["x"][:]/1000.
y = nc.variables["y"][:]/1000.
z = nc.variables["z"][:]/1000.
nlevs = 200


zind = int(78.6/1200.*nz)
zloc = (zind+0.5)*(1200./nz)
u = nc.variables["uvel"][zind,:,:,0]
v = nc.variables["vvel"][zind,:,:,0]
w = nc.variables["wvel"][zind,:,:,0]
mag = np.sqrt( u*u + v*v + w*w )
X,Y = np.meshgrid(x,y)
fig1, ax2 = plt.subplots(layout='constrained')
CS = ax2.contourf(X,Y,mag,nlevs, cmap="gist_rainbow")
ax2.set_title('Wind magnitude at z = '+str(int(zloc))+' m')
ax2.set_xlabel('$x$-location (km)')
ax2.set_ylabel('$y$-location (km)')
cbar = fig1.colorbar(CS)
cbar.ax.set_ylabel('wind magnitude (m/s)')
plt.show()
plt.close()

fig1, ax2 = plt.subplots(layout='constrained')
CS = ax2.contourf(X,Y,u,nlevs, cmap="gist_rainbow")
ax2.set_title('u at z = '+str(int(zloc))+' m')
ax2.set_xlabel('$x$-location (km)')
ax2.set_ylabel('$y$-location (km)')
cbar = fig1.colorbar(CS)
cbar.ax.set_ylabel('wind magnitude (m/s)')
plt.show()
plt.close()

fig1, ax2 = plt.subplots(layout='constrained')
CS = ax2.contourf(X,Y,u,nlevs, cmap="gist_rainbow")
ax2.set_title('v at z = '+str(int(zloc))+' m')
ax2.set_xlabel('$x$-location (km)')
ax2.set_ylabel('$y$-location (km)')
cbar = fig1.colorbar(CS)
cbar.ax.set_ylabel('wind magnitude (m/s)')
plt.show()
plt.close()



yind = ny/2
yloc = (yind+0.5)*(4000./ny)
zind2 = int(700./1200.*nz)
u = nc.variables["uvel"][:zind2,yind,:,0]
v = nc.variables["vvel"][:zind2,yind,:,0]
w = nc.variables["wvel"][:zind2,yind,:,0]
mag = np.sqrt( u*u + v*v + w*w )
X,Z = np.meshgrid(x,z[:zind2])
fig1, ax2 = plt.subplots(layout='constrained')
CS = ax2.contourf(X,Z,mag,nlevs, cmap="gist_stern")
ax2.set_title('Wind magnitude at y = '+str(int(yloc))+' m')
ax2.set_xlabel('$x$-location (km)')
ax2.set_ylabel('$z$-location (km)')
ax2.set_aspect('equal')
cbar = fig1.colorbar(CS,orientation='horizontal')
cbar.ax.set_ylabel('wind magnitude (m/s)')
plt.show()
plt.close()



u = nc.variables["uvel"][:,:,:,0]
v = nc.variables["vvel"][:,:,:,0]
w = nc.variables["wvel"][:,:,:,0]

plt.plot(np.mean(u,(1,2)),z)
plt.show()
plt.close()

plt.plot(np.mean(v,(1,2)),z)
plt.show()
plt.close()

plt.plot(np.mean(np.sqrt(u*u+v*v+w*w),(1,2)),z)
plt.show()
plt.close()

