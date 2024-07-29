from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_ind(arr,val) :
    return np.argmin(np.abs(arr-val))

def turbulent_intensity(fname) :
    nc    = Dataset(fname,"r")
    x     = nc.variables["x"][:]
    y     = nc.variables["y"][:]
    z     = nc.variables["z"][:]
    i1    = get_ind(x,5000/3.-127*2.5)
    i2    = get_ind(x,5000/3.-127*2.0)
    j1    = get_ind(y,1500/2.-127*2)
    j2    = get_ind(y,1500/2.+127*2)
    k1    = get_ind(z,89-127/2.)
    k2    = get_ind(z,89+127/2.)
    u     = nc.variables["uvel"][:,j1:j2+1,i1:i2+1]
    v     = nc.variables["vvel"][:,j1:j2+1,i1:i2+1]
    mag   = np.sqrt(u*u + v*v)
    tot   = 0
    count = 0
    for k in range(k1,k2+1) :
        tot   += np.std(mag [k,:,:])/np.mean(mag [k,:,:])
        count += 1
    return tot / count

def shear_exponent(fname) :
    nc     = Dataset(fname,"r")
    x      = nc.variables["x"][:]
    y      = nc.variables["y"][:]
    z      = nc.variables["z"][:]
    i1    = get_ind(x,5000/3.-127*2.5)
    i2    = get_ind(x,5000/3.-127*2.0)
    j1    = get_ind(y,1500/2.-127)
    j2    = get_ind(y,1500/2.+127)
    k1    = get_ind(z,89-127/2.)
    k2    = get_ind(z,89+127/2.)
    kref   = get_ind(z,89)
    zref   = z[kref]
    u      = nc.variables["avg_u"][kref,j1:j2+1,i1:i2+1]
    v      = nc.variables["avg_v"][kref,j1:j2+1,i1:i2+1]
    magref = np.mean(np.sqrt(u*u + v*v))
    tot    = 0
    count  = 0
    for k in range(k1,k2+1) :
        if (k != kref) :
            u      = nc.variables["avg_u"][k,j1:j2+1,i1:i2+1]
            v      = nc.variables["avg_v"][k,j1:j2+1,i1:i2+1]
            mag    = np.mean(np.sqrt(u*u + v*v))
            tot   += np.log(mag/magref)/np.log(z[k]/zref)
            count += 1
    return tot / count

def uhub(fname) :
    nc   = Dataset(fname,"r")
    x    = nc.variables["x"][:]
    y    = nc.variables["y"][:]
    z    = nc.variables["z"][:]
    i1   = get_ind(x,5000/3.-127*2.5)
    i2   = get_ind(x,5000/3.-127*2.0)
    j1   = get_ind(y,1500/2.-127)
    j2   = get_ind(y,1500/2.+127)
    for k in range(nc.dimensions["z"].size) :
        if (z[k] > 89) :
            k1 = k-1
            break
    k2 = k1+1
    u = nc.variables["avg_u"][k1,j1:j2+1,i1:i2+1]
    v = nc.variables["avg_v"][k1,j1:j2+1,i1:i2+1]
    mag1 = np.mean(np.sqrt(u*u+v*v))
    u = nc.variables["avg_u"][k2,j1:j2+1,i1:i2+1]
    v = nc.variables["avg_v"][k2,j1:j2+1,i1:i2+1]
    mag2 = np.mean(np.sqrt(u*u+v*v))
    return mag1*(z[k2]-89)/(z[k2]-z[k1]) + mag2*(89-z[k1])/(z[k2]-z[k1])

def dirhub(fname) :
    nc   = Dataset(fname,"r")
    x    = nc.variables["x"][:]
    y    = nc.variables["y"][:]
    z    = nc.variables["z"][:]
    i1   = get_ind(x,5000/3.-127*2.5)
    i2   = get_ind(x,5000/3.-127*2.0)
    j1   = get_ind(y,1500/2.-127)
    j2   = get_ind(y,1500/2.+127)
    kref = get_ind(z,89)
    u = nc.variables["avg_u"][kref,j1:j2+1,i1:i2+1]
    v = nc.variables["avg_v"][kref,j1:j2+1,i1:i2+1]
    return np.mean(np.arctan2(v,u)/np.pi*180)

for i in range(1,6) :
    fname = "validation_000000"+str(i).zfill(2)+".nc"
    print(fname,": ",turbulent_intensity(fname)," , ",
                     shear_exponent     (fname)," , ",
                     uhub               (fname)," , ",
                     dirhub             (fname))
print("")

fname = "validation_00000004.nc"

nc   = Dataset(fname,"r")
x    = (nc.variables["x"][:]-0.3*5000)/127
y    = (nc.variables["y"][:]-0.5*1500)/127
z    =  nc.variables["z"][:]
X,Y  = np.meshgrid(x,y)
u    = nc.variables["avg_u"][get_ind(z,89),:,:]
v    = nc.variables["avg_v"][get_ind(z,89),:,:]
mag  = np.sqrt(u*u + v*v)
CS = plt.contourf(X,Y,mag,levels=np.arange(0,10,0.1),cmap="coolwarm",extend="both")
plt.axis('scaled')
plt.margins(x=0)
plt.tight_layout()
plt.xlabel("x-location (turbine diameters)")
plt.ylabel("y-location (turbine diameters)")
plt.xlim(-5,20)
plt.ylim(-4,4)
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("bottom", size="5%", pad=0.5)
plt.colorbar(CS,orientation="horizontal",cax=cax)
plt.show()
plt.close()

nc   = Dataset(fname,"r")
x    = (nc.variables["x"][:]-0.3*5000)/127
y    = (nc.variables["y"][:]-0.5*1500)/127
z    =  nc.variables["z"][:]
X,Y  = np.meshgrid(x,y)
u    = nc.variables["uvel"][get_ind(z,89),:,:]
v    = nc.variables["vvel"][get_ind(z,89),:,:]
mag  = np.sqrt(u*u + v*v)
CS = plt.contourf(X,Y,mag,levels=np.arange(0,10,0.1),cmap="coolwarm",extend="both")
plt.axis('scaled')
plt.margins(x=0)
plt.tight_layout()
plt.xlabel("x-location (turbine diameters)")
plt.ylabel("y-location (turbine diameters)")
plt.xlim(-5,20)
plt.ylim(-4,4)
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("bottom", size="5%", pad=0.5)
plt.colorbar(CS,orientation="horizontal",cax=cax)
plt.show()
plt.close()

nc = Dataset(fname,"r")
nt = nc.dimensions["num_time_steps"].size
plt.plot([i/(nt-1)*600 for i in range(nt)],nc.variables["power_trace_turb_0"][:]*1000)
plt.xlabel("Time [s]")
plt.ylabel("Power [kW]")
plt.show()
plt.close()

print(np.mean(nc.variables["power_trace_turb_0"][:]*1000))
