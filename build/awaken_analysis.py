from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Ellipse

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
    j1    = get_ind(y,1500/2.-127*2)
    j2    = get_ind(y,1500/2.+127*2)
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

def uhub(fname,x1=2.5,x2=2.0) :
    nc   = Dataset(fname,"r")
    x    = nc.variables["x"][:]
    y    = nc.variables["y"][:]
    z    = nc.variables["z"][:]
    i1   = get_ind(x,5000/3.-127*x1)
    i2   = get_ind(x,5000/3.-127*x2)
    j1   = get_ind(y,1500/2.-127*2)
    j2   = get_ind(y,1500/2.+127*2)
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

def dirhub(fname,x1=2.5,x2=2.0) :
    nc   = Dataset(fname,"r")
    x    = nc.variables["x"][:]
    y    = nc.variables["y"][:]
    z    = nc.variables["z"][:]
    i1   = get_ind(x,5000/3.-127*x1)
    i2   = get_ind(x,5000/3.-127*x2)
    j1   = get_ind(y,1500/2.-127*2)
    j2   = get_ind(y,1500/2.+127*2)
    kref = get_ind(z,89)
    u = nc.variables["avg_u"][kref,j1:j2+1,i1:i2+1]
    v = nc.variables["avg_v"][kref,j1:j2+1,i1:i2+1]
    return np.mean(np.arctan2(v,u)/np.pi*180)

end         = 4
misfit_best = 100
for i in range(1,end+1) :
    fname = "validation_000000"+str(i).zfill(2)+".nc"
    ti = turbulent_intensity(fname)
    se = shear_exponent     (fname)
    uh = uhub               (fname)
    dh = dirhub             (fname)
    misfit = (ti-0.097)**2/0.097**2 +\
             (se-0.116)**2/0.116**2 +\
             (uh-6.27 )**2/6.27 **2 +\
             (dh-4.33 )**2/4.33 **2
    if (misfit < misfit_best) :
        ind_best = i
        misfit_best = misfit
    print(fname,": ",ti," , ",se," , ",uh," , ",dh," , ",misfit)
print("Best file: ","validation_000000"+str(ind_best).zfill(2)+".nc")

fname_best  = "validation_000000"+str(ind_best).zfill(2)+".nc"
fname_best2 = "validation_precursor_000000"+str(ind_best).zfill(2)+".nc"

# fname_best  = "validation_00000003.nc"
# fname_best2 = "validation_precursor_00000003.nc"

nc   = Dataset(fname_best,"r")
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

nc   = Dataset(fname_best,"r")
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

nc = Dataset(fname_best,"r")
nt = nc.dimensions["num_time_steps"].size
plt.plot([i/(nt-1)*600 for i in range(nt)],nc.variables["normmag_trace_turb_0"][:],label="normmag")
plt.plot([i/(nt-1)*600 for i in range(nt)],nc.variables["normmag0_trace_turb_0"][:],label="normmag0")
plt.xlabel("Time [s]")
plt.ylabel("Normal wind magnitude [m/s]")
plt.legend()
plt.xlim(0,600)
plt.show()
plt.close()

nc = Dataset(fname_best,"r")
nt = nc.dimensions["num_time_steps"].size
plt.plot([i/(nt-1)*600 for i in range(nt)],nc.variables["power_trace_turb_0"][:]*1000)
plt.xlabel("Time [s]")
plt.ylabel("Power [kW]")
plt.xlim(0,600)
plt.ylim(0,1100)
plt.show()
plt.close()

nc   = Dataset(fname_best,"r")
x    = (nc.variables["x"][:]-0.3*5000)/127
y    = (nc.variables["y"][:]-0.5*1500)/127
z    =  nc.variables["z"][:]
y1   = get_ind(y,-2.5)
y2   = get_ind(y, 2.5)
X,Y  = np.meshgrid(x,y)
u    = nc.variables["avg_u"][get_ind(z,89),y1:y2+1,:]
v    = nc.variables["avg_v"][get_ind(z,89),y1:y2+1,:]
mag  = np.sqrt(u*u + v*v)
tke  = nc.variables["avg_tke"][get_ind(z,89),y1:y2+1,:] / nc.variables["density_dry"][get_ind(z,89),y1:y2+1,:]
nc   = Dataset(fname_best2,"r")
x    = (nc.variables["x"][:]-0.3*5000)/127
y    = (nc.variables["y"][:]-0.5*1500)/127
z    =  nc.variables["z"][:]
y1   = get_ind(y,-2.5)
y2   = get_ind(y, 2.5)
X,Y  = np.meshgrid(x,y)
u    = nc.variables["avg_u"][get_ind(z,89),y1:y2+1,:]
v    = nc.variables["avg_v"][get_ind(z,89),y1:y2+1,:]
mag2  = np.sqrt(u*u + v*v)
mag   = mag / mag2
fig, ((ax11,ax12),(ax21,ax22),(ax31,ax32),(ax41,ax42),(ax51,ax52)) = plt.subplots(5, 2,sharex=True)
ax11.plot(y[y1:y2+1],mag[:,get_ind(x, 1)])
ax21.plot(y[y1:y2+1],mag[:,get_ind(x, 2)])
ax31.plot(y[y1:y2+1],mag[:,get_ind(x, 4)])
ax41.plot(y[y1:y2+1],mag[:,get_ind(x, 6)])
ax51.plot(y[y1:y2+1],mag[:,get_ind(x,10)])
ax12.plot(y[y1:y2+1],tke[:,get_ind(x, 1)])
ax22.plot(y[y1:y2+1],tke[:,get_ind(x, 2)])
ax32.plot(y[y1:y2+1],tke[:,get_ind(x, 4)])
ax42.plot(y[y1:y2+1],tke[:,get_ind(x, 6)])
ax52.plot(y[y1:y2+1],tke[:,get_ind(x,10)])
plt.show()
plt.close()

nc   = Dataset(fname_best,"r")
x    = (nc.variables["x"][:]-0.3*5000)/127
y    = (nc.variables["y"][:]-0.5*1500)/127
z    =  nc.variables["z"][:]
y1   = get_ind(y,-2.5)
y2   = get_ind(y, 2.5)
z2   = get_ind(z,250)
u    = nc.variables["avg_u"][:z2+1,y1:y2+1,:]
v    = nc.variables["avg_v"][:z2+1,y1:y2+1,:]
mag  = np.sqrt(u*u + v*v)
fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5, 1,sharex=True)
Y,Z  = np.meshgrid(y[y1:y2+1],z[:z2+1])
CS1 = ax1.contourf(Y,Z,mag[:,:,get_ind(x, 0)+6],levels=np.arange(0,10,0.1),cmap="coolwarm",extend="both")
CS2 = ax2.contourf(Y,Z,mag[:,:,get_ind(x, 2)  ],levels=np.arange(0,10,0.1),cmap="coolwarm",extend="both")
CS3 = ax3.contourf(Y,Z,mag[:,:,get_ind(x, 4)  ],levels=np.arange(0,10,0.1),cmap="coolwarm",extend="both")
CS4 = ax4.contourf(Y,Z,mag[:,:,get_ind(x, 8)  ],levels=np.arange(0,10,0.1),cmap="coolwarm",extend="both")
CS5 = ax5.contourf(Y,Z,mag[:,:,get_ind(x,10)  ],levels=np.arange(0,10,0.1),cmap="coolwarm",extend="both")
ax1.add_patch(Ellipse([0,89],1,127,fill=False,edgecolor="black",linestyle='--'))
ax2.add_patch(Ellipse([0,89],1,127,fill=False,edgecolor="black",linestyle='--'))
ax3.add_patch(Ellipse([0,89],1,127,fill=False,edgecolor="black",linestyle='--'))
ax4.add_patch(Ellipse([0,89],1,127,fill=False,edgecolor="black",linestyle='--'))
ax5.add_patch(Ellipse([0,89],1,127,fill=False,edgecolor="black",linestyle='--'))
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
fig.colorbar(CS5, cax=cbar_ax)
plt.show()


print("Average power generation [kW]              : ",np.mean(nc.variables["power_trace_turb_0"][:]*1000))
print("Average power coefficient                  : ",np.mean(nc.variables["cp_trace_turb_0"][:]))
print("Average thrust coefficient                 : ",np.mean(nc.variables["ct_trace_turb_0"][:]))
print("Average disk norm wind mag [m/s]           : ",np.mean(nc.variables["normmag_trace_turb_0"][:]))
print("Average disk freestream norm wind mag [m/s]: ",np.mean(nc.variables["normmag0_trace_turb_0"][:]))

