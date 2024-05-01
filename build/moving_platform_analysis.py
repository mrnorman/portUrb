
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

nc_orig = Dataset('no_platform_00000004.nc','r')
nc_pert = Dataset('with_platform_00000004.nc','r')
nx = nc_orig.dimensions["x"].size
ny = nc_orig.dimensions["y"].size
nz = nc_orig.dimensions["z"].size
x  = nc_orig.variables["x"][:]/1000
y  = nc_orig.variables["y"][:]/1000
z  = nc_orig.variables["z"][:]/1000
X,Y = np.meshgrid(x,y)

zhub = int(150./1000.*nz)
u_orig = nc_orig.variables["avg_u"][zhub,:,:,0]
u_pert = nc_pert.variables["avg_u"][zhub,:,:,0]
mx = np.max(np.abs(u_pert-u_orig))
CS = plt.contourf(X,Y,u_pert-u_orig,levels=np.arange(-mx,mx,0.01),cmap="seismic",extend="both")
plt.title("Time-averaged $u$: moving_platform - baseline, hub height")
plt.axis('scaled')
plt.margins(x=0)
plt.tight_layout()
plt.xlabel("x-location (km)")
plt.ylabel("y-location (km)")
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("bottom", size="5%", pad=0.5)
plt.colorbar(CS,orientation="horizontal",cax=cax)
plt.savefig("time_avg_u_hub.png",dpi=300)
plt.show()

plt.close()

u_orig = nc_orig.variables["avg_tke"][zhub,:,:,0]
u_pert = nc_pert.variables["avg_tke"][zhub,:,:,0]
mx = np.max(np.abs(u_pert-u_orig))
CS = plt.contourf(X,Y,u_pert-u_orig,levels=np.arange(-mx,mx,0.01),cmap="seismic",extend="both")
plt.title("Time-averaged unresolved TKE: moving_platform - baseline, hub height")
plt.axis('scaled')
plt.margins(x=0)
plt.tight_layout()
plt.xlabel("x-location (km)")
plt.ylabel("y-location (km)")
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("bottom", size="5%", pad=0.5)
plt.colorbar(CS,orientation="horizontal",cax=cax)
plt.savefig("time_avg_unresolved_tke_hub.png",dpi=300)
plt.show()

plt.close()

tke_orig = (nc_orig.variables["avg_up_up"][zhub,:,:,0]+\
            nc_orig.variables["avg_vp_vp"][zhub,:,:,0]+\
            nc_orig.variables["avg_wp_wp"][zhub,:,:,0])/2
tke_pert = (nc_pert.variables["avg_up_up"][zhub,:,:,0]+\
            nc_pert.variables["avg_vp_vp"][zhub,:,:,0]+\
            nc_pert.variables["avg_wp_wp"][zhub,:,:,0])/2
mx = np.max(np.abs(u_pert-u_orig))
CS = plt.contourf(X,Y,tke_pert-tke_orig,levels=np.arange(-mx,mx,0.01),cmap="seismic",extend="both")
plt.title(r"$\frac{1}{2hr}\int_0^{2hr}(\overline{u^\prime u^\prime}+\overline{v^\prime v^\prime}+\overline{w^\prime w^\prime})d\tau$;    $\vec{u}^\prime(t)=\vec{u}(t)-\frac{1}{t}\int_0^t{\vec{u}(t)}d\tau$: moving_platform - baseline, hub height")
plt.axis('scaled')
plt.margins(x=0)
plt.tight_layout()
plt.xlabel("x-location (km)")
plt.ylabel("y-location (km)")
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("bottom", size="5%", pad=0.5)
plt.colorbar(CS,orientation="horizontal",cax=cax)
plt.savefig("time_avg_resolved_tke_hub.png",dpi=300)
plt.show()

plt.close()


fig, axs = plt.subplots(2,sharex=True)
axs[0].hist(nc_orig.variables["mag_trace_turb_0"][:],bins=np.arange(7,15,0.1),density=True)
axs[1].hist(nc_pert.variables["mag_trace_turb_0"][:],bins=np.arange(7,15,0.1),density=True)
axs[0].set_title("Baseline (no moving platform)")
axs[1].set_title("Moving platform")
axs[1].set_xlabel("Wind Magnitude (m/s) Bins")
axs[0].set_ylabel("Probability Density")
axs[1].set_ylabel("Probability Density")
axs[0].set_ylim((0,4.0))
axs[1].set_ylim((0,4.0))
plt.show()
plt.close()


# fig, axs = plt.subplots(2,sharex=True)
# axs[0].hist(nc_orig.variables["normmag_trace_turb_0"][:],bins=np.arange(7,15,0.1),density=True)
# axs[1].hist(nc_pert.variables["normmag_trace_turb_0"][:],bins=np.arange(7,15,0.1),density=True)
# axs[0].set_title("Baseline (no moving platform)")
# axs[1].set_title("Moving platform")
# axs[1].set_xlabel("Wind Magnitude (m/s) Bins")
# axs[0].set_ylabel("Probability Density")
# axs[1].set_ylabel("Probability Density")
# axs[0].set_ylim((0,4.0))
# axs[1].set_ylim((0,4.0))
# plt.show()
# plt.close()


fig, axs = plt.subplots(2,sharex=True)
axs[0].hist(nc_orig.variables["normmag0_trace_turb_0"][:],bins=np.arange(7,15,0.1),density=True)
axs[1].hist(nc_pert.variables["normmag0_trace_turb_0"][:],bins=np.arange(7,15,0.1),density=True)
axs[0].set_title("Baseline (no moving platform)")
axs[1].set_title("Moving platform")
axs[1].set_xlabel("Wind Magnitude (m/s) Bins")
axs[0].set_ylabel("Probability Density")
axs[1].set_ylabel("Probability Density")
axs[0].set_ylim((0,2.5))
axs[1].set_ylim((0,2.5))
plt.show()
plt.close()

print("Orig Mean MW: ",np.mean(nc_orig.variables["power_trace_turb_0"][:]))
print("Pert Mean MW: ",np.mean(nc_pert.variables["power_trace_turb_0"][:]))

