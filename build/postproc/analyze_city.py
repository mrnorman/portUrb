
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

prefix="../../manhattan_2m/city_2m_1e-6_"
t1 = 10
t2 = 111
for t in range(t1,t2) :
  nc   = Dataset(f"{prefix}{t:08d}.nc","r")
  u    = np.array(nc["avg_u"][:,:,:])
  v    = np.array(nc["avg_v"][:,:,:])
  w    = np.array(nc["avg_w"][:,:,:])
  upup = np.array(nc["avg_up_up"][:,:,:])
  vpvp = np.array(nc["avg_vp_vp"][:,:,:])
  wpwp = np.array(nc["avg_wp_wp"][:,:,:])
  upwp = np.array(nc["avg_up_wp"][:,:,:])
  tke  = np.array(nc["avg_tke"  ][:,:,:])
  u_avg    = u    if (t==t1) else u_avg   +u
  v_avg    = v    if (t==t1) else v_avg   +v
  w_avg    = w    if (t==t1) else w_avg   +w
  upup_avg = upup if (t==t1) else upup_avg+upup
  vpvp_avg = vpvp if (t==t1) else vpvp_avg+vpvp
  wpwp_avg = wpwp if (t==t1) else wpwp_avg+wpwp
  upwp_avg = upwp if (t==t1) else upwp_avg+upwp
  tke_avg  = tke  if (t==t1) else tke_avg +tke
u_avg /= (t2-t1)
v_avg /= (t2-t1)
w_avg /= (t2-t1)
upup_avg /= (t2-t1)
vpvp_avg /= (t2-t1)
wpwp_avg /= (t2-t1)
upwp_avg /= (t2-t1)
tke_avg  /= (t2-t1)
mag_slip  = np.sqrt(u_avg*u_avg+v_avg*v_avg+w_avg*w_avg)
tkeres_slip  = (upup_avg + vpvp_avg + wpwp_avg)/2
upwp_slip = np.abs(upwp)
tke_slip  = tke_avg
x = np.array(nc["x"][:])
y = np.array(nc["y"][:])
z = np.array(nc["z"][:])
X,Y = np.meshgrid(x,y)
for k in [0] :
  var = tkeres_slip[k,:,:]
  print(z[k],np.min(var),np.max(var))
  mn = np.min(var)
  mx = np.max(var)
  CS = plt.contourf(X,Y,var,levels=np.arange(mn,mx,(mx-mn)/100),cmap="seismic",extend="both")
  plt.axis('scaled')
  plt.xlabel("x-location (km)")
  plt.ylabel("y-location (km)")
  divider = make_axes_locatable(plt.gca())
  cax = divider.append_axes("bottom", size="5%", pad=0.5)
  cbar = plt.colorbar(CS,orientation="horizontal",cax=cax)
  cbar.ax.tick_params(labelrotation=45)
  plt.margins(x=0)
  plt.tight_layout()
  plt.savefig(f"tkeres_1e-6_{z[k]}m.png",dpi=600)
  plt.show()
  plt.close()

  var = tke_slip[k,:,:]
  print(z[k],np.min(var),np.max(var))
  mn = np.min(var)
  mx = np.max(var)
  CS = plt.contourf(X,Y,var,levels=np.arange(mn,mx,(mx-mn)/100),cmap="seismic",extend="both")
  plt.axis('scaled')
  plt.xlabel("x-location (km)")
  plt.ylabel("y-location (km)")
  divider = make_axes_locatable(plt.gca())
  cax = divider.append_axes("bottom", size="5%", pad=0.5)
  cbar = plt.colorbar(CS,orientation="horizontal",cax=cax)
  cbar.ax.tick_params(labelrotation=45)
  plt.margins(x=0)
  plt.tight_layout()
  plt.savefig(f"tke_1e-6_{z[k]}m.png",dpi=600)
  plt.show()
  plt.close()


prefix="../../manhattan_2m/city_2m_5e-2_"
for t in range(t1,t2) :
  nc   = Dataset(f"{prefix}{t:08d}.nc","r")
  u    = np.array(nc["avg_u"    ][:,:,:])
  v    = np.array(nc["avg_v"    ][:,:,:])
  w    = np.array(nc["avg_w"    ][:,:,:])
  upup = np.array(nc["avg_up_up"][:,:,:])
  vpvp = np.array(nc["avg_vp_vp"][:,:,:])
  wpwp = np.array(nc["avg_wp_wp"][:,:,:])
  upwp = np.array(nc["avg_up_wp"][:,:,:])
  tke  = np.array(nc["avg_tke"  ][:,:,:])
  u_avg    = u    if (t==t1) else u_avg   +u
  v_avg    = v    if (t==t1) else v_avg   +v
  w_avg    = w    if (t==t1) else w_avg   +w
  upup_avg = upup if (t==t1) else upup_avg+upup
  vpvp_avg = vpvp if (t==t1) else vpvp_avg+vpvp
  wpwp_avg = wpwp if (t==t1) else wpwp_avg+wpwp
  upwp_avg = upwp if (t==t1) else upwp_avg+upwp
  tke_avg  = tke  if (t==t1) else tke_avg +tke
u_avg /= (t2-t1)
v_avg /= (t2-t1)
w_avg /= (t2-t1)
upup_avg /= (t2-t1)
vpvp_avg /= (t2-t1)
wpwp_avg /= (t2-t1)
upwp_avg /= (t2-t1)
tke_avg  /= (t2-t1)
mag_fric  = np.sqrt(u_avg*u_avg+v_avg*v_avg+w_avg*w_avg)
tkeres_fric  = (upup_avg + vpvp_avg + wpwp_avg)/2
upwp_fric = np.abs(upwp)
tke_fric  = tke_avg
for k in [0] :
  var = tkeres_slip[k,:,:]-tkeres_fric[k,:,:]
  print(z[k],np.min(var),np.max(var))
  mn = np.min(var)
  mx = np.max(var)
  mx = max(abs(mn),abs(mx))
  CS = plt.contourf(X,Y,var,levels=np.arange(-mx,mx,2*mx/100),cmap="seismic",extend="both")
  plt.axis('scaled')
  plt.xlabel("x-location (km)")
  plt.ylabel("y-location (km)")
  divider = make_axes_locatable(plt.gca())
  cax = divider.append_axes("bottom", size="5%", pad=0.5)
  cbar = plt.colorbar(CS,orientation="horizontal",cax=cax)
  cbar.ax.tick_params(labelrotation=45)
  plt.margins(x=0)
  plt.tight_layout()
  plt.savefig(f"tkeres_diff_5e-2_{z[k]}m.png",dpi=600)
  plt.show()
  plt.close()

  var = tke_slip[k,:,:]-tke_fric[k,:,:]
  print(z[k],np.min(var),np.max(var))
  mn = np.min(var)
  mx = np.max(var)
  mx = max(abs(mn),abs(mx))
  CS = plt.contourf(X,Y,var,levels=np.arange(-mx,mx,2*mx/100),cmap="seismic",extend="both")
  plt.axis('scaled')
  plt.xlabel("x-location (km)")
  plt.ylabel("y-location (km)")
  divider = make_axes_locatable(plt.gca())
  cax = divider.append_axes("bottom", size="5%", pad=0.5)
  cbar = plt.colorbar(CS,orientation="horizontal",cax=cax)
  cbar.ax.tick_params(labelrotation=45)
  plt.margins(x=0)
  plt.tight_layout()
  plt.savefig(f"tke_diff_5e-2_{z[k]}m.png",dpi=600)
  plt.show()
  plt.close()


prefix="../../manhattan_2m/city_2m_5e-1_"
for t in range(t1,t2) :
  nc   = Dataset(f"{prefix}{t:08d}.nc","r")
  u    = np.array(nc["avg_u"    ][:,:,:])
  v    = np.array(nc["avg_v"    ][:,:,:])
  w    = np.array(nc["avg_w"    ][:,:,:])
  upup = np.array(nc["avg_up_up"][:,:,:])
  vpvp = np.array(nc["avg_vp_vp"][:,:,:])
  wpwp = np.array(nc["avg_wp_wp"][:,:,:])
  upwp = np.array(nc["avg_up_wp"][:,:,:])
  tke  = np.array(nc["avg_tke"  ][:,:,:])
  u_avg    = u    if (t==t1) else u_avg   +u
  v_avg    = v    if (t==t1) else v_avg   +v
  w_avg    = w    if (t==t1) else w_avg   +w
  upup_avg = upup if (t==t1) else upup_avg+upup
  vpvp_avg = vpvp if (t==t1) else vpvp_avg+vpvp
  wpwp_avg = wpwp if (t==t1) else wpwp_avg+wpwp
  upwp_avg = upwp if (t==t1) else upwp_avg+upwp
  tke_avg  = tke  if (t==t1) else tke_avg +tke
u_avg /= (t2-t1)
v_avg /= (t2-t1)
w_avg /= (t2-t1)
upup_avg /= (t2-t1)
vpvp_avg /= (t2-t1)
wpwp_avg /= (t2-t1)
upwp_avg /= (t2-t1)
tke_avg  /= (t2-t1)
mag_fric  = np.sqrt(u_avg*u_avg+v_avg*v_avg+w_avg*w_avg)
tkeres_fric  = (upup_avg + vpvp_avg + wpwp_avg)/2
upwp_fric = np.abs(upwp)
tke_fric  = tke_avg
for k in [0] :
  var = tkeres_slip[k,:,:]-tkeres_fric[k,:,:]
  print(z[k],np.min(var),np.max(var))
  mn = np.min(var)
  mx = np.max(var)
  mx = max(abs(mn),abs(mx))
  CS = plt.contourf(X,Y,var,levels=np.arange(-mx,mx,2*mx/100),cmap="seismic",extend="both")
  plt.axis('scaled')
  plt.xlabel("x-location (km)")
  plt.ylabel("y-location (km)")
  divider = make_axes_locatable(plt.gca())
  cax = divider.append_axes("bottom", size="5%", pad=0.5)
  cbar = plt.colorbar(CS,orientation="horizontal",cax=cax)
  cbar.ax.tick_params(labelrotation=45)
  plt.margins(x=0)
  plt.tight_layout()
  plt.savefig(f"tkeres_diff_5e-1_{z[k]}m.png",dpi=600)
  plt.show()
  plt.close()

  var = tke_slip[k,:,:]-tke_fric[k,:,:]
  print(z[k],np.min(var),np.max(var))
  mn = np.min(var)
  mx = np.max(var)
  mx = max(abs(mn),abs(mx))
  CS = plt.contourf(X,Y,var,levels=np.arange(-mx,mx,2*mx/100),cmap="seismic",extend="both")
  plt.axis('scaled')
  plt.xlabel("x-location (km)")
  plt.ylabel("y-location (km)")
  divider = make_axes_locatable(plt.gca())
  cax = divider.append_axes("bottom", size="5%", pad=0.5)
  cbar = plt.colorbar(CS,orientation="horizontal",cax=cax)
  cbar.ax.tick_params(labelrotation=45)
  plt.margins(x=0)
  plt.tight_layout()
  plt.savefig(f"tke_diff_5e-1_{z[k]}m.png",dpi=600)
  plt.show()
  plt.close()
