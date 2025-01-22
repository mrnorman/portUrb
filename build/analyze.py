from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cmap import Colormap

def get_ind(a,v) :
    return np.argmin(np.abs(a-v))


def plot_group_1() :
  #######################################
  ## 5 MW blades versus disk, dx=2m
  #######################################
  print("*** 5 MW Disk and blades wake comparison ***")
  D = 63*2
  H = 90
  bld = Dataset("run_01_00000010.nc","r")
  dsk = Dataset("run_02_00000010.nc","r")
  x = np.array(bld["x"][:]/D-2.5)
  y = np.array(bld["y"][:]/D-1)
  z = np.array(bld["z"][:])
  k = get_ind(z,H)
  X,Y = np.meshgrid(x,y)
  var = np.array(dsk["avg_u"][k,:,:])-np.array(bld["avg_u"][k,:,:])
  mn = np.min(var)
  mx = np.max(var)
  mn = -max(abs(mn),abs(mx))
  mx =  max(abs(mn),abs(mx))
  fig = plt.figure(figsize=(6,6))
  ax  = fig.gca()
  CS = ax.contourf(X,Y,var,levels=np.arange(mn,mx,(mx-mn)/100),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
  ax.axis('scaled')
  ax.set_xlabel("x (turb diams)")
  ax.set_ylabel("y (turb diams)")
  ax.margins(x=0)
  divider = make_axes_locatable(plt.gca())
  cax = divider.append_axes("bottom", size="10%", pad=0.5)
  cbar = plt.colorbar(CS,orientation="horizontal",cax=cax)
  cbar.ax.tick_params(rotation=35)
  ax.margins(x=0)
  plt.margins(x=0)
  plt.tight_layout()
  plt.savefig("5mw_disk_vs_blades_contour_tavg_11.4mps.png",dpi=600)
  plt.show()
  plt.close()


def plot_group_2() :
  #######################################
  ## 22 MW blades versus disk, dx=2m
  #######################################
  print("*** 22 MW Disk and blades wake comparison ***")
  D = 141.60905*2
  H = 167.193
  bld = Dataset("run_03_00000010.nc","r")
  dsk = Dataset("run_04_00000010.nc","r")
  x = np.array(bld["x"][:]/D-2.5)
  y = np.array(bld["y"][:]/D-1)
  z = np.array(bld["z"][:])
  k = get_ind(z,H)
  X,Y = np.meshgrid(x,y)
  var = np.array(dsk["avg_u"][k,:,:])-np.array(bld["avg_u"][k,:,:])
  mn = np.min(var)
  mx = np.max(var)
  mn = -max(abs(mn),abs(mx))
  mx =  max(abs(mn),abs(mx))
  fig = plt.figure(figsize=(6,6))
  ax  = fig.gca()
  CS = ax.contourf(X,Y,var,levels=np.arange(mn,mx,(mx-mn)/100),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
  ax.axis('scaled')
  ax.set_xlabel("x (turb diams)")
  ax.set_ylabel("y (turb diams)")
  ax.margins(x=0)
  divider = make_axes_locatable(plt.gca())
  cax = divider.append_axes("bottom", size="10%", pad=0.5)
  cbar = plt.colorbar(CS,orientation="horizontal",cax=cax)
  cbar.ax.tick_params(rotation=35)
  ax.margins(x=0)
  plt.margins(x=0)
  plt.tight_layout()
  plt.savefig("22mw_disk_vs_blades_contour_tavg_11.4mps.png",dpi=600)
  plt.show()
  plt.close()


def plot_group_3() :
  ###########################################
  ## 22 MW blended grid spacing comparison
  ###########################################
  print("*** 22 MW blended grid spacing comparison ***")
  files =  ["run_03_00000010.nc",
            "run_06_00000010.nc",
            "run_07_00000010.nc",
            "run_08_00000010.nc",
            "run_09_00000010.nc"]
  D = 141.60905*2
  H = 167.193
  for d in [0,4,10] :
    fig = plt.figure(figsize=(6,3))
    ax  = fig.gca()
    print(f"D={d}")
    for f in range(len(files)) :
      nc = Dataset(files[f],"r")
      x = np.array(nc["x"][:]/D-2.5)
      y = np.array(nc["y"][:]/D-1)
      z = np.array(nc["z"][:])
      dx = int(nc["x"][1]-nc["x"][0])
      k = get_ind(z,H)
      i = get_ind(x,d)
      u = np.array(nc["avg_u"][k,:,i])
      u2 = u if (f==0) else u2
      print(f"dx={dx}, reldiff={np.abs(np.mean(u)-np.mean(u2))/np.mean(u2):.3e}")
      ax.plot(y,u,label=f"dx={dx}")
    ax.grid(True)
    ax.set_ylim(8.25,13.5)
    ax.set_xlabel("y-location (turbine diameters)")
    ax.set_ylabel("u-velocity (m/s)")
    ax.margins(x=0)
    fig.tight_layout()
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"22mw_blended_comparison_dx-{dx}_D-{d}.png",dpi=600)
    plt.show()
    plt.close()


def plot_group_4() :
  ##########################################
  ## 22 MW blades versus disk, 4m/s, dx=2m
  ##########################################
  print("*** 22 MW Disk and blades 4 m/s comparison ***")
  D = 141.60905*2
  H = 167.193
  bld = Dataset("run_10_00000010.nc","r")
  dsk = Dataset("run_11_00000010.nc","r")
  x = np.array(bld["x"][:]/D-2.5)
  y = np.array(bld["y"][:]/D-1)
  z = np.array(bld["z"][:])
  k = get_ind(z,H)
  X,Y = np.meshgrid(x,y)
  var = np.array(dsk["avg_u"][k,:,:])-np.array(bld["avg_u"][k,:,:])
  mn = np.min(var)
  mx = np.max(var)
  mn = -max(abs(mn),abs(mx))
  mx =  max(abs(mn),abs(mx))
  fig = plt.figure(figsize=(6,6))
  ax  = fig.gca()
  CS = ax.contourf(X,Y,var,levels=np.arange(mn,mx,(mx-mn)/100),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
  ax.axis('scaled')
  ax.set_xlabel("x (turb diams)")
  ax.set_ylabel("y (turb diams)")
  ax.margins(x=0)
  divider = make_axes_locatable(plt.gca())
  cax = divider.append_axes("bottom", size="10%", pad=0.5)
  cbar = plt.colorbar(CS,orientation="horizontal",cax=cax)
  cbar.ax.tick_params(rotation=35)
  ax.margins(x=0)
  plt.margins(x=0)
  plt.tight_layout()
  plt.savefig("22mw_disk_vs_blades_contour_tavg_4mps.png",dpi=600)
  plt.show()
  plt.close()
  bld = Dataset("run_12_00000010.nc","r")
  dsk = Dataset("run_13_00000010.nc","r")
  x = np.array(bld["x"][:]/D-2.5)
  y = np.array(bld["y"][:]/D-1)
  z = np.array(bld["z"][:])
  k = get_ind(z,H)
  X,Y = np.meshgrid(x,y)
  var = np.array(dsk["avg_u"][k,:,:])-np.array(bld["avg_u"][k,:,:])
  mn = np.min(var)
  mx = np.max(var)
  mn = -max(abs(mn),abs(mx))
  mx =  max(abs(mn),abs(mx))
  fig = plt.figure(figsize=(6,6))
  ax  = fig.gca()
  CS = ax.contourf(X,Y,var,levels=np.arange(mn,mx,(mx-mn)/100),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
  ax.axis('scaled')
  ax.set_xlabel("x (turb diams)")
  ax.set_ylabel("y (turb diams)")
  ax.margins(x=0)
  divider = make_axes_locatable(plt.gca())
  cax = divider.append_axes("bottom", size="10%", pad=0.5)
  cbar = plt.colorbar(CS,orientation="horizontal",cax=cax)
  cbar.ax.tick_params(rotation=35)
  ax.margins(x=0)
  plt.margins(x=0)
  plt.tight_layout()
  plt.savefig("22mw_disk_vs_blades_contour_tavg_4mps.png",dpi=600)
  plt.show()
  plt.close()


def plot_group_5() :
  ##########################################
  ## 22 MW blades, immersed versus not
  ##########################################
  print("*** 22 MW blades immersed vs not immersed comparison ***")
  D = 141.60905*2
  H = 167.193
  immN = Dataset("run_14_00000010.nc","r")
  immY = Dataset("run_03_00000010.nc","r")
  x = np.array(immN["x"][:]/D-2.5)
  y = np.array(immN["y"][:]/D-1)
  z = np.array(immN["z"][:])
  k = get_ind(z,H)
  X,Y = np.meshgrid(x,y)
  var = np.array(immN["avg_u"][k,:,:])-np.array(immY["avg_u"][k,:,:])
  mn = -2.5
  mx =  2.5
  fig = plt.figure(figsize=(6,6))
  ax  = fig.gca()
  CS = ax.contourf(X,Y,var,levels=np.arange(mn,mx,(mx-mn)/100),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
  ax.axis('scaled')
  ax.set_xlabel("x (turb diams)")
  ax.set_ylabel("y (turb diams)")
  ax.margins(x=0)
  divider = make_axes_locatable(plt.gca())
  cax = divider.append_axes("bottom", size="10%", pad=0.5)
  cbar = plt.colorbar(CS,orientation="horizontal",cax=cax)
  cbar.ax.tick_params(rotation=35)
  ax.margins(x=0)
  plt.margins(x=0)
  plt.tight_layout()
  plt.savefig("22mw_imm_versus_no_blades_contour_tavg.png",dpi=600)
  plt.show()
  plt.close()


def plot_group_6() :
  ##########################################
  ## 22 MW disk, immersed versus not
  ##########################################
  print("*** 22 MW disk immersed vs not immersed comparison ***")
  D = 141.60905*2
  H = 167.193
  immN = Dataset("run_16_00000010.nc","r")
  immY = Dataset("run_04_00000010.nc","r")
  x = np.array(immN["x"][:]/D-2.5)
  y = np.array(immN["y"][:]/D-1)
  z = np.array(immN["z"][:])
  k = get_ind(z,H)
  X,Y = np.meshgrid(x,y)
  var = np.array(immN["avg_u"][k,:,:])-np.array(immY["avg_u"][k,:,:])
  mn = -2.0
  mx =  2.0
  fig = plt.figure(figsize=(6,6))
  ax  = fig.gca()
  CS = ax.contourf(X,Y,var,levels=np.arange(mn,mx,(mx-mn)/100),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
  ax.axis('scaled')
  ax.set_xlabel("x (turb diams)")
  ax.set_ylabel("y (turb diams)")
  ax.margins(x=0)
  divider = make_axes_locatable(plt.gca())
  cax = divider.append_axes("bottom", size="10%", pad=0.5)
  cbar = plt.colorbar(CS,orientation="horizontal",cax=cax)
  cbar.ax.tick_params(rotation=35)
  ax.margins(x=0)
  plt.margins(x=0)
  plt.tight_layout()
  plt.savefig("22mw_imm_versus_no_disk_contour_tavg.png",dpi=600)
  plt.show()
  plt.close()


def plot_group_7() :
  ##########################################
  ## 22 MW disk, Yaw angle
  ##########################################
  print("*** 22 MW yaw angle comparison ***")
  D = 141.60905*2
  H = 167.193
  yaw00 = Dataset("run_18_00000010.nc","r")
  yaw15 = Dataset("run_19_00000010.nc","r")
  yaw30 = Dataset("run_20_00000010.nc","r")
  x = np.array(yaw00["x"][:]/D-2.5)
  y = np.array(yaw00["y"][:]/D-2)
  z = np.array(yaw00["z"][:])
  k = get_ind(z,H)
  X,Y = np.meshgrid(x,y)

  u = np.array(yaw00["avg_u"][k,:,:])
  v = np.array(yaw00["avg_v"][k,:,:])
  w = np.array(yaw00["avg_w"][k,:,:])
  var = np.sqrt(u*u+v*v+w*w)
  mn = 7
  mx = 14
  fig = plt.figure(figsize=(6,6))
  ax  = fig.gca()
  CS = ax.contourf(X,Y,var,levels=np.arange(mn,mx,(mx-mn)/100),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
  dudy = np.abs(var-np.roll(var,(-1,0),axis=(0,1)))
  x1 = 1
  x2 = 2
  y1 = y[get_ind(y,0)+np.argmax(dudy[get_ind(y,0):,get_ind(x,x1)])]
  y2 = y[get_ind(y,0)+np.argmax(dudy[get_ind(y,0):,get_ind(x,x2)])]
  ty1 = y1
  ty2 = y2
  print(f"Top: yaw00_slope: {(y2-y1)/(x2-x1)};    yaw00_theta: {np.atan2(y2-y1,x2-x1)/np.pi*180}")
  ax.plot([x1,x2],[y1,y2],color="black")
  x1 = 1
  x2 = 2
  y1 = y[np.argmax(dudy[:get_ind(y,0),get_ind(x,x1)])]
  y2 = y[np.argmax(dudy[:get_ind(y,0),get_ind(x,x2)])]
  by1 = y1
  by2 = y2
  print(f"Bot: yaw00_slope: {(y2-y1)/(x2-x1)};    yaw00_theta: {np.atan2(y2-y1,x2-x1)/np.pi*180}")
  ax.plot([x1,x2],[y1,y2],color="black",linestyle=":",linewidth=2)
  y1 = (by1+ty1)/2
  y2 = (by2+ty2)/2
  print(f"Cen: yaw00_slope: {(y2-y1)/(x2-x1)};    yaw00_theta: {np.atan2(y2-y1,x2-x1)/np.pi*180}")
  ax.plot([x1,x2],[y1,y2],color="black",linestyle="--")
  ax.axis('scaled')
  ax.set_xlabel("x (turb diams)")
  ax.set_ylabel("y (turb diams)")
  ax.margins(x=0)
  divider = make_axes_locatable(plt.gca())
  cax = divider.append_axes("bottom", size="10%", pad=0.5)
  cbar = plt.colorbar(CS,orientation="horizontal",cax=cax)
  cbar.ax.tick_params(rotation=35)
  ax.margins(x=0)
  plt.margins(x=0)
  plt.tight_layout()
  plt.savefig("22mw_yaw00_tavg.png",dpi=600)
  plt.show()
  plt.close()

  u = np.array(yaw15["avg_u"][k,:,:])
  v = np.array(yaw15["avg_v"][k,:,:])
  w = np.array(yaw15["avg_w"][k,:,:])
  var = np.sqrt(u*u+v*v+w*w)
  mn = 7
  mx = 14
  fig = plt.figure(figsize=(6,6))
  ax  = fig.gca()
  CS = ax.contourf(X,Y,var,levels=np.arange(mn,mx,(mx-mn)/100),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
  dudy = np.abs(var-np.roll(var,(-1,0),axis=(0,1)))
  x1 = 1
  x2 = 2
  y1 = y[get_ind(y,0)+np.argmax(dudy[get_ind(y,0):,get_ind(x,x1)])]
  y2 = y[get_ind(y,0)+np.argmax(dudy[get_ind(y,0):,get_ind(x,x2)])]
  ty1 = y1
  ty2 = y2
  print(f"Top: yaw15_slope: {(y2-y1)/(x2-x1)};    yaw15_theta: {np.atan2(y2-y1,x2-x1)/np.pi*180}")
  ax.plot([x1,x2],[y1,y2],color="black")
  x1 = 1
  x2 = 2
  y1 = y[np.argmax(dudy[:get_ind(y,0),get_ind(x,x1)])]
  y2 = y[np.argmax(dudy[:get_ind(y,0),get_ind(x,x2)])]
  by1 = y1
  by2 = y2
  print(f"Bot: yaw15_slope: {(y2-y1)/(x2-x1)};    yaw15_theta: {np.atan2(y2-y1,x2-x1)/np.pi*180}")
  ax.plot([x1,x2],[y1,y2],color="black",linestyle=":",linewidth=2)
  y1 = (by1+ty1)/2
  y2 = (by2+ty2)/2
  print(f"Cen: yaw15_slope: {(y2-y1)/(x2-x1)};    yaw15_theta: {np.atan2(y2-y1,x2-x1)/np.pi*180}")
  ax.plot([x1,x2],[y1,y2],color="black",linestyle="--")
  ax.axis('scaled')
  ax.set_xlabel("x (turb diams)")
  ax.set_ylabel("y (turb diams)")
  ax.margins(x=0)
  divider = make_axes_locatable(plt.gca())
  cax = divider.append_axes("bottom", size="10%", pad=0.5)
  cbar = plt.colorbar(CS,orientation="horizontal",cax=cax)
  cbar.ax.tick_params(rotation=35)
  ax.margins(x=0)
  plt.margins(x=0)
  plt.tight_layout()
  plt.savefig("22mw_yaw15_tavg.png",dpi=600)
  plt.show()
  plt.close()

  u = np.array(yaw30["avg_u"][k,:,:])
  v = np.array(yaw30["avg_v"][k,:,:])
  w = np.array(yaw30["avg_w"][k,:,:])
  var = np.sqrt(u*u+v*v+w*w)
  mn = 7
  mx = 14
  fig = plt.figure(figsize=(6,6))
  ax  = fig.gca()
  CS = ax.contourf(X,Y,var,levels=np.arange(mn,mx,(mx-mn)/100),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
  dudy = np.abs(var-np.roll(var,(-1,0),axis=(0,1)))
  x1 = 1
  x2 = 2
  y1 = y[get_ind(y,0)+np.argmax(dudy[get_ind(y,0):,get_ind(x,x1)])]
  y2 = y[get_ind(y,0)+np.argmax(dudy[get_ind(y,0):,get_ind(x,x2)])]
  ty1 = y1
  ty2 = y2
  print(f"Top: yaw30_slope: {(y2-y1)/(x2-x1)};    yaw30_theta: {np.atan2(y2-y1,x2-x1)/np.pi*180}")
  ax.plot([x1,x2],[y1,y2],color="black")
  x1 = 1
  x2 = 2
  y1 = y[np.argmax(dudy[:get_ind(y,0),get_ind(x,x1)])]
  y2 = y[np.argmax(dudy[:get_ind(y,0),get_ind(x,x2)])]
  by1 = y1
  by2 = y2
  print(f"Bot: yaw30_slope: {(y2-y1)/(x2-x1)};    yaw30_theta: {np.atan2(y2-y1,x2-x1)/np.pi*180}")
  ax.plot([x1,x2],[y1,y2],color="black",linestyle=":",linewidth=2)
  y1 = (by1+ty1)/2
  y2 = (by2+ty2)/2
  print(f"Cen: yaw30_slope: {(y2-y1)/(x2-x1)};    yaw30_theta: {np.atan2(y2-y1,x2-x1)/np.pi*180}")
  ax.plot([x1,x2],[y1,y2],color="black",linestyle="--")
  ax.axis('scaled')
  ax.set_xlabel("x (turb diams)")
  ax.set_ylabel("y (turb diams)")
  ax.margins(x=0)
  divider = make_axes_locatable(plt.gca())
  cax = divider.append_axes("bottom", size="10%", pad=0.5)
  cbar = plt.colorbar(CS,orientation="horizontal",cax=cax)
  cbar.ax.tick_params(rotation=35)
  ax.margins(x=0)
  plt.margins(x=0)
  plt.tight_layout()
  plt.savefig("22mw_yaw30_tavg.png",dpi=600)
  plt.show()
  plt.close()

  var00 = (np.array(yaw00["avg_up_up"][k,:,:]) + np.array(yaw00["avg_vp_vp"][k,:,:]) + np.array(yaw00["avg_wp_wp"][k,:,:]))/2
  var15 = (np.array(yaw15["avg_up_up"][k,:,:]) + np.array(yaw15["avg_vp_vp"][k,:,:]) + np.array(yaw15["avg_wp_wp"][k,:,:]))/2
  var30 = (np.array(yaw30["avg_up_up"][k,:,:]) + np.array(yaw30["avg_vp_vp"][k,:,:]) + np.array(yaw30["avg_wp_wp"][k,:,:]))/2
  mn = min(np.min(var00),np.min(var15),np.min(var30))
  mx = min(np.max(var00),np.max(var15),np.max(var30))
  fig = plt.figure(figsize=(6,6))
  ax  = fig.gca()
  CS = ax.contourf(X,Y,var00,levels=np.arange(mn,mx,(mx-mn)/100),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
  ax.axis('scaled')
  ax.set_xlabel("x (turb diams)")
  ax.set_ylabel("y (turb diams)")
  ax.margins(x=0)
  divider = make_axes_locatable(plt.gca())
  cax = divider.append_axes("bottom", size="10%", pad=0.5)
  cbar = plt.colorbar(CS,orientation="horizontal",cax=cax)
  cbar.ax.tick_params(rotation=35)
  ax.margins(x=0)
  plt.margins(x=0)
  plt.tight_layout()
  plt.savefig("22mw_yaw00_restke_tavg.png",dpi=600)
  plt.show()
  plt.close()
  fig = plt.figure(figsize=(6,6))
  ax  = fig.gca()
  CS = ax.contourf(X,Y,var15,levels=np.arange(mn,mx,(mx-mn)/100),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
  ax.axis('scaled')
  ax.set_xlabel("x (turb diams)")
  ax.set_ylabel("y (turb diams)")
  ax.margins(x=0)
  divider = make_axes_locatable(plt.gca())
  cax = divider.append_axes("bottom", size="10%", pad=0.5)
  cbar = plt.colorbar(CS,orientation="horizontal",cax=cax)
  cbar.ax.tick_params(rotation=35)
  ax.margins(x=0)
  plt.margins(x=0)
  plt.tight_layout()
  plt.savefig("22mw_yaw15_restke_tavg.png",dpi=600)
  plt.show()
  plt.close()
  fig = plt.figure(figsize=(6,6))
  ax  = fig.gca()
  CS = ax.contourf(X,Y,var30,levels=np.arange(mn,mx,(mx-mn)/100),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
  ax.axis('scaled')
  ax.set_xlabel("x (turb diams)")
  ax.set_ylabel("y (turb diams)")
  ax.margins(x=0)
  divider = make_axes_locatable(plt.gca())
  cax = divider.append_axes("bottom", size="10%", pad=0.5)
  cbar = plt.colorbar(CS,orientation="horizontal",cax=cax)
  cbar.ax.tick_params(rotation=35)
  ax.margins(x=0)
  plt.margins(x=0)
  plt.tight_layout()
  plt.savefig("22mw_yaw30_restke_tavg.png",dpi=600)
  plt.show()
  plt.close()


plot_group_1()
plot_group_2()
plot_group_3()
plot_group_4()
plot_group_5()
plot_group_6()
plot_group_7()

