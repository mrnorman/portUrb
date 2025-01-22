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
  colors = ["black","red","green","cyan"]
  l=0
  fig = plt.figure(figsize=(6,3))
  ax  = fig.gca()
  print("Relative difference in mean velocity")
  for d in [0,4,10] :
    i = get_ind(x,d)
    vbld = np.array(bld["avg_u"][k,:,i])
    vdsk = np.array(dsk["avg_u"][k,:,i])
    print(f"{np.abs(np.mean(vbld)-np.mean(vdsk))/np.mean(vbld):.3e}")
    ax.plot(y,vbld,label=f"blade,D={d}",color=colors[l],linestyle="-" )
    ax.plot(y,vdsk,label=f"disk,D={d}" ,color=colors[l],linestyle="--")
    l += 1
  ax.grid(True)
  ax.legend()
  ax.set_xlim(-1.2,1)
  ax.set_ylim(7,13.5)
  ax.set_xlabel("y-location (turbine diameters)")
  ax.set_ylabel("u-velocity (m/s)")
  ax.margins(x=0)
  fig.tight_layout()
  plt.tight_layout()
  plt.savefig("5mw_blade_disk_wake_comparison.png",dpi=600)
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
  colors = ["black","red","green","cyan"]
  l=0
  fig = plt.figure(figsize=(6,3))
  ax  = fig.gca()
  print("Relative difference in mean velocity")
  for d in [0,4,10] :
    i = get_ind(x,d)
    vbld = np.array(bld["avg_u"][k,:,i])
    vdsk = np.array(dsk["avg_u"][k,:,i])
    print(f"{np.abs(np.mean(vbld)-np.mean(vdsk))/np.mean(vbld):.3e}")
    ax.plot(y,vbld,label=f"blade,D={d}",color=colors[l],linestyle="-" )
    ax.plot(y,vdsk,label=f"disk,D={d}" ,color=colors[l],linestyle="--")
    l += 1
  ax.grid(True)
  ax.legend()
  ax.set_xlim(-1.2,1)
  ax.set_ylim(8.25,13.5)
  ax.set_xlabel("y-location (turbine diameters)")
  ax.set_ylabel("u-velocity (m/s)")
  ax.margins(x=0)
  fig.tight_layout()
  plt.tight_layout()
  plt.savefig("22mw_blade_disk_wake_comparison.png",dpi=600)
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
  colors = ["black","red","green","cyan"]
  l=0
  fig = plt.figure(figsize=(6,3))
  ax  = fig.gca()
  print("Relative difference in mean velocity")
  for d in [0,4,10] :
    i = get_ind(x,d)
    vbld = np.array(bld["avg_u"][k,:,i])
    vdsk = np.array(dsk["avg_u"][k,:,i])
    print(f"{np.abs(np.mean(vbld)-np.mean(vdsk))/np.mean(vbld):.3e}")
    ax.plot(y,vbld,label=f"blade,D={d}",color=colors[l],linestyle="-" )
    ax.plot(y,vdsk,label=f"disk,D={d}" ,color=colors[l],linestyle="--")
    l += 1
  ax.grid(True)
  ax.legend()
  ax.set_xlim(-1.2,1)
  ax.set_ylim(2,5)
  ax.set_xlabel("y-location (turbine diameters)")
  ax.set_ylabel("u-velocity (m/s)")
  ax.margins(x=0)
  fig.tight_layout()
  plt.tight_layout()
  plt.savefig("22mw_blade_disk_wake_comparison_4mps.png",dpi=600)
  plt.show()
  plt.close()
  ##########################################
  ## 22 MW blades versus disk, 24m/s, dx=2m
  ##########################################
  print("*** 22 MW Disk and blades 24 m/s comparison ***")
  D = 141.60905*2
  H = 167.193
  bld = Dataset("run_12_00000010.nc","r")
  dsk = Dataset("run_13_00000010.nc","r")
  x = np.array(bld["x"][:]/D-2.5)
  y = np.array(bld["y"][:]/D-1)
  z = np.array(bld["z"][:])
  k = get_ind(z,H)
  colors = ["black","red","green","cyan"]
  l=0
  fig = plt.figure(figsize=(6,3))
  ax  = fig.gca()
  print("Relative difference in mean velocity")
  for d in [0,4,10] :
    i = get_ind(x,d)
    vbld = np.array(bld["avg_u"][k,:,i])
    vdsk = np.array(dsk["avg_u"][k,:,i])
    print(f"{np.abs(np.mean(vbld)-np.mean(vdsk))/np.mean(vbld):.3e}")
    ax.plot(y,vbld,label=f"blade,D={d}",color=colors[l],linestyle="-" )
    ax.plot(y,vdsk,label=f"disk,D={d}" ,color=colors[l],linestyle="--")
    l += 1
  ax.grid(True)
  ax.legend()
  ax.set_xlim(-1.2,1)
  ax.set_ylim(23.5,24.3)
  ax.set_xlabel("y-location (turbine diameters)")
  ax.set_ylabel("u-velocity (m/s)")
  ax.margins(x=0)
  fig.tight_layout()
  plt.tight_layout()
  plt.savefig("22mw_blade_disk_wake_comparison_24mps.png",dpi=600)
  plt.show()
  plt.close()


def plot_group_5() :
  ##########################################
  ## 22 MW blades, immersed versus not
  ##########################################
  print("*** 22 MW blades immersed vs not immersed comparison ***")
  D = 141.60905*2
  H = 167.193
  bld = Dataset("run_14_00000010.nc","r")
  dsk = Dataset("run_03_00000010.nc","r")
  x = np.array(bld["x"][:]/D-2.5)
  y = np.array(bld["y"][:]/D-1)
  z = np.array(bld["z"][:])
  k = get_ind(z,H)
  # Plot contour of simulation without immersed material
  X,Y = np.meshgrid(x,y)
  var = np.array(bld["avg_u"][k,:,:])
  mn = np.min(var)
  mx = np.max(var)
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
  plt.savefig("22mw_blades_no_immersed_contour_tavg.png",dpi=600)
  plt.show()
  plt.close()
  # Plot contour of simulation with immersed material
  X,Y = np.meshgrid(x,y)
  var = np.array(dsk["avg_u"][k,:,:])
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
  plt.savefig("22mw_blades_yes_immersed_contour_tavg.png",dpi=600)
  plt.show()
  plt.close()
  # Plot contour of simulation without immersed material
  X,Y = np.meshgrid(x,y)
  var = np.array(bld["uvel"][k,:,:])
  mn = np.min(var)
  mx = np.max(var)
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
  plt.savefig("22mw_blades_no_immersed_contour_inst.png",dpi=600)
  plt.show()
  plt.close()
  # Plot contour of simulation with immersed material
  X,Y = np.meshgrid(x,y)
  var = np.array(dsk["uvel"][k,:,:])
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
  plt.savefig("22mw_blades_yes_immersed_contour_inst.png",dpi=600)
  plt.show()
  plt.close()


def plot_group_6() :
  ##########################################
  ## 22 MW disk, immersed versus not
  ##########################################
  print("*** 22 MW disk immersed vs not immersed comparison ***")
  D = 141.60905*2
  H = 167.193
  bld = Dataset("run_16_00000010.nc","r")
  dsk = Dataset("run_04_00000010.nc","r")
  x = np.array(bld["x"][:]/D-2.5)
  y = np.array(bld["y"][:]/D-1)
  z = np.array(bld["z"][:])
  k = get_ind(z,H)
  # Plot contour of simulation without immersed material
  X,Y = np.meshgrid(x,y)
  var = np.array(bld["avg_u"][k,:,:])
  mn = np.min(var)
  mx = np.max(var)
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
  plt.savefig("22mw_disk_no_immersed_contour_tavg.png",dpi=600)
  plt.show()
  plt.close()
  # Plot contour of simulation with immersed material
  X,Y = np.meshgrid(x,y)
  var = np.array(dsk["avg_u"][k,:,:])
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
  plt.savefig("22mw_disk_yes_immersed_contour_tavg.png",dpi=600)
  plt.show()
  plt.close()
  # Plot contour of simulation without immersed material
  X,Y = np.meshgrid(x,y)
  var = np.array(bld["uvel"][k,:,:])
  mn = np.min(var)
  mx = np.max(var)
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
  plt.savefig("22mw_disk_no_immersed_contour_inst.png",dpi=600)
  plt.show()
  plt.close()
  # Plot contour of simulation with immersed material
  X,Y = np.meshgrid(x,y)
  var = np.array(dsk["uvel"][k,:,:])
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
  plt.savefig("22mw_disk_yes_immersed_contour_inst.png",dpi=600)
  plt.show()
  plt.close()


def plot_group_7() :
  ##########################################
  ## 22 MW disk, immersed versus not
  ##########################################
  print("*** 22 MW yaw angle comparison ***")
  D = 141.60905*2
  H = 167.193
  yaw00 = Dataset("run_03_00000010.nc","r")
  yaw15 = Dataset("run_19_00000010.nc","r")
  yaw30 = Dataset("run_21_00000010.nc","r")
  x = np.array(bld["x"][:]/D-2.5)
  y = np.array(bld["y"][:]/D-1)
  z = np.array(bld["z"][:])
  k = get_ind(z,H)
  # Plot contour of simulation without immersed material
  X,Y = np.meshgrid(x,y)
  var = np.array(bld["avg_u"][k,:,:])
  mn = np.min(var)
  mx = np.max(var)
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
  plt.savefig("22mw_disk_no_immersed_contour_tavg.png",dpi=600)
  plt.show()
  plt.close()
  # Plot contour of simulation with immersed material
  X,Y = np.meshgrid(x,y)
  var = np.array(dsk["avg_u"][k,:,:])
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
  plt.savefig("22mw_disk_yes_immersed_contour_tavg.png",dpi=600)
  plt.show()
  plt.close()
  # Plot contour of simulation without immersed material
  X,Y = np.meshgrid(x,y)
  var = np.array(bld["uvel"][k,:,:])
  mn = np.min(var)
  mx = np.max(var)
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
  plt.savefig("22mw_disk_no_immersed_contour_inst.png",dpi=600)
  plt.show()
  plt.close()
  # Plot contour of simulation with immersed material
  X,Y = np.meshgrid(x,y)
  var = np.array(dsk["uvel"][k,:,:])
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
  plt.savefig("22mw_disk_yes_immersed_contour_inst.png",dpi=600)
  plt.show()
  plt.close()


# plot_group_1()
# plot_group_2()
# plot_group_3()
# plot_group_4()
plot_group_5()
plot_group_6()

