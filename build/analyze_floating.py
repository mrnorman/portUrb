from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray

def get_ind(arr,val) :
    return np.argmin(np.abs(arr-val))

t1 = 1
t2 = 1
times = [str(i).zfill(7) for i in range(t1,t2+1)]

winds = [2,5,8,11,14,17,20,23,26]

prefixes_fixed = [ "turbulent_3x3_wind-2.000000_fixed-_"  ,\
                   "turbulent_3x3_wind-5.000000_fixed-_"  ,\
                   "turbulent_3x3_wind-8.000000_fixed-_"  ,\
                   "turbulent_3x3_wind-11.000000_fixed-_" ,\
                   "turbulent_3x3_wind-14.000000_fixed-_" ,\
                   "turbulent_3x3_wind-17.000000_fixed-_" ,\
                   "turbulent_3x3_wind-20.000000_fixed-_" ,\
                   "turbulent_3x3_wind-23.000000_fixed-_" ,\
                   "turbulent_3x3_wind-26.000000_fixed-_" ]

prefixes_float = [ "turbulent_3x3_wind-2.000000_floating-_"  ,\
                   "turbulent_3x3_wind-5.000000_floating-_"  ,\
                   "turbulent_3x3_wind-8.000000_floating-_"  ,\
                   "turbulent_3x3_wind-11.000000_floating-_" ,\
                   "turbulent_3x3_wind-14.000000_floating-_" ,\
                   "turbulent_3x3_wind-17.000000_floating-_" ,\
                   "turbulent_3x3_wind-20.000000_floating-_" ,\
                   "turbulent_3x3_wind-23.000000_floating-_" ,\
                   "turbulent_3x3_wind-26.000000_floating-_" ]

nc_fixed = [xarray.open_mfdataset([prefix+"000000"+str(i).zfill(2)+".nc" for i in range(4,10)],concat_dim="num_time_steps",combine="nested") for prefix in prefixes_fixed]
nc_float = [xarray.open_mfdataset([prefix+"000000"+str(i).zfill(2)+".nc" for i in range(4,10)],concat_dim="num_time_steps",combine="nested") for prefix in prefixes_float]

x = np.array(nc_fixed[0]["x"])
y = np.array(nc_fixed[0]["y"])
z = np.array(nc_fixed[0]["z"])
dx = x[1]-x[0]
dy = y[1]-y[0]
dz = z[1]-z[0]
xlen = x[-1]+dx/2
ylen = y[-1]+dy/2
zlen = z[-1]+dz/2
ihub = get_ind(x,xlen/2)
jhub = get_ind(y,ylen/2)
khub = get_ind(z,90)

yaw_std_fixed   = np.array([0. for i in range(9)])
yaw_std_float   = np.array([0. for i in range(9)])
wind_mean_fixed = np.array([0. for i in range(9)])
wind_mean_float = np.array([0. for i in range(9)])
for i in range(9) :
  print(winds[i])
  # var_fixed = np.concatenate( (np.array(nc_fixed[i]["power_trace_turb_0"]) , 
  #                              np.array(nc_fixed[i]["power_trace_turb_1"]) , 
  #                              np.array(nc_fixed[i]["power_trace_turb_2"]) , 
  #                              np.array(nc_fixed[i]["power_trace_turb_3"]) , 
  #                              np.array(nc_fixed[i]["power_trace_turb_4"]) , 
  #                              np.array(nc_fixed[i]["power_trace_turb_5"]) , 
  #                              np.array(nc_fixed[i]["power_trace_turb_6"]) , 
  #                              np.array(nc_fixed[i]["power_trace_turb_7"]) , 
  #                              np.array(nc_fixed[i]["power_trace_turb_8"])) )
  # var_float = np.concatenate( (np.array(nc_float[i]["power_trace_turb_0"]) , 
  #                              np.array(nc_float[i]["power_trace_turb_1"]) , 
  #                              np.array(nc_float[i]["power_trace_turb_2"]) , 
  #                              np.array(nc_float[i]["power_trace_turb_3"]) , 
  #                              np.array(nc_float[i]["power_trace_turb_4"]) , 
  #                              np.array(nc_float[i]["power_trace_turb_5"]) , 
  #                              np.array(nc_float[i]["power_trace_turb_6"]) , 
  #                              np.array(nc_float[i]["power_trace_turb_7"]) , 
  #                              np.array(nc_float[i]["power_trace_turb_8"])) )
  # print("Power Fixed:")
  # print("  Mean:   ",np.mean(var_fixed))
  # print("  stddev: ",np.std (var_fixed))
  # print("Power Floating:")
  # print("  Mean:   ",np.mean(var_float))
  # print("  stddev: ",np.std (var_float))
  # mn = min(np.min(var_fixed),np.min(var_float))
  # mx = max(np.max(var_fixed),np.max(var_float))
  # # if (mx-mn > 0.01) :
  # #   hist_float,bin_edges  = np.histogram(var_float,bins=np.arange(mn,mx,(mx-mn)/1000),density=True)
  # #   hist_fixed,bin_edges  = np.histogram(var_fixed,bins=np.arange(mn,mx,(mx-mn)/1000),density=True)
  # #   f, (ax1,ax2,ax3) = plt.subplots(3, 1)
  # #   ax1.stairs(hist_fixed           ,bin_edges ,fill=True)
  # #   ax2.stairs(hist_float           ,bin_edges ,fill=True)
  # #   ax3.stairs(hist_float-hist_fixed,bin_edges ,fill=True)
  # #   ax1.set_ylabel("Fixed"           ,wrap=True)
  # #   ax2.set_ylabel("Floating"        ,wrap=True)
  # #   ax3.set_ylabel("Floating - fixed",wrap=True)
  # #   ax3.set_xlabel("Power Generation (MW)")
  # #   # plt.savefig("downstream_histogram_8D_wind="+str(winds[i])+".png",dpi=600)
  # #   plt.show()
  # #   plt.close()

  # var_fixed = np.concatenate( (np.array(nc_fixed[i]["yaw_trace_turb_0"]) , 
  #                              np.array(nc_fixed[i]["yaw_trace_turb_1"]) , 
  #                              np.array(nc_fixed[i]["yaw_trace_turb_2"]) , 
  #                              np.array(nc_fixed[i]["yaw_trace_turb_3"]) , 
  #                              np.array(nc_fixed[i]["yaw_trace_turb_4"]) , 
  #                              np.array(nc_fixed[i]["yaw_trace_turb_5"]) , 
  #                              np.array(nc_fixed[i]["yaw_trace_turb_6"]) , 
  #                              np.array(nc_fixed[i]["yaw_trace_turb_7"]) , 
  #                              np.array(nc_fixed[i]["yaw_trace_turb_8"])) )
  # var_float = np.concatenate( (np.array(nc_float[i]["yaw_trace_turb_0"]) , 
  #                              np.array(nc_float[i]["yaw_trace_turb_1"]) , 
  #                              np.array(nc_float[i]["yaw_trace_turb_2"]) , 
  #                              np.array(nc_float[i]["yaw_trace_turb_3"]) , 
  #                              np.array(nc_float[i]["yaw_trace_turb_4"]) , 
  #                              np.array(nc_float[i]["yaw_trace_turb_5"]) , 
  #                              np.array(nc_float[i]["yaw_trace_turb_6"]) , 
  #                              np.array(nc_float[i]["yaw_trace_turb_7"]) , 
  #                              np.array(nc_float[i]["yaw_trace_turb_8"])) )
  # yaw_std_fixed[i] = np.std (var_fixed)
  # yaw_std_float[i] = np.std (var_float)
  # print("yaw Fixed:")
  # print("  Mean:   ",np.mean(var_fixed))
  # print("  stddev: ",np.std (var_fixed))
  # print("yaw Floating:")
  # print("  Mean:   ",np.mean(var_float))
  # print("  stddev: ",np.std (var_float))
  # mn = min(np.min(var_fixed),np.min(var_float))
  # mx = max(np.max(var_fixed),np.max(var_float))
  # # if (mx-mn > 0.01) :
  # #   hist_float,bin_edges  = np.histogram(var_float,bins=np.arange(mn,mx,(mx-mn)/1000),density=True)
  # #   hist_fixed,bin_edges  = np.histogram(var_fixed,bins=np.arange(mn,mx,(mx-mn)/1000),density=True)
  # #   f, (ax1,ax2,ax3) = plt.subplots(3, 1)
  # #   ax1.stairs(hist_fixed           ,bin_edges ,fill=True)
  # #   ax2.stairs(hist_float           ,bin_edges ,fill=True)
  # #   ax3.stairs(hist_float-hist_fixed,bin_edges ,fill=True)
  # #   ax1.set_ylabel("Fixed"           ,wrap=True)
  # #   ax2.set_ylabel("Floating"        ,wrap=True)
  # #   ax3.set_ylabel("Floating - fixed",wrap=True)
  # #   ax3.set_xlabel("Power Generation (MW)")
  # #   # plt.savefig("downstream_histogram_8D_wind="+str(winds[i])+".png",dpi=600)
  # #   plt.show()
  # #   plt.close()

  u_fixed = np.concatenate( (np.array(nc_fixed[i]["u_samp_trace_turb_0"]) , 
                             np.array(nc_fixed[i]["u_samp_trace_turb_1"]) , 
                             np.array(nc_fixed[i]["u_samp_trace_turb_2"]) , 
                             np.array(nc_fixed[i]["u_samp_trace_turb_3"]) , 
                             np.array(nc_fixed[i]["u_samp_trace_turb_4"]) , 
                             np.array(nc_fixed[i]["u_samp_trace_turb_5"]) , 
                             np.array(nc_fixed[i]["u_samp_trace_turb_6"]) , 
                             np.array(nc_fixed[i]["u_samp_trace_turb_7"]) , 
                             np.array(nc_fixed[i]["u_samp_trace_turb_8"])) )
  v_fixed = np.concatenate( (np.array(nc_fixed[i]["v_samp_trace_turb_0"]) , 
                             np.array(nc_fixed[i]["v_samp_trace_turb_1"]) , 
                             np.array(nc_fixed[i]["v_samp_trace_turb_2"]) , 
                             np.array(nc_fixed[i]["v_samp_trace_turb_3"]) , 
                             np.array(nc_fixed[i]["v_samp_trace_turb_4"]) , 
                             np.array(nc_fixed[i]["v_samp_trace_turb_5"]) , 
                             np.array(nc_fixed[i]["v_samp_trace_turb_6"]) , 
                             np.array(nc_fixed[i]["v_samp_trace_turb_7"]) , 
                             np.array(nc_fixed[i]["v_samp_trace_turb_8"])) )
  u_float = np.concatenate( (np.array(nc_float[i]["u_samp_trace_turb_0"]) , 
                             np.array(nc_float[i]["u_samp_trace_turb_1"]) , 
                             np.array(nc_float[i]["u_samp_trace_turb_2"]) , 
                             np.array(nc_float[i]["u_samp_trace_turb_3"]) , 
                             np.array(nc_float[i]["u_samp_trace_turb_4"]) , 
                             np.array(nc_float[i]["u_samp_trace_turb_5"]) , 
                             np.array(nc_float[i]["u_samp_trace_turb_6"]) , 
                             np.array(nc_float[i]["u_samp_trace_turb_7"]) , 
                             np.array(nc_float[i]["u_samp_trace_turb_8"])) )
  v_float = np.concatenate( (np.array(nc_float[i]["v_samp_trace_turb_0"]) , 
                             np.array(nc_float[i]["v_samp_trace_turb_1"]) , 
                             np.array(nc_float[i]["v_samp_trace_turb_2"]) , 
                             np.array(nc_float[i]["v_samp_trace_turb_3"]) , 
                             np.array(nc_float[i]["v_samp_trace_turb_4"]) , 
                             np.array(nc_float[i]["v_samp_trace_turb_5"]) , 
                             np.array(nc_float[i]["v_samp_trace_turb_6"]) , 
                             np.array(nc_float[i]["v_samp_trace_turb_7"]) , 
                             np.array(nc_float[i]["v_samp_trace_turb_8"])) )
  var_fixed = np.sqrt(u_fixed*u_fixed+v_fixed*v_fixed)
  var_float = np.sqrt(u_float*u_float+v_float*v_float)
  wind_mean_fixed[i] = np.mean(var_fixed)
  wind_mean_float[i] = np.mean(var_float)
  print("Umag sample disk Fixed:")
  print("  Mean:   ",np.mean(var_fixed))
  print("  stddev: ",np.std (var_fixed))
  print("Umag sample disk Floating:")
  print("  Mean:   ",np.mean(var_float))
  print("  stddev: ",np.std (var_float))
  mn = min(np.min(var_fixed),np.min(var_float))
  mx = max(np.max(var_fixed),np.max(var_float))
  # if (mx-mn > 0.01) :
  #   hist_float,bin_edges  = np.histogram(var_float,bins=np.arange(mn,mx,(mx-mn)/1000),density=True)
  #   hist_fixed,bin_edges  = np.histogram(var_fixed,bins=np.arange(mn,mx,(mx-mn)/1000),density=True)
  #   f, (ax1,ax2,ax3) = plt.subplots(3, 1)
  #   ax1.stairs(hist_fixed           ,bin_edges ,fill=True)
  #   ax2.stairs(hist_float           ,bin_edges ,fill=True)
  #   ax3.stairs(hist_float-hist_fixed,bin_edges ,fill=True)
  #   ax1.set_ylabel("Fixed"           ,wrap=True)
  #   ax2.set_ylabel("Floating"        ,wrap=True)
  #   ax3.set_ylabel("Floating - fixed",wrap=True)
  #   ax3.set_xlabel("Power Generation (MW)")
  #   # plt.savefig("downstream_histogram_8D_wind="+str(winds[i])+".png",dpi=600)
  #   plt.show()
  #   plt.close()

# xind = get_ind(x,xlen/3+128)
# for i in range(9) :
#   ufixed = np.array(nc_fixed[i]["avg_u"][khub,:,xind])
#   vfixed = np.array(nc_fixed[i]["avg_v"][khub,:,xind])
#   ufloat = np.array(nc_float[i]["avg_u"][khub,:,xind])
#   vfloat = np.array(nc_float[i]["avg_v"][khub,:,xind])
#   magfixed = np.sqrt(ufixed*ufixed+vfixed*vfixed)
#   magfloat = np.sqrt(ufloat*ufloat+vfloat*vfloat)
#   plt.plot(magfixed,label="fixed")
#   plt.plot(magfloat,label="float")
#   plt.title(str(winds[i])+" , 1 diam")
#   plt.legend()
#   plt.show()
#   plt.close()
# 
# xind = get_ind(x,xlen/3+128*2)
# for i in range(9) :
#   ufixed = np.array(nc_fixed[i]["avg_u"][khub,:,xind])
#   vfixed = np.array(nc_fixed[i]["avg_v"][khub,:,xind])
#   ufloat = np.array(nc_float[i]["avg_u"][khub,:,xind])
#   vfloat = np.array(nc_float[i]["avg_v"][khub,:,xind])
#   magfixed = np.sqrt(ufixed*ufixed+vfixed*vfixed)
#   magfloat = np.sqrt(ufloat*ufloat+vfloat*vfloat)
#   plt.plot(magfixed,label="fixed")
#   plt.plot(magfloat,label="float")
#   plt.title(str(winds[i])+" , 2 diam")
#   plt.legend()
#   plt.show()
#   plt.close()
# 
# xind = get_ind(x,xlen/3+128*4)
# for i in range(9) :
#   ufixed = np.array(nc_fixed[i]["avg_u"][khub,:,xind])
#   vfixed = np.array(nc_fixed[i]["avg_v"][khub,:,xind])
#   ufloat = np.array(nc_float[i]["avg_u"][khub,:,xind])
#   vfloat = np.array(nc_float[i]["avg_v"][khub,:,xind])
#   magfixed = np.sqrt(ufixed*ufixed+vfixed*vfixed)
#   magfloat = np.sqrt(ufloat*ufloat+vfloat*vfloat)
#   plt.plot(magfixed,label="fixed")
#   plt.plot(magfloat,label="float")
#   plt.title(str(winds[i])+" , 4 diam")
#   plt.legend()
#   plt.show()
#   plt.close()

# xind = get_ind(x,xlen/3+128*8)
# for i in range(9) :
#   ufixed = np.array(nc_fixed[i]["avg_u"][khub,:,xind])
#   vfixed = np.array(nc_fixed[i]["avg_v"][khub,:,xind])
#   ufloat = np.array(nc_float[i]["avg_u"][khub,:,xind])
#   vfloat = np.array(nc_float[i]["avg_v"][khub,:,xind])
#   magfixed = np.sqrt(ufixed*ufixed+vfixed*vfixed)
#   magfloat = np.sqrt(ufloat*ufloat+vfloat*vfloat)
#   plt.plot(magfixed,label="fixed")
#   plt.plot(magfloat,label="float")
#   plt.title(str(winds[i])+" , 8 diam")
#   plt.legend()
#   plt.show()
#   plt.close()

plt.plot(winds,yaw_std_float-yaw_std_fixed,label="yaw stdev (float-fixed)")
plt.plot(winds,wind_mean_float-wind_mean_fixed,label="wind_mean (float-fixed)")
plt.legend()
plt.xlabel("Average Hub Height Wind Speed (m/s)")
plt.show()
plt.close()
