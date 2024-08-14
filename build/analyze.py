import xarray
import matplotlib.pyplot as plt
import numpy as np

nc_float = xarray.open_mfdataset(["floating_00000005.nc","floating_00000006.nc","floating_00000007.nc","floating_00000008.nc",],concat_dim="num_time_steps",combine="nested")
nc_fixed = xarray.open_mfdataset(["fixed_00000005.nc"   ,"fixed_00000006.nc"   ,"fixed_00000007.nc"   ,"fixed_00000008.nc"   ,],concat_dim="num_time_steps",combine="nested")

power_float  = np.array( nc_float["power_trace_turb_0"] )
power_float += np.array( nc_float["power_trace_turb_1"] )
power_float += np.array( nc_float["power_trace_turb_2"] )
power_float += np.array( nc_float["power_trace_turb_3"] )
power_float += np.array( nc_float["power_trace_turb_4"] )
power_float += np.array( nc_float["power_trace_turb_5"] )
power_float += np.array( nc_float["power_trace_turb_6"] )
power_float += np.array( nc_float["power_trace_turb_7"] )
power_float += np.array( nc_float["power_trace_turb_8"] )

power_fixed  = np.array( nc_fixed["power_trace_turb_0"] )
power_fixed += np.array( nc_fixed["power_trace_turb_1"] )
power_fixed += np.array( nc_fixed["power_trace_turb_2"] )
power_fixed += np.array( nc_fixed["power_trace_turb_3"] )
power_fixed += np.array( nc_fixed["power_trace_turb_4"] )
power_fixed += np.array( nc_fixed["power_trace_turb_5"] )
power_fixed += np.array( nc_fixed["power_trace_turb_6"] )
power_fixed += np.array( nc_fixed["power_trace_turb_7"] )
power_fixed += np.array( nc_fixed["power_trace_turb_8"] )

betti_pert = np.concatenate( (np.array(nc_float["betti_trace_turb_0"]),
                              np.array(nc_float["betti_trace_turb_1"]),
                              np.array(nc_float["betti_trace_turb_2"]),
                              np.array(nc_float["betti_trace_turb_3"]),
                              np.array(nc_float["betti_trace_turb_4"]),
                              np.array(nc_float["betti_trace_turb_5"]),
                              np.array(nc_float["betti_trace_turb_6"]),
                              np.array(nc_float["betti_trace_turb_7"]),
                              np.array(nc_float["betti_trace_turb_8"])) )

u_samp = np.concatenate( (np.array(nc_float["u_samp_trace_turb_0"]),
                          np.array(nc_float["u_samp_trace_turb_1"]),
                          np.array(nc_float["u_samp_trace_turb_2"]),
                          np.array(nc_float["u_samp_trace_turb_3"]),
                          np.array(nc_float["u_samp_trace_turb_4"]),
                          np.array(nc_float["u_samp_trace_turb_5"]),
                          np.array(nc_float["u_samp_trace_turb_6"]),
                          np.array(nc_float["u_samp_trace_turb_7"]),
                          np.array(nc_float["u_samp_trace_turb_8"])) )

v_samp = np.concatenate( (np.array(nc_float["v_samp_trace_turb_0"]),
                          np.array(nc_float["v_samp_trace_turb_1"]),
                          np.array(nc_float["v_samp_trace_turb_2"]),
                          np.array(nc_float["v_samp_trace_turb_3"]),
                          np.array(nc_float["v_samp_trace_turb_4"]),
                          np.array(nc_float["v_samp_trace_turb_5"]),
                          np.array(nc_float["v_samp_trace_turb_6"]),
                          np.array(nc_float["v_samp_trace_turb_7"]),
                          np.array(nc_float["v_samp_trace_turb_8"])) )

mag_float = np.sqrt(u_samp*u_samp+v_samp*v_samp);

u_samp = np.concatenate( (np.array(nc_fixed["u_samp_trace_turb_0"]),
                          np.array(nc_fixed["u_samp_trace_turb_1"]),
                          np.array(nc_fixed["u_samp_trace_turb_2"]),
                          np.array(nc_fixed["u_samp_trace_turb_3"]),
                          np.array(nc_fixed["u_samp_trace_turb_4"]),
                          np.array(nc_fixed["u_samp_trace_turb_5"]),
                          np.array(nc_fixed["u_samp_trace_turb_6"]),
                          np.array(nc_fixed["u_samp_trace_turb_7"]),
                          np.array(nc_fixed["u_samp_trace_turb_8"])) )

v_samp = np.concatenate( (np.array(nc_fixed["v_samp_trace_turb_0"]),
                          np.array(nc_fixed["v_samp_trace_turb_1"]),
                          np.array(nc_fixed["v_samp_trace_turb_2"]),
                          np.array(nc_fixed["v_samp_trace_turb_3"]),
                          np.array(nc_fixed["v_samp_trace_turb_4"]),
                          np.array(nc_fixed["v_samp_trace_turb_5"]),
                          np.array(nc_fixed["v_samp_trace_turb_6"]),
                          np.array(nc_fixed["v_samp_trace_turb_7"]),
                          np.array(nc_fixed["v_samp_trace_turb_8"])) )

mag_fixed = np.sqrt(u_samp*u_samp+v_samp*v_samp);

print("Floating:")
print("  Mean:   ",np.mean(power_float))
print("  stddev: ",np.std (power_float))
print("Fixed:")
print("  Mean:   ",np.mean(power_fixed))
print("  stddev: ",np.std (power_fixed))

hist_float,bin_edges  = np.histogram(mag_float,bins=np.arange(3,17,0.05),density=True)
hist_fixed,bin_edges  = np.histogram(mag_fixed,bins=np.arange(3,17,0.05),density=True)
hist_betti,bin_edges2 = np.histogram(betti_pert ,bins=np.arange(-0.3,0.3,0.01),density=True)

f, (ax1,ax2,ax3,ax4) = plt.subplots(4, 1)
ax1.stairs(hist_fixed           ,bin_edges ,fill=True)
ax2.stairs(hist_float           ,bin_edges ,fill=True)
ax3.stairs(hist_float-hist_fixed,bin_edges ,fill=True)
ax4.stairs(hist_betti           ,bin_edges2,fill=True)
ax1.set_ylabel("Fixed"           ,wrap=True)
ax2.set_ylabel("Floating"        ,wrap=True)
ax3.set_ylabel("Floating - fixed",wrap=True)
ax4.set_ylabel("Perturbations"   ,wrap=True)
ax4.set_xlabel("Wind speed (m/s)")
plt.show()
plt.close()

