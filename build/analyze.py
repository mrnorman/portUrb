import xarray
import matplotlib.pyplot as plt
import numpy as np

nc_float = xarray.open_mfdataset("floating/test*.nc",concat_dim="num_time_steps",combine="nested")
nc_fixed = xarray.open_mfdataset("fixed/test*.nc",concat_dim="num_time_steps",combine="nested")

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

mag_float = np.concatenate( (np.array(nc_float["normmag0_trace_turb_0"])+np.array(nc_float["betti_trace_turb_0"]),
                             np.array(nc_float["normmag0_trace_turb_1"])+np.array(nc_float["betti_trace_turb_1"]),
                             np.array(nc_float["normmag0_trace_turb_2"])+np.array(nc_float["betti_trace_turb_2"]),
                             np.array(nc_float["normmag0_trace_turb_3"])+np.array(nc_float["betti_trace_turb_3"]),
                             np.array(nc_float["normmag0_trace_turb_4"])+np.array(nc_float["betti_trace_turb_4"]),
                             np.array(nc_float["normmag0_trace_turb_5"])+np.array(nc_float["betti_trace_turb_5"]),
                             np.array(nc_float["normmag0_trace_turb_6"])+np.array(nc_float["betti_trace_turb_6"]),
                             np.array(nc_float["normmag0_trace_turb_7"])+np.array(nc_float["betti_trace_turb_7"]),
                             np.array(nc_float["normmag0_trace_turb_8"])+np.array(nc_float["betti_trace_turb_8"])) )

mag_fixed = np.concatenate( (np.array(nc_fixed["normmag0_trace_turb_0"]),
                             np.array(nc_fixed["normmag0_trace_turb_1"]),
                             np.array(nc_fixed["normmag0_trace_turb_2"]),
                             np.array(nc_fixed["normmag0_trace_turb_3"]),
                             np.array(nc_fixed["normmag0_trace_turb_4"]),
                             np.array(nc_fixed["normmag0_trace_turb_5"]),
                             np.array(nc_fixed["normmag0_trace_turb_6"]),
                             np.array(nc_fixed["normmag0_trace_turb_7"]),
                             np.array(nc_fixed["normmag0_trace_turb_8"])) )

betti_pert = np.concatenate( (np.array(nc_float["betti_trace_turb_0"]),
                              np.array(nc_float["betti_trace_turb_1"]),
                              np.array(nc_float["betti_trace_turb_2"]),
                              np.array(nc_float["betti_trace_turb_3"]),
                              np.array(nc_float["betti_trace_turb_4"]),
                              np.array(nc_float["betti_trace_turb_5"]),
                              np.array(nc_float["betti_trace_turb_6"]),
                              np.array(nc_float["betti_trace_turb_7"]),
                              np.array(nc_float["betti_trace_turb_8"])) )

print("Floating:")
print("  Mean:   ",np.mean(power_float))
print("  stddev: ",np.std (power_float))
print("Fixed:")
print("  Mean:   ",np.mean(power_fixed))
print("  stddev: ",np.std (power_fixed))

hist_float,bin_edges  = np.histogram(mag_float ,bins=np.arange(3,17,0.05),density=True)
hist_fixed,bin_edges  = np.histogram(mag_fixed ,bins=np.arange(3,17,0.05),density=True)
hist_betti,bin_edges2 = np.histogram(betti_pert,bins=np.arange(-0.3,0.3,0.01),density=True)

f, (ax1,ax2) = plt.subplots(2, 1, sharex=True)
ax1.stairs(hist_float           ,bin_edges ,fill=True)
ax2.stairs(hist_float-hist_fixed,bin_edges ,fill=True)
ax1.set_ylabel("Floating"        ,wrap=True)
ax2.set_ylabel("Floating - fixed",wrap=True)
ax2.set_xlabel("Wind speed (m/s)")
plt.show()
plt.close()

