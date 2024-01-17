from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

nc = Dataset("compare_les.nc","r")
nz = nc.dimensions["nz"].size
ny = nc.dimensions["ny"].size
nx = nc.dimensions["nx"].size

explicit_names = ["tend_explicit_u_x","tend_explicit_v_x","tend_explicit_w_x","tend_explicit_t_x",
                  "tend_explicit_u_y","tend_explicit_v_y","tend_explicit_w_y","tend_explicit_t_y",
                  "tend_explicit_u_z","tend_explicit_v_z","tend_explicit_w_z","tend_explicit_t_z"]
closure_notke_names = ["tend_closure_notke_u_x","tend_closure_notke_v_x","tend_closure_notke_w_x","tend_closure_notke_t_x",
                       "tend_closure_notke_u_y","tend_closure_notke_v_y","tend_closure_notke_w_y","tend_closure_notke_t_y",
                       "tend_closure_notke_u_z","tend_closure_notke_v_z","tend_closure_notke_w_z","tend_closure_notke_t_z"]
closure_names = ["tend_closure_u_x","tend_closure_v_x","tend_closure_w_x","tend_closure_t_x",
                 "tend_closure_u_y","tend_closure_v_y","tend_closure_w_y","tend_closure_t_y",
                 "tend_closure_u_z","tend_closure_v_z","tend_closure_w_z","tend_closure_t_z"]
tke_names     = ["tke_x","tke_x","tke_x","tke_x",
                 "tke_y","tke_y","tke_y","tke_y",
                 "tke_z","tke_z","tke_z","tke_z"]

# print("Plotting")
# for i in range(len(explicit_names)) :
#     explicit = nc.variables[explicit_names[i]][::2,::2,::2,0].flatten() + 
#     closure  = nc.variables[closure_names [i]][::2,::2,::2,0].flatten()
#     explicit = (explicit - np.min(explicit)) / (np.max(explicit) - np.min(explicit))
#     closure  = (closure  - np.min(closure )) / (np.max(closure ) - np.min(closure ))
#     # result = stats.linregress(explicit.flatten(), closure.flatten()) 
#     # print(result)
#     plt.scatter( explicit , closure , s=0.01 , c = "blue"  , edgecolors="face" )
#     # plt.xscale("log")
#     # plt.yscale("log")
#     plt.xlabel(explicit_names[i])
#     plt.ylabel(closure_names [i])
#     plt.show()
#     plt.close()

print("Getting data")
explicit = nc.variables["tend_explicit_u_x"][::4,::4,::4,0].flatten() + \
           nc.variables["tend_explicit_u_y"][::4,::4,::4,0].flatten() + \
           nc.variables["tend_explicit_u_z"][::4,::4,::4,0].flatten()
closure  = nc.variables["tend_closure_u_x" ][::4,::4,::4,0].flatten() + \
           nc.variables["tend_closure_u_y" ][::4,::4,::4,0].flatten() + \
           nc.variables["tend_closure_u_z" ][::4,::4,::4,0].flatten()
tol = 1.e-7
print("Computing non-trivial indices")
indices = range(len(explicit))
filtered_indices = list(filter( lambda i: np.abs(closure[i]) > tol , indices ))
print("Computing filter, abs, and log")
explicit = np.log(np.abs(explicit[filtered_indices]))
closure  = np.log(np.abs(closure [filtered_indices]))
print("Linear regression")
result = stats.linregress(explicit.flatten(), closure.flatten()) 
print(result)
print("Plotting")
plt.scatter( explicit , closure , s=0.00001 , c = "blue"  , edgecolors="face" )
plt.scatter( explicit , result.slope*explicit + result.intercept , s=0.00001 , c = "red"  , edgecolors="face" )
# plt.xlim(-20,0)
plt.show()


plt.close()
print("Getting data")
explicit = nc.variables["tend_explicit_v_x"][::4,::4,::4,0].flatten() + \
           nc.variables["tend_explicit_v_y"][::4,::4,::4,0].flatten() + \
           nc.variables["tend_explicit_v_z"][::4,::4,::4,0].flatten()
closure  = nc.variables["tend_closure_v_x" ][::4,::4,::4,0].flatten() + \
           nc.variables["tend_closure_v_y" ][::4,::4,::4,0].flatten() + \
           nc.variables["tend_closure_v_z" ][::4,::4,::4,0].flatten()
tol = 1.e-7
print("Computing non-trivial indices")
indices = range(len(explicit))
filtered_indices = list(filter( lambda i: np.abs(closure[i]) > tol , indices ))
print("Computing filter, abs, and log")
explicit = np.log(np.abs(explicit[filtered_indices]))
closure  = np.log(np.abs(closure [filtered_indices]))
print("Plotting")
plt.scatter( explicit , closure , s=0.00001 , c = "blue"  , edgecolors="face" )
plt.scatter( explicit , result.slope*explicit + result.intercept , s=0.00001 , c = "red"  , edgecolors="face" )
# plt.xlim(-20,0)
plt.show()


plt.close()
print("Getting data")
explicit = nc.variables["tend_explicit_w_x"][::4,::4,::4,0].flatten() + \
           nc.variables["tend_explicit_w_y"][::4,::4,::4,0].flatten() + \
           nc.variables["tend_explicit_w_z"][::4,::4,::4,0].flatten()
closure  = nc.variables["tend_closure_w_x" ][::4,::4,::4,0].flatten() + \
           nc.variables["tend_closure_w_y" ][::4,::4,::4,0].flatten() + \
           nc.variables["tend_closure_w_z" ][::4,::4,::4,0].flatten()
tol = 1.e-7
print("Computing non-trivial indices")
indices = range(len(explicit))
filtered_indices = list(filter( lambda i: np.abs(closure[i]) > tol , indices ))
print("Computing filter, abs, and log")
explicit = np.log(np.abs(explicit[filtered_indices]))
closure  = np.log(np.abs(closure [filtered_indices]))
print("Plotting")
plt.scatter( explicit , closure , s=0.00001 , c = "blue"  , edgecolors="face" )
plt.scatter( explicit , result.slope*explicit + result.intercept , s=0.00001 , c = "red"  , edgecolors="face" )
# plt.xlim(-20,0)
plt.show()


plt.close()
print("Getting data")
explicit = nc.variables["tend_explicit_t_x"][::4,::4,::4,0].flatten() + \
           nc.variables["tend_explicit_t_y"][::4,::4,::4,0].flatten() + \
           nc.variables["tend_explicit_t_z"][::4,::4,::4,0].flatten()
closure  = nc.variables["tend_closure_t_x" ][::4,::4,::4,0].flatten() + \
           nc.variables["tend_closure_t_y" ][::4,::4,::4,0].flatten() + \
           nc.variables["tend_closure_t_z" ][::4,::4,::4,0].flatten()
tol = 1.e-7
print("Computing non-trivial indices")
indices = range(len(explicit))
filtered_indices = list(filter( lambda i: np.abs(closure[i]) > tol , indices ))
print("Computing filter, abs, and log")
explicit = np.log(np.abs(explicit[filtered_indices]))
closure  = np.log(np.abs(closure [filtered_indices]))
print("Plotting")
plt.scatter( explicit , closure , s=0.00001 , c = "blue"  , edgecolors="face" )
plt.scatter( explicit , result.slope*explicit + result.intercept , s=0.00001 , c = "red"  , edgecolors="face" )
# plt.xlim(-20,0)
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter( ( explicit_u_x.flatten() - np.min(explicit_u_x) ) / (np.max(explicit_u_x)-np.min(explicit_u_x)) ,
#             ( tke_x       .flatten() - np.min(tke_x       ) ) / (np.max(tke_x       )-np.min(tke_x       )) ,
#             ( closure_u_x .flatten() - np.min(closure_u_x ) ) / (np.max(closure_u_x )-np.min(closure_u_x )) ,
#             s=0.01 )
# # ax.set_aspect("equal")
# ax.set_xscale("log")
# ax.set_zscale("log")
# ax.set_xlabel('Explicitly Calculated')
# ax.set_ylabel('TKE')
# ax.set_zlabel('LES Closure')
# plt.show()

