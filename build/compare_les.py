from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np

nc = Dataset("compare_les.nc","r")
nz = nc.dimensions["nz"].size
ny = nc.dimensions["ny"].size
nx = nc.dimensions["nx"].size

# double explicit_u_x(nz, ny, nxp1, nens) ;
# double explicit_v_x(nz, ny, nxp1, nens) ;
# double explicit_w_x(nz, ny, nxp1, nens) ;
# double explicit_t_x(nz, ny, nxp1, nens) ;
# double explicit_u_y(nz, nyp1, nx, nens) ;
# double explicit_v_y(nz, nyp1, nx, nens) ;
# double explicit_w_y(nz, nyp1, nx, nens) ;
# double explicit_t_y(nz, nyp1, nx, nens) ;
# double explicit_u_z(nzp1, ny, nx, nens) ;
# double explicit_v_z(nzp1, ny, nx, nens) ;
# double explicit_w_z(nzp1, ny, nx, nens) ;
# double explicit_t_z(nzp1, ny, nx, nens) ;
# double closure_u_x(nz, ny, nxp1, nens) ;
# double closure_v_x(nz, ny, nxp1, nens) ;
# double closure_w_x(nz, ny, nxp1, nens) ;
# double closure_t_x(nz, ny, nxp1, nens) ;
# double closure_u_y(nz, nyp1, nx, nens) ;
# double closure_v_y(nz, nyp1, nx, nens) ;
# double closure_w_y(nz, nyp1, nx, nens) ;
# double closure_t_y(nz, nyp1, nx, nens) ;
# double closure_u_z(nzp1, ny, nx, nens) ;
# double closure_v_z(nzp1, ny, nx, nens) ;
# double closure_w_z(nzp1, ny, nx, nens) ;
# double closure_t_z(nzp1, ny, nx, nens) ;
# double TKE(nz, ny, nx, nens) ;
# double density_dry(nz, ny, nx, nens) ;
# double uvel(nz, ny, nx, nens) ;
# double vvel(nz, ny, nx, nens) ;
# double wvel(nz, ny, nx, nens) ;
# double temperature(nz, ny, nx, nens) ;

print("Getting data")
explicit_u_x = nc.variables["explicit_u_x"][::2,::2,::2,0]
closure_u_x  = nc.variables["closure_u_x" ][::2,::2,::2,0]

print("Plotting")
plt.scatter( explicit_u_x.flatten() , closure_u_x.flatten() , s=0.00001 )
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Explicitly calculated")
plt.ylabel("LES Closure")
print("Saving Figure")
plt.show()
#plt.savefig("u_x.jpg")

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

