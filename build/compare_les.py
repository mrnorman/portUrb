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
explicit_u_x = nc.variables["explicit_u_x"][::5,::5,::5,0]
closure_u_x  = nc.variables["closure_u_x" ][::5,::5,::5,0]

print("Plotting")
plt.scatter( explicit_u_x.flatten() , closure_u_x.flatten() , s=0.01 )
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Explicitly calculated")
plt.ylabel("LES Closure")
print("Saving Figure")
plt.show()
#plt.savefig("u_x.jpg")

