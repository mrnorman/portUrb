import numpy as np
import Betti
from netCDF4 import Dataset

C_p, C_t, pitch_angles, TSR_values = Betti.process_rotor_performance()
nc = Dataset("Betti_NREL_5MW.nc","w")
nc.createDimension("pitch_angles",len(pitch_angles))
nc.createDimension("TSR_values"  ,len(TSR_values  ))
nc.createVariable("C_p"         ,"f4",("TSR_values","pitch_angles",))[:,:] = np.array(C_p         )
nc.createVariable("C_t"         ,"f4",("TSR_values","pitch_angles",))[:,:] = np.array(C_t         )
nc.createVariable("pitch_angles","f4",("pitch_angles",             ))[:]   = np.array(pitch_angles)
nc.createVariable("TSR_values"  ,"f4",("TSR_values"  ,             ))[:]   = np.array(TSR_values  )
nc.close()

