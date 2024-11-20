from netCDF4 import Dataset
import numpy as np

nc = Dataset("supercell_with_micro_morr_errors.nc","r")
print("qv: ",np.mean(np.array(nc["micro_morr_errs_qv"])))
print("qc: ",np.mean(np.array(nc["micro_morr_errs_qc"])))
print("qr: ",np.mean(np.array(nc["micro_morr_errs_qr"])))
print("qi: ",np.mean(np.array(nc["micro_morr_errs_qi"])))
print("qs: ",np.mean(np.array(nc["micro_morr_errs_qs"])))
print("qg: ",np.mean(np.array(nc["micro_morr_errs_qg"])))
print("ni: ",np.mean(np.array(nc["micro_morr_errs_ni"])))
print("ns: ",np.mean(np.array(nc["micro_morr_errs_ns"])))
print("nr: ",np.mean(np.array(nc["micro_morr_errs_nr"])))
print("ng: ",np.mean(np.array(nc["micro_morr_errs_ng"])))
print("t : ",np.mean(np.array(nc["micro_morr_errs_t" ])))

