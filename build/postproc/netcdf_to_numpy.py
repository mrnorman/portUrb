from netCDF4 import Dataset
import numpy as np
import os
import sys

nc = Dataset(sys.argv[1],"r")
inputs  = nc.variables["inputs" ][:,:]
outputs = nc.variables["outputs"][:,:]

num_samples = inputs .shape[0]
num_inputs  = inputs .shape[1]
num_outputs = outputs.shape[1]

permuted_indices = np.random.permutation(np.arange(0, num_samples))

inputs_shuffled  = inputs [permuted_indices[:],:]
outputs_shuffled = outputs[permuted_indices[:],:]

np.savez(sys.argv[2],inputs=inputs_shuffled,outputs=outputs_shuffled)

