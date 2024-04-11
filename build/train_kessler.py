import numpy as np
from sklearn.metrics import r2_score

import os
# The line below reduces tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, ReLU, Softmax, LeakyReLU, PReLU, ELU, ThresholdedReLU
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl
# The option below improves the speed of model.predict() (see below)
tensorflow.compat.v1.disable_eager_execution()

import matplotlib.pyplot as plt

path = f'supercell_kessler_data.npz'
data_link = "https://www.dropbox.com/scl/fi/6d1zhhqimv6e72a4kg153/supercell_kessler_data.npz?rlkey=xsyhzwjkyhxedt1wxt72y9fvn&dl=0"

# Download the data if necessary
if ( not os.path.isfile(path) ):
    print(f"Downloading data from:\n {data_link}...")
    !wget '{data_link}' -O {path}

# Load the data
fh = np.load("supercell_kessler_data.npz")
inputs  = fh['inputs' ]
outputs = fh['outputs']
num_samples = inputs .shape[0]
num_inputs  = inputs .shape[1]
num_outputs = outputs.shape[1]

# Scale the data, and dump the min,max
scaling_inputs  = np.ndarray(shape=[num_inputs ,2],dtype=np.single)
scaling_outputs = np.ndarray(shape=[num_outputs,2],dtype=np.single)
for i in range(num_inputs) :
    mn = np.amin(inputs[:,i])
    mx = np.amax(inputs[:,i])
    scaling_inputs[i,0] = mn
    scaling_inputs[i,1] = mx
    inputs[:,i] = (inputs[:,i] - mn) / (mx-mn)
for i in range(num_outputs) :
    mn = np.amin(outputs[:,i])
    mx = np.amax(outputs[:,i])
    scaling_outputs[i,0] = mn
    scaling_outputs[i,1] = mx
    outputs[:,i] = (outputs[:,i] - mn) / (mx-mn)

np.savetxt('supercell_kessler_input_scaling.txt' , scaling_inputs , fmt="%s")
np.savetxt('supercell_kessler_output_scaling.txt', scaling_outputs, fmt="%s")

# Split the data
validation_split = 0.1
split_index = int(validation_split*num_samples)
inputs_validation  = inputs [:split_index,:]
outputs_validation = outputs[:split_index,:]
inputs_train       = inputs [split_index:,:]
outputs_train      = outputs[split_index:,:]

# Create and train the Keras Neural Network (Dense, Feed-Forward)
model = Sequential()
model.add( Dense(units=10,input_dim=num_inputs,kernel_initializer="RandomUniform") )
model.add( LeakyReLU(alpha=0.1) )
model.add( Dense(units=10,kernel_initializer="RandomUniform") )
model.add( LeakyReLU(alpha=0.1) )
model.add( Dense(num_outputs,kernel_initializer="RandomUniform") )
model.compile(loss='mse', optimizer="Nadam", metrics=['mean_absolute_error'])
print(model.summary())
history = model.fit(inputs_train, outputs_train, epochs=10, batch_size=128, validation_split=0.2, verbose=1)

model_json = model.to_json()
with open('supercell_kessler_model_weights.json', 'w') as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights('supercell_kessler_model_weights.h5')
print('Saved model to disk')

