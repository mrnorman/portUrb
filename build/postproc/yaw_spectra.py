from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

def ma(a,n=3):
  return np.convolve( a , np.ones(n)/n , mode='valid' )
def spectra(T,dx) :
  spd  = np.abs( np.fft.rfft(T[:]) )**2
  freq = np.fft.rfftfreq(len(T[:]))*2*2*np.pi/(2*dx)
  return freq , spd

times = range(2,9)
N = len(times)
files = [f"run_21_{i:08d}.nc" for i in times]
for i in range(len(files)) :
  loc = np.array(Dataset(files[i],"r")["yaw_trace_turb_0"][:])
  yaw = loc if (i==0) else np.append(yaw,loc)

# time = [N*900*i/(len(yaw)-1) for i in range(len(yaw))]
# plt.plot(time,yaw)
# plt.show()
# plt.close()

w  = 2
yaw = np.append(yaw,np.flip(yaw))
freq,spd = spectra(yaw,N*900/len(yaw))
i1 = np.argmin(np.abs(freq-0.13))
i2 = np.argmin(np.abs(freq-0.50))
i3 = np.argmin(np.abs(freq-0.90))
plt.loglog(ma(freq,w) ,ma(spd,w)         ,label="Yaw Spectra",linewidth=0.5,color="black")
plt.loglog(freq[i1:i2],10*freq[i1:i2]**-8,label=r"$f^{-8}$"  ,color="red" )
plt.loglog(freq[i3:]  ,50*freq[i3:]**-4  ,label=r"$f^{-4}$"  ,color="cyan")
plt.legend()
plt.show()
plt.close()

