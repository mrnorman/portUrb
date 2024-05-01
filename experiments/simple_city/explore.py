import matplotlib.pyplot as plt
import numpy as np
import yaml

h = yaml.safe_load(open("inputs/IEA-15-240-RWT.yaml","r"))
ref_u   = h["velocity_magnitude"]
ref_C_T = h["thrust_coef"]
ref_C_P = h["power_coef"]
ref_P   = h["power_megawatts"]
ref_rot = h["rotation_rpm"]

def comp_C_T(uinf) :
  return np.interp(uinf,ref_u,ref_C_T,left=0.,right=0.)

def comp_T(uinf) :
  return np.interp(uinf,ref_u,ref_T,left=0.,right=0.)

def comp_C_P(uinf) :
  return np.interp(uinf,ref_u,ref_C_P,left=0.,right=0.)

def comp_P(uinf) :
  return np.interp(uinf,ref_u,ref_P,left=0.,right=0.)

def comp_rot(uinf) :
  return np.interp(uinf,ref_u,ref_rot,left=0.,right=0.)

def comp_a(uinf) :
  return 0.5*(1-np.sqrt(1-comp_C_T(uinf)))


u = np.arange(0.1,30,0.001)
a = comp_a(u)
for i in range(100) :
  a = comp_a(u/(1-a))

ui = u/(1-a)

plt.plot(u,u ,label="reference")
plt.plot(u,ui,label="freestream")
plt.xlabel("disk-integrated normal velocity magnitude")
plt.ylabel("free stream velocity magnitude")
plt.legend()
plt.show()
plt.close()

plt.plot(u,1/(1-a))
plt.xlabel("disk-integrated normal velocity magnitude")
plt.ylabel("freestream magnitude / disk-integrated magnitude")
plt.show()
plt.close()

plt.plot(u,comp_C_T(ui))
plt.xlabel("disk-integrated normal velocity magnitude")
plt.ylabel("coefficient of thrust")
plt.show()
plt.close()

plt.plot(u,comp_C_P(ui))
plt.xlabel("disk-integrated normal velocity magnitude")
plt.ylabel("coefficient of power")
plt.show()
plt.close()

plt.plot(u,comp_P(ui))
plt.xlabel("disk-integrated normal velocity magnitude")
plt.ylabel("total power (MW)")
plt.show()
plt.close()

plt.plot(u,0.5*1.2*comp_C_T(ui)*ui*ui*np.pi*242.23775645/2*242.23775645/2/1000/1000)
plt.xlabel("disk-integrated normal velocity magnitude")
plt.ylabel("Total Thrust (MN)")
plt.show()
plt.close()

plt.plot(u,comp_rot(ui))
plt.xlabel("disk-integrated normal velocity magnitude")
plt.ylabel("rotation rate (rpm)")
plt.show()
plt.close()

fig, axs = plt.subplots(2,sharex=True)
axs[0].hist(u ,bins=np.arange(1,30,0.1),density=True)
axs[1].hist(ui,bins=np.arange(1,30,0.1),density=True)
axs[0].set_title("disk-integrated")
axs[1].set_title("freestream")
axs[1].set_xlabel("Wind Magnitude (m/s) Bins")
axs[0].set_ylabel("Probability Density")
axs[1].set_ylabel("Probability Density")
axs[0].set_ylim((0,0.1))
axs[1].set_ylim((0,0.1))
plt.show()
plt.close()

u = np.random.normal(loc=10,scale=1.38,size=100000)
a = comp_a(u)
for i in range(100) :
  a = comp_a(u/(1-a))
ui = u/(1-a)
fig, axs = plt.subplots(2,sharex=True)
axs[0].hist(u ,bins=np.arange(6,20,0.1),density=True)
axs[1].hist(ui,bins=np.arange(6,20,0.1),density=True)
axs[0].set_title("disk-integrated")
axs[1].set_title("freestream")
axs[1].set_xlabel("Wind Magnitude (m/s) Bins")
axs[0].set_ylabel("Probability Density")
axs[1].set_ylabel("Probability Density")
axs[0].set_ylim((0,1.0))
axs[1].set_ylim((0,1.0))
plt.show()
plt.close()

