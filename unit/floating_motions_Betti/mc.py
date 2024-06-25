import matplotlib.pyplot as plt 
import numpy as np
import Betti

end_time     = 1000
dt           = 0.05
n            = int(end_time/dt)+1
Vin          = np.array([n,1])
v_w          = 12
x0           = np.array([-2, 0, 37.550, 0, 0, 0, 1]) 
v_wind       = np.array([v_w for i in range(n)]) # np.random.normal(v_w,2,n)
random_phases = 2 * np.pi * np.random.rand(400)
kd           = 1
control_mode = 0 
t, x, v_wind_out, wave_eta, Q_t, betas = Betti.main(end_time, v_w, x0, v_wind, random_phases, kd, control_mode , dt) 

# The column is each state [surge, surge_velocity, heave, heave_velocity, pitch, pitch_rate, rotor_speed]
plt.plot(t,x[:,0]    ,linewidth=0.3); plt.title("surge"         ); plt.savefig("surge.png"         ,dpi=300); plt.close()
plt.plot(t,x[:,1]    ,linewidth=0.3); plt.title("surge_velocity"); plt.savefig("surge_velocity.png",dpi=300); plt.close()
plt.plot(t,x[:,2]    ,linewidth=0.3); plt.title("heave"         ); plt.savefig("heave.png"         ,dpi=300); plt.close()
plt.plot(t,x[:,3]    ,linewidth=0.3); plt.title("heave_velocity"); plt.savefig("heave_velocity.png",dpi=300); plt.close()
plt.plot(t,x[:,4]    ,linewidth=0.3); plt.title("pitch"         ); plt.savefig("pitch.png"         ,dpi=300); plt.close()
plt.plot(t,x[:,5]    ,linewidth=0.3); plt.title("pitch_rate"    ); plt.savefig("pitch_rate.png"    ,dpi=300); plt.close()
plt.plot(t,x[:,6]    ,linewidth=0.3); plt.title("rotor_speed"   ); plt.savefig("rotor_speed.png"   ,dpi=300); plt.close()
plt.plot(t,v_wind_out,linewidth=0.3); plt.title("v_wind_out"    ); plt.savefig("v_wind_out.png"    ,dpi=300); plt.close()
plt.plot(t,wave_eta  ,linewidth=0.3); plt.title("wave_eta"      ); plt.savefig("wave_eta.png"      ,dpi=300); plt.close()
d_Ph = 5.4305  # (m) Horizontal distance between BS and BP
d_Pv = 127.5879  # (m) Vertical distance between BS and BP
d_P = np.sqrt(d_Ph**2 + d_Pv**2)
Vin = v_w + x[:,1]+ d_P*x[:,5]/180*np.pi*np.cos(x[:,4]/180*np.pi)
plt.plot(t,Vin  ,linewidth=0.3); plt.title("V_in"      ); plt.savefig("V_in.png"      ,dpi=300); plt.close()
np.savetxt('Vintime',t,delimiter=',')
np.savetxt('Vindata',Vin,delimiter=',')
