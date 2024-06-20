import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.barycenters import dtw_barycenter_averaging

#Define Functions for Dataset Import
def dataset(n):
  path = f'./Data Training/{n}.csv'
  arr = np.genfromtxt(path, delimiter=';', skip_header=1)
  arr_fixed = np.delete(arr, np.arange(1, len(arr[0]), 2), axis=1)[:,:-1]
  return arr_fixed

# Import data 1.csv-50.csv to a variable X
N = 50
filterSignal = np.array([2, 10, 11, 12, 18, 20, 41, 42, 43, 45, 46, 47, 48, 49, 51, 52, 53, 55, 56, 57, 58, 59, 64, 66, 67, 69, 71, 72, 74, 75, 83, 87, 88, 93, 94, 95, 98, 106])
X_train = np.array([dataset(i) for i in range(1,N+1)])[:,270:,filterSignal+1]

signals = 'TSIM (s);Reactor thermal power (MW);Boron concentration (ppm);Core Reactivity pcm (pcm);Core Reactivity %dk/k (%dk/k);Control rod worth pcm (pcm);Rod Reactivity %dk/k (%dk/k);Fuel Reactivity (Doppler) %dk/k (%dk/k);Moderator Temperature Reactivity %dk/k (%dk/k);Boron Reactivity %dk/k (%dk/k);Xe Reactivity pcm (pcm);Clad surface T max (ºc);Clad surface T average (ºc);Average fuel temperature (ºc);Peak fuel (centre line) T (ºc);Relief valve flow (Kg/s);ADS1 valve flow (Kg/s);ADS2 valve flow (Kg/s);ADS3 valve flow (Kg/s);Generator load (MW);ADS Total Flow (Kg/s);FWS Total Flow (Kg/s);DHR1 Pool level (%);DHR2 Pool level (%);Subpression pool level (%);Sump level (%);Containment pressure. (bar);DHR1 Pool temperature (ºc);DHR2 Pool temperature (ºc);Subpression pool temperature (ºc);Containment temperature (ºc);Containment coolant spray flow 1 (kg/s);Containment coolant spray flow 2 (Kg/s);Condenser level (%);Condenser pressure vacuum (mmHg);Coolant pump 1 power (W);Coolant pump 2 power (W);Condenser coolant flow (kg/s);Coolant Temperature at condenser inlet. (ºc);Coolant Temperature at condenser outlet (ºc);DHR train 1 (kg/s);DHR train 2 (kg/s);% Steam generator 1 valve opening (%);% Steam generator 2 valve opening (%);Pump speed 1 (rpm);Pump speed 2 (rpm);Flow to Steam Generator 1 (kg/s);Flow to Steam Generator 2 (kg/s);Flow through Heater 3 (kg/s);Flow through Heater 2 (kg/s);Flow through Heater 1 (kg/s);Water Storage Tank (%);Header FW pressure (Mpa);FW line 1 Pressure (Mpa);FW line 2 Pressure (Mpa);P inlet HX1 (Mpa);Inlet Temperature to Steam Generator 1 (ºc);Inlet Temperature to Steam Generator 2 (ºc);Heater 3 Outlet Temperature (ºc);Heater 2 Outlet Temperature (ºc);Heater 1 Outlet Temperature (ºc);GIS flow 1 (kg/s);GIS flow 2 (kg/s);GIS Tank1 level (%);GIS Tank2 level (%);% open valve turbine (%);% open valve bypass (%);Steam flow rate line 1 (kg/s);Steam flow rate line 2 (kg/s);Total flow in pressure header (kg/s);Steam flow to turbine (kg/s);Steam flow to condenser (kg/s);Steam pressure line 1 (Mpa);Steam pressure line 2 (Mpa);Pressure header (Mpa);Steam temperature line 1 (ºC);Steam temperature line 2 (ºC);Steam temperature in header (ºC);Saturation Temperature (ºc);PIS flow 1 (kg/s);PIS flow 2 (kg/s);PIS Tank1 level (%);PIS Tank2 level (%);Start up rate (%/s);Coolant flow rate (kg/s);Flow charge (kg/s);Flow Discharge (kg/s);Spray Flow (kg/s);RPV Water Level (%);PZR Level (%);Setpoint PZR Level (%);Nuclear Power (%);Power Intermediate Range (A);Power Source Range (cps);RPV Pressure (Mpa);PZR pressure (Mpa);Coolant average Temperature (ºc);Coolant Temperature at core inlet (ºc);Coolant Temperature at core outlet (ºc);Delta T (ºc);Subcooling Margin (ºc);Power RCP1 (W);Power RCP2 (W);Power RCP3 (W);Power RCP4 (W);Turbine speed (rpm);Pressure first stage (%);Reactor thermal power (%) (%);Xe reactivity %dk/k (%dk/k);Fuel Reactivity (Doppler) pcm (pcm);Moderator Temperature Reactivity pcm (pcm);Boron Reactivity pcm (pcm);Heater power (KW);Core fraction level1 (bottom) ();Core fraction level2 ();Core fraction level3 ();Core fraction level4 (top) ();'
signal = signals.split(';')[1:-1]
filteredSignal = []
for i, sig in enumerate(signal):
  if i in filterSignal:
    filteredSignal.append(sig)

seed = 0
np.random.seed(seed)

#Number of CLuster
C = 2

print("DTW k-means")
km = TimeSeriesKMeans(n_clusters=C,
                      metric='dtw', 
                      verbose=True, 
                      random_state=seed)

y_pred = km.fit_predict(X_train)

cl = [np.argwhere(y_pred==i).flatten() for i in range(C)]

#Compute barycenter
bars = [dtw_barycenter_averaging(X_train[cl[i],:,:]) for i in range(C)]

T = np.array([dataset(i) for i in range(1,N+1)])[:,270:, 0]

fig, ax = plt.subplots(X_train.shape[2], C)
fig.set_size_inches(10, 100)

for z in range(X_train.shape[2]):
    for i, X in enumerate(X_train):
        ax[z, y_pred[i]].plot(T[i,:],
                              X[:,z], 
                              color='k', 
                              alpha=.4,
                              linewidth=1)
    ax[z, 0].set_ylabel(filteredSignal[z])
    ax[z, 1].set_yticklabels([])

    ax[z, 0].set_xlabel("Waktu Simulasi (detik)")
    ax[z, 1].set_xlabel("Waktu Simulasi (detik)")

    for j in range(C):
        ax[z, j].set_ylim((np.min(X_train[:,:,z]), np.max(X_train[:,:,z])))
        ax[z, j].plot(T[-1,:], bars[j][:,z], color='r', linewidth=.8)
        ax[0, j].set_title(f"Cluster {j+1}")

plt.tight_layout()
fig.savefig(f'Hasil Training {C} Klaster.png')
plt.show()
