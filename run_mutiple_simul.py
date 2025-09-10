from simulation import *

T_range = np.arange(start=5, stop=35, step=5) # uK
dMOT_range = np.arange(start=4, stop=24, step=4) # mm

for T in T_range:
    for dMOT in dMOT_range:
        label = f'T = {T} uK, dMOT = {dMOT} mm'
        print(label)
        simulation(N=1e5, T=T, dMOT=dMOT)