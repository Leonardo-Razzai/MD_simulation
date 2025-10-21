from simulation import *
from Beams import GaussianBeam, LGBeamL1

T_range = np.arange(start=5, stop=50, step=5) # uK
dMOT_range = np.arange(start=18, stop=25, step=2) # mm

beam = GaussianBeam()

for T in T_range:
    for dMOT in dMOT_range:
        label = f'T = {T} uK, dMOT = {dMOT} mm, Beam = {beam.name}'
        print(label)
        simulation(N=1e5, T=T, dMOT=dMOT, beam=beam)