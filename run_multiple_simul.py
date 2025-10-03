from simulation import *
from Beams import GaussianBeam, LGBeamL1

T_range = np.arange(start=7.5, stop=124.5, step=5) # uK
dMOT_range = np.arange(start=2, stop=10, step=1) # mm

beam = GaussianBeam()

for T in T_range:
    for dMOT in dMOT_range:
        label = f'T = {T} uK, dMOT = {dMOT} mm, Beam = {beam.name}'
        print(label)
        simulation(N=1e5, T=T, dMOT=dMOT, beam=beam)