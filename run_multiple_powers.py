from simulation import *
from Beams import beams
import time

Powers = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]

for P in Powers:
    simulation(N=1e5, T=15, dMOT=7, beam=LGBeamL1(P_b=P), HEATING=True)
    time.sleep(120)