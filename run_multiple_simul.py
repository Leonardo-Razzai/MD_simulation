from simulation import *
from Beams import beams
from sys import argv
import time

# Select Beam
if len(argv) > 1:
    beam_name = argv[1]
else:
    print('\nSpecify a valid beam name:\nGauss : Gaussian beam\nLG : Laguerre-Gauss beam\n\n')
    exit()

if beam_name not in ['Gauss', 'LG']:
    print('\nSpecify a valid beam name:\nGauss : Gaussian beam\nLG : Laguerre-Gauss beam\n\n') 
    exit()

beam = beams[beam_name]

T_range = np.arange(start=5, stop=50, step=5) # uK
dMOT_range = np.arange(start=2, stop=22, step=2) # mm
HEATING = True

for T in T_range:
    for dMOT in dMOT_range:
        label = f'T = {T} uK, dMOT = {dMOT} mm, Beam = {beam.name}'
        print(label)
        simulation(N=1e5, T=T, dMOT=dMOT, beam=beam, Heating=HEATING)
        time.sleep(60)