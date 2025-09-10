"""
analysis_multiple_simul.py

Post-processing of multiple cold atom simulations.
Generates plots of capture fraction vs. time and 
radial density distributions at the fiber tip, 
for different MOT temperatures and displacements.

Author: [Leonardo Razzai]
"""

from analysis import *
from matplotlib import colormaps
import numpy as np
import matplotlib.pyplot as plt
import os

# Output folder for figures
img_folder = './img/'
os.makedirs(img_folder, exist_ok=True)

# Parameter ranges
T_range = np.arange(start=5, stop=35, step=5)   # MOT temperature in μK
dMOT_range = np.arange(start=4, stop=24, step=4) # MOT displacement in mm

# --- 1. Capture fraction vs. time at fixed dMOT, varying T ---
cmap = colormaps.get_cmap('inferno')
colors = [cmap(x) for x in np.linspace(0.1, 0.8, len(T_range))]

for dMOT in dMOT_range:
    for i, T in enumerate(T_range):
        label = f'T = {T} μK, dMOT = {dMOT} mm'
        print(label)

        ts, f_cap = capt_atoms_vs_t(T, dMOT)
        plot_cap_frac(ts, f_cap, label=label, color=colors[i])

    plt.legend()
    plt.savefig(img_folder + f'cap_frac_dMOT={dMOT}mm.jpg')
    plt.clf()


# --- 2. Density distributions at fiber for fixed dMOT, varying T ---
for dMOT in dMOT_range:
    for i, T in enumerate(T_range):
        label = f'T = {T} μK, dMOT = {dMOT} mm'
        print(label)

        hist_rho_step, _ = density_at_fib(step=-1, T=T, dMOT=dMOT)
        plot_density_at_fib(hist_rho_step=hist_rho_step, label=label, color=colors[i])

    plt.legend()
    plt.savefig(img_folder + f'density_at_fib_dMOT={dMOT}mm.jpg')
    plt.clf()


# --- 3. Capture fraction vs. time at fixed T, varying dMOT ---
cmap = colormaps.get_cmap('YlGnBu')
colors = [cmap(x) for x in np.linspace(0.3, 0.7, len(dMOT_range))]

for T in T_range:
    for i, dMOT in enumerate(dMOT_range):
        label = f'T = {T} μK, dMOT = {dMOT} mm'
        print(label)

        ts, f_cap = capt_atoms_vs_t(T, dMOT)
        plot_cap_frac(ts, f_cap, label=label, color=colors[i])

    plt.legend()
    plt.savefig(img_folder + f'cap_frac_T={T}uK.jpg')
    plt.clf()


# --- 4. Density distributions at fiber for fixed T, varying dMOT ---
for T in T_range:
    for i, dMOT in enumerate(dMOT_range):
        label = f'T = {T} μK, dMOT = {dMOT} mm'
        print(label)

        hist_rho_step, _ = density_at_fib(step=-1, T=T, dMOT=dMOT)
        plot_density_at_fib(hist_rho_step=hist_rho_step, label=label, color=colors[i])

    plt.legend()
    plt.savefig(img_folder + f'density_at_fib_T={T}uK.jpg')
    plt.clf()
