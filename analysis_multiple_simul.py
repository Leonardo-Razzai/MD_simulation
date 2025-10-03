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
from Beams import GaussianBeam, LGBeamL1

# Select Beam
beam = GaussianBeam()

# Output folder for figures
img_folder = './img/'
os.makedirs(img_folder, exist_ok=True)

# Parameter ranges
T_range = np.arange(start=5, stop=120, step=5)   # MOT temperature in μK
dMOT_range = np.arange(start=2, stop=11, step=1) # MOT displacement in mm
dMOT_range = np.concatenate([dMOT_range, np.arange(12, 20, 1)])

out_folder = img_folder + beam.name +'/'
os.makedirs(out_folder, exist_ok=True)

# --- 1. Capture fraction vs. time at fixed dMOT, varying T ---
cmap = colormaps.get_cmap('inferno')
colors = [cmap(x) for x in np.linspace(0.1, 0.8, 2)]

def plot_3d_TdMOTconc():

    # Create meshgrid
    T_array, dMOT_array = np.meshgrid(T_range, dMOT_range)

    # Allocate Z with same shape
    Z = np.zeros_like(T_array, dtype=float)

    # Fill Z with function values
    for i in range(T_array.shape[0]):
        for j in range(T_array.shape[1]):
            Z[i, j] = get_last_conc(T_array[i, j], dMOT_array[i, j], beam=beam)

    Z = Z * 100 # in %
    # Plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(T_array, dMOT_array, Z, cmap='viridis', edgecolor='none')

    ax.set_xlabel("Temperature (µK)")
    ax.set_ylabel("MOT distance (mm)")
    ax.set_zlabel("Captured atoms (%)")
    ax.set_title(f"Atom Capture Efficiency vs T and dMOT ({beam.name})")

    #fig.colorbar(surf, shrink=0.5, aspect=10, label="Captured atoms (%)")
    plt.tight_layout()
    plt.savefig(out_folder + f'conc_vs_T-dMOT.jpg')
    plt.clf()


def plot_cap_frac_vs_T(T_range, dMOT_range):
    """
    Plot capture fraction vs. time for fixed dMOT and varying T.
    Saves one plot per dMOT.
    """
    cmap = colormaps.get_cmap('inferno')
    colors = [cmap(x) for x in np.linspace(0.1, 0.8, 2)]

    small_T_range = [T_range.min(), T_range.max()]
    small_dMOT_range = [dMOT_range.min(), dMOT_range.max()]

    for dMOT in small_dMOT_range:
        for i, T in enumerate(small_T_range):
            label = f'T = {T} μK, dMOT = {dMOT} mm'
            print(label)

            ts, f_cap = capt_atoms_vs_t(T, dMOT, beam=beam)
            plot_cap_frac(ts, f_cap, label=label, color=colors[i])

        plt.legend()
        plt.title(f'Captured Faction vs T ({beam.name})')
        plt.tight_layout()
        plt.savefig(out_folder + f'cap_frac_dMOT={dMOT}mm.jpg')
        plt.clf()


def plot_density_vs_T(T_range, dMOT_range):
    """
    Plot radial density distribution at fiber for fixed dMOT and varying T.
    Saves one plot per dMOT.
    """
    cmap = colormaps.get_cmap('inferno')
    colors = [cmap(x) for x in np.linspace(0.1, 0.8, 2)]

    small_T_range = [T_range.min(), T_range.max()]
    small_dMOT_range = [dMOT_range.min(), dMOT_range.max()]

    for dMOT in small_dMOT_range:
        for i, T in enumerate(small_T_range):
            label = f'T = {T} μK, dMOT = {dMOT} mm'
            print(label)

            hist_rho_step, _ = density_at_fib(step=-1, T=T, dMOT=dMOT, beam=beam)
            plot_density_at_fib(hist_rho_step=hist_rho_step, label=label, color=colors[i])

        plt.legend()
        plt.title(f'Radial density vs T ({beam.name})')
        plt.tight_layout()
        plt.savefig(out_folder + f'density_at_fib_dMOT={dMOT}mm.jpg')
        plt.clf()


def plot_cap_frac_vs_dMOT(T_range, dMOT_range):
    """
    Plot capture fraction vs. time for fixed T and varying dMOT.
    Saves one plot per T.
    """
    cmap = colormaps.get_cmap('YlGnBu')
    colors = [cmap(x) for x in np.linspace(0.3, 0.7, 2)]

    small_T_range = [T_range.min(), T_range.max()]
    small_dMOT_range = [dMOT_range.min(), dMOT_range.max()]

    for T in small_T_range:
        for i, dMOT in enumerate(small_dMOT_range):
            label = f'T = {T} μK, dMOT = {dMOT} mm'
            print(label)

            ts, f_cap = capt_atoms_vs_t(T, dMOT, beam=beam)
            plot_cap_frac(ts, f_cap, label=label, color=colors[i])

        plt.legend()
        plt.title(f'Captured Faction vs dMOT ({beam.name})')
        plt.tight_layout()
        plt.savefig(out_folder + f'cap_frac_T={T}uK.jpg')
        plt.clf()


def plot_density_vs_dMOT(T_range, dMOT_range):
    """
    Plot radial density distribution at fiber for fixed T and varying dMOT.
    Saves one plot per T.
    """
    cmap = colormaps.get_cmap('YlGnBu')
    colors = [cmap(x) for x in np.linspace(0.3, 0.7, 2)]

    small_T_range = [T_range.min(), T_range.max()]
    small_dMOT_range = [dMOT_range.min(), dMOT_range.max()]

    for T in small_T_range:
        for i, dMOT in enumerate(small_dMOT_range):
            label = f'T = {T} μK, dMOT = {dMOT} mm'
            print(label)

            hist_rho_step, _ = density_at_fib(step=-1, T=T, dMOT=dMOT, beam=beam)
            plot_density_at_fib(hist_rho_step=hist_rho_step, label=label, color=colors[i])

        plt.legend()
        plt.title(f'Radial density vs dMOT ({beam.name})')
        plt.tight_layout()
        plt.savefig(out_folder + f'density_at_fib_T={T}uK.jpg')
        plt.clf()



if __name__ == "__main__":
    print(f'Analysis {beam.name} Beam.\nSaving imgs to {out_folder}\n\n')
    plot_cap_frac_vs_T(T_range, dMOT_range)
    plot_density_vs_T(T_range, dMOT_range)
    plot_cap_frac_vs_dMOT(T_range, dMOT_range)
    plot_density_vs_dMOT(T_range, dMOT_range)
    plot_3d_TdMOTconc()
