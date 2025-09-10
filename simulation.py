from Dynamics import *
from Verlet import *
import numpy as np
import os

def print_simulation_parameters(
    N, T, dMOT, RMOT, MOT_t, w0, zR, tau,
    m_Rb, kB, rho_max, zeta_min, zeta_max, t_max, dt, N_steps
):
    """
    Print the main simulation parameters and derived quantities.
    """

    # Velocity scales
    vs_rho = w0 / tau
    vs_zeta = zR / tau
    alpha = m_Rb / (2 * kB * T)
    v_bar = np.sqrt(np.pi / alpha)

    print("\n=== SIMULATION PARAMETERS ===")
    print(f"Temperature (T): {T:.2e} K")
    print(f"Number of atoms (N): {N}")
    print(f"MOT displacement (dMOT): {dMOT} mm")
    print(f"MOT radius (RMOT): {RMOT} mm")
    print(f"MOT duration (MOT_t): {MOT_t} s")

    print("\n--- Initial positions ---")
    print(f"rho_max: {rho_max:.2e} [w0 units] (r_max = {rho_max * w0:.2e} m)")
    print(f"zeta_min: {zeta_min:.2e} [zR units] (z_min = {zeta_min * zR:.2e} m)")
    print(f"zeta_max: {zeta_max:.2e} [zR units] (z_max = {zeta_max * zR:.2e} m)")

    print("\n--- Velocity scales ---")
    print(f"vs_rho: {vs_rho:.2e} m/s")
    print(f"vs_zeta: {vs_zeta:.2e} m/s")
    print(f"alpha: {alpha:.2e} (s/m)**2")
    print(f"v_bar: {v_bar:.2e} m/s")

    print("\n--- Time discretization ---")
    print(f"t_max: {t_max:.2e} [normalized units]")
    print(f"dt: {dt:.2e} [normalized units]")
    print(f"N_steps: {N_steps}")

    print("\n==============================\n")


def simulation(N=int(1e2), T=15, dMOT=5):
    """
    Run a full atom trajectory simulation.

    Parameters
    ----------
    N : int, optional
        Number of atoms (default: 1e2 for test, 1e5 recommended).
    T : float
        Temperature [µK].
    dMOT : float
        MOT–fiber distance [mm].

    Returns
    -------
    None
        Runs the integration, saves results in `./data/`.

    Notes
    -----
    - Positions initialized uniformly within MOT sphere.
    - Velocities drawn from thermal Maxwell-Boltzmann distribution.
    - Calls `verlet` integrator from `Dynamics.py`.
    - Results are saved with filenames based on T and dMOT.
    """

    # SIMULATION PARAMETERS
    N = int(N) # num of atoms
    
    # MOT
    T = T * 1e-6 # K
    dMOT = dMOT * 1e-3 # m
    RMOT = 1e-3 # m
    MOT_t = 0.1 # s

    # initial positions

    z_max = dMOT + RMOT
    z_min = dMOT - RMOT
    zeta_max = z_max / zR
    zeta_min = z_min / zR

    zeta_0 = np.random.uniform(zeta_min, zeta_max, size=N)

    rho_max = 0.25 * zeta_max # in units of w0
    rho_0 = np.random.uniform(-rho_max, rho_max, size=N)

    x0 = np.array([rho_0, zeta_0])

    # initial velocities
    vs_rho = w0 / tau
    vs_zeta = zR / tau
    alpha = m_Rb / (2 * kB * T)
    v_bar = np.sqrt(np.pi / alpha)

    v_rho_0 = np.random.normal(loc = 0, scale = 1 / (2*alpha), size = N) * vs_rho / v_bar
    v_zeta_0 = np.random.normal(loc = 0, scale = 1 / (2*alpha), size = N) * vs_zeta / v_bar 

    v0 = np.array([v_rho_0, v_zeta_0])

    # Time and Num
    t_max = MOT_t / tau / 100
    dt = t_max / 1e3
    N_steps = int(t_max / dt)

    # Call this after defining constants in your script
    if __name__=='__main__':
        print_simulation_parameters(
            N=N, T=T, dMOT=dMOT, RMOT=RMOT, MOT_t=MOT_t,
            w0=w0, zR=zR, tau=tau, m_Rb=m_Rb, kB=kB,
            rho_max=rho_max, zeta_min=zeta_min, zeta_max=zeta_max,
            t_max=t_max, dt=dt, N_steps=N_steps
        )

    res = verlet(x0, v0, acc, dt, N_steps)
    save_data(res, T, dMOT)

def save_data(res, T, dMOT):
    """
    Save raw simulation results to disk.

    Parameters
    ----------
    res : tuple
        Output of `verlet` (xs, vs, ts).
    T : float
        Temperature [K].
    dMOT : float
        MOT–fiber distance [m].

    Notes
    -----
    - Creates result folder if not existing.
    - Saves positions, velocities, times as `.npy` arrays.
    """

    res_folder = data_folder + f'res_T={T*1e6:.0f}uK_dMOT={dMOT*1e3:.0f}mm/'

    if os.path.exists(res_folder) - 1 :
        os.mkdir(res_folder)

    np.save(res_folder + pos_fname, res[0])
    np.save(res_folder + vel_fname, res[1])
    np.save(res_folder + time_fname, res[2])

if __name__ == '__main__':
    from sys import argv

    if len(argv) < 3:
        print('Specify T and dMOT')
        exit()

    T = int(argv[1])
    dMOT = int(argv[2])

    simulation(N=int(1e5), T=T, dMOT=dMOT)
