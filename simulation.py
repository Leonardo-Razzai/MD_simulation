from Dynamics import *
from Verlet import *
import numpy as np
import os
from Beams import GaussianBeam, LGBeamL1

# MOT characteristics
RMOT = 1e-3 # m
VMOT = 4/3 * np.pi * RMOT**3
dMOT_max = 20e-3 # m
h_max = dMOT_max + RMOT
R_cil = 4e-4
V_cil = 2 * np.pi * RMOT * R_cil**2

T_MAX = 80e-3
N_steps = int(4e3)
DT = T_MAX / N_steps

N_save = 30 # number of saved steps
DT_save = DT * N_save

# FLAGS
Diff_Powers = False

def print_simulation_parameters(
    N, T, dMOT, RMOT, w0, zR, tau,
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
    print(f"Temperature (T): {T*1e6:.2f} uK")
    print(f"Number of atoms (N): {N:.2e}")
    print(f"MOT displacement (dMOT): {dMOT*1e3:.2f} mm")
    print(f"MOT radius (RMOT): {RMOT*1e3:.2f} mm")

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
    print(f"t_max: {t_max*1e3:.2f} (ms)")
    print(f"dt: {dt*1e6:.2f} (us)")
    print(f"N_steps: {N_steps}")

    print("\n==============================\n")


def simulation(N=int(1e5), T=15, dMOT=5, beam=GaussianBeam()):
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

    zR = beam.zR
    vs_rho = beam.vs_rho
    vs_zeta = beam.vs_zeta
    tau = beam.tau

    # initial positions
    z_max = dMOT + RMOT
    z_min = dMOT - RMOT
    zeta_max = z_max / zR
    zeta_min = z_min / zR

    zeta_0 = np.random.uniform(zeta_min, zeta_max, size=N)

    rho_max = h_max / zR # in units of w0
    rho_0 = np.random.uniform(-rho_max, rho_max, size=N)

    x0 = np.array([rho_0, zeta_0])

    # initial velocities
    alpha = m_Rb / (2 * kB * T)
    sigma_rho = np.sqrt(1 / (2*alpha)) / vs_rho
    sigma_zeta = np.sqrt(1 / (2*alpha)) / vs_zeta
    v_rho_0 = np.random.normal(loc = 0, scale = sigma_rho, size = N)
    v_zeta_0 = np.random.normal(loc = 0, scale = sigma_zeta, size = N)

    v0 = np.array([v_rho_0, v_zeta_0])

    # Time and Num
    dt = DT / tau

    # Call this after defining constants in your script
    if __name__=='__main__':
        print_simulation_parameters(
            N=N, T=T, dMOT=dMOT, RMOT=RMOT, t_max=T_MAX,
            w0=w0, zR=zR, tau=tau, m_Rb=m_Rb, kB=kB,
            rho_max=rho_max, zeta_min=zeta_min, zeta_max=zeta_max,
            dt=DT, N_steps=N_steps
        )

    xres, vres = evolve_up_to(x0, v0, beam.acc, dt, N_steps, z_min=10)
    res = verlet(xres, vres, beam.acc, dt, N_steps)
    save_data(res, T, dMOT, N, zR, tau, beam)

def evolve_up_to(x0, v0, acc, dt, N_steps, z_min=5):
    res = verlet_up_to(x0, v0, acc, dt, N_steps, z_min=z_min)
    return res

def save_data(res, T, dMOT, N, zR, tau, beam=GaussianBeam()):
    """
    Save raw simulation results and main parameters to disk.

    Parameters
    ----------
    res : tuple
        Output of `verlet` (xs, vs, ts).
    T : float
        Temperature [K].
    dMOT : float
        MOT–fiber distance [m].
    N : float
        Number of atoms inside the cylindrical volume.

    Notes
    -----
    - Results are saved in a folder named
      `res_T={T}uK_dMOT={dMOT}mm/` inside `data_folder/`.
    - Arrays saved:
        * positions.npy : atom trajectories
        * velocities.npy : atom velocities
        * times.npy : time steps
    - Parameters saved:
        * parameters.txt : human-readable file containing
          the main simulation parameters and constants.
    """

    if Diff_Powers:
        res_folder = data_folder + f'{beam.name}/Different_Powers/res_T={T*1e6:.0f}uK_dMOT={dMOT*1e3:.0f}mm_P={P_b}W/'
    else:
        res_folder = data_folder + f'{beam.name}/res_T={T*1e6:.0f}uK_dMOT={dMOT*1e3:.0f}mm/'

    os.makedirs(res_folder, exist_ok=True)

    # Save arrays
    iterator = trange(0, 3, desc="Saving", mininterval=1.0)

    f_names = [pos_fname, vel_fname, time_fname]

    idx = np.linspace(0, len(res[0])-1, N_save, dtype=int)

    for i in iterator:
        small_res = res[i]
        small_res = small_res[idx]
        np.save(res_folder + f_names[i], small_res)

    # Save parameters in a human-readable text file
    param_file = res_folder + "parameters.txt"
    with open(param_file, "w") as f:
        f.write("=== SIMULATION PARAMETERS ===\n")
        f.write(f"Temperature (T): {T*1e6:.2f} uK\n")
        f.write(f"MOT displacement (dMOT): {dMOT*1e3:.2f} mm\n")
        f.write(f"MOT radius (RMOT): {RMOT*1e3:.2f} mm\n")
        f.write(f"Simulation Time: {T_MAX:.2f} s\n")
        f.write(f"Num. of Atoms in simulation (N): {N:.3e}\n")
        f.write(f"Beam: {beam.name}\n")
        f.write(f"Wavelength: {beam.lambda_b * 1e9} nm\n")
        f.write(f"Power: {P_b} W\n\n")

        f.write("--- Constants ---\n")
        f.write(f"w0: {w0:.3e} m\n")
        f.write(f"zR: {zR:.3e} m\n")
        f.write(f"tau: {tau:.3e} s\n")
        f.write(f"m_Rb: {m_Rb:.3e} kg\n")
        f.write(f"kB: {kB:.3e} J/K\n")

    print(f"Parameters saved in {param_file}")


if __name__ == '__main__':
    from sys import argv

    if len(argv) < 3:
        print('Specify T and dMOT')
        exit()

    T = int(argv[1])
    dMOT = int(argv[2])
    beam_name = argv[3]

    if beam_name == 'LG':
        beam = LGBeamL1()
    elif beam_name == 'Gauss':
        beam = GaussianBeam()

    simulation(N=int(1e5), T=T, dMOT=dMOT, beam=beam)
