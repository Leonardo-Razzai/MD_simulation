from Dynamics import *
from Beams import GaussianBeam, LGBeamL1

def R_sc(r_atoms: np.ndarray, beam=GaussianBeam()):
    """
    Calculate the photon scattering rate for a given laser intensity and wavelength.

    This function computes the scattering rate `R_sc` for a two-level (or multilevel) atom 
    interacting with a laser field. The rate depends on the intensity of the laser, 
    the laser wavelength, and the detuning from the atomic transition frequencies.

    Parameters
    ----------
    Beam

    Returns
    -------
    Rsc : float
        Photon scattering rate in s⁻¹.

    Notes
    -----
    - The calculation uses detunings from two possible transition frequencies 
      (`omega_1` and `omega_2`), each with associated linewidths (`Gamma_1`, `Gamma_2`).
    - The constants `c`, `hbar`, `omega_1`, `omega_2`, `Gamma_1`, and `Gamma_2`
      must be defined in the same scope or imported globally.
    """
    I = beam.intensity(r_atoms[0], r_atoms[1])

    omega = 2 * np.pi * c / beam.lambda_b
    Delta_1 = omega - omega_1  # Detuning from first transition
    Delta_2 = omega - omega_2  # Detuning from second transition
    Rsc = np.abs(np.pi * c**2 / (2 * hbar)) * (
        Gamma_1**2 / (Delta_1**2 * omega_1**3) +
        2 * Gamma_2**2 / (Delta_2**2 * omega_2**3)
    ) * I
    return Rsc

def R_sc_avg(r_atoms: np.ndarray, beam=GaussianBeam()):
    """
    Calculate the photon scattering rate for a given laser intensity and wavelength.

    This function computes the scattering rate `R_sc` for a two-level (or multilevel) atom 
    interacting with a laser field. The rate depends on the intensity of the laser, 
    the laser wavelength, and the detuning from the atomic transition frequencies.

    Parameters
    ----------
    Beam

    Returns
    -------
    Rsc : float
        Photon scattering rate in s⁻¹.

    Notes
    -----
    - The calculation uses detunings from two possible transition frequencies 
      (`omega_1` and `omega_2`), each with associated linewidths (`Gamma_1`, `Gamma_2`).
    - The constants `c`, `hbar`, `omega_1`, `omega_2`, `Gamma_1`, and `Gamma_2`
      must be defined in the same scope or imported globally.
    """
    I = np.mean(beam.intensity(r_atoms[0], r_atoms[1]))

    omega = 2 * np.pi * c / beam.lambda_b
    Delta_1 = omega - omega_1  # Detuning from first transition
    Delta_2 = omega - omega_2  # Detuning from second transition
    Rsc = np.abs(np.pi * c**2 / (2 * hbar)) * (
        Gamma_1**2 / (Delta_1**2 * omega_1**3) +
        2 * Gamma_2**2 / (Delta_2**2 * omega_2**3)
    ) * I
    return Rsc


def dsigma_v(r_atoms: np.ndarray, dt: float, beam=GaussianBeam()):
    """
    Calculate the velocity spread due to photon recoil over a time interval.

    This function estimates the change in velocity spread (`Δσ_v`) of atoms 
    resulting from random photon scattering events over a given time interval.

    Parameters
    ----------
    dt : float
        Time interval over which scattering occurs, in seconds.
    I : float
        Laser intensity in W/m^2.
    Beam : Beam class
        Laser beam (Guassian or LG).

    Returns
    -------
    float
        Velocity spread increment (Δσ_v) in m/s.

    Notes
    -----
    - The calculation is based on the recoil velocity from photon scattering 
      and assumes isotropic emission.
    - Depends on the scattering rate `R_sc(I, lambda_b)`.
    - The constants `hbar`, `m_Rb`, and others used in `R_sc` must be defined 
      in the same scope or globally.
    """
    k_b = 2 * np.pi / beam.lambda_b
    return hbar * k_b / m_Rb * np.sqrt(R_sc_avg(r_atoms, beam) * dt * beam.tau / 2) # m/s

def AddScattering(r_atoms, v_atoms, dt, beam=GaussianBeam()):

    # --- Compute scattering rate per atom ---
    R = R_sc(r_atoms, beam)  # [1/s]
    N_exp = R * dt * beam.tau                         # expected number of scatterings (mean)

    # --- Sample actual scattering counts from Poisson distribution ---
    N_sc = np.random.poisson(N_exp)                   # shape (N,), integer per atom

    # --- Physical parameters ---
    k_b = 2 * np.pi / beam.lambda_b                   # wavevector magnitude
    dv = hbar * k_b / m_Rb                            # single-photon recoil velocity

    # --- Prepare arrays ---
    N = len(r_atoms[0])
    M = np.max(N_sc)                                  # max number of scatters among atoms
    theta = np.random.uniform(0, 2 * np.pi, size=(N, M))  # random emission directions

    # --- Build index mask: only first N_sc[i] columns are True per atom ---
    index = np.tile(np.arange(1, M + 1), (N, 1)) <= N_sc[:, None]

    # --- Compute cumulative recoil components ---
    dv_rho  = np.sum(dv * np.sin(theta) * index, axis=1) / beam.vs_rho
    dv_zeta = np.sum(dv * np.cos(theta) * index, axis=1) / beam.vs_zeta

    # --- Update atomic velocities ---
    new_velocity = np.copy(v_atoms)
    new_velocity += np.array([dv_rho, dv_zeta])

    return new_velocity


def GetTemperature(v_atoms, beam = GaussianBeam()):
    "Return temperature in K"
    std_v_rho = np.std(v_atoms[0]) * beam.vs_rho
    std_v_zeta = np.std(v_atoms[1]) * beam.vs_zeta
    T_rho = std_v_rho**2 * m_Rb / kB
    T_zeta = std_v_zeta**2 * m_Rb / kB

    return T_rho, T_zeta
