import math
import numpy as np

data_folder = './Results/'
pos_fname = 'position.npy'
vel_fname = 'velocity.npy'
time_fname = 'time.npy'

def print_physical_constants(
    c, kB, g,
    P_b, R_trap,
    m_Rb, Gamma_1, Gamma_2, omega_1, omega_2, w0):

    """
    Print all physical constants, trap parameters, Rb constants,
    and derived scales used in the MOT simulation.
    """

    print("\n=== PHYSICAL CONSTANTS ===")
    print(f"Speed of light (c): {c:.2e} m/s")
    print(f"Boltzmann constant (kB): {kB:.2e} J/K")
    print(f"Gravity (g): {g:.2e} m/s²")

    print("\n=== TRAP PARAMETERS ===")
    print(f"Beam power (P_b): {P_b:.2e} W")
    print(f"Fiber dimension (R_trap): {R_trap:.2e} m")
    print(f"Beam waist (w0): {w0:.2e} m (length scale along radial coordinate)")

    print("\n=== RUBIDIUM CONSTANTS ===")
    print(f"Rb-87 mass (m_Rb): {m_Rb:.2e} kg")
    print(f"Gamma_1: {Gamma_1:.2e} Hz")
    print(f"Gamma_2: {Gamma_2:.2e} Hz")
    print(f"Transition frequency omega_1: {omega_1:.2e} Hz")
    print(f"Transition frequency omega_2: {omega_2:.2e} Hz")

    print("\n==============================\n")

# Physical constants
c = 299792458                   # light vel in vacuum (m/s)
kB = 1.38064852E-23             # Boltzmann constant (J/K)
g = 9.81                         # Grav. Acc. (m/s^2)
hbar = 1.05457182e-34           # reduced Planck constant (J*s)

#Trap
P_b = 1                         # power beam (W)
R_trap = 30E-6                  # Fiber dimension (m)

# Rb
m_Rb = 87*1.66053904E-27        # Mass Rb 87
Gamma_1 = 2 * np.pi * 6.065e6 # Natural linewidth in Hz
Gamma_2 = 2 * np.pi * 5.746e6
omega_1 = 2 * np.pi * 377.1075 * 1e12 # Transition frequency 5S1/2 -> 5P3/2 in Hz
omega_2 = 2 * np.pi * 384.2305 * 1e12 # Transition frequency 5S1/2 -> 5S3/2  in Hz

def alpha_func(lambda_b):
    omega = 2 * np.pi * c / lambda_b
    Delta_1 = omega - omega_1  # Detuning from the transition frequency
    Delta_2 = omega - omega_2  # Detuning from the transition frequency
    alpha = - np.pi * c**2 * (Gamma_1/(Delta_1*omega_1**3) + 2 * Gamma_2/(Delta_2*omega_2**3))
    return alpha

w0 = 19E-6                      # length scale along rho ( Beam waist at fiber tip )
def zR_func(lambda_b):
    # scales
    zR = np.pi * w0**2/lambda_b   # length scale along z ( Rayleigh length )
    return zR

def w(z: float, zR: float):
    """
    Gaussian beam radius.

    Parameters
    ----------
    z : float
        Axial coordinate.

    Returns
    -------
    beta : float
        Gaussian beam radius at z:
            w(z) = w0 * np.sqrt(1 + (z/zR)**2).
    """
    return w0 * np.sqrt(1 + (z/zR)**2)

if __name__ == '__main__':
    print_physical_constants(
        c, kB, g,
        P_b, R_trap,
        m_Rb, Gamma_1, Gamma_2, omega_1, omega_2,
        w0
    )