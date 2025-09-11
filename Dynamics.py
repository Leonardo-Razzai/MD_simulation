import math
import numpy as np

data_folder = './Results/'
pos_fname = 'position.npy'
vel_fname = 'velocity.npy'
time_fname = 'time.npy'

def print_physical_constants(
    c, kB, g,
    lambda_b, P_b, R_trap,
    m_Rb, Gamma_1, Gamma_2, omega_1, omega_2,
    omega, Delta_1, Delta_2, k,
    w0, zR, I0, phi, tau, acc_sc
):
    """
    Print all physical constants, trap parameters, Rb constants,
    and derived scales used in the MOT simulation.
    """

    print("\n=== PHYSICAL CONSTANTS ===")
    print(f"Speed of light (c): {c:.2e} m/s")
    print(f"Boltzmann constant (kB): {kB:.2e} J/K")
    print(f"Gravity (g): {g:.2e} m/s²")

    print("\n=== TRAP PARAMETERS ===")
    print(f"Laser wavelength (lambda_b): {lambda_b:.2e} m")
    print(f"Beam power (P_b): {P_b:.2e} W")
    print(f"Fiber dimension (R_trap): {R_trap:.2e} m")
    print(f"Beam waist (w0): {w0:.2e} m (length scale along radial coordinate)")
    print(f"Rayleigh length (zR): {zR:.2e} m (length scale along axial coordinate)")
    print(f"Intensity scale (I0): {I0:.2e} W/m²")

    print("\n=== RUBIDIUM CONSTANTS ===")
    print(f"Rb-87 mass (m_Rb): {m_Rb:.2e} kg")
    print(f"Gamma_1: {Gamma_1:.2e} Hz")
    print(f"Gamma_2: {Gamma_2:.2e} Hz")
    print(f"Transition frequency omega_1: {omega_1:.2e} Hz")
    print(f"Transition frequency omega_2: {omega_2:.2e} Hz")

    print("\n=== DERIVED QUANTITIES ===")
    print(f"Laser angular frequency (omega): {omega:.2e} rad/s")
    print(f"Detuning Delta_1: {Delta_1:.2e} rad/s")
    print(f"Detuning Delta_2: {Delta_2:.2e} rad/s")
    print(f"Coupling constant (k): {k:.2e}")

    print("\n=== SCALES ===")
    print(f"phi: {phi:.2e}")
    print(f"tau (time scale): {tau:.2e} s")
    print(f"acc_sc (acceleration scale): {acc_sc:.2e}")

    print("\n==============================\n")

# Physical constants
c = 299792458                   # light vel in vacuum (m/s)
kB = 1.38064852E-23             # Boltzmann constant (J/K)
g = 9.81                         # Grav. Acc. (m/s^2)

#Trap
lambda_b = 1064E-9              # Beam wavelength (m)
P_b = 2                         # power beam (W)
R_trap = 30E-6                  # Fiber dimension (m)

# Rb
m_Rb = 87*1.66053904E-27        # Mass Rb 87
Gamma_1 = 2 * np.pi * 6.065e6 # Natural linewidth in Hz
Gamma_2 = 2 * np.pi * 5.746e6
omega_1 = 2 * np.pi * 377.1075 * 1e12 # Transition frequency 5S1/2 -> 5P3/2 in Hz
omega_2 = 2 * np.pi * 384.2305 * 1e12 # Transition frequency 5S1/2 -> 5S3/2  in Hz
omega = 2 * np.pi * c / lambda_b
Delta_1 = omega - omega_1  # Detuning from the transition frequency
Delta_2 = omega - omega_2  # Detuning from the transition frequency
k = - np.pi * c**2 * (Gamma_1/(Delta_1*omega_1**3) + 2 * Gamma_2/(Delta_2*omega_2**3))

# scales
w0 = 19E-6                      # length scale along rho ( Beam waist at fiber tip )
zR = math.pi * w0**2/lambda_b   # length scale along z ( Rayleigh length )
I0 = 2 * P_b / (np.pi * w0**2)     # intensity scale

phi = k * I0 / (m_Rb * w0**2)
tau = 1 / np.sqrt(phi)          # time scale ( 1 / sqrt(phi) is a time )
acc_sc = lambda_b / np.pi * phi

def w(z: float):
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

def beta(zeta: np.ndarray):
    """
    Dimensionless beam divergence factor.

    Parameters
    ----------
    zeta : float or np.ndarray
        Axial coordinate (dimensionless, normalized to Rayleigh range zR).

    Returns
    -------
    beta : float or np.ndarray
        Divergence factor, defined as:
            beta(zeta) = 1 / (1 + zeta^2).

    Notes
    -----
    - Encodes the z-dependence of the beam waist:
          w(z) = w0 * sqrt(1 + zeta^2).
    - Appears in the scaling of the Gaussian optical dipole potential.
    """

    return 1 / (1 + zeta**2)

def du_drho(rho: np.ndarray, zeta: np.ndarray):
    """
    Radial derivative of the dimensionless optical potential.

    Parameters
    ----------
    rho : float or np.ndarray
        Radial coordinate (dimensionless, in units of w0).
    zeta : float or np.ndarray
        Axial coordinate (dimensionless, in units of zR).

    Returns
    -------
    dUdrho : float or np.ndarray
        Radial derivative ∂U/∂ρ (dimensionless).

    Notes
    -----
    - Expression:
        dU/dρ = -4 * beta(zeta)^2 * rho * exp[-2 * beta(zeta) * rho^2]
    - Drives atoms toward the fiber axis (ρ = 0).
    """
    return - 4 * beta(zeta)**2 * rho * np.exp(-2 * beta(zeta) * rho**2)

def du_dzeta(rho: np.ndarray, zeta: np.ndarray):
    """
    Axial derivative of the dimensionless optical potential.

    Parameters
    ----------
    rho : float or np.ndarray
        Radial coordinate (dimensionless).
    zeta : float or np.ndarray
        Axial coordinate (dimensionless).
    
    Returns
    -------
    dUdzeta : float or np.ndarray
        Axial derivative ∂U/∂ζ (dimensionless).

    Notes
    -----
    - Expression:
        dU/dζ = - 4 * (λ / (π w0)) * beta(zeta) * zeta 
                * exp[-2 * beta(zeta) * rho^2] * (1 - 2*beta(zeta)*rho^2)
    - Contains both axial intensity gradient and radial correction.
    - λ is the trapping wavelength (here Nd:YAG 1064 nm).
    - w0 is the mode field radius of the fiber mode.
    """
    return - 4 * (lambda_b/(np.pi * w0)) * beta(zeta) * zeta * np.exp(- 2 * beta(zeta) * rho**2) * (1 - 2*beta(zeta)*rho**2)

def acc(x: np.ndarray):
    """
    Compute full acceleration vector at given position.

    Parameters
    ----------
    x : np.ndarray, shape (2,) or (2, N)
        Position vector(s):
        - x[0] = rho (dimensionless radial coordinate).
        - x[1] = zeta (dimensionless axial coordinate).

    Returns
    -------
    a : np.ndarray, shape (2,) or (2, N)
        Acceleration vector:
        - a[0] = radial acceleration (dimensionless).
        - a[1] = axial acceleration (dimensionless).

    Notes
    -----
    - Radial acceleration:
        aρ = du_drho(rho, zeta).
    - Axial acceleration:
        aζ = du_dzeta(rho, zeta) - g / acc_sc.
    - The term -g/acc_sc accounts for gravity along z.
      Here, `acc_sc` is the chosen acceleration scaling unit.
    """
    rho, zeta = x
    acc_rho = du_drho(rho, zeta)
    acc_zeta = du_dzeta(rho, zeta) - g / acc_sc

    return np.array([acc_rho, acc_zeta])

if __name__ == '__main__':
    print_physical_constants(
        c, kB, g,
        lambda_b, P_b, R_trap,
        m_Rb, Gamma_1, Gamma_2, omega_1, omega_2,
        omega, Delta_1, Delta_2, k,
        w0, zR, I0, phi, tau, acc_sc
    )