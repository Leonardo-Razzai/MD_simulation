import numpy as np
from Heating  import *
from Beams import *

# from tqdm import trange

def Heating_1(xs, vs, dt, i, beam=GaussianBeam()):
    sigma_i_rho, sigma_i_zeta = np.std(vs[i], axis=1) # std across atoms
    r_atoms = xs[i]
    dsigma_i = dsigma_v(r_atoms, dt, beam)

    sigma_i_rho = sigma_i_rho * beam.vs_rho
    sigma_i_zeta = sigma_i_zeta * beam.vs_zeta
    
    new_sigma_rho = (sigma_i_rho + dsigma_i) / beam.vs_rho
    new_sigma_zeta = (sigma_i_zeta + dsigma_i) / beam.vs_zeta
    
    vs[i, 0, :] = vs[i, 0, :] + np.random.normal(0, scale=new_sigma_rho, size=len(vs[i, 0, :]))
    vs[i, 1, :] = vs[i, 1, :] + np.random.normal(0, scale=new_sigma_zeta, size=len(vs[i, 0, :]))

def Heating_2(xs, vs, dt, i, beam=GaussianBeam()):
    r_atoms = xs[i]
    v_atoms = vs[i]
    new_velocity = AddScattering(r_atoms, v_atoms, dt, beam) 
    return new_velocity

def verlet(x0, v0, a_func, dt, steps, beam=GaussianBeam(), HEATING=False, progress=True):
    """
    Integrate atomic motion using the velocity-Verlet scheme.

    The integrator evolves atomic trajectories in dimensionless units under the
    acceleration field provided by `a_func`. This implementation assumes that
    the optical potential is zero for ζ < 0 (atoms past the fiber tip).

    Parameters
    ----------
    x0 : (2,) array_like or (2, N) ndarray
        Initial positions in dimensionless units:
        - x[0] = ρ (radial coordinate in w0 units)
        - x[1] = ζ (axial coordinate in zR units).
        Supports single-particle or multi-particle arrays.
    v0 : (2,) array_like or (2, N) ndarray
        Initial velocities in dimensionless units.
    a_func : callable
        Function of the form `a_func(x)` returning acceleration components
        (aρ, aζ) at position `x`. Must support vectorized input.
    dt : float
        Time step (dimensionless units).
    steps : int
        Number of integration steps.
    progress : bool, optional
        If True, display a tqdm progress bar (default: True).

    Returns
    -------
    xs : ndarray, shape (steps+1, 2, N)
        Atomic positions at each step.
    vs : ndarray, shape (steps+1, 2, N)
        Atomic velocities at each step.
    ts : ndarray, shape (steps+1,)
        Dimensionless time values at each step.

    Notes
    -----
    - Uses a Taylor expansion for the first step.
    - Enforces boundary condition: motion only for ζ > 0.
    - Velocities are estimated via central differences.
    """

    xs = np.zeros((steps+1,) + np.shape(x0))
    vs = np.zeros((steps+1,) + np.shape(v0))
    ts = np.zeros(steps+1)

    xs[0] = x0
    vs[0] = v0
    ts[0] = 0.0

    # First step using Taylor expansion
    a0 = a_func(xs[0])
    xs[1] = xs[0] + v0*dt + 0.5*a0*dt**2

    # Loop with optional progress bar
    iterator = range(steps)
    for i in iterator:
        t = (i+1)*dt
        ts[i+1] = t

        z = xs[i, 1]
        update = z > 0

        # acceleration at current step
        a = a_func(xs[i])

        # Verlet position update
        xs[i+1] = xs[i] + (xs[i] - xs[i-1] + a*dt**2) * update

        # Velocity (estimated with central difference)
        vs[i] = (xs[i+1] - xs[i-1]) / (2*dt) * update

        if HEATING:
           vs[i] = Heating_2(xs, vs, dt, i, beam)

    # Last velocity estimation
    vs[-1] = (xs[-1] - xs[-2]) / dt * update

    return xs, vs, ts

def verlet_up_to(x0, v0, a_func, dt, steps, z_min=5, HEATING=False):
    """
    Integrate atomic motion using the velocity-Verlet scheme.

    The integrator evolves atomic trajectories in dimensionless units under the
    acceleration field provided by `a_func`. This implementation assumes that
    the optical potential is zero for ζ < 0 (atoms past the fiber tip).

    Parameters
    ----------
    x0 : (2,) array_like or (2, N) ndarray
        Initial positions in dimensionless units:
        - x[0] = ρ (radial coordinate in w0 units)
        - x[1] = ζ (axial coordinate in zR units).
        Supports single-particle or multi-particle arrays.
    v0 : (2,) array_like or (2, N) ndarray
        Initial velocities in dimensionless units.
    a_func : callable
        Function of the form `a_func(x)` returning acceleration components
        (aρ, aζ) at position `x`. Must support vectorized input.
    dt : float
        Time step (dimensionless units).
    steps : int
        Number of integration steps.
    progress : bool, optional
        If True, display a tqdm progress bar (default: True).

    Returns
    -------
    xs : ndarray, shape (steps+1, 2, N)
        Atomic positions at each step.
    vs : ndarray, shape (steps+1, 2, N)
        Atomic velocities at each step.
    ts : ndarray, shape (steps+1,)
        Dimensionless time values at each step.

    Notes
    -----
    - Uses a Taylor expansion for the first step.
    - Enforces boundary condition: motion only for ζ > 0.
    - Velocities are estimated via central differences.
    """

    xs = np.zeros((steps+1,) + np.shape(x0))
    vs = np.zeros((steps+1,) + np.shape(v0))

    xs[0] = x0
    vs[0] = v0

    # First step using Taylor expansion
    a0 = a_func(xs[0])
    xs[1] = xs[0] + v0*dt + 0.5*a0*dt**2
    z = np.max(x0[1])

    i=1
    while z > z_min:
        xstep = xs[i]
        z = np.min(xstep[1])

        # acceleration at current step
        a = a_func(xs[i])

        # Verlet position update
        xs[i+1] = xs[i] + (xs[i] - xs[i-1] + a*dt**2)

        # Velocity (estimated with central difference)
        vs[i] = (xs[i+1] - xs[i-1]) / (2*dt)

        if HEATING:
            vs[i] = Heating_2(xs, vs, dt, i, beam)

        i+=1
    # Last velocity estimation
    vs[-1] = (xs[-1] - xs[-2]) / dt
    
    return xs[i-1], vs[i-1]
