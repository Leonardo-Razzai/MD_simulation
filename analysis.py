from Dynamics import *
from simulation import VMOT, V_cil, RMOT
import matplotlib.pyplot as plt
import os
from Beams import GaussianBeam, LGBeamL1

BEAMS = [GaussianBeam(), LGBeamL1()]

def  data_fname(T, dMOT, beam=GaussianBeam()):
    return f'{beam.name}/res_T={T:.0f}uK_dMOT={dMOT:.0f}mm_beam={beam.name}/'

def compute_NMOT(N):
    """
    Compute the effective number of atoms in the cylindrical fiber volume.

    Parameters
    ----------
    N : int
        Total number of atoms in the MOT.

    Returns
    -------
    NMOT : float
        Effective number of atoms inside the cylindrical fiber volume.

    Notes
    -----
    - Uses global constants:
        * VMOT : volume of the MOT sphere
        * V_cil : effective cylindrical fiber volume
    - Formula:
        NMOT = N * VMOT / V_cil
    """
    return N * VMOT / V_cil


def capt_atoms_vs_t(T, dMOT, beam=GaussianBeam()):
    """
    Compute fraction of captured atoms vs time.

    Parameters
    ----------
    T : float
        MOT temperature [µK].
    dMOT : float
        MOT–fiber distance [mm].

    Returns
    -------
    ts : ndarray
        Time points (dimensionless).
    f_cap : ndarray
        Fraction of captured atoms at each time.

    Notes
    -----
    An atom is considered captured if:
    - ζ <= 0 (at or below fiber tip),
    - |ρ| < R_trap / w0 (within fiber mode radius).
    """

    res_folder = data_folder + data_fname(T, dMOT, beam)

    if os.path.exists(res_folder):
        try:
            xs = np.load(res_folder + pos_fname)
            ts = np.load(res_folder + time_fname)

            r_cap = R_trap / w0 # trap radius in units of w0
            n_cap = np.sum((xs[:, 1, :] <= 0) & (np.abs(xs[:, 0, :]) < r_cap), axis=1)

            N = len(xs[0, 0, :])
            NMOT = compute_NMOT(N)

            return ts, n_cap / NMOT
        
        except Exception as err:
            print(f"Error: {err=}, {type(err)=}")
            exit()
    else:
        print(f'No simualtion was run with T={T}uK and dMOT={dMOT}mm')
        exit()

def get_last_conc(T, dMOT, beam=GaussianBeam()):
    """
    Compute the final fraction of atoms captured at the fiber.

    Parameters
    ----------
    T : float
        MOT temperature [µK].
    dMOT : float
        MOT–fiber distance [mm].

    Returns
    -------
    last_conc : float
        Fraction of atoms captured at the final simulation step.

    Notes
    -----
    - Uses `capt_atoms_vs_t` to compute the time evolution of the captured fraction.
    - Returns the last value of the capture fraction array.
    - The result is normalized by the effective number of atoms inside the fiber volume.
    """
    _, conc = capt_atoms_vs_t(T, dMOT, beam)
    last_conc = conc[-1]
    return last_conc


def density_at_fib(step, T, dMOT, beam=GaussianBeam()):
    """
    Compute radial density distribution of atoms at the fiber.

    Parameters
    ----------
    step : int
        Time step index (-1 for final distribution).
    T : float
        MOT temperature [µK].
    dMOT : float
        MOT–fiber distance [mm].

    Returns
    -------
    hist_rho_step : tuple
        Histogram (counts, bins) for atoms at fiber at given step.
    hist_rho_init : tuple
        Histogram (counts, bins) for initial MOT distribution.
    """

    res_folder = data_folder + data_fname(T, dMOT, beam)

    if os.path.exists(res_folder):
        try:
            xs = np.load(res_folder + pos_fname)
            
            rho_step = xs[step, 0, :]
            zeta_step = xs[step, 1, :]
            index_at_fib = zeta_step <= 0

            rho_at_fib = rho_step[index_at_fib]

            rho_init = xs[0, 0, :]

            if len(rho_at_fib) > 0:
                hist_rho_step = np.histogram(rho_at_fib, int(np.sqrt(len(rho_at_fib))), density=True)
                hist_rho_init = np.histogram(rho_init, int(np.sqrt(len(rho_init))), density=True)

                return hist_rho_step, hist_rho_init
            else:
                return None, None
            
        except Exception as err:
            print(f"Error: {err=}, {type(err)=}")
            exit()
    else:
        print(f'No simualtion was run with T={T}uK and dMOT={dMOT}mm')
        exit()

def z_density(step, T, dMOT, beam=GaussianBeam()):
    """
    Compute axial density distribution of atoms at a given step.

    Parameters
    ----------
    step : int
        Time step index (-1 for final distribution).
    T : float
        MOT temperature [µK].
    dMOT : float
        MOT–fiber distance [mm].

    Returns
    -------
    hist_zeta_step : tuple
        Histogram (counts, bins) for atomic axial positions at given step.
    hist_zeta_init : tuple
        Histogram (counts, bins) for initial MOT distribution.
    """

    res_folder = data_folder + data_fname(T, dMOT, beam)

    if os.path.exists(res_folder):
        try:
            xs = np.load(res_folder + pos_fname)
            
            zeta_step = xs[step, 1, :]
            zeta_init = xs[0, 1, :]

            hist_zeta_step = np.histogram(zeta_step, int(np.sqrt(len(zeta_step))), density=True)
            hist_zeta_init = np.histogram(zeta_init, int(np.sqrt(len(zeta_init))), density=True)
            
            return hist_zeta_step, hist_zeta_init
        
        except Exception as err:
            print(f"Error: {err=}, {type(err)=}")
            exit()
    else:
        print(f'No simualtion was run with T={T}uK and dMOT={dMOT}mm')
        exit()

def density(T, dMOT, beam, rho_min: float, rho_max: float, 
            zeta_min: float, zeta_max: float, step=-1):


    """
    Compute the 2D spatial density histogram of atoms from simulation data.

    Parameters
    ----------
    T : float
        Temperature of the simulation in microkelvin (uK).
    dMOT : float
        MOT displacement in millimeters (mm).
    rho_min : float
        Minimum value of the radial coordinate (rho) for the histogram.
    rho_max : float
        Maximum value of the radial coordinate (rho) for the histogram.
    zeta_min : float
        Minimum value of the axial coordinate (zeta) for the histogram.
    zeta_max : float
        Maximum value of the axial coordinate (zeta) for the histogram.
    step : int, optional
        Time step index to use from the simulation data (default is -1, the last step).

    Returns
    -------
    n : ndarray
        2D array of histogram counts for each (rho, zeta) bin. Shape is (len(zeta_centers), len(rho_centers)).
    rho_centers : ndarray
        1D array of bin center positions along the rho axis.
    zeta_centers : ndarray
        1D array of bin center positions along the zeta axis.

    Notes
    -----
    The function expects simulation results to be saved in a folder
    with the naming convention: 'res_T={T}uK_dMOT={dMOT}mm/'.
    If the folder or file does not exist, the function exits.
    """

    res_folder = data_folder + data_fname(T, dMOT, beam)

    if os.path.exists(res_folder):
        try:
            xs = np.load(res_folder + pos_fname)
            rho_atoms = xs[step, 0, :]
            zeta_atoms = xs[step, 1, :]
        except Exception as err:
            print(f"Error: {err=}, {type(err)=}")
            exit()
    else:
        print(f'No simulation was run with T={T}uK and dMOT={dMOT}mm')
        exit()

    # Define bin edges
    rho_array = np.linspace(rho_min, rho_max, 501)  # 100 bins
    zeta_array = np.linspace(zeta_min, zeta_max, 501)

    # Compute 2D histogram (counts in each bin)
    n, rho_edges, zeta_edges = np.histogram2d(
        rho_atoms, zeta_atoms, bins=[rho_array, zeta_array]
    )

    # For plotting, use bin centers instead of edges
    rho_centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
    zeta_centers = 0.5 * (zeta_edges[:-1] + zeta_edges[1:])

    return n.T, rho_centers, zeta_centers

def plot_cap_frac(ts, f_cap, label='Fraction Captured', color='royalblue'):
    """
    Plot fraction of captured atoms vs time.

    Parameters
    ----------
    ts : ndarray
        Time points (dimensionless).
    f_cap : ndarray
        Fraction of captured atoms.
    label : str
        Plot label.
    color : str
        Curve color.
    """

    plt.plot(ts, f_cap*100, label=label, color=color)
    plt.title('Fraction of atoms captured at the fiber')
    plt.xlabel(r'Time ($\tau$)')
    plt.ylabel('Atoms captured (%)')

def plot_initial_density_rho(hist_rho_init):
    """
    Plot initial radial distribution of MOT atoms.

    Parameters
    ----------
    hist_rho_init : tuple
        Histogram of initial ρ distribution (counts, bins).
    """

    # Bin centers
    init_bins = hist_rho_init[1]

    init_widths = np.diff(init_bins)

    plt.bar(
        init_bins[:-1], hist_rho_init[0],
        width=init_widths, align='edge',
        color='blue', alpha=0.8, label='Initial distribution'
    )

    plt.title('Initial radial distribution of atomic positions')
    plt.xlabel(r'$\rho$ $(w_0)$')
    plt.ylabel('Probability density')
    plt.legend()

def plot_density_at_fib(hist_rho_step, label='Distribution at the fiber', color='red'):
    """
    Plot radial distribution of captured atoms at fiber.

    Parameters
    ----------
    hist_rho_step : tuple
        Histogram of ρ for atoms at fiber.
    label : str
        Plot label.
    color : str
        Bar color.
    """
    # Bin centers
    if hist_rho_step:
        step_bins = hist_rho_step[1]

        step_widths = np.diff(step_bins)

        plt.bar(
            step_bins[:-1], hist_rho_step[0],
            width=step_widths, align='edge',
            color=color, alpha=0.8, label=label
        )
    else:
        plt.plot([],[])

    plt.title('Distribution of radial position at fiber')
    plt.xlabel(r'$\rho$ $(w_0)$')
    plt.ylabel('Probability density')
    plt.legend()

def plot_initial_density_zeta(hist_zeta_init):
    """
    Plot initial axial distribution of MOT atoms.

    Parameters
    ----------
    hist_zeta_init : tuple
        Histogram of initial ρ distribution (counts, bins).
    """

    # Bin centers
    init_bins = hist_zeta_init[1]

    init_widths = np.diff(init_bins)

    plt.bar(
        init_bins[:-1], hist_zeta_init[0],
        width=init_widths, align='edge',
        color='blue', alpha=0.8, label='Initial distribution'
    )

    plt.title('Initial axial distribution of atomic positions')
    plt.xlabel(r'$\zeta$ $(z_R)$')
    plt.ylabel('Probability density')
    plt.legend()

def plot_density_zeta(hist_zeta_step, label='Distribution of axial positions', color='red'):
    """
    Plot axial distribution at the given step.

    Parameters
    ----------
    hist_zeta_step : tuple
        Histogram of ρ for atoms at fiber.
    label : str
        Plot label.
    color : str
        Bar color.
    """
    # Bin centers
    step_bins = hist_zeta_step[1]

    step_widths = np.diff(step_bins)

    plt.bar(
        step_bins[:-1], hist_zeta_step[0],
        width=step_widths, align='edge',
        color=color, alpha=0.8, label=label
    )
    plt.ylim(0, 1)
    plt.title('Distribution of radial position at fiber')
    plt.xlabel(r'$z$ $(z_R)$')
    plt.ylabel('Probability density')
    plt.legend()

def plot_density_zeta_vs_t(steps: list, T, dMOT, beam):

    """
    Plot axial distribution at given steps.

    Parameters
    ----------
    steps : list
        Steps at which computing the distributions.
    T : float
        MOT temperature [µK].
    dMOT : float
        MOT–fiber distance [mm].
    """

    from matplotlib import colormaps
    cmap = colormaps.get_cmap('inferno')
    colors = [cmap(x) for x in np.linspace(0.1, 0.8, len(steps))]

    for i, step in enumerate(steps):
        hist_zeta_step, _ = z_density(step, T, dMOT, beam)
        plot_density_zeta(hist_zeta_step, label=f'step = {step}', color=colors[i])

    plt.title('Axial position distribution at different times')

def plot_density_rho_vs_t(steps: list, T, dMOT, beam):
    """
    Plot axial distribution at given steps.

    Parameters
    ----------
    steps : list
        Steps at which computing the distributions.
    T : float
        MOT temperature [µK].
    dMOT : float
        MOT–fiber distance [mm].
    """

    from matplotlib import colormaps
    cmap = colormaps.get_cmap('inferno')
    colors = [cmap(x) for x in np.linspace(0.1, 0.8, len(steps))]

    for i, step in enumerate(steps):
        hist_rho_step, _ = density_at_fib(step, T, dMOT, beam)
        plot_density_at_fib(hist_rho_step, label=f'step = {step}', color=colors[i])

    plt.title('Distribution of atoms at fiber at different times')

def plot_density(n, rho_array, zeta_array, beam=GaussianBeam()):
    
    """
    Plot a 2D density contour of atomic distribution.

    Parameters
    ----------
    n : ndarray
        2D array of histogram counts (output of `density` function).
    rho_array : ndarray
        1D array of radial bin centers.
    zeta_array : ndarray
        1D array of axial bin centers.

    Returns
    -------
    None
        The function displays a contour plot of the density using matplotlib.

    Notes
    -----
    The radial (r) and axial (z) coordinates are converted to millimeters
    using `w0` and `zR` respectively. The density is plotted using a
    filled contour plot with 50 levels and a 'viridis' colormap.
    """
    
    # atomic density contour
    R, Z = np.meshgrid(rho_array * w0 * 1e3, zeta_array * beam.zR * 1e3)

    import matplotlib as mpl

    fig, ax = plt.subplots(figsize=(8,6))

    # density background
    cp = ax.contourf(R, Z, n, levels=50, cmap="viridis")
    fig.colorbar(cp, ax=ax, label="Atomic Density")

    # beam intensity (normalized)
    rho_dim = R / (w0 * 1e3)
    zeta_dim = Z / (beam.zR * 1e3)
    I = beam.intensity(rho_dim, zeta_dim)
    I = I / I.max()

    # overlay with alpha
    cmap = plt.cm.inferno
    cf = ax.contourf(R, Z, I, levels=50, cmap=cmap, alpha=0.1)

    ax.set_title(f'Atom and Intesity distribution ({beam.name})')
    ax.set_xlabel(r'$\rho$ (mm)')
    ax.set_ylabel('z (mm)')

    # make a mappable for the colorbar with opaque colors
    sm = mpl.cm.ScalarMappable(norm=cf.norm, cmap=cmap)
    sm.set_array([])  
    fig.colorbar(sm, ax=ax, label="Beam intensity")

if __name__ == '__main__':

    from sys import argv

    if len(argv) < 3:
        print('Specify T, dMOT, Beam (Gauss or LG)')
        exit()

    T = int(argv[1])
    dMOT = int(argv[2])
    beam = str(argv[3])

    print(f'T = {T} uK, dMOT = {dMOT} mm, beam = {beam}')

    for b in BEAMS:
        if b.name == beam:
            chosen_beam = b

    ts, f_cap = capt_atoms_vs_t(T, dMOT, beam=chosen_beam)
    plot_cap_frac(ts, f_cap)
    plt.show()

    hist_rho_step, hist_rho_init = density_at_fib(step=-1, T=T, dMOT=dMOT, beam=chosen_beam)
    plot_initial_density_rho(hist_rho_init)
    plot_density_at_fib(hist_rho_step=hist_rho_step)
    plt.show()

    plot_density_zeta_vs_t([0, 5, 10, 12, 15], T, dMOT, beam=chosen_beam)
    plt.show()

    print(f'Percentage of atoms at the fiber: {get_last_conc(T, dMOT, chosen_beam)*100:.2f} %')

    n, rho_array, zeta_array = density(T, dMOT, beam=chosen_beam, rho_min=-RMOT/w0, rho_max=RMOT/w0, zeta_min=0, zeta_max=5, step=12)
    plot_density(n, rho_array, zeta_array, beam=chosen_beam)
    plt.show()
