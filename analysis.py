from Dynamics import *
import matplotlib.pyplot as plt
import os

def capt_atoms_vs_t(T, dMOT):
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

    res_folder = data_folder + f'res_T={T:.0f}uK_dMOT={dMOT:.0f}mm/'

    if os.path.exists(res_folder):
        try:
            xs = np.load(res_folder + pos_fname)
            ts = np.load(res_folder + time_fname)

            r_cap = R_trap / w0 # trap radius in units of w0
            n_cap = np.sum((xs[:, 1, :] <= 0) & (np.abs(xs[:, 0, :]) < r_cap), axis=1)

            return ts, n_cap / len(xs[0, 0, :])
        
        except Exception as err:
            print(f"Error: {err=}, {type(err)=}")
            exit()
    else:
        print(f'No simualtion was run with T={T}uK and dMOT={dMOT}mm')
        exit()

def density_at_fib(step, T, dMOT):
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

    res_folder = data_folder + f'res_T={T:.0f}uK_dMOT={dMOT:.0f}mm/'

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
                return np.array([0]), np.array([0])
            
        except Exception as err:
            print(f"Error: {err=}, {type(err)=}")
            exit()
    else:
        print(f'No simualtion was run with T={T}uK and dMOT={dMOT}mm')
        exit()

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
    step_bins = hist_rho_step[1]

    step_widths = np.diff(step_bins)

    plt.bar(
        step_bins[:-1], hist_rho_step[0],
        width=step_widths, align='edge',
        color=color, alpha=0.8, label=label
    )

    plt.title('Distribution of radial position at fiber')
    plt.xlabel(r'$\rho$ $(w_0)$')
    plt.ylabel('Probability density')
    plt.legend()


if __name__ == '__main__':

    from sys import argv

    if len(argv) < 3:
        print('Specify T and dMOT')
        exit()

    T = int(argv[1])
    dMOT = int(argv[2])

    print(f'T = {T} uK, dMOT = {dMOT} mm')

    ts, f_cap = capt_atoms_vs_t(T, dMOT)
    plot_cap_frac(ts, f_cap)
    plt.show()

    hist_rho_step, hist_rho_init = density_at_fib(step=-1, T=T, dMOT=dMOT)
    plot_initial_density_rho(hist_rho_init)
    plot_density_at_fib(hist_rho_step=hist_rho_step)
    plt.show()

