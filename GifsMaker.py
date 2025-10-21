import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from Beams import *    

def MakeGif_density(pos: np.ndarray, density: np.ndarray, beam=GaussianBeam(), file_name='density_vs_time'):
    """
    Create a GIF animation of atomic density over time.

    Parameters:
    ----------
    file_name : str, optional
        Name of the output GIF file.
    """
    rho_array, zeta_array = pos

    R, Z = np.meshgrid(rho_array * w0 * 1e3, zeta_array * beam.zR * 1e3)

    fig, ax = plt.subplots(figsize=(5, 10))
    img = ax.imshow(density[0], extent=[R.min(), R.max(), Z.min(), Z.max()],
                    origin='lower', cmap='viridis', aspect='auto')
    
    # # overlay beam intensity
    # rho_dim = R / (w0 * 1e3)
    # zeta_dim = Z / (beam.zR * 1e3)
    # I = beam.intensity(rho_dim, zeta_dim)
    # I = I / I.max()
    # ax.contour(R, Z, I, levels=30, cmap='inferno', alpha=0.3)

    ax.set_title(f'Density and Intensity distribution ({beam.name})')
    ax.set_xlabel(r'$\rho$ (mm)')
    ax.set_ylabel('z (mm)')
    ax.set_ylim(0, Z.max())

    fig.colorbar(img, ax=ax, label="Atomic Density")

    def update(i):
        img.set_data(density[i])
        return [img]

    ani = FuncAnimation(fig, update, frames=int(0.7*len(density)), interval=100)
    ani.save('./gifs/'+file_name+'.gif', writer='pillow', fps=10)
    plt.close(fig)