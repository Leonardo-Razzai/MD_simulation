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
    ani.save(file_name+'.gif', writer='pillow', fps=3)
    plt.close(fig)



# def MakeGif_sol(self, file_name = 'cahn_hilliard.gif'):
#     """
#     Create a GIF animation of concentration field evolution.

#     Parameters:
#     ----------
#     file_name : str, optional
#         Name of the output GIF file.
#     """
#     N = len(x)
#     Nt = len(sol)
#     sol_to_plot = sol[0:Nt:step]
#     sol_to_plot.shape
    
#     fig, ax = plt.subplots(1, 2, figsize=(14, 6))

#     ax[0].plot([0, 400], [200, 200], color = 'darkorchid')
#     img = ax[0].imshow(sol_to_plot[0], cmap='inferno')
#     ax[0].set_title(f'Concentration in x-y plane')

#     ln2, = ax[1].plot(x, sol_to_plot[0][N//2], color='darkorchid')
#     ax[1].plot([np.min(x), np.max(x)], [1/np.sqrt(3), 1/np.sqrt(3)], '--', color='black', label='Spinodal')
#     ax[1].plot([np.min(x), np.max(x)], [-1/np.sqrt(3), -1/np.sqrt(3)], '--', color='black')
#     ax[1].set_title('Concentration at y=0', fontdict=title_font)
#     ax[1].set_ylabel('Conc', fontdict=base_font)
#     ax[1].set_xlabel('x (cm)', fontdict=base_font)
#     ax[1].set_ylim(-1, 1)
#     ax[1].set_xlim(x[0], x[-1])
#     ax[1].legend()

#     fig.suptitle(f'Time evolution of concentration D = {D:.2e}, a = {a:.2e}', fontdict=title_figure)

#     def update(i):
#         img.set_data(sol_to_plot[i])
#         ln2.set_data(x, sol_to_plot[i][N//2])


#     ani = FuncAnimation(fig, update, frames = len(sol_to_plot)-1)
#     ani.save(file_name, writer='pillow', fps= len(sol_to_plot)/20)
    
# def MakeGif_tot(self, file_name = 'cahn_hilliard.gif'):
#     """
#     Create a GIF animation of concentration and Fourier analysis over time.

#     Parameters:
#     ----------
#     file_name : str, optional
#         Name of the output GIF file.
#     """
#     N = len(x)
#     Nt = len(sol)
#     sol_to_plot = sol[0:Nt:step]
#     sol_to_plot.shape

#     fig = plt.figure(figsize=(15, 7))
#     grid = plt.GridSpec(4, 4, hspace=0.8, wspace=0.3)

#     # image
#     img_ax = fig.add_subplot(grid[:, :2])
#     img_ax.set_xticks([])
#     img_ax.set_yticks([])
#     img = img_ax.imshow(sol_to_plot[0], cmap='inferno')
#     img_ax.set_title(f'Concentration in x-y plane', fontdict=title_font)

#     # conc_x
#     concx_ax = fig.add_subplot(grid[:2, 2:])
#     histo = histo[0]
#     bins = histo[1]
#     counts = histo[0]
#     ln1, = concx_ax.plot(bins[:-1], counts / np.max(counts), color='darkorchid')
#     concx_ax.axvline(1/np.sqrt(3), ls='--', color='black', label='Spinodal')
#     concx_ax.axvline(-1/np.sqrt(3), ls='--', color='black')
#     concx_ax.set_title('Concentration distribution', fontdict=title_font)
#     concx_ax.set_ylabel('counts', fontdict=base_font)
#     concx_ax.set_xlabel('Conc', fontdict=base_font)
#     concx_ax.set_xlim(-1.2, 1.2)
#     concx_ax.set_ylim(0.05, 1.1)
#     concx_ax.legend(loc='upper left')

#     # ft_conc_x
#     ft_concx_ax = fig.add_subplot(grid[2:, 2:])
#     ft_concx_ax.set_ylim(0, np.max(density[:])+1)

#     dx = x[1] - x[0]
#     k = np.fft.fftfreq(N, dx) * 2*np.pi

#     ft_ln1, = ft_concx_ax.plot(k[:N//2], density[0][:N//2])

#     ft_concx_ax.axvline(1/(np.sqrt(a)), ls='--',color='red', label='Critical line\n'+r'1/$\sqrt{a}$')
#     ft_concx_ax.set_xlabel(r'k (cm$^{-1}$)', fontdict=base_font)
#     ft_concx_ax.set_ylabel('A(k, t)/A(k, 0)', fontdict=base_font)
#     ft_concx_ax.set_title('FT of concentration profile', fontdict=title_font)
#     ft_concx_ax.legend()

#     fig.suptitle('Time evolution of concentration', fontdict=title_figure)

#     def update(i):
#         img.set_data(sol_to_plot[i])
#         histo = histo[i]
#         bins = histo[1]
#         counts = histo[0]
#         ln1.set_data(bins[:-1], counts / np.max(counts))
#         ft_ln1.set_data(k[:N//2], density[i][:N//2])


#     ani = FuncAnimation(fig, update, frames = len(sol_to_plot)-1)
#     ani.save(file_name, writer='pillow', fps= len(sol_to_plot)/20)