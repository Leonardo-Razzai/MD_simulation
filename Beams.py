import numpy as np
from Dynamics import *
import matplotlib.pyplot as plt

class GaussianBeam:
    """
    Gaussian optical beam (fundamental TEM00 mode).
    Provides dimensionless accelerations in (rho, zeta).
    """
    def __init__(self):
        self.I0 = 2 * P_b / (np.pi * w0**2)
        self.lambda_b = 1064e-9
        self.zR = zR_func(self.lambda_b)
        self.alpha = alpha_func(self.lambda_b)
        self.phi = np.abs( self.alpha * self.I0 / (m_Rb * w0**2) )
        self.tau = 1 / np.sqrt(self.phi)          # time scale ( 1 / sqrt(phi) is a time )
        self.vs_rho = w0 / self.tau               # radial velocity scale
        self.vs_zeta = self.zR / self.tau              # axial velocity scale
        self.acc_sc = self.lambda_b / np.pi * self.phi

    def beta(self, zeta):
        return 1 / (1 + zeta**2)
    
    def intensity(self, rho, zeta):
        b = self.beta(zeta)
        return b**2 * np.exp(-2 * b * rho**2)

    def du_drho(self, rho, zeta):
        b = self.beta(zeta)
        return -4 * b**2 * rho * np.exp(-2 * b * rho**2)

    def du_dzeta(self, rho, zeta):
        b = self.beta(zeta)
        pref = -4 * (self.lambda_b / (np.pi * w0))
        return pref * b * zeta * np.exp(-2 * b * rho**2) * (1 - 2*b*rho**2)

    def acc(self, x):
        rho, zeta = x
        acc_rho = self.du_drho(rho, zeta)
        acc_zeta = self.du_dzeta(rho, zeta) - g/self.acc_sc
        return np.array([acc_rho, acc_zeta])
    
    def plot_trans_intensity(self):
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        x, y = np.meshgrid(x, y)

        # Filled contour plot
        rho = np.sqrt(x**2 + y**2)
        cp = plt.contourf(x, y, self.intensity(rho, 0), levels=100, cmap="viridis")  # note the .T
        # Add colorbar
        plt.colorbar(cp, label="Norm. Intensity")

        plt.xlabel(r"x ($w_0$)")
        plt.ylabel(r"y ($w_0$)")
        plt.title("Intensity contour plot Gaussian Beam")

    def plot_long_intensity(self, x=np.linspace(-2, 2, 100), z=np.linspace(0, 4, 100)):
        
        x, z = np.meshgrid(x, z)

        # Filled contour plot
        rho = np.abs(x)
        cp = plt.contourf(x, z, self.intensity(rho, z), levels=100, cmap="inferno", alpha=0.2)  # note the .T
        # Add colorbar
        plt.colorbar(cp, label="Norm. Intensity")

        plt.xlabel(r"x ($w_0$)")
        plt.ylabel(r"y ($w_0$)")
        plt.title("Long. Intensity contour plot Gaussian Beam")

class LGBeamL1:
    """
    Laguerre–Gaussian donut beam (p=0, ℓ=1).
    Provides dimensionless accelerations in (rho, zeta).
    """
    def __init__(self):
        self.I0 = 4 * P_b / (np.pi * w0**2)
        self.lambda_b = 650e-9
        self.zR = zR_func(self.lambda_b)
        self.alpha = alpha_func(self.lambda_b)
        self.phi = np.abs( self.alpha * self.I0 / (m_Rb * w0**2) )
        self.tau = 1 / np.sqrt(self.phi)          # time scale ( 1 / sqrt(phi) is a time )
        self.vs_rho = w0 / self.tau               # radial velocity scale
        self.vs_zeta = self.zR / self.tau              # axial velocity scale
        self.acc_sc = self.lambda_b / np.pi * self.phi

    def beta(self, zeta):
        return 1 / (1 + zeta**2)

    def intensity(self, rho, zeta):
        """
        Dimensionless intensity I(rho,zeta) for ℓ=1.
        """
        b = self.beta(zeta)
        s = 2 * b * rho**2
        return 2 * b * s * np.exp(-s)

    def du_drho(self, rho, zeta):
        """
        Radial derivative of the dipole potential (dimensionless).
        ℓ=1.
        """
        
        rho = rho + 1e-5 * (rho > 0) - 1e-5 * (rho < 0) 
        I = self.intensity(rho, zeta)
        b = self.beta(zeta)
        s = 2 * b * rho**2
        return I * (2/rho) * (s - 1)

    def du_dzeta(self, rho, zeta):
        """
        Axial derivative of the dipole potential (dimensionless).
        ℓ=1.
        """
        I = self.intensity(rho, zeta)
        b = self.beta(zeta)
        s = 2 * b * rho**2
        return 4 * (self.lambda_b / (np.pi * w0)) * I * 2 * zeta * b * (2 - s)

    def acc(self, x):
        rho, zeta = x
        acc_rho = self.du_drho(rho, zeta)
        acc_zeta = self.du_dzeta(rho, zeta) - g/self.acc_sc
        return np.array([acc_rho, acc_zeta])
    
    def plot_trans_intensity(self, x=np.linspace(-2, 2, 100), y=np.linspace(-2, 2, 100)):
        
        x, y = np.meshgrid(x, y)

        # Filled contour plot
        rho = np.sqrt(x**2 + y**2)
        cp = plt.contourf(x, y, self.intensity(rho, 0), levels=100, cmap="viridis")  # note the .T
        # Add colorbar
        plt.colorbar(cp, label="Norm. Intensity")

        plt.xlabel(r"x ($w_0$)")
        plt.ylabel(r"y ($w_0$)")
        plt.title("Intensity contour plot LG Beam")
    
    def plot_long_intensity(self, x=np.linspace(-2, 2, 100), z=np.linspace(0, 4, 100)):
        
        x, z = np.meshgrid(x, z)

        # Filled contour plot
        rho = np.abs(x)
        cp = plt.contourf(x, z, self.intensity(rho, z), levels=100, cmap="inferno", alpha=0.2)  # note the .T
        # Add colorbar
        plt.colorbar(cp, label="Norm. Intensity")

        plt.xlabel(r"x ($w_0$)")
        plt.ylabel(r"y ($w_0$)")
        plt.title("Long. Intensity contour plot LG Beam")

if __name__ == "__main__":
    beam = LGBeamL1()
    beam.plot_trans_intensity()
    plt.show()

    beam.plot_long_intensity()
    plt.show()

    beam = GaussianBeam()
    beam.plot_trans_intensity()
    plt.show()

    beam.plot_long_intensity()
    plt.show()