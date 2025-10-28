import numpy as np
import matplotlib.pyplot as plt
from Dynamics import *

class Beam:
    """
    Base class for optical beams.
    Defines shared initialization, scaling, and plotting utilities.
    """

    def __init__(self, name, P_b=1, lambda_b=1064e-9, w0_b=19e-6):
        self.name = name
        self.P_b = P_b
        self.lambda_b = lambda_b
        self.w0_b = w0_b
        self.update_props()

    def update_props(self):
        """Update derived optical and scaling properties."""
        self.I0 = 4 * self.P_b / (np.pi * self.w0_b**2)
        self.zR = zR_func(self.lambda_b)
        self.alpha = alpha_func(self.lambda_b)
        self.phi = np.abs(self.alpha * self.I0 / (m_Rb * self.w0_b**2))
        self.tau = 1 / np.sqrt(self.phi)
        self.vs_rho = self.w0_b / self.tau
        self.vs_zeta = self.zR / self.tau
        self.as_rho = self.w0_b / self.tau**2
        self.as_zeta = self.zR / self.tau**2

    # --- setters ---
    def Set_Power(self, P_b: float):
        self.P_b = P_b
        self.update_props()

    def Set_Lambda(self, lambda_b: float):
        self.lambda_b = lambda_b
        self.update_props()

    def Set_w0(self, w0_b: float):
        self.w0_b = w0_b
        self.update_props()

    # --- generic beam geometry ---
    def w(self, z: float):
        """Beam radius as function of zeta (dimensionless)."""
        return np.sqrt(1 + z**2)
    
    def beta(self, zeta):
        return 1 / (1 + zeta**2)

    # --- plotting utilities ---
    def plot_trans_intensity(self, x=np.linspace(-2, 2, 100), y=np.linspace(-2, 2, 100)):
        x, y = np.meshgrid(x, y)
        rho = np.sqrt(x**2 + y**2)
        cp = plt.contourf(x, y, self.intensity(rho, 0), levels=100, cmap="viridis")
        plt.colorbar(cp, label="Norm. Intensity")
        plt.xlabel(r"x ($w_0$)")
        plt.ylabel(r"y ($w_0$)")
        plt.title(f"Intensity contour plot {self.name} Beam")

    def plot_long_intensity(self, x=np.linspace(-2, 2, 100), z=np.linspace(0, 4, 100)):
        x, z = np.meshgrid(x, z)
        rho = np.abs(x)
        cp = plt.contourf(x, z, self.intensity(rho, z), levels=100, cmap="inferno", alpha=0.7)
        plt.colorbar(cp, label="Norm. Intensity")
        plt.xlabel(r"x ($w_0$)")
        plt.ylabel(r"z ($z_R$)")
        plt.title(f"Longitudinal Intensity contour plot {self.name} Beam")


# -------------------------------------------------------------------------
# Subclasses implementing specific beam profiles
# -------------------------------------------------------------------------

class GaussianBeam(Beam):
    """
    Gaussian optical beam (fundamental TEM00 mode).
    Provides dimensionless accelerations in (rho, zeta).
    """

    def __init__(self, P_b=1, lambda_b=1064e-9, w0_b=19e-6):
        super().__init__("Gauss", P_b, lambda_b, w0_b)

    def intensity(self, rho, zeta):
        b = self.beta(zeta)
        return b * np.exp(-2 * b * rho**2)

    def du_drho(self, rho, zeta):
        b = self.beta(zeta)
        return 4 * b * rho * self.intensity(rho, zeta)

    def du_dzeta(self, rho, zeta):
        b = self.beta(zeta)
        pref = self.lambda_b**2 / (np.pi**2 * self.w0_b**2)
        return pref * 2 * b * zeta * (1 - 2 * b * rho**2) * self.intensity(rho, zeta)

    def acc(self, x):
        rho, zeta = x
        acc_rho = -self.du_drho(rho, zeta)
        acc_zeta = -self.du_dzeta(rho, zeta) - g / self.as_zeta
        return np.array([acc_rho, acc_zeta])


class LGBeamL1(Beam):
    """
    Laguerre–Gaussian donut beam (p=0, ℓ=1).
    Provides dimensionless accelerations in (rho, zeta).
    """

    def __init__(self, P_b=1, lambda_b=650e-9, w0_b=19e-6):
        super().__init__("LG", P_b, lambda_b, w0_b)

    def intensity(self, rho, zeta):
        b = self.beta(zeta)
        s = 2 * b * rho**2
        return 2 * b * s * np.exp(-s)

    def du_drho(self, rho, zeta):
        rho = rho + 1e-5 * np.sign(rho)
        I = self.intensity(rho, zeta)
        b = self.beta(zeta)
        s = 2 * b * rho**2
        return I * (2 / rho) * (s - 1)

    def du_dzeta(self, rho, zeta):
        I = self.intensity(rho, zeta)
        b = self.beta(zeta)
        s = 2 * b * rho**2
        prefix = self.lambda_b**2 / (np.pi**2 * self.w0_b**2)
        return prefix * I * 2 * zeta * b * (2 - s)

    def acc(self, x):
        rho, zeta = x
        acc_rho = self.du_drho(rho, zeta)
        acc_zeta = self.du_dzeta(rho, zeta) - g / self.as_zeta
        return np.array([acc_rho, acc_zeta])


# -------------------------------------------------------------------------
# Beam dictionary for easy lookup
# -------------------------------------------------------------------------
beams = {
    "Gauss": GaussianBeam(),
    "LG": LGBeamL1()
}

# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    for name, beam in beams.items():
        beam.plot_trans_intensity()
        plt.show()
        beam.plot_long_intensity()
        plt.show()
