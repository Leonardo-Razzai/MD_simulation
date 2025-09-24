import numpy as np

class GaussianBeam:
    """
    Gaussian optical beam (fundamental TEM00 mode).
    Provides dimensionless accelerations in (rho, zeta).
    """
    def __init__(self, w0, lambda_b, g=9.81, acc_sc=1.0):
        self.w0 = w0
        self.lambda_b = lambda_b
        self.g = g
        self.acc_sc = acc_sc
        self.zR = np.pi * w0**2 / lambda_b

    def beta(self, zeta):
        return 1 / (1 + zeta**2)

    def du_drho(self, rho, zeta):
        b = self.beta(zeta)
        return -4 * b**2 * rho * np.exp(-2 * b * rho**2)

    def du_dzeta(self, rho, zeta):
        b = self.beta(zeta)
        pref = -4 * (self.lambda_b / (np.pi * self.w0))
        return pref * b * zeta * np.exp(-2 * b * rho**2) * (1 - 2*b*rho**2)

    def acc(self, x):
        rho, zeta = x
        acc_rho = self.du_drho(rho, zeta)
        acc_zeta = self.du_dzeta(rho, zeta) - self.g/self.acc_sc
        return np.array([acc_rho, acc_zeta])


class LGBeamL1:
    """
    Laguerre–Gaussian donut beam (p=0, ℓ=1).
    Provides dimensionless accelerations in (rho, zeta).
    """
    def __init__(self, w0, lambda_b, g=9.81, acc_sc=1.0):
        self.w0 = w0
        self.lambda_b = lambda_b
        self.g = g
        self.acc_sc = acc_sc
        self.zR = np.pi * w0**2 / lambda_b

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
        if rho < 1e-5:
            return 0.0
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
        return I * (2*zeta)/(1+zeta**2) * (2 - s)

    def acc(self, x):
        rho, zeta = x
        acc_rho = self.du_drho(rho, zeta)
        acc_zeta = self.du_dzeta(rho, zeta) - self.g/self.acc_sc
        return np.array([acc_rho, acc_zeta])
