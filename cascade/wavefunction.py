import json

import mpmath

import numpy as np

import os

import scipy.special as scp
import scipy.integrate as spi

import time

from typing import Optional


class WaveFunction:

    def __init__(self, Z: int = 1, mu: float = 1.0, rmin: Optional[float] = None, rmax: float = 50.0, dx: float = 0.005):
        """
        Initialize the WaveFunction object with given parameters.
        
        Parameters:
        Z (int): Atomic number.
        mu (float): Reduced mass.
        rmin (float, optional): Minimum radial distance. Defaults to 1e-6 / (Z * mu).
        rmax (float): Maximum radial distance.
        dx (float): Step size for the radial grid.
        """
        if rmin is None:
            rmin = 1e-6 / (Z * mu)
        
        self.Z = Z
        self.mu = mu
        self.rmin = rmin
        self.rmax = rmax
        self.dx = dx
        
        # Calculate the logarithmic grid
        xmin = np.log(Z * rmin)
        xmax = np.log(Z * rmax)

        x_i = np.arange(xmin, xmax, dx)
        self.r_i = np.exp(x_i) / Z

        self.ngpts = self.r_i.shape[0]
        
        # Determine the package directory
        package_dir = os.path.dirname(__file__)
        # Path to save the JSON file within the data directory
        self.radial_integrals_path = os.path.join(package_dir, 'data', 'radial_integrals.json')

    def associated_laguerre(self, n: int, l: int, rho: np.ndarray) -> np.ndarray:
        """
        Calculate the associated Laguerre polynomial.

        Parameters:
        n (int): Principal quantum number.
        l (int): Azimuthal quantum number.
        rho (np.ndarray): Radial variable.

        Returns:
        np.ndarray: Evaluated Laguerre polynomial.
        """
        return scp.genlaguerre(n, l)(rho)

    def calculate_radial_integral(self, nmax=40):

        r_i = self.r_i

        # Initialize the nested list to hold radial integrals
        radial_integrals = []

        start = time.time()

        # Fill the nested list with computed radial integrals
        for ni in np.arange(nmax + 1):
            ni_list = []  # List for the current ni
            for li in np.arange(ni):
                li_list = []  # List for the current li
                for nf in np.arange(ni):
                    nf_list = []  # List for the current nf
                    for p, lf in enumerate([li-1, li+1]):
                        
                        if nf >= ni or lf < 0 or lf >= nf or ni == 0:
                            nf_list.append(None)  # Append None for invalid cases
                            continue

                        ui_i = self.get_discrete_wf(n=ni, l=li) * r_i
                        uf_i = self.get_discrete_wf(n=nf, l=lf) * r_i

                        radial = self.integrate(ui_i * uf_i * r_i, r_i)

                        nf_list.append(radial)

                    li_list.append(nf_list)
                ni_list.append(li_list)
            radial_integrals.append(ni_list)

        radial_integrals[0] = None

        end = time.time()

        print(f'Time for RI: {end-start:.3f} s')

        return radial_integrals
    
    def get_continuum_wf(self, k: float, l: int = 0, maxterms: int = 1000) -> np.ndarray:
        """
        Calculate the continuum wavefunction for given parameters.

        Parameters:
        k (float): Wave number.
        l (int): Azimuthal quantum number.
        maxterms (int): Maximum terms for hypergeometric series.

        Returns:
        np.ndarray: Continuum wavefunction values.
        """
        Z = self.Z
        r_i = self.r_i
        Rkl_i = np.zeros(self.ngpts, dtype=complex)
        
        prefactor = 2**(l+1) / mpmath.factorial(2*l + 1)
        exponent = np.exp(np.pi * Z / (2 * k))
        gamma_term = abs(mpmath.gamma(l + 1 + 1j * Z / k))
        
        for i, r in enumerate(r_i):
            hyper_term = mpmath.hyp1f1(l + 1 + 1j * Z / k, 2*l + 2, 2 * 1j * k * r, maxterms=maxterms)
            R = prefactor * exponent * k**(l + 0.5) * gamma_term * r**l * np.exp(-1j * k * r) * hyper_term
            Rkl_i[i] = complex(R)
        
        return Rkl_i

    def get_discrete_wf(self, n: int = 1, l: int = 0) -> np.ndarray:
        """
        Calculate the radial wavefunction for given quantum numbers.

        Parameters:
        n (int): Principal quantum number.
        l (int): Azimuthal quantum number.

        Returns:
        np.ndarray: Radial wavefunction values.
        """
        Z = self.Z
        mu = self.mu
        r_i = self.r_i
        
        rho_i = 2 * Z * mu * r_i / n

        # Radial wavefunction
        normalization = self.get_normalization(Z, mu, n, l)
        exponential = np.exp(-rho_i / 2)
        polynomial = rho_i**l
        laguerre = self.associated_laguerre(n - l - 1, 2 * l + 1, rho_i)

        Rnl_i = normalization * exponential * polynomial * laguerre

        return Rnl_i
    
    def get_normalization(self, Z: float, mu: float, n: int, l: int) -> float:
        """
        Computes the normalization factor for a wavefunction using a numerically stable method.

        Parameters:
        ----------
        Z : float
            Atomic number or scaling factor.
        mu : float
            Reduced mass or relevant factor.
        n : int
            Principal quantum number.
        l : int
            Angular momentum quantum number.

        Returns:
        -------
        float
            The computed normalization factor. Returns `np.nan` if calculation fails.
        """
        log_term1 = 3 * np.log(2 * Z * mu / n)
        log_term2 = scp.loggamma(n - l)
        log_term3 = scp.loggamma(n + l + 1)
        log_normalization = 0.5 * (log_term1 + log_term2 - log_term3 - np.log(2 * n))
        
        # Exponentiate the result to get the normalization
        normalization = np.exp(log_normalization)
        return normalization

    def load_radial_integral(self):

        radial_integrals_path = self.radial_integrals_path

        # Load the radial integrals from the JSON file
        if not os.path.exists(radial_integrals_path):
            raise FileNotFoundError(f"Radial integrals file not found at: {file_path}")

        with open(radial_integrals_path, "r") as f:
            radial_integrals = json.load(f)

        return radial_integrals
    
    def integrate(self, y_i: np.ndarray = 1, x_i: np.ndarray = 1, method: str = 'simps') -> float:
        """
        Integrate the given function using the specified method.

        Parameters:
        y_i (np.ndarray): Function values to integrate.
        x_i (np.ndarray): Grid where to perform the integration.
        method (str): Integration method ('simps' for Simpson's rule).

        Returns:
        float: Result of the integration.
        """

        if method == 'simps':
            I = spi.simpson(y_i, x=x_i)
        
        return I

    def save_radial_integrals(self, nmax=40, rmax=3000):

        self.rmax = rmax
        radial_integrals = self.calculate_radial_integral(nmax)

        # Save the radial integrals to a JSON file
        with open(self.radial_integrals_path, "w") as f:
            json.dump(radial_integrals, f, indent=4)
