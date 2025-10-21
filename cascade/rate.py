from math import pi

import numpy as np

from particle import Particle

from scipy.constants import alpha, eV, hbar, physical_constants

from typing import Optional

# Constants
mass_e = Particle.from_name(name='e-').mass
hartree = physical_constants['Hartree energy in eV'][0]
_aut = physical_constants['atomic unit of time'][0]

class Rate:
    """
    A class to calculate various rates and energy changes in an exotic atom.

    Parameters:
    -----------
    ExoticAtom : object
        An object representing the exotic atom, which must contain an 'exotic_particle'
        attribute with a 'lifetime' property.
    nmax : int, optional
        The maximum principal quantum number for calculations (default is 40).
    unit : float, optional
        The unit for the decay rate calculation (default is 1e-12).
    """

    def __init__(self, 
                 ExoticAtom: Optional[object] = None,
                 WaveFunction: Optional[object] = None,
                 nmax: int = 40, 
                 unit: float = 1e-12) -> None:
        
        if ExoticAtom is None:
            raise ValueError("ExoticAtom must be provided and cannot be None.")
        
        if not hasattr(ExoticAtom, 'exotic_particle'):
            raise AttributeError("ExoticAtom must have an 'exotic_particle' attribute.")
        
        if WaveFunction is not None:
            self.wf = WaveFunction
        
        self.ExoticAtom = ExoticAtom
        self.nmax = nmax
        self.unit = unit
        self.mu = self.ExoticAtom.get_reduced_mass()
        self.Z = self.ExoticAtom.Z

    def get_auger_energy(self, ni: int = 2, nf: int = 1, Ze: float = 1.0, nshell: int = 1) -> float:
        """
        Calculate the Auger energy for an electron transition between two energy levels.

        Parameters:
        -----------
        ni : int, optional, default=2
            Initial principal quantum number (energy level) of the electron.
        nf : int, optional, default=1
            Final principal quantum number (energy level) of the electron after the transition.
        Ze : float, optional, default=1.0
            Effective nuclear charge seen by the electron after the transition.
        nshell : int, optional, default=1
            Principal quantum number of the shell from which the Auger electron is emitted.

        Returns:
        --------
        float
            The calculated Auger energy for the given transition, in atomic units.
        """

        # Compute the energy difference for the transition
        delta_energy = self.get_delta_energy(ni, nf)
        # Calculate the exotic term
        exotic_term = (self.mu / mass_e) * self.Z**2 * delta_energy
        # Calculate the electron shell term
        electron_term = Ze**2 / (2 * nshell**2)
        
        # Return the Auger energy
        return exotic_term - electron_term

    def get_auger_probability(self, Ze: int = 1, y: float = 1.0, radial: float = 1.0, shell: str = '1s'):
        """
        Calculates the Auger probability for a given shell.

        Parameters:
            Ze (int): Effective nuclear charge.
            y (float): Dimensionless parameter related to energy.
            radial (float): Radial overlap factor.
            shell (str): Electron shell ('1s', '2s', '2p').

        Returns:
            float: Auger probability.

        References:
            AKYLAS, VR. "Cascade of negative muons in atoms, Ph. D. Thesis." (1978).
        """

        # Dictionary to store shell-specific parameters
        shell_params = {
            '1s': {
                'factor_coeff': 32 / 3,
                'ypoly_func': lambda y: np.power(y, 2) / (1 + np.power(y, 2)),
                'exponent_func': lambda y: np.exp(y * (4 * np.arctan(y) - pi)) / np.sinh(pi * y)
            },
            '2s': {
                'factor_coeff': 64 / 3,
                'ypoly_func': lambda y: np.power(y, 2) * (1 + np.power(y, 2)) / np.power(4 + np.power(y, 2), 2),
                'exponent_func': lambda y: np.exp(y * (4 * np.arctan(y / 2) - pi)) / np.sinh(pi * y)
            },
            '2p': {
                'factor_coeff': 2 / 3 * 8 / 6,
                'ypoly_func': lambda y: np.power(y, 4) * (12 + 11 * np.power(y, 2)) / np.power(4 + np.power(y, 2), 3),
                'exponent_func': lambda y: np.exp(y * (4 * np.arctan(y / 2) - pi)) / np.sinh(pi * y)
            }
        }

        if shell not in shell_params:
            raise ValueError(f"Unsupported shell '{shell}'. Supported shells: '1s', '2s', '2p'.")

        params = shell_params[shell]
        factor = params['factor_coeff'] * np.power(Ze / self.Z, 2) * pi * np.power(mass_e / self.mu, 2)
        ypoly = params['ypoly_func'](y)
        exponent = params['exponent_func'](y)

        P_auger = factor * radial**2 * ypoly * exponent / _aut * self.unit

        return P_auger
    
    def get_auger_rate(self, shell: str = '1s', Ze: float = 1):
        """
        Calculate the Auger decay rate matrix (Gamma_aug_nlmp) for different quantum states
        of the exotic atom.

        This method computes the Auger decay rates for transitions between different energy levels
        and orbital angular momentum states in the atom. The rates are calculated based on the radial
        integrals, the reduced mass of the system, and other physical constants.

        Returns:
        --------
        Gamma_aug_nlmp : list
            A nested list structure containing the Auger decay rates for different quantum numbers (n, l, m, p).
            The structure is organized as follows:
            - Gamma_aug_nlmp[ni][li][nf][p] gives the Auger rate for the transition from state (ni, li) to (nf, lf).
            - The first element (Gamma_aug_nlmp[0]) is set to None to handle the special case where ni = 0.
        """
        mu = self.ExoticAtom.get_reduced_mass()
        Z = self.ExoticAtom.Z
        nshell = int(shell[0])

        radial_integrals = self.wf.load_radial_integral()

        if radial_integrals is None:
            raise ValueError("Radial integrals could not be loaded.")

        Gamma_aug_nlmp = []

        print(f'Calcualting Auger Rate for shell {shell} n={nshell} and Ze={Ze}')

        # Fill the nested list with computed Auger rates
        for ni in np.arange(self.nmax + 1):
            ni_list = []  # List for the current ni
            for li in np.arange(ni):
                li_list = []  # List for the current li
                for nf in np.arange(ni):
                    nf_list = []  # List for the current nf
                    for p, lf in enumerate([li-1, li+1]):

                        if lf < 0 or lf >= nf:
                            nf_list.append(None)
                            continue

                        radial = radial_integrals[int(ni)][int(li)][int(nf)][p] * np.sqrt(li / (2*li + 1))
                        energy = self.get_auger_energy(ni, nf, Ze, nshell)

                        if energy > 0:
                            y = Ze / np.sqrt(2 * energy)
                            Gamma_aug = self.get_auger_probability(Ze, y, radial, shell)

                        else:
                            Gamma_aug = 0

                        nf_list.append(Gamma_aug)

                    li_list.append(nf_list)
                ni_list.append(li_list)
            Gamma_aug_nlmp.append(ni_list)

        Gamma_aug_nlmp[0] = None

        return Gamma_aug_nlmp

    def get_decay_rate(self) -> float:
        """
        Calculates the decay rate of the exotic particle in the atom.

        Returns:
        --------
        Gamma_decay : float
            The decay rate of the exotic particle, scaled by the unit factor.
        """

        if not hasattr(self.ExoticAtom.exotic_particle, 'lifetime'):
            raise AttributeError("The 'exotic_particle' attribute must have a 'lifetime' property.")
    
        # Convert exotic particle lifetime from nanoseconds to seconds
        lifetime_seconds = self.ExoticAtom.exotic_particle.lifetime * 1e-9

        # Calculate decay rate (Gamma) in the specified unit
        Gamma_decay = 1 / lifetime_seconds * self.unit

        return Gamma_decay

    def get_delta_energy(self, ni: int = 2, nf: int = 1) -> float:
        """
        Calculates the energy difference between two energy levels.

        Parameters:
        -----------
        ni : int, optional
            Initial principal quantum number (default is 2).
        nf : int, optional
            Final principal quantum number (default is 1).

        Returns:
        --------
        delta_energy : float
            The energy difference between the two levels.
        """
        delta_energy = 0.5 * abs(1 / ni**2 - 1 / nf**2)
        return delta_energy

    def get_nuclear_rate(self, Gamma_nuc_values):
        """
        Computes the nuclear absorption rate matrix (gnuc_nl) based on the given
        gnuc_values. This method uses vectorized operations for efficient computation.

        Parameters:
        -----------
        Gamma_nuc_values : array-like
            The input values related to nuclear absorption, expected to be a 1D array.

        Returns:
        --------
        Gamma_nuc_nl : numpy.ndarray
            A 2D array of shape (nmax+1, nmax+1) containing the computed nuclear absorption rates.
            The matrix is filled according to the specific formula derived from gnuc_values.
        """
        nmax = self.nmax

        # Initialize gnuc_circ with zeros and populate it with gnuc_values
        Gamma_nuc_circ = np.zeros(nmax)
        Gamma_nuc_circ[:len(gnuc_values)] = gnuc_values

        # Initialize the gnuc_nl matrix
        Gamma_nuc_nl = np.zeros((nmax + 1, nmax + 1))

        # Compute the first column of gnuc_nl
        Gamma_nuc_nl[1:nmax, 0] = Gamma_nuc_circ[:nmax - 1] / (units.Hartree * units._aut * 1e12)

        # Vectorized calculation for subsequent elements
        for l in range(1, nmax):
            n = np.arange(l + 1, nmax + 1)
            factor = (n / (n + 1)) ** (2 * l + 4) * (n + l + 1) / (n - l)
            Gamma_nuc_nl[l + 1:, l] = Gamma_nuc_nl[l:, l - 1] * factor

        return Gamma_nuc_nl

    def get_radiative_rate(self):
        """
        Calculate the radiative decay rate matrix (Gamma_rad_nlmp) for different quantum states
        of the exotic atom.

        This method computes the radiative decay rates for transitions between different energy levels
        and orbital angular momentum states in the atom. The rates are calculated based on the radial
        integrals, the reduced mass of the system, and other physical constants.

        Returns:
        --------
        Gamma_rad_nlmp : list
            A nested list structure containing the radiative decay rates for different quantum numbers (n, l, m, p).
            The structure is organized as follows:
            - Gamma_rad_nlmp[ni][li][nf][p] gives the radiative rate for the transition from state (ni, li) to (nf, lf).
            - The first element (Gamma_rad_nlmp[0]) is set to None to handle the special case where ni = 0.
        """
        mu = self.ExoticAtom.get_reduced_mass()
        Z = self.ExoticAtom.Z

        radial_integrals = self.wf.load_radial_integral()

        if radial_integrals is None:
            raise ValueError("Radial integrals could not be loaded.")

        # Precompute constants
        pre_factor = Z**4 * (mu / mass_e) * 4.0 / 3.0 * alpha**3 / _aut * self.unit

        Gamma_rad_nlmp = []

        # Fill the nested list with computed radial integrals
        for ni in np.arange(self.nmax + 1):
            ni_list = []  # List for the current ni
            for li in np.arange(ni):
                li_list = []  # List for the current li
                for nf in np.arange(ni):
                    nf_list = []  # List for the current nf
                    for p, lf in enumerate([li-1, li+1]):

                        if lf < 0 or lf >= nf:
                            nf_list.append(None)
                            continue
                        
                        # Compute the energy difference
                        deltaE = self.get_delta_energy(ni, nf)
                        # Calculate the radiative rate using the radial integral
                        radial_integral = radial_integrals[int(ni)][int(li)][int(nf)][p]
                        Gamma_rad = pre_factor * li / (2.0*li + 1.0) * radial_integral**2 * deltaE**3

                        nf_list.append(Gamma_rad)

                    li_list.append(nf_list)
                ni_list.append(li_list)
            Gamma_rad_nlmp.append(ni_list)

        Gamma_rad_nlmp[0] = None

        return Gamma_rad_nlmp
