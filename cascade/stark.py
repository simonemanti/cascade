import ase.units as units

from math import pi

import numpy as np

from odeintw import odeintw

from scipy.integrate import solve_ivp


class Stark:
    """
    Class for calculating rates for stark mixing.

    ...

    Parameters
    ----------

    rho: float

    Attributes
    ----------

    velocity: float
    
    """

    def __init__(self, rho=1, kenergy=1, tmax=100, ntpts=100):
        self.rho = rho
        self.kenergy = kenergy
        self.tmax = tmax
        self.ntpts = ntpts    
        self.velocity = np.sqrt(2 * self.kenergy / units.Hartree)
        self.R0 = self.velocity * tmax / 2
        self.t_i = np.linspace(0, tmax, ntpts)
        self.theta_i = np.arctan2(self.velocity * self.t_i - self.R0, rho) * 180 / pi

    
    def evolve(self, t_i, n, method='fixed'):
        """
        Evolve the time-dependent Schrodinger equation to calculate transition probabilities        
        due to the interaction with the electric field, withing the impact parameter method.

        ...

        Parameters
        ----------

        t_i: ndarray
            Time array in atomic units
        n: int
            Principal quantum number
        method: str
            Fixed field or Rotating field method

        Returns
        -------

        c_ni: ndarray
            Occupation over the basis for different times

        """
 
        self.method = method 
     
        # Initial conditions.
        c0_ni = np.zeros((n**2), dtype=np.complex_)
        c0_ni[0] = 1.

        solution = solve_ivp(fun=self.get_evolution_function,
                             t_span=(t_i[0],t_i[-1]),
                             t_eval=t_i,
                             y0=c0_ni,
                             args=(n,))

        c_ni = solution.y

        return c_ni


    def get_evolution_function(self, t, c_ni, n):
    
        F = self.get_hydrogen_efield(t)
        C_nm = -1.j * F * self.get_fixed_matrix(n)

        if self.method == 'rotating':
            dtheta = self.rho * self.velocity / (self.rho**2 + self.velocity**2 * t**2)
            C_nm += dtheta * self.get_rotating_matrix()

        return C_nm.dot(c_ni)


    def get_fixed_factors(self, n, l, m):
        """ Factors for the fixed field model. """
    
        a_lminus = np.sqrt(((l**2 - m**2) * (n**2 - l**2)) / ((2*l + 1) * (2*l - 1)))
        a_lplus = np.sqrt((((l+1)**2 - m**2) * (n**2 - (l+1)**2)) / ((2*l + 3) * (2*l + 1)))
    
        if l - 1 < 0:
            a_lminus = 0
        if l + 1 == n:
            a_lplus = 0
    
        return a_lplus, a_lminus
    
    
    def get_fixed_matrix(self, n, F=0.1):
        """ Matrix for the fixed field model. """
    
        C_nm = np.zeros((n**2, n**2), dtype=np.complex_)
    
        for l in range(n):
            for m in range(-l, l+1):
                ap, am = self.get_fixed_factors(n, l, m)
                if ap != 0.:
                    C_nm[(l+1)*l + m, (l+1)*(l+2) + m] = ap
                if am != 0.:
                    C_nm[(l+1)*l + m, l*(l-1) + m] = am 
    
        return 3 / 2 * n * C_nm
    

    def get_hydrogen_efield(self, t):
        """ Return the electric field due to an hydrogen atom. """

        # in Bohr
        R = np.sqrt((self.velocity * t - self.R0)**2 + self.rho**2) 
        # in a.u. of electric field
        F = np.exp(-2 * R) / R**2 * (1 + 2 * R + 2 * R**2)

        return F
    

    def get_rotating_factors(self, n, l, m):
        """ Factors for the rotating field model. """
    
        b_mminus = np.sqrt(l * (l + 1) - m * (m - 1))
        b_mplus = np.sqrt(l * (l + 1) - m * (m + 1))
    
        return b_mplus, b_mminus
    
    
    def get_rotating_matrix(self, n):
        """ Matrix for the rotating field model. """
    
        C_nm = np.zeros((n**2, n**2), dtype=np.complex_)
    
        for l in range(n):
            for m in range(-l, l+1):
                bp, bm = self.get_rotating_factors(n, l, m)
                if bp != 0.:
                    C_nm[(l+1)*l + m, (l+1)*l + m + 1] = -bp 
                if bm != 0.:
                    C_nm[(l+1)*l + m, (l+1)*l + m - 1] = bm 
    
        return -1.j / 2 * C_nm
