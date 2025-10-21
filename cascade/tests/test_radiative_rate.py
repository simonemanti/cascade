import pytest

from cascade.rate import Rate
from cascade.exotic import ExoticAtom
from cascade.wavefunction import WaveFunction

import numpy as np

from particle import Particle

from scipy.constants import alpha, hbar, eV, physical_constants

# Constants
mass_e = Particle.from_name(name='e-').mass
hartree = physical_constants["Hartree energy"][0] / eV
_aut = hbar / physical_constants["Hartree energy"][0]

def test_radiative_rate():
    # Set up the ExoticAtom and WaveFunction objects
    KN = ExoticAtom(symbol='N', exotic='K-')
    Z = KN.Z
    mu = KN.get_reduced_mass()
    wf = WaveFunction()

    # Create the Rate object
    rate = Rate(KN, wf)

    # Calculate the radiative rates using the get_radiative_rate method
    Gamma_rad_nlm = rate.get_radiative_rate()

    ni_n = np.arange(2,rate.nmax+1,1, dtype=float)
    # Loop through different ni values and compare with analytical results
    for n, ni in enumerate(ni_n):
        # Analytical radiative rate calculation
        Gamma_rad_ana = (
            (mu / mass_e) * Z**4 * alpha**3 / 3 *
            (np.power(2, 4 * ni) * np.power(ni, 2 * ni - 4) * np.power(ni - 1, 2 * ni - 2)) /
             np.power(2 * ni - 1, 4 * ni - 1) / _aut / 1e12
        )

        # Extract the computed radiative rate from the method output
        Gamma_rad = Gamma_rad_nlm[int(ni)][int(ni) - 1][int(ni) - 1][0]

        # Use pytest's approx for floating-point comparison
        assert pytest.approx(Gamma_rad, rel=1e-9) == Gamma_rad_ana, f"Mismatch for ni={ni}: {Gamma_rad} != {Gamma_rad_ana}"

if __name__ == '__main__':
    pytest.main()
