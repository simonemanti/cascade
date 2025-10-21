import pytest

from cascade.rate import Rate
from cascade.exotic import ExoticAtom
from cascade.wavefunction import WaveFunction
from cascade.rate import mass_e

import numpy as np


def test_exotic_atom():
    KN = ExoticAtom(symbol='N', exotic='K-')
    wf = WaveFunction()
    rate = Rate(KN, wf)

    Ze = KN.Z - 1
    nshell = 1

    for ni in range(2, rate.nmax):
        nf = ni - 1 
        energy1 = rate.get_auger_energy(ni, nf, Ze, nshell=nshell)
        energy2 = KN.Z**2 * KN.mu / mass_e / 2 * (1 / np.power(ni-1, 2) - 1 / np.power(ni, 2)) - Ze**2 / (2 * nshell**2)
   
        assert pytest.approx(energy1, rel=1e-9) == energy2

if __name__ == '__main__':
    pytest.main()
