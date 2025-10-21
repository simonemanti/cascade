import pytest
from cascade.exotic import ExoticAtom

def test_exotic_atom():
    # Create an ExoticAtom instance with known parameters
    KN = ExoticAtom(symbol='N', exotic='K-')

    # Assert properties of the exotic particle
    assert KN.exotic_particle.name == 'K-'
    assert pytest.approx(KN.exotic_particle.mass, rel=1e-9) == 493.677
    assert pytest.approx(KN.exotic_particle.lifetime, rel=1e-9) == 12.37938606264635

    # Assert properties of the exotic atom
    assert KN.symbol == 'N'
    assert KN.Z == 7
    assert pytest.approx(KN.nucleus_mass, rel=1e-4) == 13047.43789259694
    assert pytest.approx(KN.mu, rel=1e-4) == 475.6787050101065

if __name__ == '__main__':
    pytest.main()
