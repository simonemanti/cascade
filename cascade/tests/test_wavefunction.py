import pytest
from cascade.wavefunction import WaveFunction

def test_wavefunction():
    # Create a WaveFunction instance
    wf = WaveFunction()

    # Get the discrete radial positions
    r_i = wf.r_i

    # Get the discrete wavefunction for n=1, l=0 and multiply by r_i
    u10_i = wf.get_discrete_wf(n=1, l=0) * r_i

    # Check normalization of 1s state
    integral_1 = wf.integrate(u10_i**2, r_i)
    # Check radial average of 1s state
    integral_2 = wf.integrate(u10_i**2 * r_i, r_i)

    # Expected values
    norm = 1.0
    rave = 1.5

    # Assert that the calculated integrals match the expected values
    assert pytest.approx(integral_1, rel=1e-9) == norm
    assert pytest.approx(integral_2, rel=1e-9) == rave

if __name__ == '__main__':
    pytest.main()
