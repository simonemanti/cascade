import pytest

from cascade.exotic import ExoticAtom
from cascade.rate import Rate

def test_get_decay_rate():
    # Create an ExoticAtom instance with known parameters
    KN = ExoticAtom(symbol='N', exotic='K-')

    # Instantiate the Rate class with the ExoticAtom
    rate = Rate(KN)

    # Compute the decay rate
    computed_decay_rate = rate.get_decay_rate()

    # Define the expected decay rate
    expected_decay_rate = 8.077945020370656e-05

    # Assert that the computed decay rate matches the expected value
    assert pytest.approx(computed_decay_rate, rel=1e-9) == expected_decay_rate

if __name__ == '__main__':
    pytest.main()
