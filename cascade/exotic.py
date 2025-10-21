from mendeleev import element

from particle import Particle

from scipy.constants import physical_constants


class ExoticAtom:

    def __init__(self, symbol='H', Z=None, exotic='K-') -> None:
    
        self.symbol = symbol
        self.element = element(symbol)

        # Get atomic number (Z) from element symbol
        if Z is None:
            Z = self.element.atomic_number

        self.Z = Z

        # Initialize the exotic particle
        self.exotic = exotic
        self.exotic_particle = self.get_exotic_particle()
        self.nucleus_mass = self.get_nucleus_mass()
        self.mu = self.get_reduced_mass()

    def get_exotic_particle(self):
        # Get the exotic particle from the Particle package using its name

        exotic_particle = Particle.from_name(self.exotic)
        
        return exotic_particle
    
    def get_nucleus_mass(self):

        u2MeV = physical_constants['atomic mass constant energy equivalent in MeV'][0]
        nucleus_mass_u = self.element.atomic_weight  # in atomic mass units (u)

        # Convert to MeV/c^2
        nucleus_mass = nucleus_mass_u * u2MeV

        return nucleus_mass
    
    def get_reduced_mass(self):

        nucleus_mass = self.nucleus_mass
        exotic_mass = self.exotic_particle.mass

        mu = nucleus_mass * exotic_mass / (nucleus_mass + exotic_mass)

        return mu
