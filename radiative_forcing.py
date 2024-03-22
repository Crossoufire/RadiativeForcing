from copy import copy
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


""" --- CONSTANTS ------------------------------------------------------------------------------------------ """

# Planck's constant [J*s]
h = 6.62607015 * 10 ** -34

# Speed of light [m/s]
c = 2.998 * 10 ** 8

# Boltzmann's constant [J/K]
kB = 1.380649* 10 ** -23

# Pressure at sea level [Pa]
P0 = 101325

# Scale height [m]
H = 8500

""" -------------------------------------------------------------------------------------------------------- """


def temperature_uniform(z: float):
    """ Considering only a uniform temperature at sea level """
    T0 = 288.2  # Temperature at sea level in Kelvin
    return T0 * np.ones_like(z)


def temperature_simple(z: float):
    """ Calculates the temperature at a given altitude using a simple atmospheric model """
    T0 = 288.2          # Temperature at sea level in Kelvin
    z_trop = 11000      # Tropopause height in meter
    Gamma = -0.0065     # Temperature gradient in Kelvin/meter

    T_trop = T0 + Gamma * z_trop
    return np.piecewise(z, [z < z_trop, z >= z_trop], [lambda z: T0 + Gamma * z, lambda z: T_trop])


def temperature_US1976(z: float):
    """ Calculates the atmospheric temperature based on the 1976 United States Standard Atmosphere model. 
    This model divides the atmosphere into distinct layers, each with its own temperature profile. """

    # Convert altitude to [km] for easier comparisons
    z_km = z / 1000

    # Troposphere (0 to 11 km)
    T0 = 288.15
    z_trop = 11

    # Tropopause (11 to 20 km)
    T_tropopause = 216.65
    z_tropopause = 20

    # Stratosphere 1 (20 to 32 km)
    T_strat1 = T_tropopause
    z_strat1 = 32

    # Stratosphere 2 (32 to 47 km)
    T_strat2 = 228.65
    z_strat2 = 47

    # Stratopause (47 to 51 km)
    T_stratopause = 270.65
    z_stratopause = 51

    # Mesosphere 1 (51 to 71 km)
    T_meso1 = T_stratopause
    z_meso1 = 71

    # Mesosphere 2 (71 to +∞)
    T_meso2 = 214.65

    conds = [
        z_km < z_trop, 
        (z_km >= z_trop) & (z_km < z_tropopause), 
        (z_km >= z_tropopause) & (z_km < z_strat1),
        (z_km >= z_strat1) & (z_km < z_strat2), 
        (z_km >= z_strat2) & (z_km < z_stratopause),
        (z_km >= z_stratopause) & (z_km < z_meso1), 
        z_km >= z_meso1,
    ]

    functions = [
        lambda z: T0 - 6.5 * z, 
        lambda z: T_tropopause, 
        lambda z: T_strat1 + 1 * (z - z_tropopause),
        lambda z: T_strat2 + 2.8 * (z - z_strat1), 
        lambda z: T_stratopause,
        lambda z: T_meso1 - 2.8 * (z - z_stratopause), 
        lambda z: T_meso2 - 2 * (z - z_meso1),
    ]

    return np.piecewise(z_km, conds, functions)


class RadiativeTransfer:
    """ Radiative transfer simulation class """

    def __init__(self, CO2_frac: float, z_max: float, delta_z: float, lambda_min: float, lambda_max: float, delta_lambda: float, model):
        self.CO2_frac = CO2_frac
        self.z_max = z_max
        self.delta_z = delta_z
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.delta_lambda = delta_lambda
        self.temp_model = model

        # Define altitude and grid steps
        self.z_range = np.arange(0, self.z_max, self.delta_z)
        self.lambda_range = np.arange(self.lambda_min, self.lambda_max, self.delta_lambda)

        # Initialize arrays
        self.upward_flux = np.zeros((len(self.z_range), len(self.lambda_range)))
        self.optical_thickness = np.zeros((len(self.z_range), len(self.lambda_range)))

    @staticmethod
    def pressure(z: float):
        """ Exponential decay of the pressure """
        return P0 * np.exp(-z / H)

    def planck_function(self, T: float):
        """ Black body radiation formula """

        term1 = (2 * h * c ** 2) / self.lambda_range ** 5
        term2 = np.exp((h * c) / (self.lambda_range * kB * T)) - 1

        return term1 / term2

    def outward_vertical_flux(self, z: float):
        """ Boundary condition. Compute outward vertical flux emitted by Earth's surface for all wavelengths """

        earth_flux = np.pi * self.planck_function(self.temp_model(z)) * self.delta_lambda
        print(f"Total earth flux at z = {z}; in wavelength range: {earth_flux.sum():.2f} W/m^2")

        return earth_flux

    def air_number_density(self, z):
        return self.pressure(z) / (kB * self.temp_model(z))

    def cross_section_CO2(self):
        """ Cross-section of the CO2 """

        LAMBDA_0 = 15.0e-6 # Band center in m
        exponent = -22.5 - 24 * np.abs((self.lambda_range - LAMBDA_0) / LAMBDA_0)

        return 10 ** exponent

    def simulate_radiative_transfer(self):
        """ Radiative transfer simulation. All wavelengths are treated in parallel using vectorization """

        flux_in = self.outward_vertical_flux(z=0)

        for i, z in enumerate(tqdm(self.z_range, ncols=70)):
            # Number density of CO2 molecules and absorption coefficient
            n_CO2 = self.air_number_density(z) * self.CO2_frac
            kappa = self.cross_section_CO2() * n_CO2

            # Compute fluxes within layer
            self.optical_thickness[i, :] = kappa * self.delta_z
            absorbed_flux = np.minimum(kappa * self.delta_z * flux_in, flux_in)
            emitted_flux = self.optical_thickness[i, :] * np.pi * self.planck_function(self.temp_model(z)) * self.delta_lambda
            self.upward_flux[i, :] = flux_in - absorbed_flux + emitted_flux

            # Flux leaving layer becomes flux entering next layer
            flux_in = self.upward_flux[i, :]

        print(f"Total outgoing flux at the top of the atmosphere: {self.upward_flux[-1, :].sum():.2f} W/m^2")


if __name__ == "__main__":

    # Parameters to use the RadiativeTransfer class
    params = {
        "CO2_frac": 280 * 10 ** -6,
        "z_max": 80000,
        "delta_z": 10,
        "lambda_min": 0.1 * 10 ** -6,
        "lambda_max": 100 * 10 ** -6,
        "delta_lambda": 0.01 * 10 ** -6,
        "model": temperature_uniform,
    }

    params2 = copy(params)
    params2["CO2_frac"] = params["CO2_frac"] * 2

    rt = RadiativeTransfer(**params)
    rt.simulate_radiative_transfer()

    rt2 = RadiativeTransfer(**params2)
    rt2.simulate_radiative_transfer()

    # Plot top of atmosphere spectrum
    plt.figure(figsize=(14, 9))

    # Blackbody spectrum at Earth's surface
    plt.plot(1e6 * rt.lambda_range, np.pi * rt.planck_function(rt.temp_model(z=0)) / 1e6, ls="--", c="k")

    # Blackbody spectrum for 216K
    # plt.plot(1e6 * rt2.lambda_range, np.pi * rt2.planck_function(216) / 1e6, ls="--", c="k")

    # Blackbody spectrum in space with N qty CO2
    plt.plot(1e6 * rt.lambda_range, rt.upward_flux[-1, :] / rt.delta_lambda / 1e6, c="tab:green")

    # Blackbody spectrum in space with N * 2 qty CO2
    plt.plot(1e6 * rt2.lambda_range, rt2.upward_flux[-1, :] / rt2.delta_lambda / 1e6, c="tab:red")

    # Fill between both spectrums
    plt.fill_between(1e6 * rt.lambda_range, rt.upward_flux[-1, :] / rt.delta_lambda / 1e6,
                     rt2.upward_flux[-1, :] / rt2.delta_lambda / 1e6, color="yellow", alpha=1)

    plt.xlabel("Wavelength [μm]")
    plt.ylabel("Spectral Radiance [W/m²/μm/sr]")
    plt.xlim(0, 50)
    plt.ylim(0, 30)
    plt.grid(True)

    plt.show()
