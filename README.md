# Radiative Transfer Simulation
- Python implementation of a radiative transfer simulation, which models the interaction of light with the Earth's atmosphere
- Particularly focusing on the influence of CO2 concentration on the outgoing radiation spectrum
- Forked from the original implementation of [Science Etonnante](https://github.com/scienceetonnante)

## Introduction
- Radiative transfer simulations are essential tools in atmospheric science and climate modeling. 
- They help understand how different factors, such as greenhouse gas concentrations, affect the Earth's energy balance and climate.

## Features
- Simulates the propagation of electromagnetic radiation through the Earth's atmosphere
- Only considers the absorption and emission of radiation by CO2 molecules
- Uses various atmospheric temperature models:
  - Uniform temperature model
  - Simple atmospheric model
  - US1976 atmospheric model
- Generates different matplotlib plots at the top of the atmosphere for different CO2 concentrations

## Requirements
- Python 3.7+

## Installation
- Clone the repository
```bash
git clone https://github.com/Crossoufire/radiative-forcing.git
```
- Install the required dependencies
```bash
pip install -r requirements.txt
```

## Usage
- You can adapt parameters to test different settings
```python
params = {
    "CO2_frac": 280 * 10 ** -6,
    "z_max": 80000,
    "delta_z": 10,
    "lambda_min": 0.1 * 10 ** -6,
    "lambda_max": 100 * 10 ** -6,
    "delta_lambda": 0.01 * 10 ** -6,
    "model": temperature_uniform,
}
```
- Run the script
```bash
python radiative_forcing.py
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
