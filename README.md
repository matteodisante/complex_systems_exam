# Fractional Ornstein-Uhlenbeck Process Simulation

[![Python Tests](https://github.com/OWNER/REPO/actions/workflows/python-package.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/python-package.yml)

This repository contains Python scripts to simulate the fractional Ornstein-Uhlenbeck (fOU) process, with a focus on cases where the fractional exponent (β) is 1/2 and 1/3, and comparisons with the non-fractional case (β=1). The goal is to provide a clear, self-contained implementation to calculate and visualize the probability density function (PDF) of this process.

## Repository Contents

- **`scripts/ou_fractional_scripts/main.py`**: Main entry point for generating figures.
- **`scripts/ou_fractional_scripts/core_computations.py`**: Core numerical routines (kernel, subordination, Mittag-Leffler, spectral series).
- **`scripts/ou_fractional_scripts/helpers.py`**: Plotting helpers and caching logic.
- **`scripts/ou_fractional_scripts/test_core_computations.py`**: Unit tests for the core functions.
- **`scripts/`**: Additional notebooks and scripts for related simulations and plots.

## What the code does

The main script implements two approaches to calculate the PDF of the fOU process:

1.  **Integral Map Method**: This method is based on the analytical Smirnov form of the Lévy density. The PDF `P(x,t)` is calculated as an integral of the convolution between the waiting time PDF `n(s,t)` and the Gaussian kernel of the Ornstein-Uhlenbeck process `P1(x,s)`. The integration is performed numerically over a grid of `s` values.

2.  **Spectral Series Method**: This alternative approach calculates the PDF as a series of eigenfunctions of the Ornstein-Uhlenbeck process (Hermite functions). The time evolution is captured by a factor that includes the Mittag-Leffler function.

The script generates the following outputs (saved under `scripts/ou_fractional_scripts/figures/`):

- **`fig6_beta_0_500.png`**: Time evolution of the PDF for β = 1/2.
- **`fig6_beta_0_333.png`**: Time evolution of the PDF for β = 1/3.
- **`fig6_comparison_panels.png`**: Direct comparison between β = 1/2 and β = 1/3 at different time points.
- **`fig6_spectral_vs_integral_beta_1_3.png`**: Spectral series vs integral map for β = 1/3.
- **`fig6_spectral_vs_integral_beta_0_5.png`**: Spectral series vs integral map for β = 1/2.
- **`fig6_fractional_vs_nonfractional.png`**: Fractional (β=1/2, β=1/3) vs non-fractional (β=1) comparison.
- **`fig_timing_vs_N_beta_1_3.png`**: Computation time vs number of spectral terms (linear scale).

## How to use the repository

### Prerequisites

Make sure you have the following Python libraries installed:

- `numpy`
- `matplotlib`
- `scipy`

You can install them using pip:
```bash
pip install numpy matplotlib scipy
```

### Run the simulation

To run the simulation and generate the plots, execute:

```bash
python scripts/ou_fractional_scripts/main.py
```

Plots are saved under `scripts/ou_fractional_scripts/figures/`.

#### Caching

Intermediate results are cached under `scripts/ou_fractional_scripts/data/`. If the cache files exist, the script loads them to avoid recomputation. Use `--no-cache` to force recomputation.

### Run the tests

To verify the correctness of the code, you can run the test suite:

```bash
pytest scripts/ou_fractional_scripts/test_core_computations.py
```

## Purpose of the project

The purpose of this repository is to provide a clear, working, and verified example of how to simulate the fractional Ornstein-Uhlenbeck process. It can be useful for students, researchers, or anyone interested in stochastic processes and complex systems, both as a learning tool and as a basis for further research.