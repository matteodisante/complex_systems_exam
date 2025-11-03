# Fractional Ornstein-Uhlenbeck Process Simulation

This repository contains a Python script to simulate the fractional Ornstein-Uhlenbeck (fOU) process, with a focus on cases where the fractional exponent (α) is 1/2 and 1/3. The goal is to provide a minimal, self-contained, and correct implementation to calculate and visualize the probability density function (PDF) of this process.

## Repository Contents

- **`code2.py`**: The main script that runs the simulation. It calculates the PDF of the fOU process using two different methods and generates a series of plots to visualize the results.
- **`test_code2.py`**: A suite of unit tests for the `code2.py` script to ensure the correctness of the implemented functions.
- **`hermite_check.py`**: A utility script to verify the implementation of the Hermite functions used in the spectral series method.

## What the code does

The `code2.py` script implements two approaches to calculate the PDF of the fOU process:

1.  **Integral Map Method**: This method is based on the analytical Smirnov form of the Lévy density. The PDF `P(x,t)` is calculated as an integral of the convolution between the waiting time PDF `n(s,t)` and the Gaussian kernel of the Ornstein-Uhlenbeck process `P1(x,s)`. The integration is performed numerically over a grid of `s` values.

2.  **Spectral Series Method**: This alternative approach calculates the PDF as a series of eigenfunctions of the Ornstein-Uhlenbeck process (Hermite functions). The time evolution is captured by a factor that includes the Mittag-Leffler function.

The script generates the following outputs:

- **`fig6_alpha_0_500.png`**: Time evolution of the PDF for α = 1/2.
- **`fig6_alpha_0_333.png`**: Time evolution of the PDF for α = 1/3.
- **`fig6_comparison_panels.png`**: A direct comparison between the PDFs for α = 1/2 and α = 1/3 at different time points.
- **`fig6_spectral_vs_integral.png`**: A comparison between the integral map method and the spectral series method for α = 1/3, showing the convergence as the number of terms in the series increases.

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

To run the simulation and generate the plots, execute the `code2.py` script:

```bash
python code2.py
```

The plots will be saved in the main directory of the repository.

### Run the tests

To verify the correctness of the code, you can run the test suite:

```bash
python test_code2.py
```

### Verify the Hermite functions

To check the implementation of the Hermite functions, you can run the `hermite_check.py` script:

```bash
python hermite_check.py
```
This script will compare the recurrence-based implementation with the one that uses the direct formula from `scipy`, printing the differences.

## Purpose of the project

The purpose of this repository is to provide a clear, working, and verified example of how to simulate the fractional Ornstein-Uhlenbeck process. It can be useful for students, researchers, or anyone interested in stochastic processes and complex systems, both as a learning tool and as a basis for further research.