# Complex Systems Exam - Stochastic Processes Simulations

[![Python Tests](https://github.com/matteodisante/complex_systems_exam/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/matteodisante/complex_systems_exam/actions/workflows/python-package.yml)

This repository contains Python implementations of various stochastic processes and anomalous diffusion phenomena studied in the Complex Systems course. The main focus is on fractional processes, Lévy flights, Continuous Time Random Walks (CTRW), and the fractional Ornstein-Uhlenbeck process.

## 📁 Repository Structure

```
complex_systems_exam/
├── .github/workflows/       # CI/CD pipeline
│   └── python-package.yml   # Automated testing
├── scripts/
│   ├── CTRW_sims/           # Continuous Time Random Walk simulations
│   │   ├── mc_sims.py       # Monte Carlo simulations
│   │   └── mittag_leffler_plot.py
│   ├── integral-map_subordination/  # Subordination processes
│   │   └── subordination.py # Time subordination visualization
│   ├── levy_mittag-leffler/ # Lévy and Mittag-Leffler distributions
│   │   ├── levy_mittag_gen.py      # Random generators
│   │   └── plot_phi_1.py           # Density plots
│   └── ou_fractional_scripts/      # Fractional Ornstein-Uhlenbeck
│       ├── main.py                 # Main entry point
│       ├── core_computations.py    # Core numerical routines
│       ├── helpers.py              # Plotting and caching utilities
│       └── test_core_computations.py # Unit tests
├── requirements.txt         # Python dependencies
├── pytest.ini              # Test configuration
└── README.md               # This file
```

## 🎯 Project Components

### 1. Fractional Ornstein-Uhlenbeck Process (`ou_fractional_scripts/`)

Main implementation of the fractional Ornstein-Uhlenbeck (fOU) process with focus on β = 1/2 and β = 1/3.

**Methods implemented:**
- **Integral Map Method**: Based on Smirnov's Lévy density form, computing P(x,t) via numerical convolution
- **Spectral Series Method**: Hermite function expansion with Mittag-Leffler time evolution

**Generated figures:**
- Time evolution of PDF for different β values
- Spectral vs integral map comparisons
- Fractional vs non-fractional process comparisons
- Computation time analysis

**Run:**
```bash
python scripts/ou_fractional_scripts/main.py
```

### 2. Continuous Time Random Walks (`CTRW_sims/`)

Monte Carlo simulations of CTRWs with Lévy-stable jumps and Mittag-Leffler waiting times.

**Features:**
- Lévy-stable jump generator (Chambers-Mallows-Stuck algorithm)
- Mittag-Leffler waiting time generator
- Ensemble simulations with configurable parameters
- Mean Square Displacement (MSD) analysis

**Run:**
```bash
python scripts/CTRW_sims/mc_sims.py
```

### 3. Lévy and Mittag-Leffler Distributions (`levy_mittag-leffler/`)

Generation and visualization of Lévy-stable and Mittag-Leffler distributions.

**Features:**
- Random number generators for both distributions
- Power-law tail analysis
- Density function plots
- Theoretical vs empirical distribution comparison

**Run:**
```bash
python scripts/levy_mittag-leffler/levy_mittag_gen.py
python scripts/levy_mittag-leffler/plot_phi_1.py
```

### 4. Time Subordination (`integral-map_subordination/`)

Visualization of subordination processes and inverse Lévy subordinators.

**Features:**
- Single trajectory visualization
- Ensemble analysis
- Subordinator T(τ) plots
- Physical time vs operational time mapping

**Run:**
```bash
python scripts/integral-map_subordination/subordination.py
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/matteodisante/complex_systems_exam.git
cd complex_systems_exam
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🧪 Running Tests

The repository includes automated tests for the fractional OU process:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest scripts/ou_fractional_scripts/test_core_computations.py
```

Tests are automatically run on every push via GitHub Actions CI/CD pipeline.

## 📊 Output Files

Generated figures and data are saved in subdirectories:

- `scripts/ou_fractional_scripts/figures/` - fOU process visualizations
- `scripts/ou_fractional_scripts/data/` - Cached computation results
- `scripts/CTRW_sims/*.png` - CTRW simulation outputs
- `scripts/levy_mittag-leffler/*.png` - Distribution plots
- `scripts/integral-map_subordination/*.png` - Subordination visualizations

## � Dependencies

- `numpy>=2.3.3` - Numerical computations
- `scipy>=1.16.2` - Special functions and integration
- `matplotlib>=3.10.7` - Plotting and visualization
- `mpmath>=1.3.0` - High-precision arithmetic
- `pytest>=8.4.2` - Testing framework
- `flake8>=7.1.0` - Code linting

## 🤝 Contributing

This is an academic project for the Complex Systems course. For issues or improvements, please open an issue or pull request.

## 📝 License

Academic use only - part of university coursework.

## 👤 Author

Matteo Di Sante - Complex Systems Exam, University Project

## 📌 Notes

- **Caching**: The fOU scripts cache intermediate results in `data/` directories to speed up repeated computations
- **CI/CD**: Automated tests run on every push to ensure code correctness
- **Reproducibility**: Random seeds are set for reproducible results where applicable

---

*Repository created for Complex Systems examination - First Semester, First Year*