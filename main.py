import numpy as np
import argparse
import multiprocessing
from helpers import (
    generate_main_figure,
    generate_comparison_panels,
    generate_spectral_comparison_plot,
    generate_timing_plot,
    generate_fractional_vs_nonfractional_plot,
)

def main():
    """Generate Figure 6 plots for α=1/2 (Smirnov) and α=1/3 and compare them with α=0."""

    parser = argparse.ArgumentParser(description="Generate plots for the fractional OU process.")
    parser.add_argument("--cores", type=int, default=multiprocessing.cpu_count(), help="Number of cores to use for parallel computation.")
    args = parser.parse_args()
    num_cores = args.cores

    print("=" * 70)
    print(
        "Figure 6: Fractional OU Process (normalized Lévy densities) and "
        "Comparison with Non-Fractional Case"
    )
    print(f"Using {num_cores} cores for parallel computation.")
    print("=" * 70)

    alphas = [0.5, 1.0 / 3.0]
    m, omega, k_B, T, gamma = 1.0, 1.0, 1.0, 1.0, 1.0
    K_beta = 1.0
    x0 = 0.5
    times = [0.01, 0.1, 1.0, 10.0, 100.0]
    x_values = np.linspace(-3.0, 3.0, 300)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for alpha in alphas:
        generate_main_figure(alpha, times, colors, x_values, x0, gamma, K_beta)

    panel_times = [0.01, 0.1, 1.0, 10.0, 100.0]
    generate_comparison_panels(panel_times, x0, gamma, K_beta)

    Ns_list = [5, 20, 100, 200]
    n_repeats = 5
    times_spec = times
    generate_spectral_comparison_plot(1.0/3.0, times_spec, Ns_list, n_repeats, x0, gamma, K_beta, m, omega, k_B, T, num_cores)
    generate_spectral_comparison_plot(0.5, times_spec, Ns_list, n_repeats, x0, gamma, K_beta, m, omega, k_B, T, num_cores)

    generate_timing_plot(num_cores)

    comparison_alphas = [0.5, 1.0 / 3.0, 0.0]
    comparison_times = [0.01, 0.1, 1.0, 10.0]
    generate_fractional_vs_nonfractional_plot(comparison_alphas, comparison_times, x0, gamma, K_beta)

    print("\n" + "=" * 70)
    print("All figures generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()