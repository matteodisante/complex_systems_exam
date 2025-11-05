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

def main(use_cache, num_cores):
    """Generate Figure 6 plots for α=1/2 (Smirnov) and α=1/3 and compare them with α=0."""

    print("=" * 70)
    print(
        "Figure 6: Fractional OU Process (normalized Lévy densities) and "
        "Comparison with Non-Fractional Case"
    )
    print("=" * 70)

    alphas = [0.5, 1.0 / 3.0]
    m, omega, k_B, T, gamma = 1.0, 1.0, 1.0, 1.0, 1.0
    K_beta = 1.0
    x0 = 0.5
    times = [0.01, 0.1, 1.0, 10.0, 100.0]
    x_values = np.linspace(-3.0, 3.0, 300)
    colors = ["#1B7BBE", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    

    for alpha in alphas:
        generate_main_figure(alpha, times, colors, x_values, x0, gamma, K_beta, use_cache=use_cache)

    panel_times = [0.01, 0.1, 1.0, 10.0, 100.0]
    generate_comparison_panels(panel_times, x0, gamma, K_beta, use_cache=use_cache)

    Ns_list = [5, 20, 100, 200]
    n_repeats = 5
    times_spec = times
    generate_spectral_comparison_plot(1.0/3.0, times_spec, Ns_list, n_repeats, x0, gamma, K_beta, m, omega, k_B, T, num_cores=num_cores, use_cache=use_cache)
    generate_spectral_comparison_plot(0.5, times_spec, Ns_list, n_repeats, x0, gamma, K_beta, m, omega, k_B, T, num_cores=num_cores, use_cache=use_cache)

    generate_timing_plot(num_cores=num_cores, use_cache=use_cache)

    comparison_alphas = [0.5, 1.0 / 3.0, 0.0]
    comparison_times = [0.01, 0.1, 1.0, 10.0]
    generate_fractional_vs_nonfractional_plot(comparison_alphas, comparison_times, x0, gamma, K_beta, use_cache=use_cache)

    print("\n" + "=" * 70)
    print("All figures generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for the Fractional OU Process.")
    parser.add_argument('--no-cache', dest='use_cache', action='store_false', help="Force re-computation of all data, ignoring cached files.")
    parser.add_argument(
        '--cores',
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of CPU cores to use for parallel computations. Defaults to all available cores."
    )
    parser.set_defaults(use_cache=True)
    args = parser.parse_args()

    print(f"Using {args.cores} cores for parallel computations.")

    main(args.use_cache, args.cores)
