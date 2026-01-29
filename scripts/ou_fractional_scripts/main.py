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
    """
    Main function to generate all the figures for the project.

    This script generates a series of plots related to the fractional Ornstein-Uhlenbeck (OU) process.
    It visualizes the time evolution of the probability density function (PDF) for different
    stability parameters (beta), compares different beta values, and analyzes the performance
    of the computation methods.

    Args:
        use_cache (bool): If True, the script will use pre-computed data from the 'data' directory
                          to speed up figure generation. If False, all data will be recomputed.
        num_cores (int): The number of CPU cores to use for parallel computations.
    """

    print("=" * 70)
    print(
        "Figure 6: Fractional OU Process (normalized Lévy densities) and "
        "Comparison with Non-Fractional Case"
    )
    print("=" * 70)

    # --- General Parameters ---
    betas = [0.5, 1.0 / 3.0]  # The stability parameters to be analyzed
    theta = 1.0  # Relaxation rate
    K_beta = 1.0  # Diffusion coefficient
    x0 = 0.5  # Initial position
    times = [0.01, 0.1, 1.0, 10.0, 100.0]  # Time points for the main figures
    x_values = np.linspace(-3.0, 3.0, 300)  # X-axis grid for plots
    colors = ["#1B7BBE", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"] # Plotting colors
    
    # --- Figure Generation ---

    # Generate the main PDF evolution figures for each beta
    for beta in betas:
        generate_main_figure(beta, times, colors, x_values, x0, theta, K_beta, use_cache=use_cache)

    # Generate the 2x2 panel comparing beta=1/2 and beta=1/3
    panel_times = [0.01, 0.1, 1.0, 10.0]
    generate_comparison_panels(panel_times, x0, theta, K_beta, use_cache=use_cache)

    # Generate the spectral method vs. integral method comparison plots
    Ns_list = [5, 20, 100, 200]  # Number of terms in the spectral series
    n_repeats = 5  # Number of times to repeat the timing for averaging
    times_spec = times
    generate_spectral_comparison_plot(1.0/3.0, times_spec, Ns_list, n_repeats, x0, theta, K_beta, num_cores=num_cores, use_cache=use_cache)
    generate_spectral_comparison_plot(0.5, times_spec, Ns_list, n_repeats, x0, theta, K_beta, num_cores=num_cores, use_cache=use_cache)

    # Generate the plot showing computation time vs. N for the spectral method
    generate_timing_plot(num_cores=num_cores, use_cache=use_cache)

    # Generate the plot comparing fractional cases with the standard non-fractional (beta=1) case
    comparison_betas = [0.5, 1.0 / 3.0, 1.0]
    comparison_times = [0.01, 0.1, 1.0, 10.0]
    generate_fractional_vs_nonfractional_plot(comparison_betas, comparison_times, x0, theta, K_beta, use_cache=use_cache)

    print("\n" + "=" * 70)
    print("All figures generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    # --- Command-Line Argument Parsing ---
    # This allows the user to control the script's behavior from the terminal.
    parser = argparse.ArgumentParser(description="Generate plots for the Fractional OU Process.")
    
    # Argument to disable caching
    parser.add_argument('--no-cache', dest='use_cache', action='store_false', 
                        help="Force re-computation of all data, ignoring cached files.")
    
    # Argument to specify the number of CPU cores
    parser.add_argument(
        '--cores',
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of CPU cores to use for parallel computations. Defaults to all available cores."
    )
    
    parser.set_defaults(use_cache=True)
    args = parser.parse_args()

    print(f"Using {args.cores} cores for parallel computations.")
    if not args.use_cache:
        print("Cache is disabled. All data will be recomputed.")

    # Run the main function with the parsed arguments
    main(args.use_cache, args.cores)