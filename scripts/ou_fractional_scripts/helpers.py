import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pickle
import multiprocessing
import time
from core_computations import (
    compute_pdf_vectorized,
    spectral_series_pdf,
)

# Set global matplotlib parameters for consistent styling across all plots.
plt.rcParams.update(
    {
        "figure.dpi": 350,
        "axes.titlesize": 45,           # Main title size (set_title for single plots, suptitle for multi-panel)
        "axes.labelsize": 32,           # X and Y axis labels
        "xtick.labelsize": 32,          # X tick labels
        "ytick.labelsize": 32,          # Y tick labels
        "legend.fontsize": 27,          # Legend entries
        "legend.title_fontsize": 27,    # Legend title
        "lines.linewidth": 3.0,
    }
)

# Custom font sizes for specific plot elements
SUBPLOT_TITLE_SIZE = 38      # Titles of individual subplots in multi-panel figures
ANNOTATION_SIZE = 12         # Size for annotations (L1, timing info)
SPECTRAL_YLABEL_SIZE = 38    # Y-axis labels in spectral grid plots
SPECTRAL_LABEL_SIZE = 28     # Shared X/Y labels in spectral grid plots

def print_progress(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Creates and displays a terminal progress bar.
    Useful for tracking the progress of long-running computations.

    Args:
        iteration (int): The current iteration number.
        total (int): The total number of iterations.
        prefix (str, optional): A string to display before the progress bar. Defaults to ''.
        suffix (str, optional): A string to display after the progress bar. Defaults to ''.
        decimals (int, optional): Number of decimal places for the percentage. Defaults to 1.
        length (int, optional): The character length of the bar. Defaults to 50.
        fill (str, optional): The character used to fill the bar. Defaults to '█'.
    """
    percent = ("{:0." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

def compute_spec_and_time(args):
    """
    A wrapper function to compute the spectral series PDF and measure the execution time.
    Designed to be used with multiprocessing.Pool for parallel computation.

    Args:
        args (tuple): A tuple containing all the necessary arguments for spectral_series_pdf.

    Returns:
        tuple: A tuple containing the input time, N, the computation time, and the resulting PDF.
    """
    t, N, x_spec, x0, beta_spec, theta, K_beta = args
    start_time = time.perf_counter()
    spec = spectral_series_pdf(x_spec, t, x0, beta_spec, N, theta=theta, K_beta=K_beta)
    end_time = time.perf_counter()
    return (t, N, end_time - start_time, spec)

def generate_main_figure(beta, times, colors, x_values, x0, theta, K_beta, use_cache=True):
    """
    Generates the main figure showing the time evolution of the PDF for a given beta.

    This function computes (or loads from cache) the PDF of the fractional OU process
    at different time points and plots them, along with the stationary distribution.

    Args:
        beta (float): The stability parameter.
        times (list): A list of time points to plot.
        colors (list): A list of colors for the different time plots.
        x_values (np.ndarray): The grid of x-values for the plot.
        x0 (float): The initial position.
        theta (float): The relaxation rate.
        K_beta (float): The diffusion coefficient.
        use_cache (bool, optional): Whether to use cached data if available. Defaults to True.
    """
    cache_dir = "data"
    fname_beta_str = f"{beta:.3f}".replace("0.", "").replace(".", "_")
    cache_filename = os.path.join(cache_dir, f"main_figure_data_beta_{fname_beta_str}.pkl")

    # Load data from cache or compute it if not available
    if use_cache and os.path.exists(cache_filename):
        print(f"Loading data from cache: {cache_filename}")
        with open(cache_filename, 'rb') as f:
            pdfs = pickle.load(f)
    else:
        print(f"Computing data for beta={beta:.3f}...")
        pdfs = {}
        for t in times:
            pdfs[t] = compute_pdf_vectorized(
                x_values, t, x0, beta=beta, theta=theta, K_beta=K_beta, Ns=800
            )
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_filename, 'wb') as f:
            pickle.dump(pdfs, f)

    # --- Plotting ---
    fig = plt.figure(figsize=(18, 10))
    ax_main = fig.add_subplot(1, 1, 1)
    print(f"\nGenerating figure for beta={beta:.3f}...")

    # Plot PDF for each time point
    for t, c in zip(times, colors):
        ax_main.plot(x_values, pdfs[t], color=c, lw=2.5, label=f"t = {t}", alpha=0.85)

    # Plot the stationary distribution (Gaussian)
    variance_stat = K_beta / theta
    stationary = (
        1.0
        / np.sqrt(2.0 * np.pi * variance_stat)
        * np.exp(-0.5 * theta * x_values**2 / K_beta)
    )
    ax_main.plot(
        x_values,
        stationary,
        color="black",
        linestyle="--",
        lw=2.2,
        label="Stationary",
        alpha=0.7,
    )
    # Mark the initial position
    ax_main.axvline(
        x=x0,
        color="gray",
        linestyle=":",
        linewidth=2,
        alpha=0.6,
        label=f"x₀ = {x0}",
    )

    ax_main.set_xlabel("x", fontweight="bold")
    ax_main.set_ylabel("P(x,t)", fontweight="bold")

    # Set a custom title based on the beta value
    if abs(beta - 0.5) < 1e-12:
        title_beta = r"β = 1/2 (Smirnov)"
    elif abs(beta - 1.0 / 3.0) < 1e-12:
        title_beta = r"β = 1/3"
    else:
        title_beta = f"β = {beta}"

    ax_main.set_title(
        f"Fractional OU process: {title_beta}",
        fontweight="bold",
        pad=20,
    )
    ax_main.grid(True, alpha=0.35, linestyle="--", linewidth=0.7)
    ax_main.set_ylim(bottom=0)

    # Add a legend
    lg = ax_main.legend(
        title="Times",
        loc="upper right",
        framealpha=0.95,
        edgecolor="gray",
    )
    lg.get_frame().set_linewidth(1.5)

    # Save the figure
    plt.tight_layout()
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    fname = os.path.join(figures_dir, f"fig6_beta_{beta:.3f}.png".replace("0.", "").replace(".", "_"))
    fig.savefig(fname, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def generate_comparison_panels(panel_times, x0, theta, K_beta, use_cache=True):
    """
    Generates a 2x2 panel figure comparing the PDFs for beta=1/2 and beta=1/3 at different times.

    Args:
        panel_times (list): A list of 4 time points for the panels.
        x0 (float): The initial position.
        theta (float): The relaxation rate.
        K_beta (float): The diffusion coefficient.
        use_cache (bool, optional): Whether to use cached data. Defaults to True.
    """
    print("\nGenerating comparison panels: β = 1/2 vs β = 1/3")
    x_panel = np.linspace(-0.5, 1.5, 600)
    cache_dir = "data"
    cache_filename = os.path.join(cache_dir, "comparison_panels_data.pkl")
    
    # Load or compute data
    if use_cache and os.path.exists(cache_filename):
        print(f"Loading data from cache: {cache_filename}")
        with open(cache_filename, 'rb') as f:
            panel_data = pickle.load(f)
    else:
        print("Computing data for comparison panels...")
        panel_data = {}
        for t in panel_times:
            p_half = compute_pdf_vectorized(
                x_panel, t, x0, beta=0.5, theta=theta, K_beta=K_beta, Ns=800
            )
            p_third = compute_pdf_vectorized(
                x_panel, t, x0, beta=1.0 / 3.0, theta=theta, K_beta=K_beta, Ns=800
            )
            panel_data[t] = (p_half, p_third)
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_filename, 'wb') as f:
            pickle.dump(panel_data, f)

    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey=True)
    axes = axes.flatten()

    for ax, t in zip(axes, panel_times):
        p_half, p_third = panel_data[t]
        ax.plot(
            x_panel, p_half,
            color="#1f77b4",
            linestyle="-",
            lw=2.5,
            label=r"β = 1/2",
            alpha=0.85,
        )
        ax.plot(
            x_panel, p_third,
            color="#ff7f0e",
            linestyle="--",
            lw=2.5,
            label=r"β = 1/3",
            alpha=0.85,
        )

        ax.set_title(f"t = {t}", fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold", pad=12)
        ax.set_xlabel("x", fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
        ax.set_ylim(bottom=0)

        # Calculate and display the L1 distance between the two PDFs
        diff = p_third - p_half
        L1 = np.trapezoid(np.abs(diff), x_panel)
        # Place L1 in a free corner (not under the legend) to avoid covering curves.
        ax.text(
            0.03,
            0.92,
            rf"$L_{{1}} = {L1:.2e}$",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=plt.rcParams["legend.fontsize"],
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                alpha=0.85,
                edgecolor="gray",
            ),
        )

        ax.legend(loc="upper right", framealpha=0.95, edgecolor="gray")

    axes[0].set_ylabel("P(x,t)", fontweight="bold")
    axes[2].set_ylabel("P(x,t)", fontweight="bold")

    plt.suptitle(
        "Comparison: β = 1/2 vs β = 1/3", fontsize=plt.rcParams["axes.titlesize"], fontweight="bold", y=0.995
    )
    plt.tight_layout()
    # Save the figure
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(
        os.path.join(figures_dir, "fig6_comparison_panels.png"), dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)

def _load_or_compute_spectral_data(cache_filename, title_beta_str, tasks, num_cores, use_cache):
    """Helper to abstract away data loading and computation for spectral plots."""
    if use_cache and os.path.exists(cache_filename):
        print(f"Loading data from cache: {cache_filename}")
        with open(cache_filename, 'rb') as f:
            return pickle.load(f)

    print(f"Computing data for spectral comparison ({title_beta_str})...")
    # Use a multiprocessing pool to run computations in parallel
    pool = multiprocessing.Pool(processes=num_cores)
    results = []
    total_steps = len(tasks)
    print_progress(0, total_steps, prefix=f'Progress ({title_beta_str}):', suffix='Complete', length=50)
    for i, result in enumerate(pool.imap_unordered(compute_spec_and_time, tasks)):
        results.append(result)
        print_progress(i + 1, total_steps, prefix=f'Progress ({title_beta_str}):', suffix='Complete', length=50)
    pool.close()
    pool.join()

    # Process results into dictionaries
    timings = {}
    specs = {}
    for t, N, timing, spec_result in results:
        if (t, N) not in timings:
            timings[(t, N)] = []
        timings[(t, N)].append(timing)
        specs[(t, N)] = spec_result
    
    # Cache the computed data
    os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
    with open(cache_filename, 'wb') as f:
        pickle.dump((timings, specs), f)
    
    return timings, specs

def _plot_spectral_subplot(ax, x_spec, ref_pdf, spec_pdf, N, timings_for_N, t):
    """Helper to plot a single panel in the spectral comparison figure."""
    avg_time = np.mean(timings_for_N)
    std_time = np.std(timings_for_N)

    # Plot the reference (integral) and spectral series PDFs
    ax.plot(x_spec, ref_pdf, color="black", lw=2.5, label="integral", alpha=0.8)
    ax.plot(x_spec, spec_pdf, color="#d62728", lw=2, linestyle="--", label=f"N={N}", alpha=0.8)

    # Display L1 distance and computation time on the plot
    L1 = np.trapezoid(np.abs(spec_pdf - ref_pdf), x_spec)
    bbox_props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="gray")
    
    ax.text(0.95, 0.95, rf"$L_{{1}}={L1:.2e}$", transform=ax.transAxes, fontsize=ANNOTATION_SIZE,
            verticalalignment='top', horizontalalignment='right', bbox=bbox_props)

    ax.text(0.05, 0.95, f"Time: {avg_time:.3f} ± {std_time:.3f} s", transform=ax.transAxes,
            fontsize=ANNOTATION_SIZE, verticalalignment='top', bbox=bbox_props)

    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
    ax.set_ylim(bottom=0)


def _get_beta_strings(beta_spec):
    """Helper to get beta strings for titles and filenames."""
    if abs(beta_spec - 0.5) < 1e-12:
        title_beta_str = "β = 1/2 (Smirnov)"
        fname_beta_str = "beta_0_5"
    elif abs(beta_spec - 1.0 / 3.0) < 1e-12:
        title_beta_str = "β = 1/3"
        fname_beta_str = "beta_1_3"
    else:
        title_beta_str = f"β = {beta_spec}"
        fname_beta_str = f"beta_{beta_spec:.3f}".replace(".", "_")
    return title_beta_str, fname_beta_str

def generate_spectral_comparison_plot(
    beta_spec, times_spec, Ns_list, n_repeats, x0, theta, K_beta, num_cores, use_cache=True
):
    """Generates the spectral series vs integral map comparison plot for a given beta."""
    title_beta_str, fname_beta_str = _get_beta_strings(beta_spec)

    print(f"\nGenerating spectral series comparison for {title_beta_str}...")
    x_spec = np.linspace(-0.5, 1.5, 400)
    cache_dir = "data"
    cache_filename = os.path.join(cache_dir, f"spectral_comparison_data_{fname_beta_str}.pkl")

    # Define tasks for parallel computation
    tasks = [(t, N, x_spec, x0, beta_spec, theta, K_beta)
             for t in times_spec for N in Ns_list for _ in range(n_repeats)]
    timings, specs = _load_or_compute_spectral_data(cache_filename, title_beta_str, tasks, num_cores, use_cache)

    # --- Plotting ---
    n_rows = len(times_spec)
    n_cols = len(Ns_list)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 15), sharex=True, sharey=True)

    for i, t in enumerate(times_spec):
        # Compute the reference PDF using the integral method
        ref_pdf = compute_pdf_vectorized(x_spec, t, x0, beta=beta_spec, theta=theta, K_beta=K_beta, Ns=800)
        for j, N in enumerate(Ns_list):
            # Handle different subplot indexing for 1D and 2D arrays
            ax = axes[i, j] if n_rows > 1 and n_cols > 1 else (axes[j] if n_cols > 1 else axes[i])
            _plot_spectral_subplot(ax, x_spec, ref_pdf, specs[(t, N)], N, timings[(t, N)], t)

            # Set titles and labels for the outer plots
            if i == 0:
                ax.set_title(
                    f"N = {N}", fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold", pad=10
                )
            if j == 0:
                ax.set_ylabel(f"t = {t}", fontsize=SPECTRAL_YLABEL_SIZE, fontweight="bold")

            ax.tick_params(axis="both", labelsize=12)

    # Add shared axis labels
    fig.text(0.5, 0.02, "x", ha="center", fontsize=SPECTRAL_LABEL_SIZE, fontweight="bold")
    fig.text(
        0.02,
        0.5,
        "P(x,t)",
        va="center",
        rotation="vertical",
        fontsize=SPECTRAL_LABEL_SIZE,
        fontweight="bold",
    )

    plt.suptitle(
        f"Spectral Series vs Integral Map – {title_beta_str}",
        fontsize=plt.rcParams["axes.titlesize"],
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.99])
    # Save the figure
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(
        os.path.join(figures_dir, f"fig6_spectral_vs_integral_{fname_beta_str}.png"), dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)

def _load_or_compute_timing_data(cache_filename, times_spec, Ns_list_timing, n_repeats, x_spec, x0, beta_spec, theta, K_beta, num_cores, use_cache):
    """Helper to load or compute timing data for the timing plot."""
    if use_cache and os.path.exists(cache_filename):
        print(f"Loading data from cache: {cache_filename}")
        with open(cache_filename, 'rb') as f:
            return pickle.load(f)

    print("Computing data for timing plot...")
    tasks = [(t, N, x_spec, x0, beta_spec, theta, K_beta)
             for t in times_spec for N in Ns_list_timing for _ in range(n_repeats)]

    pool = multiprocessing.Pool(processes=num_cores)
    results = []
    total_steps = len(tasks)
    print_progress(0, total_steps, prefix='Timing Plot Progress:', suffix='Complete', length=50)
    for i, result in enumerate(pool.imap_unordered(compute_spec_and_time, tasks)):
        results.append(result)
        print_progress(i + 1, total_steps, prefix='Timing Plot Progress:', suffix='Complete', length=50)
    pool.close()
    pool.join()

    os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
    with open(cache_filename, 'wb') as f:
        pickle.dump(results, f)
    return results

def _plot_timing_data(timings, times_spec, Ns_list_timing):
    """Helper to plot the computation time vs. N."""
    fig, ax = plt.subplots(figsize=(13, 9))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for t, color in zip(times_spec, colors):
        avg_times = [np.mean(timings.get((t, N), [np.nan])) for N in Ns_list_timing]
        std_times = [np.std(timings.get((t, N), [np.nan])) for N in Ns_list_timing]
        ax.errorbar(
            Ns_list_timing,
            avg_times,
            yerr=std_times,
            label=f"t = {t}",
            color=color,
            capsize=3,
            marker='o',
            markersize=5,
        )

    ax.set_xlabel("N (Number of terms in spectral series)", fontweight="bold")
    ax.set_ylabel("Average computation time (s)", fontweight="bold")
    ax.set_title("Computation time vs. N for β = 1/3", fontweight="bold")
    ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.7)
    ax.legend(title="Time (t)")
    # Linear scale on both axes for presentation clarity
    plt.tight_layout()

    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(os.path.join(figures_dir, "fig_timing_vs_N_beta_1_3.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def generate_timing_plot(num_cores, use_cache=True):
    """Generates the timing vs. N plot for beta = 1/3."""
    print("\nGenerating timing vs. N plot for β = 1/3...")
    beta_spec = 1.0 / 3.0
    cache_dir = "data"
    cache_filename = os.path.join(cache_dir, "timing_plot_data_beta_1_3.pkl")
    times_spec = [0.01, 0.1, 1.0, 10.0, 100.0]
    Ns_list_timing = [5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    n_repeats = 5
    x_spec = np.linspace(-0.5, 1.5, 400)
    theta, K_beta = 1.0, 1.0
    x0 = 0.5

    # Load or compute the timing data
    results = _load_or_compute_timing_data(
        cache_filename, times_spec, Ns_list_timing, n_repeats, x_spec, x0,
        beta_spec, theta, K_beta, num_cores, use_cache
    )

    # Process results into a dictionary for easier plotting
    timings = {}
    for t, N, timing, _ in results:
        if (t, N) not in timings:
            timings[(t, N)] = []
        timings[(t, N)].append(timing)

    _plot_timing_data(timings, times_spec, Ns_list_timing)

def _load_or_compute_frac_vs_nonfrac_data(cache_filename, comparison_times, comparison_betas, comparison_x, x0, theta, K_beta, use_cache):
    """Helper to load or compute data for the fractional vs non-fractional plot."""
    if use_cache and os.path.exists(cache_filename):
        print(f"Loading data from cache: {cache_filename}")
        with open(cache_filename, 'rb') as f:
            return pickle.load(f)

    print("Computing data for fractional vs non-fractional plot...")
    plot_data = {}
    for t in comparison_times:
        pdfs = {}
        for beta in comparison_betas:
            # The non-fractional case (beta=1) is handled analytically in the plotting function
            if abs(beta - 1.0) > 1e-12:
                pdfs[beta] = compute_pdf_vectorized(
                    comparison_x, t, x0, beta=beta, theta=theta, K_beta=K_beta, Ns=800
                )
        plot_data[t] = pdfs
    
    os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
    with open(cache_filename, 'wb') as f:
        pickle.dump(plot_data, f)
    return plot_data

def _plot_frac_vs_nonfrac_data(plot_data, comparison_times, comparison_betas, comparison_x, x0, theta, K_beta):
    """Helper to plot the fractional vs non-fractional comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharey=True)
    axes = axes.flatten()

    colors_comp = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    line_styles = ["-", "--", ":"]

    for i, t in enumerate(comparison_times):
        ax = axes[i]
        for beta_idx, beta in enumerate(comparison_betas):
            # Handle the standard OU case (beta=1) analytically
            if abs(beta - 1.0) < 1e-12:
                mean = x0 * np.exp(-theta * t)
                variance = (K_beta / theta) * (1.0 - np.exp(-2.0 * theta * t))
                nf_pdf = (1.0 / np.sqrt(2.0 * np.pi * variance)) * np.exp(-0.5 * (comparison_x - mean) ** 2 / variance)
                label = "β = 1 (Standard)"
                pdf_to_plot = nf_pdf
            # Handle fractional cases
            else:
                beta_str = "1/2" if abs(beta - 0.5) < 1e-12 else "1/3"
                label = f"β = {beta_str}"
                pdf_to_plot = plot_data[t][beta]

            ax.plot(
                comparison_x, pdf_to_plot, color=colors_comp[beta_idx], linestyle=line_styles[beta_idx],
                linewidth=2.5, alpha=0.85, label=label
            )

        ax.set_title(f"t = {t}", fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold", pad=12)
        ax.set_xlabel("x", fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
        ax.set_ylim(bottom=0)
        if i % 2 == 0:
            ax.set_ylabel("P(x,t)", fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.95, edgecolor="gray")

    plt.suptitle(
        "Comparison: fractional (β ≠ 0) vs standard (β = 1) cases",
        fontsize=plt.rcParams["axes.titlesize"], fontweight="bold", y=0.995
    )
    plt.tight_layout()
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(
        os.path.join(figures_dir, "fig6_fractional_vs_nonfractional.png"),
        dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)

def generate_fractional_vs_nonfractional_plot(comparison_betas, comparison_times, x0, theta, K_beta, use_cache=True):
    """Generates a plot comparing fractional and non-fractional (standard) OU processes."""
    print("\nGenerating comparison: Fractional vs Non-Fractional cases")
    comparison_x = np.linspace(-1.0, 2.0, 400)
    cache_dir = "data"
    cache_filename = os.path.join(cache_dir, "frac_vs_nonfrac_data.pkl")
    
    # Load or compute the necessary data
    plot_data = _load_or_compute_frac_vs_nonfrac_data(
        cache_filename, comparison_times, comparison_betas, comparison_x, x0, theta, K_beta, use_cache
    )
    
    # Generate the plot
    _plot_frac_vs_nonfrac_data(plot_data, comparison_times, comparison_betas, comparison_x, x0, theta, K_beta)