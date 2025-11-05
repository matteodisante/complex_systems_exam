#!/usr/bin/env python3
"""
Minimal, self-contained script to reproduce some key figures for the
fractional OU, (α=1/2 and α=1/3) and compare them with the non-fractional
case (α=0).

Method:
 - Use analytical normalized Smirnov form for the Lévy density l_{1/2}(z).
 - Build an s-grid (log-spaced at small s, log-spaced to large s) and
   compute n(s,t) on that grid.
 - Compute P1(x,s) (OU Gaussian kernel) on a vectorized grid and evaluate
   P(x,t) = ∫ n(s,t) P1(x,s) ds using a weighted dot product
   (trapezoidal rule).

Lévy densities are already normalized, so no post-normalization of P(x,t) is
needed.
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from scipy.special import kv, factorial
from scipy.special import eval_hermite

# Improve default plotting for slide readability
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "legend.title_fontsize": 12,
        "lines.linewidth": 3.0,
    }
)


def l_alpha(alpha, z):
    """Normalized analytic Lévy densities for α=1/2 (Smirnov) and α=1/3.

    For α=1/2 (Smirnov): l(z) = (1/(2√π)) z^{-3/2} exp(-1/(4z))
    For α=1/3: l(z) = (1/(3π)) z^{-3/2} K_{1/3}(2 / √(27z))

    These PDFs are ALREADY NORMALIZED
    """
    z = np.asarray(z)
    out = np.zeros_like(z, dtype=float)
    mask = z > 0
    if not np.any(mask):
        return out

    if abs(alpha - 0.5) < 1e-12:
        # Smirnov normalized
        out[mask] = (
            (1.0 / (2.0 * math.sqrt(math.pi)))
            * z[mask] ** (-1.5)
            * np.exp(-1.0 / (4.0 * z[mask]))
        )
        return out

    if abs(alpha - 1.0 / 3.0) < 1e-12:
        # α=1/3 normalized (with Bessel K)
        arg = 2.0 / np.sqrt(27.0 * z[mask])
        with np.errstate(over="ignore", invalid="ignore"):
            kv_vals = kv(1.0 / 3.0, arg)
        kv_vals = np.where(np.isfinite(kv_vals), kv_vals, 0.0)
        out[mask] = (1.0 / (3.0 * math.pi)) * z[mask] ** (-1.5) * kv_vals
        out[out < 0] = 0.0
        return out

    raise ValueError(f"Unsupported alpha: {alpha}")


def n_function_s_array(s, t, alpha=0.5):
    """Compute n(s,t) on array s for given t and alpha.

    n(s,t) = (1/α) * (t / s^{1+1/α}) * l_α(t / s^{1/α})

    For α=1/2: 1/α=2, 1+1/α=3, z = t / s^2
    For α=1/3: 1/α=3, 1+1/α=4, z = t / s^3
    """
    s = np.asarray(s)
    alpha = float(alpha)
    z = t / (s ** (1.0 / alpha))
    lz = l_alpha(alpha, z)  # Already normalized

    # Compute n(s,t)
    out = np.zeros_like(s, dtype=float)
    mask = s > 0  # Avoid division by zero
    out[mask] = (1.0 / alpha) * (t / (s[mask] ** (1.0 + 1.0 / alpha))) * lz[mask]
    out[~np.isfinite(out)] = 0.0  # Handle inf
    return out


def ou_kernel(x_grid, s_grid, x0, gamma=1.0, K_beta=1.0):
    """Return P1(x,s) array with shape (len(x_grid), len(s_grid)).

    P1(x,s) = Normal(x; mean = x0 e^{-γ s},
    variance = (K_beta/γ)(1 - e^{-2 γ s}))
    """
    x = np.asarray(x_grid)
    s = np.asarray(s_grid)

    mean = x0 * np.exp(-gamma * s)
    variance = (K_beta / gamma) * (1.0 - np.exp(-2.0 * gamma * s))
    variance[variance <= 1e-16] = 1e-16

    norm = 1.0 / np.sqrt(2.0 * np.pi * variance)
    X = x[:, None]
    M = mean[None, :]
    V = variance[None, :]
    P1 = norm[None, :] * np.exp(-0.5 * (X - M) ** 2 / V)
    return P1


def compute_pdf_vectorized(
    x_grid, t, x0, alpha=0.5, gamma=1.0, K_beta=1.0, s_max=None, Ns=1000
):
    """Compute P(x,t) for array x_grid using vectorized s-grid integration.

    Returns pdf array of same length as x_grid.
    """
    if s_max is None:
        s_max = max(200.0, 50.0 * t**alpha)

    # Create a composite s-grid for numerical integration.
    # The function n(s,t) varies over many orders of magnitude, so a non-uniform
    # grid is necessary for an accurate and efficient integration.
    # We use a high density of points at small s, where n(s,t) can be singular
    # or change rapidly, and a sparser, log-spaced grid for larger s.

    # Part 1: High-resolution log-spaced grid for small s.
    s_small = np.logspace(-8, -2, Ns // 4)
    # Part 2: Log-spaced grid for the rest of the range up to s_max.
    s_mid = np.logspace(-2, np.log10(max(1.0, s_max)), 3 * Ns // 4)
    # Combine the two parts and remove duplicates to form the final grid.
    s = np.unique(np.concatenate([s_small, s_mid]))

    # Compute n(s,t) with already-normalized Lévy densities
    n_vals = n_function_s_array(s, t, alpha=alpha)

    # Compute OU kernel P1(x,s)
    P1 = ou_kernel(x_grid, s, x0, gamma=gamma, K_beta=K_beta)

    # Integrate: for each x, P(x,t) = sum_s P1(x,s) * n(s)*ds using dot product.
    pdf = np.trapezoid(P1 * n_vals, s)

    # Ensure pdf is non-negative
    pdf[pdf < 0] = 0.0
    return pdf


def mittag_leffler(alpha, z):
    """Compute Mittag-Leffler function E_alpha(z)."""

    try:
        import mpmath

        mp = mpmath.mp
        mp.dps = max(50, mp.dps)
        alpha_mp = mp.mpf(alpha)
        z_mp = mp.mpf(z)

        def term(k):
            kmp = mp.mpf(k)
            return (z_mp**kmp) / mp.gamma(alpha_mp * kmp + 1)

        val = mpmath.nsum(term, [0, mp.inf])
        return float(val)
    except Exception:
        pass


def spectral_series_pdf(x_grid, t, x0, alpha, N, m, omega, k_B, T, gamma=1.0):
    """
    Exact implementation of Equation (18) from the paper for harmonic potential.

    This computes the PDF for a fractional OU process with harmonic potential
    V(x) = (1/2)*m*ω²*x²
    using the exact spectral series with non-normalized Hermite polynomials.

    Equation (18) from the paper:
    W = √(mω²/2πk_B T) * Σ_{n=0}^∞ [1/(2^n n!)] * E_alpha(-lambda_n * t^alpha)
        * H_n(x̃/√2) * H_n(x̃'/√2) * exp(-x̃²/2)
    Parameters:
    -----------
    x_grid : array
        Spatial grid points
    t : float
        Time
    x0 : float
        Initial position
    alpha : float
        Fractional order (0 < alpha ≤ 1)
    N : int
        Number of terms in the series
    m : float
        Mass
    omega : float
        Angular frequency
    k_B : float
        Boltzmann constant
    T : float
        Temperature
    gamma : float
        omega^2/eta_beta

    Returns:
    --------
    pdf : array
        The probability density function at x_grid
    """
    x_grid = np.asarray(x_grid)

    # Dimensionless coordinates (scaled as in eq. 18)
    x_tilde = x_grid * np.sqrt(m * omega**2 / (k_B * T))
    x0_tilde = x0 * np.sqrt(m * omega**2 / (k_B * T))

    # Normalization factor (from eq. 18)
    norm_factor = np.sqrt(m * omega**2 / (2.0 * np.pi * k_B * T))

    # Initialize the PDF
    pdf = np.zeros_like(x_grid, dtype=float)

    # Sum over the spectral series
    for n in range(N):
        # Eigenvalue: λ_{n,beta} = n * γ
        lambda_n = n * gamma

        # Mittag-Leffler temporal factor: E_α(-λ_n * t^α)
        ml_arg = -lambda_n * t**alpha
        E_n = mittag_leffler(alpha, ml_arg)

        # Coefficient from the series: 1/(2^n * n!)
        coeff = 1.0 / (2**n * factorial(n))

        # Non-normalized Hermite polynomials evaluated at scaled coordinates
        H_n_x_tilde = eval_hermite(n, x_tilde / np.sqrt(2.0))
        H_n_x0_tilde = eval_hermite(n, x0_tilde / np.sqrt(2.0))

        # Gaussian factor: exp(-x̃²/2)
        gaussian = np.exp(-(x_tilde**2) / 2.0)

        # Add this term to the series
        pdf += norm_factor * coeff * E_n * H_n_x_tilde * H_n_x0_tilde * gaussian

    # Ensure non-negative
    pdf[pdf < 0] = 0.0
    return pdf


def main():
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
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # ========== Main figures for each alpha ==========
    for alpha in alphas:
        fig = plt.figure(figsize=(14, 8))
        ax_main = fig.add_subplot(1, 1, 1)
        print(f"Generating figure for alpha={alpha:.3f}...")

        for t, c in zip(times, colors):
            print(f"  Computing pdf for t={t}...")
            pdf = compute_pdf_vectorized(
                x_values, t, x0, alpha=alpha, gamma=gamma, K_beta=K_beta, Ns=800
            )
            ax_main.plot(x_values, pdf, color=c, lw=2.5, label=f"t = {t}", alpha=0.85)

            # Print PDF values
            idx0 = int(np.argmin(np.abs(x_values - 0.0)))
            idx1 = int(np.argmin(np.abs(x_values - 1.0)))
            print(
                f"    P(x=0, t={t}) = {pdf[idx0]:.6e}, "
                f"P(x=1, t={t}) = {pdf[idx1]:.6e}"
            )

        # Stationary distribution and x0 marker
        variance_stat = K_beta / gamma
        stationary = (
            1.0
            / np.sqrt(2.0 * np.pi * variance_stat)
            * np.exp(-0.5 * gamma * x_values**2 / K_beta)
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
        ax_main.axvline(
            x=x0,
            color="gray",
            linestyle=":",
            linewidth=2,
            alpha=0.6,
            label=f"x₀ = {x0}",
        )

        ax_main.set_xlabel("x", fontsize=14, fontweight="bold")
        ax_main.set_ylabel("P(x,t)", fontsize=14, fontweight="bold")

        # Title with α as fraction
        if abs(alpha - 0.5) < 1e-12:
            title_alpha = r"α = 1/2 (Smirnov)"
        elif abs(alpha - 1.0 / 3.0) < 1e-12:
            title_alpha = r"α = 1/3"
        else:
            title_alpha = f"α = {alpha}"

        ax_main.set_title(
            f"Fractional OU Process: {title_alpha}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax_main.grid(True, alpha=0.35, linestyle="--", linewidth=0.7)
        ax_main.set_ylim(bottom=0)

        lg = ax_main.legend(
            title="Times",
            fontsize=11,
            title_fontsize=12,
            loc="upper right",
            framealpha=0.95,
            edgecolor="gray",
        )
        lg.get_frame().set_linewidth(1.5)

        plt.tight_layout()
        fname = f"fig6_alpha_{alpha:.3f}.png".replace("0.", "").replace(".", "_")
        fig.savefig(fname, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved {fname}\n")
        plt.close(fig)

    # ========== Comparison panels for α=1/2 vs α=1/3 ==========
    print("Generating comparison panels: α = 1/2 vs α = 1/3")
    panel_times = [0.01, 0.1, 1.0, 10.0, 100.0]
    x_panel = np.linspace(-0.5, 1.5, 600)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()

    for ax, t in zip(axes, panel_times):
        p_half = compute_pdf_vectorized(
            x_panel, t, x0, alpha=0.5, gamma=gamma, K_beta=K_beta, Ns=800
        )
        p_third = compute_pdf_vectorized(
            x_panel, t, x0, alpha=1.0 / 3.0, gamma=gamma, K_beta=K_beta, Ns=800
        )

        ax.plot(
            x_panel,
            p_half,
            color="#1f77b4",
            linestyle="-",
            lw=2.5,
            label=r"α = 1/2",
            alpha=0.85,
        )
        ax.plot(
            x_panel,
            p_third,
            color="#ff7f0e",
            linestyle="--",
            lw=2.5,
            label=r"α = 1/3",
            alpha=0.85,
        )

        ax.set_title(f"t = {t}", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("x", fontsize=12)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
        ax.set_ylim(bottom=0)

        diff = p_third - p_half
        L1 = np.trapezoid(np.abs(diff), x_panel)
        ax.text(
            0.65,
            0.95,
            f"L¹ = {L1:.2e}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=11,
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor="white",
                alpha=0.85,
                edgecolor="gray",
            ),
        )

        ax.legend(loc="upper right", fontsize=11, framealpha=0.95, edgecolor="gray")

    axes[0].set_ylabel("P(x,t)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("P(x,t)", fontsize=12, fontweight="bold")

    plt.suptitle(
        "Comparison: α = 1/2 vs α = 1/3", fontsize=16, fontweight="bold", y=0.995
    )
    plt.tight_layout()
    fig.savefig(
        "fig6_comparison_panels.png", dpi=300, bbox_inches="tight", facecolor="white"
    )
    print("Saved fig6_comparison_panels.png\n")
    plt.close(fig)

    # ========== Spectral series vs integral solution (α=1/3) ==========
    print("Spectral series comparison (α = 1/3) vs integral map solution")
    alpha_spec = 1.0 / 3.0
    times_spec = times
    Ns_list = [5, 20, 100, 200]
    n_rows = len(times_spec)
    n_cols = len(Ns_list)
    x_spec = np.linspace(-0.5, 1.5, 400)

    # New: For timing
    n_repeats = 5

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12), sharex=True, sharey=True)

    for i, t in enumerate(times_spec):
        ref = compute_pdf_vectorized(
            x_spec, t, x0, alpha=alpha_spec, gamma=gamma, K_beta=K_beta, Ns=800
        )
        for j, N in enumerate(Ns_list):
            ax = axes[i, j] if n_rows > 1 else axes[j]
            print(f"  time={t}, N={N}: computing spectral series")

            # New: timing loop
            timings = []
            for _ in range(n_repeats):
                start_time = time.perf_counter()
                spec = spectral_series_pdf(
                    x_spec, t, x0, alpha_spec, N, m, omega, k_B, T, gamma
                )
                end_time = time.perf_counter()
                timings.append(end_time - start_time)
            
            avg_time = np.mean(timings)
            std_time = np.std(timings)

            ax.plot(x_spec, ref, color="black", lw=2.5, label="integral", alpha=0.8)
            ax.plot(
                x_spec,
                spec,
                color="#d62728",
                lw=2,
                linestyle="--",
                label=f"N={N}",
                alpha=0.8,
            )

            L1 = np.trapezoid(np.abs(spec - ref), x_spec)
            ax.text(
                0.95,
                0.95,
                f"L¹={L1:.2e}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="gray",
                ),
            )

            # New: Add timing information
            ax.text(
                0.05,
                0.95,
                f"Time: {avg_time:.3f} ± {std_time:.3f} s",
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='top',
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="gray",
                ),
            )

            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
            ax.set_ylim(bottom=0)

            if i == 0:
                ax.set_title(
                    f"N = {N}", fontsize=13, fontweight="bold", pad=10
                )
            if j == 0:
                ax.set_ylabel(f"t = {t}", fontsize=12, fontweight="bold")

    fig.text(0.5, 0.02, "x", ha="center", fontsize=13, fontweight="bold")
    fig.text(
        0.02,
        0.5,
        "P(x,t)",
        va="center",
        rotation="vertical",
        fontsize=13,
        fontweight="bold",
    )

    plt.suptitle(
        "Spectral Series (Eq.18) vs Integral Map – α = 1/3",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.99])
    fig.savefig(
        "fig6_spectral_vs_integral.png", dpi=300, bbox_inches="tight", facecolor="white"
    )
    print("Saved fig6_spectral_vs_integral.png\n")
    plt.close(fig)

    # ========== Comparison between fractional and non-fractional cases ==========
    print("Generating comparison: Fractional vs Non-Fractional cases")
    comparison_alphas = [0.5, 1.0 / 3.0, 0.0]
    comparison_times = [0.01, 0.1, 1.0, 10.0]
    comparison_x = np.linspace(-1.0, 2.0, 400)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    axes = axes.flatten()

    colors_comp = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    line_styles = ["-", "--", ":"]

    for i, t in enumerate(comparison_times):
        for alpha_idx, alpha in enumerate(comparison_alphas):
            if abs(alpha - 0.0) < 1e-12:
                # Non-fractional case (α=0)
                mean = x0 * np.exp(-gamma * t)
                variance = (K_beta / gamma) * (1.0 - np.exp(-2.0 * gamma * t))
                nf_pdf = (
                    1.0
                    / np.sqrt(2.0 * np.pi * variance)
                    * np.exp(-0.5 * (comparison_x - mean) ** 2 / variance)
                )
                axes[i].plot(
                    comparison_x,
                    nf_pdf,
                    color=colors_comp[alpha_idx],
                    linestyle=line_styles[alpha_idx],
                    linewidth=2.5,
                    alpha=0.85,
                    label="α = 0 (Standard)",
                )
            else:
                # Fractional cases
                frac_pdf = compute_pdf_vectorized(
                    comparison_x, t, x0, alpha=alpha, gamma=gamma, K_beta=K_beta, Ns=800
                )

                alpha_str = "1/2" if abs(alpha - 0.5) < 1e-12 else "1/3"
                alpha_label = f"α = {alpha_str}"

                axes[i].plot(
                    comparison_x,
                    frac_pdf,
                    color=colors_comp[alpha_idx],
                    linestyle=line_styles[alpha_idx],
                    linewidth=2.5,
                    alpha=0.85,
                    label=alpha_label,
                )

        axes[i].set_title(f"t = {t}", fontsize=14, fontweight="bold", pad=12)
        axes[i].set_xlabel("x", fontsize=12)
        axes[i].grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
        axes[i].set_ylim(bottom=0)

        if i == 0 or i == 2:
            axes[i].set_ylabel("P(x,t)", fontsize=12, fontweight="bold")

        axes[i].legend(
            loc="upper right", fontsize=11, framealpha=0.95, edgecolor="gray"
        )

    plt.suptitle(
        "Comparison: Fractional (α ≠ 0) vs Non-Fractional (α = 0) Cases",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    fig.savefig(
        "fig6_fractional_vs_nonfractional.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    print("Saved fig6_fractional_vs_nonfractional.png\n")
    plt.close(fig)

    print("=" * 70)
    print("All figures generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()