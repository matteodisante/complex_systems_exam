#!/usr/bin/env python3
"""
Minimal, self-contained script to reproduce Figure 6 (fractional OU, α=1/2 and α=1/3).

Method:
 - Use analytical normalized Smirnov form for the Lévy density l_{1/2}(z).
 - Build an s-grid (log-spaced at small s, log-spaced to large s) and compute
   n(s,t) on that grid.
 - Compute P1(x,s) (OU Gaussian kernel) on a vectorized grid and evaluate
   P(x,t) = ∫ n(s,t) P1(x,s) ds using a weighted dot product (trapezoidal rule).

Lévy densities are treated as already normalized: ∫_0^∞ l_α(z) dz = 1

This is designed to be fast and portable; it only requires numpy, scipy, matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math
from scipy.special import kv
from scipy.special import eval_hermite, gammaln

# Mittag-Leffler function wrapper
_mittag_method = None  # Track which method is being used

def mittag_leffler(alpha, z):
    """Compute Mittag-Leffler function E_alpha(z)."""
    global _mittag_method
    
    # Try scipy first (preferred)
    if _mittag_method != "scipy":
        try:
            from scipy.special import mittag_leffler as _scipy_mittag
            _mittag_method = "scipy"
            return _scipy_mittag(alpha, z)
        except Exception:
            pass

    # Try mpmath (high precision)
    if _mittag_method != "mpmath":
        try:
            import mpmath
            mp = mpmath.mp
            mp.dps = max(50, mp.dps)
            if hasattr(mpmath, 'mittag_leffler'):
                _mittag_method = "mpmath"
                return float(mpmath.mittag_leffler(alpha, z))
            if hasattr(mpmath, 'mittag'):
                _mittag_method = "mpmath"
                return float(mpmath.mittag(alpha, z))
            alpha_mp = mp.mpf(alpha)
            z_mp = mp.mpf(z)
            def term(k):
                kmp = mp.mpf(k)
                return (z_mp ** kmp) / mp.gamma(alpha_mp * kmp + 1)
            val = mpmath.nsum(term, [0, mp.inf])
            _mittag_method = "mpmath"
            return float(val)
        except Exception:
            pass

    # Fallback: series in log-space
    _mittag_method = "series"
    try:
        zf = float(z)
    except Exception:
        zf = complex(z)

    if zf == 0.0:
        return 1.0

    max_terms = 2000
    tol = 1e-12
    log_abs_z = math.log(abs(zf))
    s = 0.0
    
    for k in range(max_terms):
        log_term = k * log_abs_z - gammaln(alpha * k + 1.0)
        if log_term < -50 and k > 10:
            break
        try:
            term_mag = math.exp(log_term)
        except OverflowError:
            break
        term = term_mag * ((-1) ** k) if zf < 0 else term_mag
        s += term
        if abs(term) < tol * max(1.0, abs(s)):
            break

    return s

# Improve default plotting for slide readability
plt.rcParams.update({
    'figure.dpi': 150,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'legend.title_fontsize': 12,
    'lines.linewidth': 3.0,
})


def l_alpha(alpha, z):
    """Normalized analytic Lévy densities for α=1/2 (Smirnov) and α=1/3.

    For α=1/2 (Smirnov): l(z) = (1/(2√π)) z^{-3/2} exp(-1/(4z))
    For α=1/3: l(z) = (1/(3π)) z^{-3/2} K_{1/3}(2 / √(27z))
    
    These are ALREADY NORMALIZED: ∫_0^∞ l(z) dz = 1
    """
    z = np.asarray(z)
    out = np.zeros_like(z, dtype=float)
    mask = z > 0
    if not np.any(mask):
        return out

    if abs(alpha - 0.5) < 1e-12:
        # Smirnov normalized
        out[mask] = (1.0 / (2.0 * math.sqrt(math.pi))) * z[mask]**(-1.5) * np.exp(-1.0 / (4.0 * z[mask]))
        return out

    if abs(alpha - 1.0/3.0) < 1e-12:
        # α=1/3 normalized (with Bessel K)
        arg = 2.0 / np.sqrt(27.0 * z[mask])
        with np.errstate(over='ignore', invalid='ignore'):
            kv_vals = kv(1.0/3.0, arg)
        kv_vals = np.where(np.isfinite(kv_vals), kv_vals, 0.0)
        out[mask] = (1.0 / (3.0 * math.pi)) * z[mask]**(-1.5) * kv_vals
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
    z = t / (s**(1.0/alpha))
    lz = l_alpha(alpha, z)  # Already normalized
    
    # Compute n(s,t)
    out = np.zeros_like(s, dtype=float)
    mask = s > 0
    out[mask] = (1.0/alpha) * (t / (s[mask]**(1.0 + 1.0/alpha))) * lz[mask]
    out[~np.isfinite(out)] = 0.0
    return out


def ou_kernel(x_grid, s_grid, x0, gamma=1.0, T=1.0):
    """Return P1(x,s) array with shape (len(x_grid), len(s_grid)).
    
    P1(x,s) = Normal(x; mean = x0 e^{-γ s}, variance = (T/γ)(1 - e^{-2 γ s}))
    """
    x = np.asarray(x_grid)
    s = np.asarray(s_grid)
    
    mean = x0 * np.exp(-gamma * s)
    variance = (T / gamma) * (1.0 - np.exp(-2.0 * gamma * s))
    variance[variance <= 1e-16] = 1e-16
    
    norm = 1.0 / np.sqrt(2.0 * np.pi * variance)
    X = x[:, None]
    M = mean[None, :]
    V = variance[None, :]
    P1 = norm[None, :] * np.exp(-0.5 * (X - M)**2 / V)
    return P1


def compute_pdf_vectorized(x_grid, t, x0, alpha=0.5, gamma=1.0, T=1.0, s_max=None, Ns=800):
    """Compute P(x,t) for array x_grid using vectorized s-grid integration.
    
    Returns pdf array of same length as x_grid.
    """
    if s_max is None:
        s_max = max(200.0, 50.0 * t**alpha)

    # s-grid: log-spaced at small s and up to s_max
    s_small = np.logspace(-8, -2, Ns//4)
    s_mid = np.logspace(-2, np.log10(max(1.0, s_max)), 3*Ns//4)
    s = np.unique(np.concatenate([s_small, s_mid]))

    # Compute n(s,t) with already-normalized Lévy densities
    n_vals = n_function_s_array(s, t, alpha=alpha)

    # Trapezoidal weights for s integration
    ds = np.diff(s)
    weights = np.empty_like(s)
    weights[0] = ds[0] / 2.0
    weights[-1] = ds[-1] / 2.0
    weights[1:-1] = 0.5 * (ds[:-1] + ds[1:])

    integrand_weights = n_vals * weights

    # Compute OU kernel P1(x,s)
    P1 = ou_kernel(x_grid, s, x0, gamma=gamma, T=T)

    # Integrate: for each x, P(x,t) = sum_s P1(x,s) * n(s)*ds
    pdf = P1.dot(integrand_weights)
    pdf[pdf < 0] = 0.0
    return pdf


def spectral_series_pdf(x_grid, t, x0, alpha, N, omega2_over_eta=1.0):
    """Compute spectral series approximation (Eq.18-like) for fractional OU.

    Uses eigenfunctions (Hermite functions) and Mittag-Leffler temporal factor
    E_alpha(-lambda_n t^alpha). Assumes lambda_n = n * omega2_over_eta.
    """
    x = np.asarray(x_grid)
    
    # Orthonormal Hermite functions via recurrence
    psi_nm1_x = (np.pi ** -0.25) * np.exp(-0.5 * x**2)
    psi_n_x = np.sqrt(2.0) * x * psi_nm1_x
    psi_nm1_x0 = (np.pi ** -0.25) * math.exp(-0.5 * x0**2)
    psi_n_x0 = math.sqrt(2.0) * x0 * psi_nm1_x0

    series = np.zeros(x.size, dtype=float)
    
    # n=0 term (eigenvalue λ_0 = 0)
    lambda_0 = 0.0
    E0 = mittag_leffler(alpha, -lambda_0 * (t**alpha))
    series += E0 * psi_nm1_x0 * psi_nm1_x

    if N == 1:
        series[series < 0] = 0.0
        return series

    # n=1 term
    lambda_1 = 1.0 * omega2_over_eta
    E1 = mittag_leffler(alpha, -lambda_1 * (t**alpha))
    series += E1 * psi_n_x0 * psi_n_x

    # Recurrence for n >= 2
    psi_prev_x = psi_nm1_x
    psi_curr_x = psi_n_x
    psi_prev_x0 = psi_nm1_x0
    psi_curr_x0 = psi_n_x0

    for n in range(1, N-1):
        nn = n
        coef1 = math.sqrt(2.0 / (nn + 1.0))
        coef2 = math.sqrt(nn / (nn + 1.0))
        psi_next_x = coef1 * x * psi_curr_x - coef2 * psi_prev_x
        psi_next_x0 = coef1 * x0 * psi_curr_x0 - coef2 * psi_prev_x0

        lambda_n = (nn + 1) * omega2_over_eta
        z = -lambda_n * (t**alpha)
        E_n = mittag_leffler(alpha, z)

        series += E_n * psi_next_x0 * psi_next_x

        psi_prev_x, psi_curr_x = psi_curr_x, psi_next_x
        psi_prev_x0, psi_curr_x0 = psi_curr_x0, psi_next_x0

    series[series < 0] = 0.0
    return series


def main():
    """Generate Figure 6 plots for α=1/2 (Smirnov) and α=1/3."""
    global _mittag_method
    
    print("="*70)
    print("Figure 6: Fractional OU Process (normalized Lévy densities)")
    print("="*70)
    print(f"Using Mittag-Leffler method: {_mittag_method if _mittag_method else 'auto-detecting...'}\n")
    
    alphas = [0.5, 1.0/3.0]
    gamma = 1.0
    T = 1.0
    x0 = 0.5
    times = [0.02, 0.2, 2.0, 20.0, 200.0]
    x_values = np.linspace(-3.0, 3.0, 300)
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    # --- Main figure: separate plots for each α ---
    for alpha in alphas:
        fig = plt.figure(figsize=(12, 7))
        ax_main = fig.add_subplot(1, 1, 1)
        print(f"Generating figure for alpha={alpha:.3f}...")
        
        for t, c in zip(times, colors):
            print(f"  Computing pdf for t={t}...")
            pdf = compute_pdf_vectorized(x_values, t, x0, alpha=alpha, gamma=gamma, T=T, Ns=800)
            ax_main.plot(x_values, pdf, color=c, lw=3, label=f't = {t}')
            
            # Print PDF values
            idx0 = int(np.argmin(np.abs(x_values - 0.0)))
            idx1 = int(np.argmin(np.abs(x_values - 1.0)))
            print(f"    P(x=0, t={t}) = {pdf[idx0]:.6e}, P(x=1, t={t}) = {pdf[idx1]:.6e}")

        # Stationary distribution and x0 marker
        variance_stat = T / gamma
        stationary = 1.0 / np.sqrt(2.0 * np.pi * variance_stat) * np.exp(-0.5 * gamma * x_values**2 / T)
        ax_main.plot(x_values, stationary, color='k', linestyle=':', lw=2, label='Stationary')
        ax_main.axvline(x=x0, color='0.5', linestyle=':', label=f'x0 = {x0}')

        ax_main.set_xlabel('x')
        ax_main.set_ylabel('P(x,t)')
        
        # Title with α as fraction
        if abs(alpha - 0.5) < 1e-12:
            title_alpha = r'α = 1/2'
        elif abs(alpha - 1.0/3.0) < 1e-12:
            title_alpha = r'α = 1/3'
        else:
            title_alpha = f'α = {alpha}'
        
        ax_main.set_title(f'Figure ({title_alpha})')
        ax_main.grid(True, alpha=0.35)
        ax_main.set_ylim(bottom=0)
        lg = ax_main.legend(title='Times', fontsize=12, loc='upper right')
        lg.get_frame().set_alpha(0.9)

        plt.tight_layout()
        fname = f'fig6_alpha_{alpha:.3f}.png'.replace('0.', '').replace('.', '_')
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        print(f'  Saved {fname}\n')
        plt.close(fig)

    # --- Comparison panels: α=1/2 vs α=1/3 ---
    panel_times = [0.02, 0.2, 20.0, 200.0]
    x_panel = np.linspace(-0.5, 1.5, 600)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()
    
    for ax, t in zip(axes, panel_times):
        p_half = compute_pdf_vectorized(x_panel, t, x0, alpha=0.5, gamma=gamma, T=T, Ns=800)
        p_third = compute_pdf_vectorized(x_panel, t, x0, alpha=1.0/3.0, gamma=gamma, T=T, Ns=800)
        ax.plot(x_panel, p_half, color='C0', linestyle='-', lw=3, label=r'$\alpha=1/2$')
        ax.plot(x_panel, p_third, color='C0', linestyle='--', lw=3, label=r'$\alpha=1/3$')
        ax.set_title(f't = {t}', fontsize=16)
        ax.set_xlabel('x', fontsize=14)
        ax.grid(True, alpha=0.25)
        
        diff = p_third - p_half
        L1 = np.trapezoid(np.abs(diff), x_panel)
        ax.text(0.02, 0.92, f'L1 = {L1:.2e}', transform=ax.transAxes,
                verticalalignment='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        ax.legend(loc='upper right', fontsize=10)
    
    axes[0].set_ylabel('P(x,t)', fontsize=14)
    plt.suptitle('Comparison panels: α = 1/2 vs α = 1/3', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('fig6_comparison_panels.png', dpi=300, bbox_inches='tight')
    print('Saved fig6_comparison_panels.png\n')
    plt.close(fig)

    # --- Spectral series vs integral solution (α=1/3) ---
    print('Spectral series comparison (α = 1/3) vs integral map solution')
    alpha_spec = 1.0/3.0
    times_spec = times
    Ns_list = [5, 20, 100, 500]
    n_rows = len(times_spec)
    n_cols = len(Ns_list)
    x_spec = np.linspace(-0.5, 1.5, 400)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 2.5*n_rows), sharex=True, sharey=True)
    
    for i, t in enumerate(times_spec):
        ref = compute_pdf_vectorized(x_spec, t, x0, alpha=alpha_spec, gamma=gamma, T=T, Ns=800)
        for j, N in enumerate(Ns_list):
            ax = axes[i, j] if n_rows > 1 else axes[j]
            print(f'  time={t}, N={N}: computing spectral series')
            spec = spectral_series_pdf(x_spec, t, x0, alpha_spec, N, omega2_over_eta=1.0)
            
            ax.plot(x_spec, ref, color='k', lw=2.5, label='integral')
            ax.plot(x_spec, spec, color='C1', lw=2, linestyle='--', label=f'series N={N}')
            
            L1 = np.trapezoid(np.abs(spec - ref), x_spec)
            ax.text(0.02, 0.92, f'L1={L1:.2e}', transform=ax.transAxes, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))
            
            if i == 0:
                ax.set_title(f'N={N}', fontsize=14)
            if j == 0:
                ax.set_ylabel(f't={t}', fontsize=12)
            ax.grid(True, alpha=0.25)
    
    plt.suptitle('Spectral series (Eq.18) vs integral map — α = 1/3', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('fig6_spectral_vs_integral.png', dpi=300, bbox_inches='tight')
    print('Saved fig6_spectral_vs_integral.png')
    plt.close(fig)


if __name__ == '__main__':
    main()