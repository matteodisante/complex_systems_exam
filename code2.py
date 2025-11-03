#!/usr/bin/env python3
"""
Minimal, self-contained script to reproduce Figure 6 (fractional OU, α=1/2).

Method:
 - Use analytical Smirnov form for the Lévy density l_{1/2}(z).
 - Build an s-grid (log-spaced at small s, log-spaced to large s) and compute
   n(s,t) on that grid.
 - Compute P1(x,s) (OU Gaussian kernel) on a vectorized grid and evaluate
   P(x,t) = ∫ n(s,t) P1(x,s) ds using a weighted dot product (trapezoidal rule).

This is designed to be fast and portable; it only requires numpy, scipy, matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math
from scipy.special import kv
from scipy.special import eval_hermite, gammaln
# Mittag-Leffler function wrapper:
# Prefer SciPy's implementation (if available). Otherwise try mpmath. If
# neither exposes a callable, fall back to a direct series expansion
# E_alpha(z) = sum_{k=0}^∞ z^k / Gamma(alpha*k + 1).
def mittag_leffler(alpha, z):
    # Attempt SciPy implementation first (present in newer SciPy)
    try:
        from scipy.special import mittag_leffler as _scipy_mittag
        return _scipy_mittag(alpha, z)
    except Exception:
        pass

    # Try mpmath (names vary across versions). Use mpmath's nsum on the
    # power-series if a direct mittag function isn't available; this avoids
    # overflow/underflow issues for moderate-to-large |z|.
    try:
        import mpmath
        mp = mpmath.mp
        # set a reasonable precision for these special sums
        mp.dps = max(50, mp.dps)
        if hasattr(mpmath, 'mittag_leffler'):
            return float(mpmath.mittag_leffler(alpha, z))
        if hasattr(mpmath, 'mittag'):
            return float(mpmath.mittag(alpha, z))

        # otherwise use direct nsum of the series: sum_{k=0}^∞ z^k / Gamma(alpha*k + 1)
        alpha_mp = mp.mpf(alpha)
        z_mp = mp.mpf(z)
        def term(k):
            kmp = mp.mpf(k)
            return (z_mp ** kmp) / mp.gamma(alpha_mp * kmp + 1)

        val = mpmath.nsum(term, [0, mp.inf])
        return float(val)
    except Exception:
        # fall back to a log-space series (less robust for huge |z| but
        # acceptable in many practical cases)
        pass

    # Final fallback: compute series expansion in log-space to reduce over/underflow
    try:
        zf = float(z)
    except Exception:
        zf = complex(z)

    max_terms = 2000
    tol = 1e-12
    if zf == 0.0:
        return 1.0

    # use log-space to avoid intermediate overflow for z^k when |z|>1
    log_abs_z = math.log(abs(zf))
    s = 0.0
    for k in range(0, max_terms):
        # log_term = k*log(|z|) - log(Gamma(alpha*k + 1))
        log_term = k * log_abs_z - gammaln(alpha * k + 1.0)
        # if log_term indicates extremely small term, break
        if log_term < -50 and k > 10:
            break
        try:
            term_mag = math.exp(log_term)
        except OverflowError:
            # extremely large intermediate; abort and return current sum
            break
        if zf < 0:
            term = term_mag * ((-1) ** k)
        else:
            term = term_mag
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


def l_smirnov_unnormalized(z):
    """Unnormalized Smirnov Lévy density for α=1/2.
    l(z) ∝ z^{-3/2} exp(-1/(4 z)) for z>0
    """
    z = np.asarray(z)
    out = np.zeros_like(z, dtype=float)
    mask = z > 0
    out[mask] = (1.0 / (2.0 * math.sqrt(math.pi))) * z[mask]**(-1.5) * np.exp(-1.0 / (4.0 * z[mask]))
    return out

def l_alpha_unnormalized(alpha, z):
    """Unnormalized analytic Lévy densities for supported alphas (1/2 and 1/3).

    For α=1/2 (Smirnov): l(z) = (1/(2 sqrt(pi))) z^{-3/2} exp(-1/(4z)).
    For α=1/3: l(z) = (1/(3π)) z^{-3/2} K_{1/3}(2 / sqrt(27 z)).
    """
    z = np.asarray(z)
    out = np.zeros_like(z, dtype=float)
    mask = z > 0
    if not np.any(mask):
        return out

    if abs(alpha - 0.5) < 1e-12:
        out[mask] = (1.0 / (2.0 * math.sqrt(math.pi))) * z[mask]**(-1.5) * np.exp(-1.0 / (4.0 * z[mask]))
        return out

    if abs(alpha - 1.0/3.0) < 1e-12:
        # arg for Bessel K
        arg = 2.0 / np.sqrt(27.0 * z[mask])
        # safe evaluation
        with np.errstate(over='ignore', invalid='ignore'):
            kv_vals = kv(1.0/3.0, arg)
        # some kv may be nan/inf for tiny arg; set to 0 there
        kv_vals = np.where(np.isfinite(kv_vals), kv_vals, 0.0)
        out[mask] = (1.0 / (3.0 * math.pi)) * z[mask]**(-1.5) * kv_vals
        # numerical floor
        out[out < 0] = 0.0
        return out

    # unsupported alpha
    raise ValueError(f"Unsupported alpha: {alpha}")


def compute_l_normalization():
    """Compute normalization constant ∫_0^∞ l_unnorm(z) dz (should be 1 theoretically).
    We integrate numerically from 0 to large value.
    """
    f = lambda z: (1.0 / (2.0 * math.sqrt(math.pi))) * z**(-1.5) * math.exp(-1.0 / (4.0 * z)) if z>0 else 0.0
    val, err = quad(f, 0, 200, limit=200)
    return val

def compute_l_normalization(alpha):
    """Compute normalization ∫_0^∞ l_alpha_unnormalized(z) dz numerically."""
    if abs(alpha - 0.5) < 1e-12:
        f = lambda z: (1.0 / (2.0 * math.sqrt(math.pi))) * z**(-1.5) * math.exp(-1.0 / (4.0 * z)) if z>0 else 0.0
    elif abs(alpha - 1.0/3.0) < 1e-12:
        def f(z):
            if z <= 0:
                return 0.0
            arg = 2.0 / math.sqrt(27.0 * z)
            try:
                kvv = float(kv(1.0/3.0, arg))
            except Exception:
                kvv = 0.0
            val = (1.0 / (3.0 * math.pi)) * z**(-1.5) * kvv
            return max(val, 0.0)
    else:
        raise ValueError(f"Unsupported alpha for normalization: {alpha}")

    val, err = quad(f, 0, 500, limit=400)
    return val


def n_function_s_array(s, t, alpha=0.5, l_norm=None):
    """Compute n(s,t) on array s for given t and alpha=1/2.
    n(s,t) = (1/alpha) * (t / s^{1+1/alpha}) * l_alpha( t / s^{1/alpha} )
    For α=1/2: 1/alpha=2, 1+1/alpha=3, z = t / s^{2}
    """
    s = np.asarray(s)
    alpha = float(alpha)
    z = t / (s**(1.0/alpha))
    lz = l_alpha_unnormalized(alpha, z)
    if l_norm is not None:
        lz = lz / l_norm
    # Avoid division by zero for s->0 by masking
    out = np.zeros_like(s, dtype=float)
    mask = s > 0
    out[mask] = (1.0/alpha) * (t / (s[mask]**(1.0 + 1.0/alpha))) * lz[mask]
    # replace any non-finite with zero
    out[~np.isfinite(out)] = 0.0
    return out


def ou_kernel(x_grid, s_grid, x0, gamma=1.0, T=1.0):
    """Return P1(x,s) array with shape (len(x_grid), len(s_grid)).
    P1(x,s) = Normal(x; mean = x0 e^{-γ s}, variance = (T/γ)(1 - e^{-2 γ s}))
    """
    x = np.asarray(x_grid)
    s = np.asarray(s_grid)
    # shape (len(x), len(s)) via broadcasting
    mean = x0 * np.exp(-gamma * s)            # shape (len(s),)
    variance = (T / gamma) * (1.0 - np.exp(-2.0 * gamma * s))  # shape (len(s),)
    # when s is extremely small, variance ~ 0; avoid zero by setting tiny floor
    variance[variance <= 1e-16] = 1e-16
    norm = 1.0 / np.sqrt(2.0 * np.pi * variance)
    # Compute (x - mean)^2 / (2 variance) with broadcasting
    X = x[:, None]
    M = mean[None, :]
    V = variance[None, :]
    P1 = norm[None, :] * np.exp(-0.5 * (X - M)**2 / V)
    return P1


def compute_pdf_vectorized(x_grid, t, x0, alpha=0.5, gamma=1.0, T=1.0,
                           s_max=None, Ns=800):
    """Compute P(x,t) for array x_grid using vectorized s-grid integration.
    Returns pdf array of same length as x_grid.
    """
    # prepare s-grid: capture very small s with log-spacing and larger s with log to s_max
    if s_max is None:
        s_max = max(200.0, 50.0 * t**alpha)

    # small s region to capture cusp contribution
    s_small = np.logspace(-8, -2, Ns//4)
    s_mid = np.logspace(-2, np.log10(max(1.0, s_max)), 3*Ns//4)
    s = np.unique(np.concatenate([s_small, s_mid]))

    # normalization for l(z)
    l_norm = compute_l_normalization(alpha)

    n_vals = n_function_s_array(s, t, alpha=alpha, l_norm=l_norm)

    # trapezoidal weights for s integration
    ds = np.diff(s)
    # create weights array same length as s
    weights = np.empty_like(s)
    weights[0] = ds[0] / 2.0
    weights[-1] = ds[-1] / 2.0
    weights[1:-1] = 0.5 * (ds[:-1] + ds[1:])

    # integrand weights = n(s) * ds
    integrand_weights = n_vals * weights

    # compute OU kernel P1(x,s)
    P1 = ou_kernel(x_grid, s, x0, gamma=gamma, T=T)

    # integrate: for each x, P(x,t) = sum_s P1(x,s) * n(s)*ds
    pdf = P1.dot(integrand_weights)
    # ensure non-negative (numerical noise)
    pdf[pdf < 0] = 0.0
    return pdf


def spectral_series_pdf(x_grid, t, x0, alpha, N, omega2_over_eta=1.0):
    """Compute spectral series approximation (Eq.18-like) for fractional OU.

    We use the eigenfunctions of the OU (Hermite functions) and Mittag-Leffler
    temporal factor E_alpha(-lambda_n t^alpha). Assumes lambda_n = n * omega2_over_eta
    and gamma (friction) incorporated in omega2_over_eta if needed.
    """
    x = np.asarray(x_grid)
    # We'll compute orthonormal Hermite functions psi_n(x) via a
    # stable three-term recurrence:
    #   psi_0(x) = pi^{-1/4} * exp(-x^2/2)
    #   psi_1(x) = sqrt(2) * x * psi_0(x)
    #   psi_{n+1} = (sqrt(2)/(sqrt(n+1))) * x * psi_n - sqrt(n/(n+1)) * psi_{n-1}
    # This avoids evaluating large Hermite polynomials and explicit factorials
    # and is numerically stable for moderately large n.

    x = np.asarray(x)
    M = x.size
    # prepare recurrence for psi_n(x) and psi_n(x0) while summing series
    psi_nm1_x = (np.pi ** -0.25) * np.exp(-0.5 * x**2)        # psi_0(x)
    psi_n_x = np.sqrt(2.0) * x * psi_nm1_x                   # psi_1(x)
    psi_nm1_x0 = (np.pi ** -0.25) * math.exp(-0.5 * x0**2)   # psi_0(x0)
    psi_n_x0 = math.sqrt(2.0) * x0 * psi_nm1_x0              # psi_1(x0)

    # initialize series with n=0 term
    series = np.zeros(M, dtype=float)
    # eigenvalue for n=0 is 0 -> E_alpha(0) = 1
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

    # iterate for n >= 2 using recurrence and add terms incrementally
    psi_prev_x = psi_nm1_x
    psi_curr_x = psi_n_x
    psi_prev_x0 = psi_nm1_x0
    psi_curr_x0 = psi_n_x0

    for n in range(1, N-1):
        # compute psi_{n+1}
        nn = n
        coef1 = math.sqrt(2.0 / (nn + 1.0))
        coef2 = math.sqrt(nn / (nn + 1.0))
        psi_next_x = coef1 * x * psi_curr_x - coef2 * psi_prev_x
        psi_next_x0 = coef1 * x0 * psi_curr_x0 - coef2 * psi_prev_x0

        # eigenvalue and Mittag-Leffler factor
        lambda_n = (nn + 1) * omega2_over_eta
        z = -lambda_n * (t**alpha)
        E_n = mittag_leffler(alpha, z)

        series += E_n * psi_next_x0 * psi_next_x

        # shift for next iteration
        psi_prev_x, psi_curr_x = psi_curr_x, psi_next_x
        psi_prev_x0, psi_curr_x0 = psi_curr_x0, psi_next_x0

    # numerical floor
    series[series < 0] = 0.0
    return series


def main():
    # Produce separate slide-ready figures for each alpha
    alphas = [0.5, 1.0/3.0]
    gamma = 1.0
    T = 1.0
    x0 = 0.5
    times = [0.02, 0.2, 2.0, 20.0, 200.0]

    x_values = np.linspace(-3.0, 3.0, 300)

    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    for alpha in alphas:
        fig = plt.figure(figsize=(12, 7))
        ax_main = fig.add_subplot(1, 1, 1)
        print(f"Generating figure for alpha={alpha:.3f}...")
        for t, c in zip(times, colors):
            print(f"  Computing pdf for t={t}...")
            pdf = compute_pdf_vectorized(x_values, t, x0, alpha=alpha, gamma=gamma, T=T, Ns=800)
            ax_main.plot(x_values, pdf, color=c, lw=3, label=f't = {t}')
            # print PDF values at x=0 and x=1 (nearest grid points)
            idx0 = int(np.argmin(np.abs(x_values - 0.0)))
            idx1 = int(np.argmin(np.abs(x_values - 1.0)))
            print(f"    P(x=0, t={t}) = {pdf[idx0]:.6e}, P(x=1, t={t}) = {pdf[idx1]:.6e}")

        # stationary and x0 marker
        variance_stat = T / gamma
        stationary = 1.0 / np.sqrt(2.0 * np.pi * variance_stat) * np.exp(-0.5 * gamma * x_values**2 / T)
        ax_main.plot(x_values, stationary, color='k', linestyle=':', lw=2, label='Stationary')
        ax_main.axvline(x=x0, color='0.5', linestyle=':', label=f'x0 = {x0}')

        ax_main.set_xlabel('x')
        ax_main.set_ylabel('P(x,t)')
        # Title with alpha written as a fraction for readability on slides
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
        # make legend box a little larger and readable
        lg.get_frame().set_alpha(0.9)

        plt.tight_layout()
        fname = f'fig6_alpha_{alpha:.3f}.png'.replace('0.', '').replace('.', '_')
        # Save slide-ready PNG
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        print(f'  Saved {fname}')
        plt.close(fig)

    # (removed concise comparison summary per user request)

    # (removed shaded-overlay comparison per user request)

    # --- Option A: 2x2 panels for times short, medium, long, very long ---
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
        # compute L1 difference on the plotted window and annotate
        diff = p_third - p_half
        L1 = np.trapezoid(np.abs(diff), x_panel)
        ax.text(0.02, 0.92, f'L1 = {L1:.2e}', transform=ax.transAxes,
                verticalalignment='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        ax.legend(loc='upper right', fontsize=10)
    axes[0].set_ylabel('P(x,t)', fontsize=14)
    plt.suptitle('Comparison panels: α = 1/2 vs α = 1/3', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('fig6_comparison_panels.png', dpi=300, bbox_inches='tight')
    print('Saved fig6_comparison_panels.png')
    plt.close(fig)

    # --- Spectral series comparison (Eq.18) for alpha = 1/3 ---
    print('\nSpectral series comparison (α = 1/3) vs integral map solution')
    alpha_spec = 1.0/3.0
    times_spec = times  # reuse [0.02, 0.2, 2, 20, 200]
    Ns_list = [5, 20, 100, 500]
    # plotting grid: rows = times, cols = Ns
    n_rows = len(times_spec)
    n_cols = len(Ns_list)
    x_spec = np.linspace(-0.5, 1.5, 400)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 2.5*n_rows), sharex=True, sharey=True)
    for i, t in enumerate(times_spec):
        # compute reference integral solution once per time
        ref = compute_pdf_vectorized(x_spec, t, x0, alpha=alpha_spec, gamma=gamma, T=T, Ns=800)
        for j, N in enumerate(Ns_list):
            ax = axes[i, j] if n_rows>1 else axes[j]
            print(f'  time={t}, N={N}: computing spectral series (may take a moment)')
            try:
                spec = spectral_series_pdf(x_spec, t, x0, alpha_spec, N, omega2_over_eta=1.0)
            except RuntimeError as e:
                print('  ERROR computing Mittag-Leffler:', e)
                raise
            ax.plot(x_spec, ref, color='k', lw=2.5, label='integral')
            ax.plot(x_spec, spec, color='C1', lw=2, linestyle='--', label=f'series N={N}')
            # annotate L1 difference on plotted window
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

    # (removed wide-range L1 check and plot per user request)


if __name__ == '__main__':
    main()
