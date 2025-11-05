
import numpy as np
import math
from scipy.special import kv, factorial, eval_hermite

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

    s_small = np.logspace(-8, -2, Ns // 4)
    s_mid = np.logspace(-2, np.log10(max(1.0, s_max)), 3 * Ns // 4)
    s = np.unique(np.concatenate([s_small, s_mid]))

    n_vals = n_function_s_array(s, t, alpha=alpha)

    P1 = ou_kernel(x_grid, s, x0, gamma=gamma, K_beta=K_beta)

    pdf = np.trapezoid(P1 * n_vals, s)

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
    """
    x_grid = np.asarray(x_grid)

    x_tilde = x_grid * np.sqrt(m * omega**2 / (k_B * T))
    x0_tilde = x0 * np.sqrt(m * omega**2 / (k_B * T))

    norm_factor = np.sqrt(m * omega**2 / (2.0 * np.pi * k_B * T))

    pdf = np.zeros_like(x_grid, dtype=float)

    for n in range(N):
        lambda_n = n * gamma

        ml_arg = -lambda_n * t**alpha
        E_n = mittag_leffler(alpha, ml_arg)

        coeff = 1.0 / (2**n * factorial(n))

        H_n_x_tilde = eval_hermite(n, x_tilde / np.sqrt(2.0))
        H_n_x0_tilde = eval_hermite(n, x0_tilde / np.sqrt(2.0))

        gaussian = np.exp(-(x_tilde**2) / 2.0)

        pdf += norm_factor * coeff * E_n * H_n_x_tilde * H_n_x0_tilde * gaussian

    pdf[pdf < 0] = 0.0
    return pdf
