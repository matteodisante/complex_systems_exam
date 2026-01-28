import numpy as np
import math
from scipy.special import kv, eval_hermite, gammaln

def l_beta(beta, z):
    """
    Computes the normalized analytic Lévy densities for specific beta values.
    
    THEORY CONNECTION:
    This function corresponds to the probability density of the stable subordinator.
    It is the building block for the memory kernel M(t) in the Time-Fractional Fokker-Planck Equation (T-FFPE).
    
    Notation:
    - beta: The fractional order of the time derivative (0 < beta <= 1).
    - z: The scaling variable related to time.

    Args:
        beta (float): Stability parameter (fractional exponent), expected in (0, 1].
        z (array_like): Positive scaling variable; shape (N,) or scalar.

    Returns:
        np.ndarray: Lévy PDF values; same shape as `z`.
    """
    z = np.asarray(z)
    out = np.zeros_like(z, dtype=float)
    mask = z > 0
    if not np.any(mask):
        return out

    # Case for beta = 1/2 (Smirnov density).
    # Used in the "Blue" curves in the presentation.
    # Formula:
    # l_{1/2}(z) = (1 / (2 * sqrt(pi))) * z^{-3/2} * exp(-1 / (4 z))
    if abs(beta - 0.5) < 1e-12:
        out[mask] = (
            (1.0 / (2.0 * math.sqrt(math.pi)))
            * z[mask] ** (-1.5)
            * np.exp(-1.0 / (4.0 * z[mask]))
        )
        return out

    # Case for beta = 1/3.
    # Used in the "Orange" curves. Represents a more anomalous process with heavier tails.
    # Formula:
    # l_{1/3}(z) = (1 / (3 * pi)) * z^{-3/2} * K_{1/3}(2 / sqrt(27 z))
    if abs(beta - 1.0 / 3.0) < 1e-12:
        arg = 2.0 / np.sqrt(27.0 * z[mask])
        with np.errstate(over="ignore", invalid="ignore"):
            kv_vals = kv(1.0 / 3.0, arg)
        kv_vals = np.where(np.isfinite(kv_vals), kv_vals, 0.0)
        out[mask] = (1.0 / (3.0 * math.pi)) * z[mask] ** (-1.5) * kv_vals
        out[out < 0] = 0.0  # Ensure positivity
        return out

    raise ValueError(f"Unsupported beta: {beta}. Only 0.5 and 1/3 are implemented.")


def n_function_s_array(s, t, beta=0.5):
    """
    Computes the memory kernel n(s, t) for the subordination integral.

    THEORY CONNECTION:
    This function maps the physical time 't' to the operational time 's' (internal clock).
    It allows the solution of the fractional equation to be written as an integral over 
    the standard Markovian solution (Subordination Map).

    Args:
        s (array_like): Operational/subordination time grid; shape (Ns,) or scalar.
        t (float): Physical time (scalar).
        beta (float, optional): Stability parameter (fractional exponent).

    Returns:
        np.ndarray: Memory kernel values n(s, t); same shape as `s`.
    """
    s = np.asarray(s)
    beta = float(beta)
    
    # Change of variables: z = t / s^(1/beta) used inside l_beta.
    # This ties the operational time distribution to the physical time.
    z = t / (s ** (1.0 / beta))
    lz = l_beta(beta, z)

    # Assemble kernel: n(s,t) = (1/beta) * (t / s^(1+1/beta)) * l_beta(z)
    out = np.zeros_like(s, dtype=float)
    mask = s > 0
    out[mask] = (1.0 / beta) * (t / (s[mask] ** (1.0 + 1.0 / beta))) * lz[mask]
    out[~np.isfinite(out)] = 0.0 
    return out


def ou_kernel(x_grid, s_grid, x0, theta=1.0, K_beta=1.0):
    """
    Computes the propagator P1(x, s | x0) for the standard Ornstein-Uhlenbeck process.

    THEORY CONNECTION (Slide 114):
    This corresponds to the limit beta = 1.
    It describes a Gaussian packet relaxing in a harmonic potential with rate theta.
    
    Notation:
    - theta: Relaxation rate (drift coefficient).
    - K_beta: Generalized diffusion coefficient.

    Args:
        x_grid (array_like): Spatial grid; shape (Nx,).
        s_grid (array_like): Operational time grid; shape (Ns,).
        x0 (float): Initial position (scalar).
        theta (float): Relaxation rate (drift coefficient).
        K_beta (float): Diffusion coefficient.

    Returns:
        np.ndarray: OU propagator P1(x, s | x0); shape (Nx, Ns).
    """
    x = np.asarray(x_grid)
    s = np.asarray(s_grid)

    # Mean and variance of the OU process at operational time s
    # Mean decays exponentially: x0 * exp(-theta * s)
    mean = x0 * np.exp(-theta * s)
    
    # Variance saturates to K_beta / theta as s -> infinity
    variance = (K_beta / theta) * (1.0 - np.exp(-2.0 * theta * s))
    variance[variance <= 1e-16] = 1e-16 

    norm = 1.0 / np.sqrt(2.0 * np.pi * variance)
    
    X = x[:, None]
    M = mean[None, :]
    V = variance[None, :]
    
    # Gaussian propagator P1(x, s | x0)
    P1 = norm[None, :] * np.exp(-0.5 * (X - M) ** 2 / V)
    return P1


def compute_pdf_vectorized(
    x_grid, t, x0, beta=0.5, theta=1.0, K_beta=1.0, s_max=None, Ns=1000
):
    """
    Computes the PDF P(x, t) via numerical integration of the Integral Map (Barkai's method).
    
    THEORY CONNECTION:
    P(x, t) = ∫ P1(x, s) * n(s, t) ds
    This mixes the Markovian Gaussian P1 with the non-Markovian memory kernel n(s, t).

    Args:
        x_grid (array_like): Spatial grid; shape (Nx,).
        t (float): Physical time (scalar).
        x0 (float): Initial position (scalar).
        beta (float, optional): Stability parameter (fractional exponent).
        theta (float, optional): Relaxation rate.
        K_beta (float, optional): Diffusion coefficient.
        s_max (float, optional): Upper cutoff for operational time integration.
        Ns (int, optional): Number of integration points for s.

    Returns:
        np.ndarray: PDF values P(x, t); shape (Nx,).
    """
    if s_max is None:
        s_max = max(200.0, 50.0 * t**beta)

    # Non-uniform grid for efficient integration over s
    # - s_small resolves the sharp behavior near s ≈ 0
    # - s_mid captures the long-time tail up to s_max
    s_small = np.logspace(-8, -2, Ns // 4)
    s_mid = np.logspace(-2, np.log10(max(1.0, s_max)), 3 * Ns // 4)
    s = np.unique(np.concatenate([s_small, s_mid]))

    n_vals = n_function_s_array(s, t, beta=beta)
    P1 = ou_kernel(x_grid, s, x0, theta=theta, K_beta=K_beta)

    # Numerical integration (Trapezoidal rule) over s
    pdf = np.trapezoid(P1 * n_vals, s)
    pdf[pdf < 0] = 0.0 
    return pdf


def mittag_leffler(beta, z):
    """
    Computes the Mittag-Leffler function E_beta(z).
    
    THEORY CONNECTION:
    This function replaces the standard exponential relaxation in fractional systems.
    - For small t: Stretched exponential (fast initial decay).
    - For large t: Power-law decay (heavy tail/memory).

    Args:
        beta (float): Fractional exponent (typically 0 < beta <= 1).
        z (float): Argument of the function (scalar).

    Returns:
        float | None: E_beta(z) if computed, otherwise None.
    """
    try:
        import mpmath
        mp = mpmath.mp
        mp.dps = max(50, mp.dps) 
        beta_mp = mp.mpf(beta)
        z_mp = mp.mpf(z)

        def term(k):
            kmp = mp.mpf(k)
            return (z_mp**kmp) / mp.gamma(beta_mp * kmp + 1)

        val = mpmath.nsum(term, [0, mp.inf])
        return float(val)
    except ImportError:
        print("Warning: mpmath library not found.")
        return None
    except Exception as e:
        print(f"Error in Mittag-Leffler: {e}")
        return None


def spectral_series_pdf(x_grid, t, x0, beta, N, theta=1.0, K_beta=1.0):
    """
    Computes the PDF using the Spectral Series Expansion (Slide 113).
    
     ====================== THEORY MAPPING (SLIDE 113) ======================
     1. Prefactor:
         sqrt(theta / (2 pi K_beta))

     2. Gaussian factor (outside the sum):
         exp( - theta x^2 / (2 K_beta) )

     3. Hermite arguments:
         sqrt(theta / (2 K_beta)) * x  and  sqrt(theta / (2 K_beta)) * x0
     ========================================================================

    Args:
        x_grid (array_like): Spatial grid; shape (Nx,).
        t (float): Physical time (scalar).
        x0 (float): Initial position (scalar).
        beta (float): Fractional exponent (0 < beta <= 1).
        N (int): Number of terms in the truncated series.
        theta (float, optional): Relaxation rate.
        K_beta (float, optional): Generalized diffusion coefficient.

    Returns:
        np.ndarray: PDF values from spectral series; shape (Nx,).
    """
    x_grid = np.asarray(x_grid)

    # Scale factor for Hermite arguments and Gaussian width
    # sqrt(theta / (2 K_beta)) matches the slide definition.
    scale = np.sqrt(theta / (2.0 * K_beta))

    # Prefactor from the stationary Gaussian distribution
    norm_factor = np.sqrt(theta / (2.0 * np.pi * K_beta))

    pdf = np.zeros_like(x_grid, dtype=float)

    # Sum over the eigenstates (truncated series at N terms)
    for n in range(N):
        # Eigenvalue lambda_n = n * theta (Linear spectrum)
        lambda_n = n * theta

        # Temporal relaxation term: E_beta(-n * theta * t^beta)
        ml_arg = -lambda_n * t**beta
        E_n = mittag_leffler(beta, ml_arg)
        
        if E_n is None:
            raise RuntimeError("Mittag-Leffler computation failed.")

        # Normalization coefficient for Hermite polynomials (1 / 2^n n!).
        # Use log-space to avoid overflow in n! and 2^n for large n:
        # log(coeff) = -n * log(2) - log(n!) with log(n!) = gammaln(n + 1).
        # This is numerically equivalent but stable; cost is negligible vs. Hermite evals.
        log_coeff = -n * np.log(2.0) - gammaln(n + 1.0)
        coeff = np.exp(log_coeff)

        # Hermite polynomials evaluated at scaled positions
        H_n_x = eval_hermite(n, scale * x_grid)
        H_n_x0 = eval_hermite(n, scale * x0)

        # Gaussian weight outside the sum: exp(-theta x^2 / (2 K_beta))
        gaussian = np.exp(-(theta * x_grid**2) / (2.0 * K_beta))

        # Summing eigenstates: prefactor * gaussian * H_n(x) H_n(x0)
        pdf += norm_factor * coeff * E_n * H_n_x * H_n_x0 * gaussian

    pdf[pdf < 0] = 0.0
    return pdf