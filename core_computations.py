import numpy as np
import math
from scipy.special import kv, factorial, eval_hermite

def l_alpha(alpha, z):
    """
    Computes the normalized analytic Lévy densities for specific alpha values.

    This function provides the probability density functions (PDFs) for stable distributions
    used in the context of fractional calculus. It currently supports alpha=1/2 (Smirnov distribution)
    and alpha=1/3. These distributions are fundamental for describing subordinating processes
    in fractional Fokker-Planck equations.

    Args:
        alpha (float): The stability parameter. Must be close to 0.5 or 1/3.
        z (array_like): The random variable, must be positive.

    Returns:
        np.ndarray: The value of the Lévy PDF at each point in z.
    """
    z = np.asarray(z)
    out = np.zeros_like(z, dtype=float)
    mask = z > 0
    if not np.any(mask):
        return out

    # Case for alpha = 1/2, the Smirnov distribution.
    if abs(alpha - 0.5) < 1e-12:
        # Formula: l(z) = (1 / (2 * sqrt(pi))) * z^(-3/2) * exp(-1 / (4z))
        out[mask] = (
            (1.0 / (2.0 * math.sqrt(math.pi)))
            * z[mask] ** (-1.5)
            * np.exp(-1.0 / (4.0 * z[mask]))
        )
        return out

    # Case for alpha = 1/3, using the modified Bessel function of the second kind (kv).
    if abs(alpha - 1.0 / 3.0) < 1e-12:
        # Formula: l(z) = (1 / (3 * pi)) * z^(-3/2) * K_{1/3}(2 / sqrt(27z))
        arg = 2.0 / np.sqrt(27.0 * z[mask])
        with np.errstate(over="ignore", invalid="ignore"):
            kv_vals = kv(1.0 / 3.0, arg)
        kv_vals = np.where(np.isfinite(kv_vals), kv_vals, 0.0)
        out[mask] = (1.0 / (3.0 * math.pi)) * z[mask] ** (-1.5) * kv_vals
        out[out < 0] = 0.0  # Ensure positivity
        return out

    raise ValueError(f"Unsupported alpha: {alpha}. Only 0.5 and 1/3 are implemented.")


def n_function_s_array(s, t, alpha=0.5):
    """
    Computes the memory kernel n(s, t) for a given time t and stability parameter alpha.

    This function represents the distribution of subordination times 's' at a physical time 't'.
    It is derived from the Lévy density l_alpha and is a key component in solving the
    fractional Fokker-Planck equation via subordination.

    Args:
        s (array_like): The subordination time variable.
        t (float): The physical time.
        alpha (float, optional): The stability parameter. Defaults to 0.5.

    Returns:
        np.ndarray: The value of the memory kernel for each s.
    """
    s = np.asarray(s)
    alpha = float(alpha)
    
    # Transformation of variables for l_alpha
    z = t / (s ** (1.0 / alpha))
    lz = l_alpha(alpha, z)

    # Formula: n(s,t) = (1/α) * (t / s^(1+1/α)) * l_α(t / s^(1/α))
    out = np.zeros_like(s, dtype=float)
    mask = s > 0  # Avoid division by zero
    out[mask] = (1.0 / alpha) * (t / (s[mask] ** (1.0 + 1.0 / alpha))) * lz[mask]
    out[~np.isfinite(out)] = 0.0  # Clean up any non-finite values
    return out


def ou_kernel(x_grid, s_grid, x0, gamma=1.0, K_beta=1.0):
    """
    Computes the propagator for the standard Ornstein-Uhlenbeck (OU) process.

    This function gives the probability P1(x, s | x0) of finding the particle at position x
    at time s, given it started at x0. This is the solution to the standard Fokker-Planck
    equation, which will be subordinated to solve the fractional case.

    Args:
        x_grid (array_like): Grid of spatial points x.
        s_grid (array_like): Grid of subordination times s.
        x0 (float): The initial position.
        gamma (float, optional): The friction/damping coefficient. Defaults to 1.0.
        K_beta (float, optional): The diffusion coefficient (related to temperature). Defaults to 1.0.

    Returns:
        np.ndarray: A 2D array of shape (len(x_grid), len(s_grid)) containing the probabilities.
    """
    x = np.asarray(x_grid)
    s = np.asarray(s_grid)

    # Mean and variance of the OU process at time s
    mean = x0 * np.exp(-gamma * s)
    variance = (K_beta / gamma) * (1.0 - np.exp(-2.0 * gamma * s))
    variance[variance <= 1e-16] = 1e-16  # Avoid division by zero for small s

    # Gaussian PDF for the OU process
    norm = 1.0 / np.sqrt(2.0 * np.pi * variance)
    
    # Use broadcasting to efficiently compute the PDF over the grids
    X = x[:, None]
    M = mean[None, :]
    V = variance[None, :]
    P1 = norm[None, :] * np.exp(-0.5 * (X - M) ** 2 / V)
    return P1


def compute_pdf_vectorized(
    x_grid, t, x0, alpha=0.5, gamma=1.0, K_beta=1.0, s_max=None, Ns=1000
):
    """
    Computes the PDF P(x, t) of the fractional OU process via numerical integration.

    This function implements the subordination method: P(x,t) = integral from 0 to inf of
    P1(x,s) * n(s,t) ds, where P1 is the standard OU propagator and n(s,t) is the memory kernel.
    The integration is performed numerically over a non-uniform grid for s.

    Args:
        x_grid (array_like): Grid of spatial points x.
        t (float): The physical time.
        x0 (float): The initial position.
        alpha (float, optional): The stability parameter. Defaults to 0.5.
        gamma (float, optional): The friction coefficient. Defaults to 1.0.
        K_beta (float, optional): The diffusion coefficient. Defaults to 1.0.
        s_max (float, optional): The upper limit for the s-integration. If None, it's estimated.
        Ns (int, optional): The number of points for the s-integration grid.

    Returns:
        np.ndarray: The computed PDF P(x,t) for each point in x_grid.
    """
    if s_max is None:
        # Heuristic for the upper integration limit, needs to be large enough
        s_max = max(200.0, 50.0 * t**alpha)

    # Create a non-uniform grid for s to capture behavior at both small and large s
    s_small = np.logspace(-8, -2, Ns // 4)
    s_mid = np.logspace(-2, np.log10(max(1.0, s_max)), 3 * Ns // 4)
    s = np.unique(np.concatenate([s_small, s_mid]))

    # Compute the two components of the integrand
    n_vals = n_function_s_array(s, t, alpha=alpha)
    P1 = ou_kernel(x_grid, s, x0, gamma=gamma, K_beta=K_beta)

    # Perform the trapezoidal integration over s
    pdf = np.trapezoid(P1 * n_vals, s)

    pdf[pdf < 0] = 0.0  # Ensure positivity
    return pdf


def mittag_leffler(alpha, z):
    """
    Computes the Mittag-Leffler function E_alpha(z) using the mpmath library.

    The Mittag-Leffler function is a generalization of the exponential function and appears
    frequently in the solutions of fractional differential equations.
    E_alpha(z) = sum_{k=0 to inf} (z^k / Gamma(alpha*k + 1)).

    Args:
        alpha (float): The alpha parameter of the function.
        z (float): The argument of the function.

    Returns:
        float: The computed value of E_alpha(z). Returns None if mpmath is not available.
    """
    try:
        import mpmath
        # Use high-precision arithmetic for accurate summation
        mp = mpmath.mp
        mp.dps = max(50, mp.dps)  # Set decimal precision
        alpha_mp = mp.mpf(alpha)
        z_mp = mp.mpf(z)

        # Define the k-th term of the series
        def term(k):
            kmp = mp.mpf(k)
            return (z_mp**kmp) / mp.gamma(alpha_mp * kmp + 1)

        # Sum the series to infinity
        val = mpmath.nsum(term, [0, mp.inf])
        return float(val)
    except ImportError:
        print("Warning: mpmath library not found. Mittag-Leffler function is not available.")
        return None
    except Exception as e:
        print(f"An error occurred in Mittag-Leffler computation: {e}")
        return None


def spectral_series_pdf(x_grid, t, x0, alpha, N, m, omega, k_B, T, gamma=1.0):
    """
    Computes the PDF P(x,t) using the spectral series expansion (Equation 18 from the paper).

    This provides an exact solution for the fractional OU process in a harmonic potential,
    expressed as a series of Hermite polynomials. It is computationally intensive but serves
    as a benchmark for the numerical integration method.

    Args:
        x_grid (array_like): Grid of spatial points x.
        t (float): The physical time.
        x0 (float): The initial position.
        alpha (float): The stability parameter.
        N (int): The number of terms to include in the series expansion.
        m (float): Mass of the particle.
        omega (float): Angular frequency of the harmonic potential.
        k_B (float): Boltzmann constant.
        T (float): Temperature.
        gamma (float, optional): The friction coefficient. Defaults to 1.0.

    Returns:
        np.ndarray: The computed PDF P(x,t) from the series expansion.
    """
    x_grid = np.asarray(x_grid)

    # Rescale variables to their dimensionless form (tilde variables in the paper)
    x_tilde = x_grid * np.sqrt(m * omega**2 / (k_B * T))
    x0_tilde = x0 * np.sqrt(m * omega**2 / (k_B * T))

    # Normalization factor for the PDF
    norm_factor = np.sqrt(m * omega**2 / (2.0 * np.pi * k_B * T))

    pdf = np.zeros_like(x_grid, dtype=float)

    # Sum the first N terms of the spectral series
    for n in range(N):
        # Eigenvalue of the Fokker-Planck operator
        lambda_n = n * gamma

        # Argument for the Mittag-Leffler function
        ml_arg = -lambda_n * t**alpha
        E_n = mittag_leffler(alpha, ml_arg)
        
        if E_n is None:
            raise RuntimeError("Mittag-Leffler computation failed. Check if mpmath is installed.")

        # Coefficient for the n-th term
        coeff = 1.0 / (2**n * factorial(n))

        # Hermite polynomials evaluated at rescaled positions
        H_n_x_tilde = eval_hermite(n, x_tilde / np.sqrt(2.0))
        H_n_x0_tilde = eval_hermite(n, x0_tilde / np.sqrt(2.0))

        # Gaussian part of the basis function
        gaussian = np.exp(-(x_tilde**2) / 2.0)

        # Add the n-th term to the total PDF
        pdf += norm_factor * coeff * E_n * H_n_x_tilde * H_n_x0_tilde * gaussian

    pdf[pdf < 0] = 0.0  # Ensure positivity
    return pdf