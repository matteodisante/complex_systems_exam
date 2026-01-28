import unittest
import numpy as np
from scipy.integrate import quad
from core_computations import (
    mittag_leffler,
    l_beta,
    n_function_s_array,
    ou_kernel,
    compute_pdf_vectorized,
    spectral_series_pdf,
)


class TestCoreComputations(unittest.TestCase):
    """
    Unit tests for the functions in core_computations.py.
    These tests verify the correctness of the mathematical implementations.
    """

    def test_mittag_leffler(self):
        """Tests the Mittag-Leffler function against known values."""
        # For beta=1, E_1(z) is the standard exponential function exp(z).
        self.assertAlmostEqual(mittag_leffler(1, 1), np.exp(1), places=5)
        self.assertAlmostEqual(mittag_leffler(1, 0), 1, places=5)
        
        # For beta=2, E_2(z) is cosh(sqrt(z)). For z=-1, this is cos(1).
        self.assertAlmostEqual(mittag_leffler(2, 0), 1, places=5)
        self.assertAlmostEqual(mittag_leffler(2, -1), np.cos(1), places=5)

    def test_l_beta(self):
        """Tests the Lévy density function l_beta for correctness and normalization."""
        z = np.array([1, 2, 3])

        # Test for beta = 0.5 (Smirnov distribution)
        result_half = l_beta(0.5, z)
        self.assertEqual(result_half.shape, z.shape)
        self.assertTrue(np.all(result_half >= 0), "PDF values should be non-negative.")

        # Test for beta = 1/3
        result_third = l_beta(1.0 / 3.0, z)
        self.assertEqual(result_third.shape, z.shape)
        self.assertTrue(np.all(result_third >= 0), "PDF values should be non-negative.")

        # Test that an unsupported beta raises a ValueError
        with self.assertRaises(ValueError):
            l_beta(0.8, z)

        # Test that the PDFs are normalized (integrate to 1)
        norm_half, _ = quad(lambda z_int: l_beta(0.5, z_int), 0, np.inf)
        print(f"\nNumerical normalization for beta=0.5: {norm_half}")
        self.assertAlmostEqual(norm_half, 1.0, places=4, msg="l_beta for beta=0.5 should be normalized.")

        norm_third, _ = quad(lambda z_int: l_beta(1.0 / 3.0, z_int), 0, np.inf)
        print(f"Numerical normalization for beta=1/3: {norm_third}")
        self.assertAlmostEqual(norm_third, 1.0, places=4, msg="l_beta for beta=1/3 should be normalized.")

    def test_n_function_s_array(self):
        """Tests the memory kernel function n(s, t) for basic properties."""
        s = np.array([1, 2, 3])
        t = 1.0
        result = n_function_s_array(s, t)
        self.assertEqual(result.shape, s.shape)
        self.assertTrue(np.all(np.isfinite(result)), "All values should be finite.")

    def test_ou_kernel(self):
        """Tests the Ornstein-Uhlenbeck propagator for shape and finite values."""
        x_grid = np.linspace(-1, 1, 10)
        s_grid = np.linspace(0.1, 1, 5)
        x0 = 0.0
        result = ou_kernel(x_grid, s_grid, x0)
        self.assertEqual(result.shape, (10, 5))
        self.assertTrue(np.all(np.isfinite(result)), "All values should be finite.")

    def test_compute_pdf_vectorized(self):
        """Tests the vectorized PDF computation for shape and non-negativity."""
        x_grid = np.linspace(-1, 1, 10)
        t = 1.0
        x0 = 0.0
        pdf = compute_pdf_vectorized(x_grid, t, x0)
        self.assertEqual(pdf.shape, x_grid.shape)
        self.assertTrue(np.all(pdf >= 0), "PDF values should be non-negative.")

    def test_spectral_series_pdf(self):
        """Tests the spectral series PDF for basic properties."""
        x_grid = np.linspace(-1, 1, 10)
        t = 1.0
        x0 = 0.0
        beta = 0.5
        N = 10
        m, omega, k_B, T, gamma = 1.0, 1.0, 1.0, 1.0, 1.0
        pdf = spectral_series_pdf(x_grid, t, x0, beta, N, m, omega, k_B, T, gamma)
        self.assertEqual(pdf.shape, x_grid.shape)
        self.assertTrue(np.all(pdf >= 0), "PDF values should be non-negative.")

    def test_spectral_series_pdf_normalization(self):
        """Tests that the spectral series PDF is properly normalized (integrates to 1)."""
        x_grid = np.linspace(-5, 5, 500)  # Use a wide grid for accurate integration
        t = 1.0
        x0 = 0.5
        beta = 1.0 / 3.0
        N = 20  # A reasonable number of terms for convergence
        m, omega, k_B, T, gamma = 1.0, 1.0, 1.0, 1.0, 1.0

        pdf = spectral_series_pdf(x_grid, t, x0, beta, N, m, omega, k_B, T, gamma)

        # Numerically integrate the PDF using the trapezoidal rule
        integral_val = np.trapezoid(pdf, x_grid)
        print(
            f"\nNumerical normalization for spectral series (N={N}, t={t}): "
            f"{integral_val}"
        )
        self.assertAlmostEqual(integral_val, 1.0, places=3, msg="The spectral series PDF should integrate to 1.")

    def test_spectral_series_stationary_state(self):
        """Tests if the spectral series PDF converges to the known stationary state for large t."""
        x_grid = np.linspace(-5, 5, 500)
        t_large = 100.0  # A large time for the system to reach equilibrium
        x0 = 0.5
        beta = 1.0 / 3.0
        N = 5  # Few terms are needed as higher-order terms decay quickly
        m, omega, k_B, T, gamma = 1.0, 1.0, 1.0, 1.0, 1.0

        # 1. Calculate the PDF from the spectral series at a large time
        pdf_spectral = spectral_series_pdf(
            x_grid, t_large, x0, beta, N, m, omega, k_B, T, gamma
        )

        # 2. The analytical stationary solution is the Boltzmann distribution for a harmonic potential.
        # V(x) = 0.5 * m * omega^2 * x^2. P_st(x) ~ exp(-V(x)/(k_B T)).
        # For the given parameters, this simplifies to a standard normal distribution.
        pdf_stationary = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_grid**2)

        # 3. Compare the two distributions using the L1 norm (integral of absolute difference)
        l1_diff = np.trapezoid(np.abs(pdf_spectral - pdf_stationary), x_grid)
        print(f"L1 difference from stationary state at t={t_large}: {l1_diff}")
        self.assertLess(l1_diff, 1e-2, "The PDF at large t should be close to the stationary distribution.")

    def test_spectral_series_vs_standard_ou(self):
        """
        Tests that for beta=1, the spectral series solution matches the standard OU process.
        
        The Mittag-Leffler function E_1(-z) is exp(-z), so the spectral solution should reduce
        to the standard Ornstein-Uhlenbeck solution when beta is 1.
        """
        x_grid = np.linspace(-3, 3, 500)
        t = 1.0
        x0 = 0.5
        beta = 1.0
        N = 50  # Use a higher number of terms for better accuracy
        m, omega, k_B, T, gamma = 1.0, 1.0, 1.0, 1.0, 1.0
        K_beta = 1.0

        # 1. Calculate the PDF from the spectral series with beta=1
        pdf_spectral = spectral_series_pdf(
            x_grid, t, x0, beta, N, m, omega, k_B, T, gamma
        )

        # 2. Calculate the analytical PDF for the standard OU process
        mean_ou = x0 * np.exp(-gamma * t)
        variance_ou = (K_beta / gamma) * (1 - np.exp(-2 * gamma * t))
        pdf_ou_analytical = (1.0 / np.sqrt(2 * np.pi * variance_ou)) * np.exp(
            -0.5 * (x_grid - mean_ou) ** 2 / variance_ou
        )

        # 3. Compare the two results
        l1_diff = np.trapezoid(np.abs(pdf_spectral - pdf_ou_analytical), x_grid)
        print(
            f"L1 difference between spectral (beta=1) and standard OU at t={t}: "
            f"{l1_diff}"
        )
        self.assertLess(l1_diff, 1e-3, "For beta=1, spectral solution should match standard OU solution.")


if __name__ == "__main__":
    # This allows the tests to be run from the command line
    unittest.main()