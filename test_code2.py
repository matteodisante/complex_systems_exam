import unittest
import numpy as np
from scipy.integrate import quad
from code2 import (
    # Functions from code2.py to be tested
    mittag_leffler,
    l_alpha,
    n_function_s_array,
    ou_kernel,
    compute_pdf_vectorized,
    spectral_series_pdf,
)

class TestCode2(unittest.TestCase):
    def test_mittag_leffler(self):
        # Test with known values
        self.assertAlmostEqual(mittag_leffler(1, 1), np.exp(1), places=5)
        self.assertAlmostEqual(mittag_leffler(1, 0), 1, places=5)
        self.assertAlmostEqual(mittag_leffler(2, 0), 1, places=5)
        self.assertAlmostEqual(mittag_leffler(2, -1), np.cos(1), places=5)

    def test_l_alpha(self):
        # Test for alpha = 0.5 (Smirnov)
        z = np.array([1, 2, 3])
        result = l_alpha(0.5, z)
        self.assertEqual(result.shape, z.shape)
        self.assertTrue(np.all(result > 0))

        # Test for alpha = 1/3
        z = np.array([1, 2, 3])
        result = l_alpha(1.0/3.0, z)
        self.assertEqual(result.shape, z.shape)
        self.assertTrue(np.all(result > 0))

        # Test for unsupported alpha
        with self.assertRaises(ValueError):
            l_alpha(0.8, z)

        # Test normalization
        norm_half, _ = quad(lambda z: l_alpha(0.5, z), 0, np.inf)
        print(f"Numerical normalization for alpha=0.5: {norm_half}")
        self.assertAlmostEqual(norm_half, 1.0, places=5)

        norm_third, _ = quad(lambda z: l_alpha(1.0/3.0, z), 0, np.inf)
        print(f"Numerical normalization for alpha=1/3: {norm_third}")
        self.assertAlmostEqual(norm_third, 1.0, places=5)

    def test_n_function_s_array(self):
        s = np.array([1, 2, 3])
        t = 1.0
        result = n_function_s_array(s, t)
        self.assertEqual(result.shape, s.shape)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_ou_kernel(self):
        x_grid = np.linspace(-1, 1, 10)
        s_grid = np.linspace(0.1, 1, 5)
        x0 = 0.0
        result = ou_kernel(x_grid, s_grid, x0)
        self.assertEqual(result.shape, (10, 5))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_compute_pdf_vectorized(self):
        x_grid = np.linspace(-1, 1, 10)
        t = 1.0
        x0 = 0.0
        pdf = compute_pdf_vectorized(x_grid, t, x0)
        self.assertEqual(pdf.shape, x_grid.shape)
        self.assertTrue(np.all(pdf >= 0))

    def test_spectral_series_pdf(self):
        # Test with the new signature
        x_grid = np.linspace(-1, 1, 10)
        t = 1.0
        x0 = 0.0
        alpha = 0.5
        N = 10
        m, omega, k_B, T, gamma = 1.0, 1.0, 1.0, 1.0, 1.0
        pdf = spectral_series_pdf(x_grid, t, x0, alpha, N, m, omega, k_B, T, gamma)
        self.assertEqual(pdf.shape, x_grid.shape)
        self.assertTrue(np.all(pdf >= 0))

    def test_spectral_series_pdf_normalization(self):
        # Test that the PDF integrates to 1
        x_grid = np.linspace(-5, 5, 500) # Wider grid for better integration
        t = 1.0
        x0 = 0.5
        alpha = 1.0/3.0
        N = 20 # Use a reasonable number of terms
        m, omega, k_B, T, gamma = 1.0, 1.0, 1.0, 1.0, 1.0
        
        pdf = spectral_series_pdf(x_grid, t, x0, alpha, N, m, omega, k_B, T, gamma)
        
        integral_val = np.trapezoid(pdf, x_grid)
        print(f"Numerical normalization for spectral series (N={N}, t={t}): {integral_val}")
        self.assertAlmostEqual(integral_val, 1.0, places=3)

    def test_spectral_series_stationary_state(self):
        # For large t, the PDF should approach the stationary distribution
        x_grid = np.linspace(-5, 5, 500)
        t_large = 100000.0 
        x0 = 0.5
        alpha = 1.0/3.0
        N = 5 # Only a few terms needed as higher terms decay faster
        m, omega, k_B, T, gamma = 1.0, 1.0, 1.0, 1.0, 1.0

        # Calculate from spectral series at large t
        pdf_spectral = spectral_series_pdf(x_grid, t_large, x0, alpha, N, m, omega, k_B, T, gamma)

        # Calculate the analytical stationary solution (Boltzmann distribution for harmonic potential)
        # V(x) = 0.5 * m * omega^2 * x^2
        # P_st(x) = Z * exp(-V(x)/(k_B T))
        # With m=1, omega=1, k_B=1, T=1, this is (1/sqrt(2*pi)) * exp(-x^2/2)
        pdf_stationary = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_grid**2)
        
        # Compare the two distributions using L1 norm
        l1_diff = np.trapezoid(np.abs(pdf_spectral - pdf_stationary), x_grid)
        print(f"L1 difference from stationary at t={t_large}: {l1_diff}")
        self.assertLess(l1_diff, 1e-2)

    def test_spectral_series_vs_standard_ou(self):
        # For alpha=1, the spectral series should match the standard OU solution.
        x_grid = np.linspace(-3, 3, 500)
        t = 1.0
        x0 = 0.5
        alpha = 1.0
        N = 50 # Use a higher N for better accuracy
        m, omega, k_B, T, gamma = 1.0, 1.0, 1.0, 1.0, 1.0
        K_beta = 1.0 # As used in code2.py

        # 1. Calculate PDF from spectral series with alpha=1
        pdf_spectral = spectral_series_pdf(x_grid, t, x0, alpha, N, m, omega, k_B, T, gamma)

        # 2. Calculate analytical PDF for standard OU process
        mean_ou = x0 * np.exp(-gamma * t)
        # Variance for standard OU is (K_beta/gamma) * (1 - exp(-2*gamma*t))
        variance_ou = (K_beta / gamma) * (1 - np.exp(-2 * gamma * t))
        pdf_ou_analytical = (1.0 / np.sqrt(2 * np.pi * variance_ou)) * \
                            np.exp(-0.5 * (x_grid - mean_ou)**2 / variance_ou)

        # 3. Compare the two results
        l1_diff = np.trapezoid(np.abs(pdf_spectral - pdf_ou_analytical), x_grid)
        print(f"L1 difference between spectral (alpha=1) and standard OU at t={t}: {l1_diff}")
        self.assertLess(l1_diff, 1e-3) # Set a reasonably strict tolerance

if __name__ == '__main__':
    unittest.main()