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
        x_grid = np.linspace(-1, 1, 10)
        t = 1.0
        x0 = 0.0
        alpha = 0.5
        N = 10
        pdf = spectral_series_pdf(x_grid, t, x0, alpha, N)
        self.assertEqual(pdf.shape, x_grid.shape)
        self.assertTrue(np.all(pdf >= 0))

if __name__ == '__main__':
    unittest.main()