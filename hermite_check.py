import numpy as np
from scipy.special import eval_hermite, factorial
import math

# build psi_n via direct formula from H_n (physicists'):
# psi_n(x) = (1 / sqrt(2^n n! sqrt(pi))) * H_n(x) * exp(-x^2/2)

def psi_from_H(n, x):
    Hn = eval_hermite(n, x)
    norm = math.sqrt((2.0**n) * math.factorial(n) * math.sqrt(math.pi))
    return Hn * np.exp(-0.5 * x**2) / norm

# build psi_n via recurrence (orthonormal Hermite functions)
# psi_0 = pi^{-1/4} e^{-x^2/2}
# psi_1 = sqrt(2) x psi_0
# psi_{n+1} = sqrt(2/(n+1)) x psi_n - sqrt(n/(n+1)) psi_{n-1}

def psi_from_recurrence(n, x):
    x = np.asarray(x)
    psi0 = (np.pi ** -0.25) * np.exp(-0.5 * x**2)
    if n == 0:
        return psi0
    psi1 = np.sqrt(2.0) * x * psi0
    if n == 1:
        return psi1
    psi_nm1 = psi0
    psi_n = psi1
    for k in range(1, n):
        coef1 = math.sqrt(2.0 / (k + 1.0))
        coef2 = math.sqrt(k / (k + 1.0))
        psi_np1 = coef1 * x * psi_n - coef2 * psi_nm1
        psi_nm1, psi_n = psi_n, psi_np1
    return psi_n

# grid
x = np.linspace(-4, 4, 801)

print('n, max_abs_diff, max_rel_diff')
for n in range(0, 11):
    a = psi_from_H(n, x)
    b = psi_from_recurrence(n, x)
    diff = a - b
    max_abs = np.max(np.abs(diff))
    # avoid division by zero in relative; use max(|a|,|b|)
    denom = np.maximum(np.abs(a), np.abs(b))
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.abs(diff) / np.where(denom==0, 1.0, denom)
    max_rel = np.nanmax(rel)
    print(f"{n}, {max_abs:.3e}, {max_rel:.3e}")
