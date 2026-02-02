import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def mittag_leffler_series(z, beta, terms=100):
    """
    Mittag-Leffler function using series expansion
    E_β(z) = Σ_{k=0}^∞ z^k / Γ(βk + 1)
    """
    result = 0
    for k in range(terms):
        term = z**k / gamma(beta * k + 1)
        result += term
        # Stop if term becomes negligible
        if abs(term) < 1e-15 * abs(result) and k > 10:
            break
    return result

def Psi(tau, beta):
    """
    Survival function: Ψ(τ) = E_β[-(τ)^β]
    """
    z = -(tau)**beta
    return mittag_leffler_series(z, beta)

def phi_numerical(tau, beta, dtau=1e-6):
    """
    Probability density: φ(t) = -dΨ/dt
    Using numerical differentiation
    """
    Psi_plus = Psi(tau + dtau, beta)
    Psi_minus = Psi(tau - dtau, beta)
    return -(Psi_plus - Psi_minus) / (2 * dtau)

# Beta values in (0, 1]
beta_values = [0.3, 0.5, 0.7, 1.0]
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(beta_values)))

# Define tau ranges
tau_linear = np.linspace(0.01, 2, 400)
tau_loglog = np.logspace(-2, np.log10(5), 400)

print("Computing probability density φ(t)...")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# ============================================================================
# LEFT PLOT: LINEAR SCALE
# ============================================================================
for beta, color in zip(beta_values, colors):
    print(f"  β = {beta} (linear)")
    phi_vals = np.array([phi_numerical(t, beta) for t in tau_linear])
    ax1.plot(tau_linear, phi_vals, label=f'β = {beta}', 
             color=color, linewidth=3.5)

ax1.set_xlabel(r'$t$', fontsize=22, fontweight='bold')
ax1.set_ylabel(r'$\phi(t)$', fontsize=22, fontweight='bold')
ax1.set_title(r'Densità di Probabilità $\phi(t)$ - Scala Lineare', 
              fontsize=24, pad=20, fontweight='bold')
ax1.legend(fontsize=17, loc='best', framealpha=0.95, 
          edgecolor='black', fancybox=True)
ax1.grid(True, alpha=0.35, linewidth=1.2)
ax1.tick_params(labelsize=16, width=1.5, length=8)
ax1.set_xlim(0, 2)
ax1.set_ylim(bottom=0)

# Add spines
for spine in ax1.spines.values():
    spine.set_linewidth(1.5)

# ============================================================================
# RIGHT PLOT: LOG-LOG SCALE
# ============================================================================
for beta, color in zip(beta_values, colors):
    print(f"  β = {beta} (log-log)")
    phi_vals = np.array([phi_numerical(t, beta) for t in tau_loglog])
    mask = phi_vals > 0
    ax2.loglog(tau_loglog[mask], phi_vals[mask], label=f'β = {beta}', 
               color=color, linewidth=3.5)

ax2.set_xlabel(r'$t$', fontsize=22, fontweight='bold')
ax2.set_ylabel(r'$\phi(t)$', fontsize=22, fontweight='bold')
ax2.set_title(r'Densità di Probabilità $\phi(t)$ - Scala Log-Log', 
              fontsize=24, pad=20, fontweight='bold')
ax2.legend(fontsize=17, loc='best', framealpha=0.95,
          edgecolor='black', fancybox=True)
ax2.grid(True, alpha=0.35, which='both', linewidth=1.2)
ax2.tick_params(labelsize=16, width=1.5, length=8)

# Add spines
for spine in ax2.spines.values():
    spine.set_linewidth(1.5)

plt.tight_layout()
plt.savefig('phi_density.png', dpi=300, bbox_inches='tight')
print("\n✓ Grafico salvato: phi_density.png")

print("\nNote:")
print("- φ(t): densità di probabilità derivata dalla funzione di Mittag-Leffler")
print("- β ∈ (0, 1]: rilassamento sub-esponenziale")
print("- β = 1: rilassamento esponenziale standard")
print("- β < 1: code pesanti (heavy tails), memoria a lungo termine")
