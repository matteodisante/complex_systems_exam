import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.integrate import quad

def phi_integral(t, beta):
    """
    Probability density using integral representation (C.8):
    φ(t) = (sin(βπ)/π) ∫₀^∞ [r^β e^(-rt)] / [r^(2β) + 2r^β cos(βπ) + 1] dr
    
    Special case: β = 1 gives exponential distribution φ(t) = e^(-t)
    """
    if t <= 0:
        return 0.0
    
    # Special case for β = 1 (exponential distribution)
    if abs(beta - 1.0) < 1e-10:
        return np.exp(-t)
    
    def integrand(r):
        numerator = r**beta * np.exp(-r * t)
        denominator = r**(2*beta) + 2 * r**beta * np.cos(beta * np.pi) + 1
        return numerator / denominator
    
    coefficient = np.sin(beta * np.pi) / np.pi
    result, _ = quad(integrand, 0, np.inf, limit=100)
    return coefficient * result

# Beta values in (0, 1]
beta_values = [0.3, 0.5, 0.7, 1.0]
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(beta_values)))

# Define tau ranges
tau_linear = np.linspace(0.01, 2, 400)
tau_loglog = np.logspace(-2, 2, 400)

print("Computing probability density φ(t)...")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# ============================================================================
# LEFT PLOT: LINEAR SCALE
# ============================================================================
for beta, color in zip(beta_values, colors):
    print(f"  β = {beta} (linear)")
    phi_vals = np.array([phi_integral(t, beta) for t in tau_linear])
    ax1.plot(tau_linear, phi_vals, label=f'β = {beta}', 
             color=color, linewidth=3.5)

ax1.set_xlabel(r'$\tau$', fontsize=22, fontweight='bold')
ax1.set_ylabel(r'$\phi(\tau)$', fontsize=22, fontweight='bold')
ax1.set_title(r'Densità di Probabilità $\phi(\tau)$ - Scala Lineare', 
              fontsize=18, pad=15, fontweight='bold')
ax1.legend(fontsize=17, loc='best', framealpha=0.95, 
          edgecolor='black', fancybox=True)
ax1.grid(True, alpha=0.35, linewidth=1.2)
ax1.tick_params(labelsize=16, width=1.5, length=8)
ax1.set_xlim(0, 2)
ax1.set_ylim(bottom=0, top=1)

# Add spines
for spine in ax1.spines.values():
    spine.set_linewidth(1.5)

# ============================================================================
# RIGHT PLOT: LOG-LOG SCALE
# ============================================================================
for beta, color in zip(beta_values, colors):
    print(f"  β = {beta} (log-log)")
    phi_vals = np.array([phi_integral(t, beta) for t in tau_loglog])
    mask = phi_vals > 0
    ax2.loglog(tau_loglog[mask], phi_vals[mask], label=f'β = {beta}', 
               color=color, linewidth=3.5)

ax2.set_xlabel(r'$\tau$', fontsize=22, fontweight='bold')
ax2.set_ylabel(r'$\phi(\tau)$', fontsize=22, fontweight='bold')
ax2.set_title(r'Densità di Probabilità $\phi(\tau)$ - Scala Log-Log', 
              fontsize=18, pad=15, fontweight='bold')
ax2.legend(fontsize=17, loc='best', framealpha=0.95,
          edgecolor='black', fancybox=True)
ax2.grid(True, alpha=0.35, which='both', linewidth=1.2)
ax2.tick_params(labelsize=16, width=1.5, length=8)
ax2.set_ylim(bottom=1e-4, top=1)

# Add spines
for spine in ax2.spines.values():
    spine.set_linewidth(1.5)

plt.tight_layout()
plt.savefig('phi_density.png', dpi=300, bbox_inches='tight')
print("\n✓ Grafico salvato: phi_density.png")

print("\nNote:")
print("- φ(τ): densità di probabilità calcolata con rappresentazione integrale (C.8)")
print("- Formula: φ(τ) = (sin(βπ)/π) ∫₀^∞ [r^β e^(-rτ)] / [r^(2β) + 2r^β cos(βπ) + 1] dr")
print("- β ∈ (0, 1]: rilassamento sub-esponenziale")
print("- β = 1: rilassamento esponenziale standard")
print("- β < 1: code pesanti (heavy tails), memoria a lungo termine")
