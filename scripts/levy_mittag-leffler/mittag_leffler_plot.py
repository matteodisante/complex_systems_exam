import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.integrate import quad

# 1. Impostazione dei Parametri
beta = 0.9
t = np.logspace(-3, 3, 100)

# Costanti
a = gamma(beta + 1)
b = gamma(beta) * np.sin(beta * np.pi) / np.pi

# 2. Calcolo delle curve di approssimazione
y_weibull = np.exp(-(t**beta) / a)
y_power = b * (t**(-beta))

# 3. Calcolo della Funzione di Mittag-Leffler Analitica
def g_mu(mu, beta):
    numerator = (1/np.pi) * np.sin(beta * np.pi)
    denominator = mu**(1+beta) + 2 * np.cos(beta * np.pi) * mu + mu**(1-beta)
    return numerator / denominator

def mittag_leffler_integral(t_val, beta):
    # Special case for β = 1: E_1(-t) = exp(-t)
    if abs(beta - 1.0) < 1e-10:
        return np.exp(-t_val)
    
    integrand = lambda mu: np.exp(-mu * t_val) * g_mu(mu, beta)
    res, error = quad(integrand, 0, np.inf, limit=200)
    return res

y_analytical = [mittag_leffler_integral(val, beta) for val in t]

# 4. Creazione del Grafico Migliorato
plt.figure(figsize=(10, 7)) # Aumento dimensione figura

# Plot Analitico
plt.loglog(t, y_analytical, 'r-', linewidth=2.5, label=r'$E_\beta(-t^\beta)$, analytical')

# Plot Approssimazione Weibull
plt.loglog(t, y_weibull, color='lightgreen', linestyle='--', linewidth=2.5, label=r'$\exp(-t^\beta/a)$')

# Plot Approssimazione Power Law
plt.loglog(t, y_power, color='blue', linestyle=':', linewidth=2.5, label=r'$b t^{-\beta}$')

# --- Miglioramenti Estetici ---
font_size_labels = 16
font_size_ticks = 14
font_size_legend = 14

plt.xlabel(r'$\tau$', fontsize=font_size_labels)
# Asse Y con il simbolo della Mittag-Leffler
plt.ylabel(r'$E_\beta(-\tau^\beta)$', fontsize=font_size_labels)
plt.title(r'$\beta = 0.9$', fontsize=18, pad=15)

# Limiti assi
plt.xlim(0.001, 1000)
plt.ylim(0.0001, 2)

# Legenda migliorata
plt.legend(loc='lower left', fontsize=font_size_legend, frameon=True, framealpha=0.9, edgecolor='gray')

# Ticks (Numeri sugli assi) più grandi
plt.tick_params(axis='both', which='major', labelsize=font_size_ticks)
# Opzionale: gestire anche i minor ticks se necessario, ma loglog lo fa in automatico spesso

# Griglia
plt.grid(True, which="major", ls="-", alpha=0.4)
plt.grid(True, which="minor", ls=":", alpha=0.2)

plt.tight_layout()
plt.savefig('reproduction_fig2_improved.png', dpi=150)
plt.show()


# ============================================================================
# PLOT Ψ(t) = E_β(-t^β) - Funzione di Sopravvivenza
# ============================================================================

# Beta values in (0, 1]
beta_values = [0.3, 0.5, 0.7, 0.9, 1.0]
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(beta_values)))

# Define tau ranges
tau_linear = np.linspace(0.01, 4, 400)
tau_loglog = np.logspace(-2, 2, 400)

print("Computing survival function Ψ(t) = E_β(-t^β)...")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# ============================================================================
# LEFT PLOT: LINEAR SCALE
# ============================================================================
for beta_val, color in zip(beta_values, colors):
    print(f"  β = {beta_val} (linear)")
    psi_vals = np.array([mittag_leffler_integral(t_val, beta_val) for t_val in tau_linear])
    ax1.plot(tau_linear, psi_vals, label=f'β = {beta_val}', 
             color=color, linewidth=3.5)

ax1.set_xlabel(r'$\tau$', fontsize=22, fontweight='bold')
ax1.set_ylabel(r'$\Psi(\tau)$', fontsize=22, fontweight='bold')
ax1.set_title(r'$\Psi(\tau) = E_\beta(-\tau^\beta)$ - Scala Lineare', 
              fontsize=18, pad=15, fontweight='bold')
ax1.legend(fontsize=17, loc='best', framealpha=0.95, 
          edgecolor='black', fancybox=True)
ax1.grid(True, alpha=0.35, linewidth=1.2)
ax1.tick_params(labelsize=16, width=1.5, length=8)
ax1.set_xlim(0, 4)
ax1.set_ylim(bottom=0, top=1)


# Add spines
for spine in ax1.spines.values():
    spine.set_linewidth(1.5)

# ============================================================================
# RIGHT PLOT: LOG-LOG SCALE
# ============================================================================
for beta_val, color in zip(beta_values, colors):
    print(f"  β = {beta_val} (log-log)")
    psi_vals = np.array([mittag_leffler_integral(t_val, beta_val) for t_val in tau_loglog])
    mask = psi_vals > 0
    ax2.loglog(tau_loglog[mask], psi_vals[mask], label=f'β = {beta_val}', 
               color=color, linewidth=3.5)

ax2.set_xlabel(r'$\tau$', fontsize=22, fontweight='bold')
ax2.set_ylabel(r'$\Psi(\tau)$', fontsize=22, fontweight='bold')
ax2.set_title(r'$\Psi(\tau) = E_\beta(-\tau^\beta)$ - Scala Log-Log', 
              fontsize=18, pad=15, fontweight='bold')
ax2.legend(fontsize=17, loc='best', framealpha=0.95,
          edgecolor='black', fancybox=True)
ax2.grid(True, alpha=0.35, which='both', linewidth=1.2)
ax2.tick_params(labelsize=16, width=1.5, length=8)
ax2.set_ylim(bottom=1e-3, top=1e0)

# Add spines
for spine in ax2.spines.values():
    spine.set_linewidth(1.5)

plt.tight_layout()
plt.savefig('psi_survival_function.png', dpi=300, bbox_inches='tight')
print("\n✓ Grafico Ψ(t) salvato: psi_survival_function.png")
plt.show()