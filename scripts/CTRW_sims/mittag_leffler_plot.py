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

plt.xlabel('t', fontsize=font_size_labels)
# Asse Y con il simbolo della Mittag-Leffler
plt.ylabel(r'$E_\beta(-t^\beta)$', fontsize=font_size_labels)
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