import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

# --- CONFIGURAZIONE STILE ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 2

# --- 1. GENERAZIONE DATI SINGOLA TRAIETTORIA ---
np.random.seed(123)
beta = 0.6          
n_steps = 80        

dt = levy_stable.rvs(alpha=beta, beta=1, loc=0, scale=1, size=n_steps)
dt = np.abs(dt)
T_tau = np.cumsum(dt) 
tau_axis = np.arange(n_steps)

t_max = T_tau[-1]
time_physical = np.linspace(0, t_max, 2000)
S_t = np.searchsorted(T_tau, time_physical)

# --- 2. SIMULAZIONE ENSEMBLE ---
num_particles = 100000 
n_ens_steps = 500 

dt_ens = levy_stable.rvs(alpha=beta, beta=1, loc=0, scale=1, size=(num_particles, n_ens_steps))
dt_ens = np.abs(dt_ens)
T_ens = np.cumsum(dt_ens, axis=1)

t_fix = t_max / 2

has_reached_target = T_ens[:, -1] > t_fix
valid_T = T_ens[has_reached_target]
s_vals = np.argmax(valid_T > t_fix, axis=1)

# --- 3. PLOTTING ---
fig = plt.figure(figsize=(18, 6), constrained_layout=True)
gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1])

# (A)
ax1 = fig.add_subplot(gs[0])
ax1.step(tau_axis, T_tau, where='post', color='blue', lw=2)
ax1.set_title(r'(A) Subordinatore $T(\tau)$')
ax1.set_xlabel(r'Tempo operativo $\tau$')
ax1.set_ylabel(r'Tempo fisico $t$')
ax1.grid(True, linestyle=':', alpha=0.6)

jump_idx = np.argmax(dt[:60])
jump_height = dt[jump_idx]
ax1.annotate('', xy=(jump_idx, T_tau[jump_idx]), xytext=(jump_idx, T_tau[jump_idx]-jump_height),
             arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax1.text(jump_idx + 5, T_tau[jump_idx] - jump_height/2, "Lunga Attesa", 
         color='red', va='center', fontsize=10, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

# (B)
ax2 = fig.add_subplot(gs[1])
ax2.plot(time_physical, S_t, color='red', lw=2)
ax2.set_title(r'(B) Processo inverso $S(t)$')
ax2.set_xlabel(r'Tempo fisico $t$')
ax2.set_ylabel(r'Tempo operativo $S$')
ax2.grid(True, linestyle=':', alpha=0.6)

t_start = T_tau[jump_idx-1] if jump_idx > 0 else 0
t_end = T_tau[jump_idx]
rect = patches.Rectangle((t_start, 0), t_end-t_start, jump_idx+5, color='red', alpha=0.15)
ax2.add_patch(rect)
ax2.text((t_start+t_end)/2, jump_idx + 8, "Trapping", ha='center', color='red', fontsize=10, fontweight='bold')
ax2.set_xlim(0, t_max)
ax2.set_ylim(0, n_steps)

# --- (C) PDF CON BINNING CORRETTO ---
ax3 = fig.add_subplot(gs[2])

limit_val = np.percentile(s_vals, 99)
data_plot = s_vals[s_vals <= limit_val]

# --- FIX: CALCOLO BIN INTERI ---
# 1. Decidiamo quanti bin vogliamo APPROSSIMATIVAMENTE (es. 50)
target_bins = 50
data_range = data_plot.max() - data_plot.min()

# 2. Calcoliamo la larghezza intera (step) più vicina
#    Math.ceil assicura che lo step sia almeno 1
bin_step = max(1, int(round(data_range / target_bins)))

# 3. Creiamo i bordi dei bin allineati agli interi
#    Sottraiamo 0.5 per centrare i bin sugli interi (es. bin tra 0.5 e 3.5 cattura 1, 2, 3)
bins_integers = np.arange(data_plot.min(), data_plot.max() + bin_step, bin_step) - 0.5

hist_counts, hist_bins = np.histogram(data_plot, bins=bins_integers, density=True)
bin_centers = 0.5 * (hist_bins[:-1] + hist_bins[1:])
bin_width = hist_bins[1] - hist_bins[0]

ax3.bar(bin_centers, hist_counts, width=bin_width*0.9, 
        color='green', alpha=0.6, edgecolor='darkgreen', linewidth=1, label='Simulazione')

ax3.set_title(rf'(C) PDF $n(\tau, t)$ a $t={t_fix:.0f}$')
ax3.set_xlabel(r'Tempo operativo $\tau$')
ax3.set_ylabel(r'Densità di Probabilità')
ax3.grid(True, linestyle=':', alpha=0.6)
ax3.set_xlim(0, limit_val)

peak_idx = np.argmax(hist_counts)
ax3.annotate('Picco a $\\tau$ piccoli', 
             xy=(bin_centers[peak_idx], hist_counts[peak_idx]), 
             xytext=(bin_centers[peak_idx] + 15, hist_counts[peak_idx]),
             arrowprops=dict(facecolor='black', shrink=0.05),
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

fig.suptitle('Dinamica della Subordinazione: Dai salti alla densità', fontsize=16, y=1.05)
plt.show()