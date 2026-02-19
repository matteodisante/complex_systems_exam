import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.special import gamma

# --- PRESENTATION STYLE SETTINGS ---
def setup_presentation_style():
    plt.style.use('seaborn-v0_8-paper') # Start with a clean base style
    plt.rcParams.update({
        'font.size': 16,              # Base font size
        'axes.labelsize': 18,         # Axis labels (x, y)
        'axes.titlesize': 20,         # Subplot titles
        'xtick.labelsize': 16,        # X-axis numbers
        'ytick.labelsize': 16,        # Y-axis numbers
        'legend.fontsize': 16,        # Legend text
        'lines.linewidth': 4,         # Thicker lines for visibility
        'lines.markersize': 10,       # Larger markers
        'figure.titlesize': 24,       # Overall figure title
        'figure.figsize': (16, 7),    # Larger default figure size
        'axes.grid': True,            # Turn on grid by default
        'grid.alpha': 0.3,            # Subtle grid
        'axes.titleweight': 'bold',   # Bold titles
        'axes.labelweight': 'bold',   # Bold labels
    })

setup_presentation_style()

# --- Generators (Unchanged) ---
def generate_levy_stable(n, alpha, gamma_x):
    epsilon = 1e-10
    u = np.random.uniform(epsilon, 1.0, n)
    v = np.random.uniform(epsilon, 1.0, n)
    phi = np.pi * (v - 0.5)
    term1 = (-np.log(u) * np.cos(phi)) / np.cos((1 - alpha) * phi)
    term2 = np.power(term1, 1 - 1/alpha)
    term3 = np.sin(alpha * phi) / np.cos(phi)
    return gamma_x * term2 * term3

def generate_mittag_leffler(n, beta, gamma_t):
    epsilon = 1e-10
    u = np.random.uniform(epsilon, 1.0, n)
    v = np.random.uniform(epsilon, 1.0, n)
    inner_term = (np.sin(beta * np.pi) / np.tan(beta * np.pi * v)) - np.cos(beta * np.pi)
    tau = -gamma_t * np.log(u) * np.power(inner_term, 1/beta)
    return tau

# --- Helper 1: Fit Slope for Histogram (Density) ---
def extract_slope_and_plot(ax, data, bins, theoretical_index, percentile_cutoff=95):
    # 1. Calculate Density
    counts, edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    
    # 2. Filter for valid bins
    valid = counts > 0
    x_vals = bin_centers[valid]
    y_vals = counts[valid]
    
    # 3. Determine Tail Cutoff
    tail_threshold_val = np.percentile(data, percentile_cutoff)
    is_in_tail = x_vals > tail_threshold_val
    
    # Fallback
    if np.sum(is_in_tail) < 5:
        tail_threshold_val = np.percentile(data, 80) 
        is_in_tail = x_vals > tail_threshold_val
        
    x_tail = x_vals[is_in_tail]
    y_tail = y_vals[is_in_tail]
    
    # 4. Linear Regression
    log_x = np.log10(x_tail)
    log_y = np.log10(y_tail)
    slope, intercept, _, _, _ = linregress(log_x, log_y)
    
    # 5. Plot Fitted Line (Bright Red Dashed)
    x_line = np.logspace(np.log10(x_tail[0]), np.log10(x_tail[-1]), 10)
    fit_y = 10**(intercept + slope * np.log10(x_line))
    
    ax.plot(x_line, fit_y, color='red', linestyle='--', linewidth=4, label=f'Fit: {slope:.2f}')
    
    # 6. Plot Theoretical Line (Cyan Dotted)
    theo_slope = -1 - theoretical_index
    mid_idx = len(x_line) // 2
    anchor_x = x_line[mid_idx]
    anchor_y = fit_y[mid_idx]
    
    y_theo = anchor_y * (x_line / anchor_x)**(theo_slope)
    
    ax.plot(x_line, y_theo, color='cyan', linestyle=':', linewidth=4, label=f'Theory: {theo_slope:.2f}')
    
    return slope

# --- Helper 2: Fit Slope for Survival Function (CCDF) ---
def fit_survival_tail(ax, sorted_data, ccdf, beta, percentile_cutoff=99.5):
    # 1. Determine Tail Region
    n = len(sorted_data)
    cutoff_index = int(n * (percentile_cutoff / 100))
    
    t_tail = sorted_data[cutoff_index:]
    ccdf_tail = ccdf[cutoff_index:]
    
    # Filter 0s
    mask = ccdf_tail > 0
    t_tail = t_tail[mask]
    ccdf_tail = ccdf_tail[mask]
    
    # 2. Linear Regression
    log_t = np.log10(t_tail)
    log_ccdf = np.log10(ccdf_tail)
    slope, intercept, _, _, _ = linregress(log_t, log_ccdf)
    
    # 3. Plot Fitted Line
    x_plot = t_tail[::10] 
    fit_y = 10**(intercept + slope * np.log10(x_plot))
    
    ax.plot(x_plot, fit_y, color='red', linestyle='--', linewidth=4, label=f'Fit: {slope:.2f}')
    
    return slope

# --- Configuration ---
N_SAMPLES = 2000000
ALPHA = 1.5    
BETA = 0.8     
GAMMA_X = 1.0  
GAMMA_T = 1.0  

# --- Generation ---
levy_samples = generate_levy_stable(N_SAMPLES, ALPHA, GAMMA_X)
ml_samples = generate_mittag_leffler(N_SAMPLES, BETA, GAMMA_T)

# ==========================================
# FIGURE 1: LÉVY ALPHA-STABLE
# ==========================================
fig1, ax1 = plt.subplots(1, 2, figsize=(18, 7)) # Wider figure for keynote

# Left: Linear Scale
ax1[0].hist(levy_samples, bins=200, range=(-10, 10), density=True, 
            color='skyblue', edgecolor='black', alpha=0.6)
ax1[0].set_title(fr'Lévy $\alpha$-Stable (Linear)' + '\n' + fr'$\alpha={ALPHA}$')
ax1[0].set_xlabel(r'Jump value $\xi$')
ax1[0].set_ylabel('Density')

# Right: Log-Log Scale
levy_pos = levy_samples[levy_samples > 0]
bins_levy = np.logspace(np.log10(np.min(levy_pos)), np.log10(np.max(levy_pos)), 60)
ax1[1].hist(levy_pos, bins=bins_levy, density=True, 
            color='skyblue', edgecolor='black', alpha=0.3)
ax1[1].set_xscale('log')
ax1[1].set_yscale('log')
ax1[1].set_title(r'Lévy Positive Tail (Log-Log)' + '\n' + r'Expected Slope $\approx -2.5$')
ax1[1].set_xlabel(r'Jump value $\xi$')

extract_slope_and_plot(ax1[1], levy_pos, bins_levy, ALPHA, percentile_cutoff=95)
ax1[1].legend(frameon=True, framealpha=1, shadow=True) # Pop-out legend

plt.tight_layout()
plt.show()

# ==========================================
# FIGURE 2: MITTAG-LEFFLER
# ==========================================
fig2, ax2 = plt.subplots(1, 2, figsize=(18, 7))

# --- Subplot 1 (Left): PDF Histogram (Log-Log Scale) ---
ml_pos = ml_samples[ml_samples > 0]
bins_ml_pdf = np.logspace(np.log10(np.min(ml_pos)), np.log10(np.max(ml_pos)), 100)

ax2[0].hist(ml_pos, bins=bins_ml_pdf, density=True, 
            color='salmon', edgecolor='black', alpha=0.6, label='Data PDF')

ax2[0].set_xscale('log')
ax2[0].set_yscale('log')
ax2[0].set_title(r'Mittag-Leffler PDF' + '\n' + fr'Exp. Slope $\approx -{1+BETA}$')
ax2[0].set_xlabel(r'Waiting Time $\tau$')
ax2[0].set_ylabel(r'Density $\psi(\tau)$')

extract_slope_and_plot(ax2[0], ml_pos, bins_ml_pdf, BETA, percentile_cutoff=80)
ax2[0].legend(frameon=True, framealpha=1, shadow=True)

# --- Subplot 2 (Right): Survival Function (Log-Log Scale) ---
sorted_data = np.sort(ml_samples)
n = len(sorted_data)
ccdf = 1.0 - np.arange(1, n + 1) / n 
ax2[1].loglog(sorted_data, ccdf, color='#fa8072', linewidth=5, label='Data CCDF') # Thicker line

# Theoretical Asymptotes
t_min, t_max = sorted_data[0], sorted_data[-1]
t_small = np.logspace(np.log10(t_min), 0.5, 100) 
a_const = gamma(BETA + 1)
y_weibull = np.exp(-(t_small**BETA) / a_const)

t_large = np.logspace(0, np.log10(t_max), 100)
b_const = (gamma(BETA) * np.sin(BETA * np.pi)) / np.pi
y_powerlaw = b_const * (t_large**(-BETA))

ax2[1].loglog(t_small, y_weibull, 'b--', linewidth=4, label=r'Small $t$ Theory')
ax2[1].loglog(t_large, y_powerlaw, 'k:', linewidth=4, label=fr'Large $t$ Theory')

fit_survival_tail(ax2[1], sorted_data, ccdf, BETA, percentile_cutoff=99.5)

ax2[1].set_title(r'Survival Function $\Psi(t)$' + '\n' + fr'Theoretical Slope $\approx -{BETA}$')
ax2[1].set_xlabel(r'Waiting Time $\tau$')
ax2[1].set_ylabel(r'$P(\tau > t)$')
ax2[1].set_ylim(1e-5, 1.5)
ax2[1].set_xlim(t_min, t_max)
ax2[1].legend(frameon=True, framealpha=1, shadow=True)

plt.tight_layout()
plt.show()