import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import gamma
from scipy.integrate import quad

# ==============================================================================
# 1. GENERATORI DI NUMERI CASUALI (Basati su Eq. 19 e 20 del Paper)
# ==============================================================================

def generate_levy_stable(alpha, gamma_x, size):
    """
    Genera salti distribuiti secondo Levy (simmetrici) usando l'algoritmo di Chambers-Mallows-Stuck.
    Eq. (19) del paper.
    """
    # Caso Gaussiano (Alpha = 2)
    if alpha >= 2.0:
        return np.random.normal(0, np.sqrt(2)*gamma_x, size)
    
    # Caso Levy (Alpha < 2)
    u = np.random.uniform(0, 1, size)
    v = np.random.uniform(0, 1, size)
    phi = np.pi * (v - 0.5)
    
    # Calcolo spezzato per leggibilità
    numer = np.sin(alpha * phi)
    denom = np.cos(phi)
    term1 = numer / denom
    
    term2 = np.cos((1 - alpha) * phi)
    term3 = -np.log(u) * np.cos(phi)
    
    # Gestione divisioni per zero o valori estremi
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = (term3 / term2)
        # Assicuriamoci che la base sia positiva per la potenza
        xi = gamma_x * np.sign(term1) * (np.abs(term1)) * (factor ** (1 - 1/alpha))
        
    return np.nan_to_num(xi, nan=0.0)

def generate_mittag_leffler(beta, gamma_t, size):
    """
    Genera tempi di attesa distribuiti secondo Mittag-Leffler.
    Eq. (20) del paper.
    """
    # Caso Esponenziale (Beta = 1)
    if beta >= 1.0:
        return -gamma_t * np.log(np.random.uniform(0, 1, size))

    u = np.random.uniform(0, 1, size)
    v = np.random.uniform(0, 1, size)
    
    pi_beta = np.pi * beta
    # Evitiamo singolarità esatte in v=0 o v=1
    v = np.clip(v, 1e-6, 1-1e-6)
    
    term_trig = (np.sin(pi_beta) / np.tan(pi_beta * v)) - np.cos(pi_beta)
    term_trig = np.maximum(term_trig, 1e-9) # Evita numeri negativi/zero
    
    tau = -gamma_t * np.log(u) * (term_trig ** (1/beta))
    return tau

# ==============================================================================
# 2. MOTORI DI SIMULAZIONE (OTTIMIZZATI PER MEMORIA)
# ==============================================================================

def simulate_trajectory(alpha, beta, gamma_t, gamma_x, n_jumps=1000):
    """Simula una singola traiettoria completa (t, x) per la Figura 1."""
    taus = generate_mittag_leffler(beta, gamma_t, n_jumps)
    jumps = generate_levy_stable(alpha, gamma_x, n_jumps)
    
    # Tempo cumulativo e posizione cumulativa
    t = np.concatenate(([0], np.cumsum(taus)))
    x = np.concatenate(([0], np.cumsum(jumps)))
    
    return t, x

def simulate_final_positions(alpha, beta, gamma_t, gamma_x, t_target, n_walkers=10000):
    """
    Calcola SOLO la posizione finale di n_walkers al tempo t_target.
    Usa blocchi per risparmiare memoria se n_walkers è alto.
    """
    # Stima euristica dei passi necessari:
    # Se beta è piccolo, i tempi sono lunghi -> servono pochi passi.
    # Se beta è grande (~1), servono più passi.
    estimated_steps = int(20 + 50 * t_target / gamma_t) if gamma_t > 0 else 1000
    if estimated_steps < 100: estimated_steps = 100
    if estimated_steps > 10000: estimated_steps = 10000 # Cap di sicurezza
    
    final_pos = np.zeros(n_walkers)
    
    # Generazione vettoriale
    taus = generate_mittag_leffler(beta, gamma_t, (n_walkers, estimated_steps))
    jumps = generate_levy_stable(alpha, gamma_x, (n_walkers, estimated_steps))
    
    times = np.cumsum(taus, axis=1)
    positions = np.cumsum(jumps, axis=1)
    
    # Logica vettoriale per trovare la posizione a t_target
    # Creiamo una maschera dove il tempo è ANCORA minore del target
    mask_still_walking = times < t_target
    
    # Contiamo quanti passi ha fatto ogni walker (somma dei True lungo l'asse)
    steps_taken = np.sum(mask_still_walking, axis=1)
    
    # L'indice da prendere è steps_taken - 1. 
    # Se steps_taken è 0, il walker è ancora all'origine (posizione 0).
    for i in range(n_walkers):
        idx = steps_taken[i] - 1
        if idx >= 0:
            final_pos[i] = positions[i, idx]
        else:
            final_pos[i] = 0.0
            
    return final_pos

# ==============================================================================
# 3. FUNZIONI TEORICHE (Per Validazione Fig. 4)
# ==============================================================================

def g_mu(mu, beta):
    # Densità spettrale per Mittag-Leffler (Eq. 26)
    num = (1/np.pi) * np.sin(beta * np.pi)
    den = mu**(1+beta) + 2 * np.cos(beta * np.pi) * mu + mu**(1-beta)
    return num / den

def mittag_leffler_func(t_val, beta):
    # Calcolo integrale numerico di E_beta(-t)
    if beta == 1.0: return np.exp(-t_val)
    integrand = lambda mu: np.exp(-mu * t_val) * g_mu(mu, beta)
    res, _ = quad(integrand, 0, np.inf, limit=50)
    return res

def get_theoretical_W(xi_vals, alpha, beta):
    """Calcola la curva teorica W(xi) tramite Trasformata di Fourier inversa."""
    # Se Gaussiana
    if alpha == 2.0 and beta == 1.0:
        return (1/np.sqrt(4*np.pi)) * np.exp(-xi_vals**2 / 4)

    # Altrimenti integrazione numerica
    k_vals = np.linspace(0, 10, 100) # Grid di integrazione k
    
    # Pre-calcoliamo E_beta(-k^alpha)
    ml_vals = np.array([mittag_leffler_func(k**alpha, beta) for k in k_vals])
    
    w_res = []
    for xi in xi_vals:
        # Integrale coseno: (1/pi) * int( cos(k*xi) * ML(k) )
        term = np.cos(k_vals * xi) * ml_vals
        val = np.trapz(term, k_vals) / np.pi
        w_res.append(val)
    return np.array(w_res)

# ==============================================================================
# 4. PLOTTING DELLE FIGURE
# ==============================================================================

def plot_fig_1():
    print("--- Generazione Figura 1 ---")
    alpha, beta = 1.7, 0.8
    gamma_t = 0.1
    gamma_x = gamma_t**(beta/alpha)
    
    t, x = simulate_trajectory(alpha, beta, gamma_t, gamma_x, n_jumps=500)
    
    plt.figure(figsize=(10, 6))
    plt.step(t, x, where='post', color='navy', linewidth=1.2)
    plt.xlabel(r'Time $t$', fontsize=14)
    plt.ylabel(r'Position $x(t)$', fontsize=14)
    plt.title(rf'Fig. 1: Trajectory ($\alpha={alpha}, \beta={beta}$)', fontsize=16)
    plt.xlim(0, np.percentile(t, 95)) # Zoom automatico per evitare outlier estremi
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_fig_3():
    print("--- Generazione Figura 3 (3D) ---")
    alpha, beta = 1.7, 0.8
    gamma_t = 0.1
    gamma_x = gamma_t**(beta/alpha)
    
    # Parametri simulazione
    n_walkers = 15000
    times = np.linspace(0.1, 10, 20) # 20 slice temporali
    bins = np.linspace(-3, 3, 50)    # Griglia spaziale
    
    # Matrice per Z
    Z = []
    
    for t in times:
        pos = simulate_final_positions(alpha, beta, gamma_t, gamma_x, t, n_walkers)
        hist, _ = np.histogram(pos, bins=bins, density=True)
        Z.append(hist)
    
    Z = np.array(Z)
    X_grid, Y_grid = np.meshgrid((bins[:-1]+bins[1:])/2, times)
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Surface con stile simile al paper
    surf = ax.plot_surface(X_grid, Y_grid, Z, cmap=cm.viridis, 
                           edgecolor='k', linewidth=0.2, alpha=0.9, rstride=1, cstride=1)
    
    ax.set_xlabel(r'$x$', fontsize=14)
    ax.set_ylabel(r'$t$', fontsize=14)
    ax.set_zlabel(r'$p(x,t)$', fontsize=14)
    ax.set_title(rf'Fig. 3: PDF Evolution ($\alpha={alpha}, \beta={beta}$)', fontsize=16)
    
    # View angle ottimizzato
    ax.view_init(elev=35, azim=-70)
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, np.max(Z)*1.1)
    
    plt.tight_layout()
    plt.show()

def plot_fig_4():
    print("--- Generazione Figura 4 (Scaling) - Attendere prego... ---")
    # Parametri: (Alpha, Beta, Label)
    cases = [
        (2.0, 1.0, 'Standard Diffusion'),
        (1.7, 0.8, 'Fractional 1'),
        (1.0, 0.9, 'Fractional 2')
    ]
    
    # Tempi diversi per verificare lo scaling
    gamma_vals = [1.0, 0.2, 0.05] 
    t_fixed = 5.0
    n_walkers = 20000
    
    fig, axes = plt.subplots(3, 1, figsize=(8, 14))
    
    for i, (alpha, beta, label) in enumerate(cases):
        ax = axes[i]
        print(f"   Calcolo caso {i+1}/3: alpha={alpha}, beta={beta}")
        
        # 1. Simulazioni (Punti colorati)
        for gt in gamma_vals:
            gx = gt**(beta/alpha)
            pos = simulate_final_positions(alpha, beta, gt, gx, t_fixed, n_walkers)
            
            # Scaling Variable: xi = x / t^(beta/alpha)
            scale_factor = t_fixed**(beta/alpha)
            scaled_pos = pos / scale_factor
            
            # Istogramma
            y_hist, bin_edges = np.histogram(scaled_pos, bins=60, range=(-4, 4), density=True)
            x_hist = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # PDF scalata: P_scaled = P_real * scale_factor
            # Nota: numpy density=True fa già l'integrale a 1 sullo spazio scalato, quindi è ok.
            
            ax.plot(x_hist, y_hist, '.', label=rf'$\gamma_t={gt}$')
            
        # 2. Teoria (Linea Nera)
        xi_theory = np.linspace(-4, 4, 60)
        try:
            w_theory = get_theoretical_W(xi_theory, alpha, beta)
            ax.plot(xi_theory, w_theory, 'k-', linewidth=2, label='Theory')
        except Exception as e:
            print(f"Errore calcolo teoria: {e}")

        ax.set_title(rf'Scaling: $\alpha={alpha}, \beta={beta}$', fontsize=14)
        ax.set_xlabel(r'$\xi = x / t^{\beta/\alpha}$', fontsize=12)
        ax.set_ylabel(r'$t^{\beta/\alpha} p(x,t)$', fontsize=12)
        ax.set_xlim(-4, 4)
        ax.grid(True, alpha=0.3)
        if i==0: ax.legend()
        
    plt.tight_layout()
    plt.show()

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    # Eseguiamo le funzioni. Puoi commentarne alcune se vuoi testarne solo una.
    plot_fig_1()
    plot_fig_3()
    plot_fig_4()