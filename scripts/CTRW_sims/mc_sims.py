import os

# Limita numpy a 1 thread per evitare oversubscription con multiprocessing
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import gamma
from scipy.integrate import quad, simpson
from multiprocessing import Pool, cpu_count

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

def simulate_final_positions(alpha, beta, gamma_t, gamma_x, t_target, n_walkers=10000, block_size=2000):
    """
    Calcola SOLO la posizione finale di n_walkers al tempo t_target.
    Usa blocchi per risparmiare memoria se n_walkers è alto.
    """
    # Stima euristica dei passi necessari:
    # Se beta è piccolo, i tempi sono lunghi -> servono pochi passi.
    # Se beta è grande (~1), servono più passi.
    estimated_steps = int(20 + 30 * t_target / gamma_t) if gamma_t > 0 else 1000
    if estimated_steps < 100: estimated_steps = 100
    if estimated_steps > 5000: estimated_steps = 5000 # Cap ridotto per risparmiare RAM
    
    final_pos = np.zeros(n_walkers)

    # Elaborazione a blocchi per ridurre la RAM e migliorare la parallelizzazione
    if block_size is None or block_size <= 0:
        block_size = n_walkers

    for start in range(0, n_walkers, block_size):
        end = min(start + block_size, n_walkers)
        batch_size = end - start

        # Generazione vettoriale sul blocco
        taus = generate_mittag_leffler(beta, gamma_t, (batch_size, estimated_steps))
        jumps = generate_levy_stable(alpha, gamma_x, (batch_size, estimated_steps))

        times = np.cumsum(taus, axis=1)
        positions = np.cumsum(jumps, axis=1)

        # Logica vettoriale per trovare la posizione a t_target
        mask_still_walking = times < t_target
        steps_taken = np.sum(mask_still_walking, axis=1)

        idx = steps_taken - 1
        idx_clipped = np.clip(idx, 0, estimated_steps - 1)
        batch_positions = positions[np.arange(batch_size), idx_clipped]
        batch_positions[idx < 0] = 0.0

        final_pos[start:end] = batch_positions

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

    # Altrimenti integrazione numerica con quad (integratore adattivo)
    # Integrazione per segmenti per ridurre instabilità ai k alti
    w_res = []
    k_max = 120.0  # Aumentato intervallo in k per ridurre fluttuazioni
    segments = [(0, 20), (20, 40), (40, 60), (60, 80), (80, k_max)]
    for xi in xi_vals:
        if np.isclose(xi, 0.0):
            integrand = lambda k: mittag_leffler_func(k**alpha, beta)
            val, _ = quad(integrand, 0, k_max, limit=600, epsabs=1e-10, epsrel=1e-8)
        else:
            integrand = lambda k: mittag_leffler_func(k**alpha, beta)
            total = 0.0
            for a, b in segments:
                seg, _ = quad(
                    integrand,
                    a,
                    b,
                    weight='cos',
                    wvar=xi,
                    limit=300,
                    epsabs=1e-10,
                    epsrel=1e-8,
                )
                total += seg
            val = total
        w_res.append(val / np.pi)
    return np.array(w_res)

# ==============================================================================
# 4. PLOTTING DELLE FIGURE
# ==============================================================================

def _compute_single_curve(args):
    """Worker function per parallelizzare il calcolo di una singola curva gamma_t."""
    case_idx, alpha, beta, gt, t_fixed, n_walkers, block_size, bins = args
    gx = gt**(beta/alpha)
    pos = simulate_final_positions(alpha, beta, gt, gx, t_fixed, n_walkers, block_size=block_size)
    
    # Scaling Variable: xi = x / t^(beta/alpha)
    scale_factor = t_fixed**(beta/alpha)
    scaled_pos = pos / scale_factor
    
    # Istogramma
    y_hist, bin_edges = np.histogram(scaled_pos, bins=bins, range=(-4, 4), density=True)
    x_hist = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return case_idx, x_hist, y_hist, gt

def plot_fig_1(alpha=1.7, beta=0.8, n_realizations=4, n_jumps=500, gamma_t=0.1):
    print("--- Generazione Figura 1 (4 realizzazioni) ---")

    gamma_x = gamma_t**(beta/alpha)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    axes = axes.ravel()

    for idx, ax in enumerate(axes[:n_realizations], start=1):
        t, x = simulate_trajectory(alpha, beta, gamma_t, gamma_x, n_jumps=n_jumps)

        ax.step(t, x, where='post', color='navy', linewidth=1.1)
        ax.set_xlabel(r'Time $t$', fontsize=14)
        ax.set_ylabel(r'Position $x(t)$', fontsize=14)
        ax.set_title(rf'Realization {idx}: $\alpha={alpha}, \beta={beta}$', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlim(0, np.percentile(t, 95))
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.show()


def plot_fig_4(n_cores=None, n_walkers=40000, bins=120, seed=1234, block_size=2000):
    """
    Genera Figura 4 con scaling laws.
    
    Parameters:
    -----------
    n_cores : int, optional
        Numero di core da usare per la parallelizzazione.
        Se None, usa tutti i core disponibili.
        Se 1, esegue in modo seriale (utile per debug).
    """
    if n_cores is None:
        n_cores = cpu_count()
    
    print(f"--- Generazione Figura 4 (Scaling) - Usando {n_cores} core(s) ---")
    print(f"Configurazione: n_walkers={n_walkers}, bins={bins}, block_size={block_size}, t_fixed={5.0}, seed={seed}")
    np.random.seed(seed)
    
    # Parametri: (Alpha, Beta, Label, gamma_t values)
    cases = [
        (2.0, 1.0, 'Standard Diffusion', [0.01, 0.80, 1.00, 1.20, 1.40]),
        (1.7, 0.8, 'Fractional 1', [0.0001, 0.1000, 0.3000, 0.5000, 0.7000]),
        (1.0, 0.9, 'Fractional 2', [0.004, 0.600, 0.800, 1.000, 1.200])
    ]
    
    t_fixed = 5.0
    
    fig, axes = plt.subplots(3, 1, figsize=(8, 14))
    
    # Prepara TUTTI i task per TUTTI i casi contemporaneamente
    all_tasks = []
    for i, (alpha, beta, label, gamma_vals) in enumerate(cases):
        print(f"   Caso {i+1}: alpha={alpha}, beta={beta}, gamma_t={gamma_vals}")
        for gt in gamma_vals:
            all_tasks.append((i, alpha, beta, gt, t_fixed, n_walkers, block_size, bins))
    
    print(f"   Totale task da eseguire: {len(all_tasks)}")
    
    # Esegui TUTTI i task in parallelo
    if n_cores > 1:
        with Pool(n_cores) as pool:
            all_results = pool.map(_compute_single_curve, all_tasks)
    else:
        all_results = [_compute_single_curve(task) for task in all_tasks]
    
    # Raggruppa i risultati per caso
    results_by_case = {0: [], 1: [], 2: []}
    for case_idx, x_hist, y_hist, gt in all_results:
        results_by_case[case_idx].append((x_hist, y_hist, gt))
    
    # Plot dei risultati
    for i, (alpha, beta, label, gamma_vals) in enumerate(cases):
        ax = axes[i]
        print(f"   Plotting caso {i+1}/3: alpha={alpha}, beta={beta}")
        
        # Plot simulazioni
        for x_hist, y_hist, gt in results_by_case[i]:
            ax.plot(x_hist, y_hist, '-', linewidth=1.2, label=rf'$\gamma_t={gt}$')
            
        # 2. Teoria (Linea Nera)
        xi_theory = np.linspace(-4, 4, 200)  # Aumentato sampling per curva liscia
        try:
            w_theory = get_theoretical_W(xi_theory, alpha, beta)
            ax.plot(xi_theory, w_theory, 'k-', linewidth=1.8, label='Theory')
        except Exception as e:
            print(f"Errore calcolo teoria: {e}")

        ax.set_title(rf'Scaling: $\alpha={alpha}, \beta={beta}$', fontsize=18, fontweight='bold')
        ax.set_xlabel(r'$\xi = x / t^{\beta/\alpha}$', fontsize=16)
        ax.set_ylabel(r'$t^{\beta/\alpha} p(x,t)$', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlim(-4, 4)
        ax.grid(True, alpha=0.3)
        if i==0: ax.legend(fontsize=12)
        
    plt.tight_layout()
    plt.show()

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    # Configurazione numero di core
    # None = usa tutti i core disponibili
    # 1 = esecuzione seriale (no parallelizzazione)
    # 4 = usa 4 core
    N_CORES = 8  # Cambia questo valore per controllare il parallelismo
    
    print(f"Core disponibili: {cpu_count()}")
    if N_CORES is None:
        print(f"Verranno usati tutti i {cpu_count()} core disponibili")
    else:
        print(f"Verranno usati {N_CORES} core")
    
    # Eseguiamo le funzioni. Puoi commentarne alcune se vuoi testarne solo una.
    plot_fig_1()
    plot_fig_4(n_cores=N_CORES)