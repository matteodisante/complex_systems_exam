# Figure Notes (Keynote-ready)

Uso consigliato in Keynote:
- Copia/incolla ogni sezione “Slide” nelle **note del relatore** della slide corrispondente.
- Le prime 1–2 righe sono pensate come *talk track*, i bullet come *callout*.

## Slide — `fig6_beta_50png.png` (β = 1/2)
**File output:** `scripts/ou_fractional_scripts/figures/fig6_beta_50png.png`

**Generata da**
- Funzione: `generate_main_figure(beta, times, colors, x_values, x0, theta, K_beta, use_cache)`
- Codice: `scripts/ou_fractional_scripts/helpers.py`
- Chiamata da: `scripts/ou_fractional_scripts/main.py` (loop su `betas`)

**Parametri (da `main.py`)**
- β = 0.5
- θ = 1.0, Kβ = 1.0, x0 = 0.5
- Tempi: t ∈ {0.01, 0.1, 1, 10, 100}
- Griglia x: `linspace(-3, 3, 300)`
- Metodo numerico per le PDF: `compute_pdf_vectorized(..., Ns=800)`
- Cache: `data/main_figure_data_beta_50*.pkl` (se `use_cache=True`)

**Cosa mostra / a cosa serve**
- Evoluzione temporale della PDF del processo OU frazionario (β=1/2) + confronto con la distribuzione stazionaria.

**Aspetti salienti da notare**
- *Rilassamento lento* rispetto all’OU standard: le code e la forma impiegano più tempo a stabilizzarsi.
- Linea tratteggiata nera = stazionaria gaussiana con varianza `Kβ/θ`.
- La linea verticale marca la condizione iniziale x0.
- Nota “metodo”: ogni curva a tempo fissato è ottenuta numericamente con `Ns=800` (accuratezza vs costo).
- Lettura qualitativa: a tempi piccoli la PDF è più “concentrata” vicino a x0, poi tende a una forma simmetrica centrata in 0.
- Messaggio chiave (slide): β controlla la lentezza del rilassamento (memoria/subdiffusione).

---

## Slide — `fig6_beta_333_png.png` (β = 1/3)
**File output:** `scripts/ou_fractional_scripts/figures/fig6_beta_333_png.png`

**Generata da**
- Funzione: `generate_main_figure(...)` in `scripts/ou_fractional_scripts/helpers.py`
- Chiamata da: `scripts/ou_fractional_scripts/main.py`

**Parametri (da `main.py`)**
- β = 1/3
- θ = 1.0, Kβ = 1.0, x0 = 0.5
- Tempi: t ∈ {0.01, 0.1, 1, 10, 100}
- Griglia x: `linspace(-3, 3, 300)`
- `compute_pdf_vectorized(..., Ns=800)`
- Cache: `data/main_figure_data_beta_333*.pkl` (se `use_cache=True`)

**Cosa mostra / a cosa serve**
- Come sopra, ma per β=1/3: serve a evidenziare l’effetto della “più forte” memoria/subdiffusione (β più piccolo).

**Aspetti salienti da notare**
- Con β=1/3 l’avvicinamento alla stazionarietà è ancora più lento.
- Confronto diretto con β=1/2: stessa (θ, Kβ, x0), cambia solo β.
- Ai tempi piccoli si evidenzia maggiormente la differenza di forma tra i due β; ai tempi lunghi la differenza si attenua.
- Se serve una frase da slide: “β più piccolo ⇒ dinamica più ‘memory-driven’ ⇒ rilassamento più lento”.

---

## Slide — `fig6_comparison_panels.png` (β=1/2 vs β=1/3)
**File output:** `scripts/ou_fractional_scripts/figures/fig6_comparison_panels.png`

**Generata da**
- Funzione: `generate_comparison_panels(panel_times, x0, theta, K_beta, use_cache)`
- Codice: `scripts/ou_fractional_scripts/helpers.py`
- Chiamata da: `scripts/ou_fractional_scripts/main.py`

**Parametri principali**
- Pannelli 2×2 per t ∈ {0.01, 0.1, 1, 10}
- β confrontati: 1/2 (blu) vs 1/3 (arancio)
- θ = 1.0, Kβ = 1.0, x0 = 0.5
- Griglia: `x_panel = linspace(-0.5, 1.5, 600)`
- PDF calcolate con `compute_pdf_vectorized(..., Ns=800)`
- Cache: `data/comparison_panels_data.pkl`

**Cosa mostra / a cosa serve**
- Confronto “a parità di tutto” tra due β: rende immediata la differenza di forma della PDF ai vari tempi.

**Aspetti salienti da notare**
- Box L1: `L1 = ∫ |p_{β=1/3}(x,t) - p_{β=1/2}(x,t)| dx` (calcolata via trapezi sulla griglia).
- L1 tende a ridursi ai tempi lunghi: entrambe le dinamiche convergono verso la stessa stazionaria.
- Il pannello rende evidente *dove* le curve differiscono: tipicamente in prossimità del picco e nelle code.
- Nota numerica: L1 è un indicatore robusto e interpretabile (area tra le curve), non dipende dal segno della differenza.

---

## Slide — `fig6_spectral_vs_integral_beta_0_5.png` (β=1/2, griglia spettrale)
**File output:** `scripts/ou_fractional_scripts/figures/fig6_spectral_vs_integral_beta_0_5.png`

**Generata da**
- Funzione: `generate_spectral_comparison_plot(beta_spec, times_spec, Ns_list, n_repeats, ..., num_cores, use_cache)`
- Codice: `scripts/ou_fractional_scripts/helpers.py`
- Chiamata da: `scripts/ou_fractional_scripts/main.py`

**Parametri principali (da `main.py` + `helpers.py`)**
- β = 0.5
- Tempi: `times_spec = [0.01, 0.1, 1, 10, 100]`
- Troncamento serie spettrale: `Ns_list = [5, 20, 100, 200]` (colonne)
- Ripetizioni timing: `n_repeats = 5` (per stimare media e deviazione std del tempo)
- Griglia: `x_spec = linspace(-0.5, 1.5, 400)`
- Confronto tra:
  - **integral map**: `compute_pdf_vectorized(..., Ns=800)` (curva nera)
  - **serie spettrale**: `spectral_series_pdf(..., N)` (curva rossa tratteggiata)
- Parallelizzazione: `multiprocessing.Pool(processes=num_cores)` (da CLI `--cores`, default = tutti)
- Cache: `data/spectral_comparison_data_beta_0_5.pkl`

**Cosa mostra / a cosa serve**
- Validazione numerica: quanto bene la serie spettrale approssima la soluzione “integral map” al variare di N e t.

**Aspetti salienti da notare**
- Box L1 in ogni pannello: errore `∫|p_spec - p_ref| dx`.
- Box Time: tempo medio ± std su `n_repeats=5` (misurato con `time.perf_counter`).
- Trade-off chiave: aumentando N l’errore scende ma il costo computazionale cresce.
- Lettura della griglia: righe = tempi (dinamica), colonne = N (accuratezza/costo della serie spettrale).
- Regola pratica da enfatizzare: scegliere N minimo che porta L1 sotto una soglia “accettabile” per la presentazione.
- Nota su parallelizzazione: i calcoli per (t, N) sono indipendenti e vengono distribuiti su `--cores`.

---

## Slide — `fig6_spectral_vs_integral_beta_1_3.png` (β=1/3, griglia spettrale)
**File output:** `scripts/ou_fractional_scripts/figures/fig6_spectral_vs_integral_beta_1_3.png`

**Generata da**
- Stessa pipeline della figura β=1/2, ma con β=1/3.

**Parametri principali**
- β = 1/3
- Tempi: `times_spec = [0.01, 0.1, 1, 10, 100]`
- `Ns_list = [5, 20, 100, 200]`, `n_repeats=5`
- `x_spec = linspace(-0.5, 1.5, 400)`
- Cache: `data/spectral_comparison_data_beta_1_3.pkl`

**Cosa mostra / a cosa serve**
- Come cambia accuratezza/costo della serie spettrale quando β è più piccolo.

**Aspetti salienti da notare**
- A N bassi l’errore può essere più visibile ai tempi piccoli.
- Stesso trade-off: accuratezza vs tempo.
- Confronto tra β: a parità di N, l’accuratezza richiesta può cambiare (β più piccolo può “stressare” di più l’approssimazione ai tempi brevi).
- Messaggio chiave (slide): la validazione non è solo “funziona/non funziona” ma quantifica errore (L1) e costo (Time).

---

## Slide — `fig_timing_vs_N_beta_1_3.png` (tempo vs N)
**File output:** `scripts/ou_fractional_scripts/figures/fig_timing_vs_N_beta_1_3.png`

**Generata da**
- Funzione: `generate_timing_plot(num_cores, use_cache)` → `_load_or_compute_timing_data(...)` → `_plot_timing_data(...)`
- Codice: `scripts/ou_fractional_scripts/helpers.py`
- Chiamata da: `scripts/ou_fractional_scripts/main.py`

**Parametri principali (da `helpers.py`)**
- β fissato: 1/3
- Tempi: `times_spec = [0.01, 0.1, 1, 10, 100]`
- N testati: `Ns_list_timing = [5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]`
- Ripetizioni: `n_repeats = 5`
- Griglia: `x_spec = linspace(-0.5, 1.5, 400)`
- Tempo misurato sulla **serie spettrale** (`spectral_series_pdf`) tramite `compute_spec_and_time`
- Barre di errore: deviazione standard sui 5 run
- Cache: `data/timing_plot_data_beta_1_3.pkl`

**Cosa mostra / a cosa serve**
- Quantifica il costo computazionale della serie spettrale al crescere di N e al variare di t.

**Aspetti salienti da notare**
- Crescita del tempo con N (quasi monotona): utile per scegliere un N “budget-aware”.
- Le barre d’errore riflettono variabilità runtime (scheduler, cache CPU, OS).
- Ogni punto è media su `n_repeats=5`; le barre sono deviazione standard (non errore standard della media).
- Lettura per decisione: fissato un tempo massimo per pannello (es. <1s), si ricava un range di N utilizzabile.
- Nota di riproducibilità: con cache attiva si riusano i tempi salvati; con `--no-cache` si ricomputa tutto.

---

## Slide — `fig6_fractional_vs_nonfractional.png` (frazionario vs standard)
**File output:** `scripts/ou_fractional_scripts/figures/fig6_fractional_vs_nonfractional.png`

**Generata da**
- Funzione: `generate_fractional_vs_nonfractional_plot(comparison_betas, comparison_times, x0, theta, K_beta, use_cache)`
- Helper: `_load_or_compute_frac_vs_nonfrac_data(...)` + `_plot_frac_vs_nonfrac_data(...)`
- Codice: `scripts/ou_fractional_scripts/helpers.py`
- Chiamata da: `scripts/ou_fractional_scripts/main.py`

**Parametri principali**
- β confrontati: {1/2, 1/3, 1 (standard)}
- Tempi: t ∈ {0.01, 0.1, 1, 10}
- θ = 1.0, Kβ = 1.0, x0 = 0.5
- Griglia: `comparison_x = linspace(-1.0, 2.0, 400)`
- PDF frazionarie: `compute_pdf_vectorized(..., Ns=800)`
- Caso standard (β=1): **gaussiana analitica** dell’OU classico
  - media: `x0 * exp(-θ t)`
  - varianza: `(Kβ/θ) * (1 - exp(-2 θ t))`
- Cache: `data/frac_vs_nonfrac_data.pkl`

**Cosa mostra / a cosa serve**
- Evidenzia la differenza tra dinamica standard (Markoviana) e dinamiche frazionarie (memoria) a parità di parametri.

**Aspetti salienti da notare**
- Lo standard OU “si apre” con varianza che cresce verso `Kβ/θ`, mentre i frazionari mostrano rilassamento più lento.
- A tempi lunghi le curve tendono a essere più simili (stessa stazionaria).
- Caso standard (β=1): è la soluzione analitica gaussiana, utile come baseline “senza memoria”.
- Differenza concettuale da dire a voce: nel frazionario la subordinazione/time-change rende la dinamica non-Markoviana.
- Suggerimento per narrazione: mostrare prima t piccoli (differenze massime), poi t grandi (convergenza).
