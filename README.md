# Simulazione del Processo Frazionario di Ornstein-Uhlenbeck

Questa repository contiene uno script Python per simulare il processo frazionario di Ornstein-Uhlenbeck (fOU), con un focus sui casi in cui l'esponente frazionario (α) è 1/2 e 1/3. Lo scopo è fornire un'implementazione minima, autoconsistente e corretta per calcolare e visualizzare la funzione di densità di probabilità (PDF) di questo processo.

## Contenuto della Repository

- **`code2.py`**: Lo script principale che esegue la simulazione. Calcola la PDF del processo fOU utilizzando due metodi differenti e genera una serie di grafici per visualizzare i risultati.
- **`test_code2.py`**: Una suite di unit test per lo script `code2.py` per garantire la correttezza delle funzioni implementate.
- **`hermite_check.py`**: Uno script di utilità per verificare l'implementazione delle funzioni di Hermite utilizzate nel metodo delle serie spettrali.

## Cosa fa il codice

Lo script `code2.py` implementa due approcci per calcolare la PDF del processo fOU:

1.  **Metodo della Mappa Integrale**: Questo metodo si basa sulla forma analitica di Smirnov per la densità di Lévy. La PDF `P(x,t)` viene calcolata come un integrale della convoluzione tra la PDF `n(s,t)` dei tempi di attesa e il kernel Gaussiano del processo di Ornstein-Uhlenbeck `P1(x,s)`. L'integrazione viene eseguita numericamente su una griglia di valori di `s`.

2.  **Metodo delle Serie Spettrali**: Questo approccio alternativo calcola la PDF come una serie di autofunzioni del processo di Ornstein-Uhlenbeck (funzioni di Hermite). L'evoluzione temporale è catturata da un fattore che include la funzione di Mittag-Leffler.

Lo script genera i seguenti output:

- **`fig6_alpha_0_500.png`**: Evoluzione temporale della PDF per α = 1/2.
- **`fig6_alpha_0_333.png`**: Evoluzione temporale della PDF per α = 1/3.
- **`fig6_comparison_panels.png`**: Un confronto diretto tra le PDF per α = 1/2 e α = 1/3 a diversi istanti di tempo.
- **`fig6_spectral_vs_integral.png`**: Un confronto tra il metodo della mappa integrale e il metodo delle serie spettrali per α = 1/3, mostrando la convergenza all'aumentare del numero di termini nella serie.

## Come usare la repository

### Prerequisiti

Assicurati di avere installato le seguenti librerie Python:

- `numpy`
- `matplotlib`
- `scipy`

Puoi installarle usando pip:
```bash
pip install numpy matplotlib scipy
```

### Eseguire la simulazione

Per eseguire la simulazione e generare i grafici, esegui lo script `code2.py`:

```bash
python code2.py
```

I grafici verranno salvati nella directory principale della repository.

### Eseguire i test

Per verificare la correttezza del codice, puoi eseguire la suite di test:

```bash
python test_code2.py
```

### Verificare le funzioni di Hermite

Per controllare l'implementazione delle funzioni di Hermite, puoi eseguire lo script `hermite_check.py`:

```bash
python hermite_check.py
```
Questo script confronterà l'implementazione basata sulla ricorrenza con quella che utilizza la formula diretta di `scipy`, stampando le differenze.

## Scopo del progetto

Lo scopo di questa repository è fornire un esempio chiaro, funzionante e verificato di come simulare il processo frazionario di Ornstein-Uhlenbeck. Può essere utile a studenti, ricercatori o chiunque sia interessato a processi stocastici e sistemi complessi, sia come strumento di apprendimento che come base per ulteriori ricerche.
