# 1. Parti da un'immagine Python 3.14 leggera
FROM python:3.14-slim

# 2. Imposta la cartella di lavoro all'interno del container
WORKDIR /app

# 3. Copia SOLO il file dei requisiti
COPY requirements.txt .

# 4. Installa le dipendenze
#    (lo facciamo prima di copiare il codice per la cache)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Ora copia tutto il resto del tuo codice nel container
COPY . .

# 6. Comando di default (opzionale, ma buona pratica)
CMD ["python", "main.py"]