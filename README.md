# veneto_cf_verifica-risoluzione-immagini-in-otter

Pipeline automatizzata per applicare **super-risoluzione** e **downscaling personalizzato** su immagini contenenti righelli cromatici, tipicamente utilizzate nel contesto del riconoscimento CF e verifica qualità immagini.

---

## 📦 Installazione della repository

Aprire il terminale ed eseguire il comando:

```bash
git clone https://github.com/DTimperiCSA/veneto_cf_verifica-risoluzione-immagini-in-otter
cd veneto_cf_verifica-risoluzione-immagini-in-otter
````

---

## 🔧 Inizializzazione e dipendenze

Il progetto utilizza [**PDM**](https://pdm.fming.dev) come gestore di pacchetti Python.

### Requisiti:

* Python ≥ 3.10
* pip (per installare `pdm`)

### Passaggi:

1. Installare `pdm` (se non lo si ha già):

```bash
pip install pdm
```

2. Installare tutte le dipendenze del progetto:

```bash
pdm install
```

---

## 🚀 Lancio del progetto

Per avviare il progetto:

```bash
pdm run python -m src.main
```

---

## 🔁 Workflow della pipeline

### Benchmark automatico

* Alla prima esecuzione, viene effettuato un **benchmark automatico** utilizzando **le 5 immagini più pesanti**.
* Vengono testate più configurazioni: device (CPU/GPU), numero di processi e numero di thread.
* I risultati vengono salvati in `logs/benchmark_results.csv`.
* Viene selezionata e salvata la **configurazione ottimale** in `logs/best_config.json`.

### Elaborazione immagini

1. La pipeline legge tutte le immagini presenti nella directory `images/input`.
2. Per ogni **sottocartella** di input, vengono processate le immagini contenute al suo interno.
3. Per ciascuna immagine:

   * Viene applicata la **super-risoluzione** (x2, x3 o x4) usando il modello configurato in `config.py`.
   * Viene applicato il **downscaling personalizzato** in base al PPI (es. 400\_PPI, 600\_PPI) rilevato dalla **sottocartella**.
   * Le dimensioni in millimetri vengono **convertite in pollici** prima del calcolo, con l'aggiunta di un **fattore correttivo empirico** (`CORRETTIVO_INCH_...`).
4. I risultati finali vengono salvati in:

   * `images/output/sr_xN/...` per le immagini super-risolute
   * `images/output/downscaling/...` per le immagini finali downscaled
5. I log di errore vengono salvati solo per le immagini **fallite** in `logs/failures.csv`.

---

## 📂 Struttura delle cartelle

```
veneto_cf_verifica-risoluzione-immagini-in-otter/
│
├── images/
│   ├── input/               # immagini da elaborare
│   └── output/
│       ├── sr_x2/           # immagini super-risolute
│       └── downscaling/     # immagini finali dopo il downscaling
│
├── logs/
│   ├── failures.csv         # log delle immagini fallite
│   ├── benchmark_results.csv
│   └── best_config.json     # configurazione ottimale trovata
│
├── model/
│   └── SR_Script/           # modello super-resolution
│
├── src/
│   ├── main.py              # entrypoint principale
│   ├── benchmark.py         # script di benchmark
│   ├── processing.py        # logica di super-risoluzione e downscaling
│   ├── worker.py            # classe ImageWorker
│   ├── utils.py             # utility varie
│   ├── config.py            # configurazioni (PPI, scale, parametri)
│   └── paths.py             # gestione percorsi
│
└── pyproject.toml           # definizioni PDM
```

---

## 📌 Note aggiuntive

* Tutti i percorsi vengono gestiti con `pathlib.Path`.
* Il progetto è compatibile con ambienti **Windows, Linux e macOS**.
* Il logging è altamente dettagliato per il debug.
* L'intero flusso supporta **multi-threading** e **multi-processing**.
* Il downscaling si basa **sulla cartella in cui si trova l'immagine**, non sul nome file.