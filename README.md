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
│   .gitignore
│   .pdm-python
│   pdm.lock
│   pyproject.toml
│   README.md
│
├───benchmark
│   │   benchmark.py
│   │   benchmark_log.csv
│   │   benchmark_results.json
│   │
│   └───__pycache__
│           benchmark.cpython-313.pyc
│
├───images
│   ├───input
│   │   ├───400_PPI
│   │   │       file.svg
│   │   │       immagine_a_400 - Copia (2).tif
│   │   │       immagine_a_400 - Copia (3).tif
│   │   │       immagine_a_400 - Copia (4).tif
│   │   │       immagine_a_400 - Copia.tif
│   │   │       immagine_a_400.tif
│   │   │
│   │   └───600_PPI
│   │           immagine_a_600 - Copia (2).tif
│   │           immagine_a_600 - Copia (3).tif
│   │           immagine_a_600 - Copia (4).tif
│   │           immagine_a_600 - Copia.svg
│   │           immagine_a_600.tif
│   │           sample-50-MB-pdf-file.pdf
│   │
│   ├───original
│   │       immagine_a_400.tif
│   │       immagine_a_600.tif
│   │
│   └───output
│       ├───downscaled_x2
│       │   ├───400_PPI
│       │   └───600_PPI
│       └───sr_x2
│           ├───400_PPI
│           └───600_PPI
├───logs
│   │   logger.py
│   │   processing_log.csv
│   │   processing_log_benchmark.csv
│   │   test_log.csv
│   │
│   └───__pycache__
│           logger.cpython-313.pyc
│
├───metadata
│       metadati_immagine_a_400.json
│       metadati_immagine_a_600.json
│
├───model
│   └───SR_Script
│       │   .gitignore
│       │   pdm.lock
│       │   process_sr.py
│       │   pyproject.toml
│       │   README.md
│       │   requirements.txt
│       │   super_resolution.py
│       │   tiling_image_loader.py
│       │
│       ├───input
│       │       00001 copy.png
│       │       00001.png
│       │
│       ├───output
│       │       00001 copy_x2.png
│       │       00001_x2.png
│       │
│       ├───src
│       │   └───sr_script
│       │           __init__.py
│       │
│       ├───super_res
│       │       edsr_2x.ven
│       │       edsr_3x.ven
│       │       edsr_4x.ven
│       │
│       ├───tests
│       │       __init__.py
│       │
│       └───__pycache__
│               super_resolution.cpython-313.pyc
│               super_resolution.cpython-39.pyc
│               tiling_image_loader.cpython-313.pyc
│
├───src
│   │   config.py
│   │   image_processing.py
│   │   main.py
│   │   metadata_manager.py
│   │   paths.py
│   │   utils.py
│   │   worker.py
│   │   __init__.py
│   │
│   └───__pycache__
│           config.cpython-313.pyc
│           image_processing.cpython-313.pyc
│           main.cpython-313.pyc
│           paths.cpython-313.pyc
│           test.cpython-313.pyc
│           utils.cpython-313.pyc
│           worker.cpython-313.pyc
│           __init__.cpython-313.pyc
│
└───tests
    │   test_benchmark_thread.py
    │   test_downscaling.py
    │   test_run_csv_logger.py
    │   test_super_resolution_x2.py
    │
    └───__pycache__
            test_benchmark_thread.cpython-313.pyc
            test_run_csv_logger.cpython-313.pyc
```

---

## 📌 Note aggiuntive

* Tutti i percorsi vengono gestiti con `pathlib.Path`.
* Il progetto è compatibile con ambienti **Windows, Linux e macOS**.
* Il logging è altamente dettagliato per il debug.
* L'intero flusso supporta **multi-threading** e **multi-processing**.
* Il downscaling si basa **sulla cartella in cui si trova l'immagine**, non sul nome file.
