import sys
import time
import csv
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, set_start_method, Manager
from functools import partial
from more_itertools import chunked

from src.utils import *
from src.paths import *
from src.config import *
from src.worker import ImageWorker
from logs.logger import CSVLogger
from model.SR_Script.super_resolution import SA_SuperResolution
from benchmark.benchmark import benchmark

MAX_ATTEMPTS = 10
RETRY_DELAY = 5  # seconds

# Modifica della funzione per aggiornare via queue
def process_batch(images, threads, super_resolution_dir, downscaling_dir, model_path, logger_path, progress_queue):
    model = SA_SuperResolution(
        models_dir=model_path,
        model_scale=SUPER_RESOLUTION_PAR,
        tile_size=128,
        gpu_id=0,
        verbosity=False,
    )
    logger = CSVLogger(logger_path)
    worker = ImageWorker(logger, super_resolution_dir, downscaling_dir, model)

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(worker.run, img): img for img in images}

        for future in as_completed(futures):
            img = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.log(img.name, "run", success=False, error=f"Thread error: {e}")
            finally:
                progress_queue.put(1)  # segnala un'immagine completata

    logger.stop()


def run_standard_processing(processes=1, threads=4):
    
    print("🔍 Verifica iniziale: caricamento del modello di super-risoluzione...")

    try:
        _ = SA_SuperResolution(
            models_dir=SR_SCRIPT_MODEL_DIR,
            model_scale=SUPER_RESOLUTION_PAR,
            tile_size=128,
            gpu_id=0,
            verbosity=True,
        )
    except Exception as e:
        raise RuntimeError(f"Errore durante il caricamento del modello di super-risoluzione: {e}")

    print("🚀 Avvio del processo multi-processo e multi-thread...")

    super_resolution_dir, downscaling_dir = find_output_dir()
    input_images = count_all_images(INPUT_IMAGES_DIR)

    if not input_images:
        print("⚠️ Nessuna immagine trovata nella cartella di input. Uscita.")
        return

    images_to_process = []
    for img in input_images:
        relative = img.relative_to(INPUT_IMAGES_DIR)
        subdir = relative.parts[0] if len(relative.parts) > 1 else ""
        final_output = downscaling_dir / subdir / img.name
        if not final_output.exists():
            images_to_process.append(img)

    print(f"📦 Totale immagini trovate:        {len(input_images)}")
    print(f"✅ Immagini già elaborate:         {len(input_images) - len(images_to_process)}")
    print(f"🕐 Immagini da elaborare ora:      {len(images_to_process)}")

    chunks = list(chunked(images_to_process, max(1, len(images_to_process) // processes)))

    manager = Manager()
    progress_queue = manager.Queue()

    target = partial(
        process_batch,
        threads=threads,
        super_resolution_dir=super_resolution_dir,
        downscaling_dir=downscaling_dir,
        model_path=SR_SCRIPT_MODEL_DIR,
        logger_path=CSV_LOG_PATH,
        progress_queue=progress_queue,
    )

    try:
        set_start_method("spawn", force=True)
        with Pool(processes) as pool:
            result = pool.map_async(target, chunks)

            with tqdm(total=len(images_to_process), desc="📷 Immagini elaborate") as pbar:
                completed = 0
                while completed < len(images_to_process):
                    try:
                        progress_queue.get(timeout=0.5)
                        completed += 1
                        pbar.update(1)
                    except:
                        if result.ready():
                            break
                result.wait()
    except KeyboardInterrupt:
        print("\n[🚪] Interrotto manualmente dall'utente. Uscita.")
        sys.exit(0)

    # Log finale
    error_count = 0
    if CSV_LOG_PATH.exists():
        with open(CSV_LOG_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            error_count = sum(1 for row in reader if row["status"] == "false")

    success_count = len(input_images) - error_count

    print("\n📊 Risultato finale:")
    print(f"✅ Immagini processate con successo: {success_count}")
    print(f"❌ Immagini con errore:              {error_count}")

def main():
    parser = argparse.ArgumentParser(description="Processa immagini o esegui benchmark.")
    parser.add_argument("--benchmark", action="store_true", help="Esegui benchmark multiprocesso e multithread")
    args = parser.parse_args()

    if args.benchmark:
        benchmark()
        return

    if not JSON_BENCHMARK_BEST_CONFIG_PATH.exists():
        print("⚠️ Nessuna configurazione ottimale trovata. Eseguo benchmark...")
        benchmark()

    with JSON_BENCHMARK_BEST_CONFIG_PATH.open("r", encoding="utf-8") as f:
        best_config = json.load(f)

    processes = int(best_config["processes"])
    threads = int(best_config["threads"])
    print(f"\n📌 Uso della configurazione ottimale: {processes} processi, {threads} thread")

    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                print(f"\n🔁 Tentativo {attempt} di {MAX_ATTEMPTS}...\n")
                run_standard_processing(processes, threads)
                print("✅ Elaborazione completata con successo.")
                break
            except KeyboardInterrupt:
                print("\n[🚪] Interrotto manualmente dall'utente. Uscita.")
                sys.exit(0)
            except Exception as e:
                print(f"\n❌ Crash: {e}")
                if attempt < MAX_ATTEMPTS:
                    print(f"⏳ Nuovo tentativo in {RETRY_DELAY} secondi...")
                    time.sleep(RETRY_DELAY)
                else:
                    print("\n❌ Numero massimo di tentativi raggiunto. Uscita.")
                    sys.exit(1)
    except KeyboardInterrupt:
        print("\n[🚪] Interrotto manualmente dall'utente. Uscita.")
        sys.exit(0)


if __name__ == "__main__":
    main()