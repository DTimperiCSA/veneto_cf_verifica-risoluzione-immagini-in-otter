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
from math import ceil

from src.utils import *
from src.paths import *
from src.config import *
from src.estimate_ppi_from_ruler import *
from src.worker import ImageWorker
from logs.logger import CSVLogger
from model.SR_Script.super_resolution import SA_SuperResolution
from benchmark.benchmark import benchmark

MAX_ATTEMPTS = 10
RETRY_DELAY = 5  # seconds

# ---------- Process batch ----------
def process_batch(images, threads, super_resolution_dir, downscaling_dir, model_path, logger, progress_queue):
    """Process a chunk of images in threads."""
    model = SA_SuperResolution(
        models_dir=model_path,
        model_scale=SUPER_RESOLUTION_PAR,
        tile_size=128,
        gpu_id=0,
        verbosity=False,
    )

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
                progress_queue.put(1)
    logger.stop()

# ---------- Standard processing ----------
def run_standard_processing(processes, threads, logger: CSVLogger):
    print("🔍 Caricamento modello di super-risoluzione (test iniziale)...")
    try:
        _ = SA_SuperResolution(
            models_dir=SR_SCRIPT_MODEL_DIR,
            model_scale=SUPER_RESOLUTION_PAR,
            tile_size=128,
            gpu_id=0,
            verbosity=True,
        )
    except Exception as e:
        logger.log_crash(f"Errore caricamento modello SR: {e}")
        raise RuntimeError(f"Errore nel caricamento modello SR: {e}")

    print("\n📂 Scansione cartelle da elaborare...")
    super_resolution_dir, downscaling_dir = find_output_dir()
    folders = [f for f in INPUT_IMAGES_DIR.rglob("*") if f.is_dir()]
    folder_to_images = {}

    for folder in folders:
        images = [img for img in folder.glob("*") if is_valid_image_file(img) and img.name.lower() != "thumbs.db"]
        images_to_process = []
        for img in images:
            rel = img.relative_to(INPUT_IMAGES_DIR)
            subdir = rel.parent
            out_path = downscaling_dir / subdir / img.name
            if not out_path.exists():
                images_to_process.append(img)
        if images_to_process:
            folder_to_images[folder] = images_to_process
        else:
            logger.log(folder.name, "no_images_to_process", success=False, error="Nessuna immagine da processare", full_path=str(folder))

    if not folder_to_images:
        print("✅ Tutte le immagini risultano già elaborate.")
        return

    manager = Manager()
    progress_queue = manager.Queue()

    total_success = 0
    total_error = 0

    for folder, images in folder_to_images.items():
        print(f"\n📂 Cartella: {folder} ({len(images)} immagini da processare)")

        ppi = estimate_ppi_for_folder(folder)
        if not ppi:
            logger.log(folder.name, "estimate_ppi", success=False, error="Impossibile stimare PPI", full_path=str(folder))
            print(f"⚠️ Impossibile stimare PPI per {folder}. Skip cartella.")
            continue

        chunk_size = ceil(len(images) / processes)
        chunks = list(chunked(images, chunk_size))

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
                completed = 0
                with tqdm(total=len(images), desc="📷 Immagini elaborate", ncols=80) as pbar:
                    while completed < len(images):
                        try:
                            progress_queue.get(timeout=1)
                            completed += 1
                            pbar.update(1)
                        except:
                            if result.ready():
                                while not progress_queue.empty():
                                    progress_queue.get_nowait()
                                    completed += 1
                                    pbar.update(1)
                                break
                            continue
                result.wait()
        except KeyboardInterrupt:
            print("\n[🚪] Interrotto manualmente dall'utente. Uscita.")
            sys.exit(0)
        except Exception as e:
            logger.log_crash(f"Errore multiprocessing: {e}")
            raise

        # Conta successi e fallimenti
        folder_error_count = 0
        if CSV_LOG_PATH.exists():
            with open(CSV_LOG_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                folder_error_count = sum(
                    1 for row in reader
                    if row["status"] == "false" and Path(row["full_path"]).parent.name == folder.name
                )

        folder_success = len(images) - folder_error_count
        total_success += folder_success
        total_error += folder_error_count

        print(f"   ✅ Successi: {folder_success} | ❌ Errori: {folder_error_count}")

    print("\n📊 Risultato finale:")
    print(f"✅ Immagini processate con successo: {total_success}")
    print(f"❌ Immagini con errore:              {total_error}")

    logger.sort_itslef()

# ---------- Main ----------
def main():
    if CSV_LOG_PATH.exists():
        print(f"📜 Log esistente trovato: {CSV_LOG_PATH}. Rimuovo per una nuova esecuzione.")
        CSV_LOG_PATH.unlink()
    logger = CSVLogger(CSV_LOG_PATH)

    # ---------- Check or run benchmark ----------
    if not JSON_BENCHMARK_BEST_CONFIG_PATH.exists():
        print("⚠️ Nessuna configurazione ottimale trovata. Eseguo benchmark...")
        benchmark()

    # ---------- Load best config ----------
    with JSON_BENCHMARK_BEST_CONFIG_PATH.open("r", encoding="utf-8") as f:
        best_config = json.load(f)

    processes = int(best_config["processes"])
    threads = int(best_config["threads"])
    print(f"\n📌 Uso della configurazione ottimale: {processes} processi, {threads} thread")

    # ---------- Standard processing with retries ----------
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"\n🔁 Tentativo {attempt} di {MAX_ATTEMPTS}...\n")
            run_standard_processing(processes, threads, logger)
            print("✅ Elaborazione completata con successo.")
            break
        except KeyboardInterrupt:
            print("\n[🚪] Interrotto manualmente dall'utente. Uscita.")
            sys.exit(0)
        except Exception as e:
            logger.log_crash(f"Crash generale: {e}")
            print(f"\n❌ Crash: {e}")
            if attempt < MAX_ATTEMPTS:
                print(f"⏳ Nuovo tentativo in {RETRY_DELAY} secondi...")
                time.sleep(RETRY_DELAY)
            else:
                print("\n❌ Numero massimo di tentativi raggiunto. Uscita.")
                sys.exit(1)
        finally:
            tmp_dir = OUTPUT_TMP_DIR # or any directory you want to remove
            if tmp_dir.exists() and tmp_dir.is_dir():
                print(f"🗑️ Pulizia della directory temporanea: {tmp_dir}")
                shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.stop()


if __name__ == "__main__":
    main()
