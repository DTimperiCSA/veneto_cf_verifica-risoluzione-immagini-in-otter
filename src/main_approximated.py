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
from src.image_segmentation_pipeline import *
from src.image_processing import *
from src.worker import ImageWorker
from src.segmentation.unet import UNet
from logs.logger import CSVLogger
from model.SR_Script.super_resolution import SA_SuperResolution
from benchmark.benchmark import benchmark
from src.image_segmentation_pipeline_approssimated import *

MAX_ATTEMPTS = 10
RETRY_DELAY = 5  # seconds
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Process batch ----------
def process_batch(images, threads, super_resolution_dir, downscaling_dir, model_path, logger_path, progress_queue, analysis_result):
    from src.worker import ImageWorker
    from model.SR_Script.super_resolution import SA_SuperResolution

    logger = CSVLogger(logger_path)

    sr_model = SA_SuperResolution(
        models_dir=model_path,
        model_scale=SUPER_RESOLUTION_PAR,
        tile_size=128,
        gpu_id=0,
        verbosity=False,
    )

    worker = ImageWorker(logger, super_resolution_dir, downscaling_dir, sr_model, analysis_result)
    
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
    
    print("🔍 Caricamento modello UNet...")
    try:
        unet_model = UNet(n_channels=3, n_classes=1)
        checkpoint = torch.load(
            SAVE_PATH,
            map_location=DEVICE  # or "cuda" if you want GPU
        )

        # load weights properly
        if "model_state_dict" in checkpoint:
            unet_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            unet_model.load_state_dict(checkpoint)

        # move to GPU (or CPU)
        unet_model = unet_model.to(DEVICE)
    except Exception as e:
        raise RuntimeError(f"Errore nel caricamento UNet: {e}")

    manager = Manager()
    progress_queue = manager.Queue()

    total_success = 0
    total_error = 0

    PATHS_TO_SKIP_FOR_NOW = {
        Path(r"Z:\Digital Library\Conservatorio Benedetto Marcello\Conservatorio\B001.001"),
        Path(r"Z:\Digital Library\Conservatorio Benedetto Marcello\Conservatorio\B001.002"),
        Path(r"Z:\Digital Library\Conservatorio Benedetto Marcello\Conservatorio\B078.006"),
        Path(r"Z:\Digital Library\Conservatorio Benedetto Marcello\Conservatorio\B081.004"),
        Path(r"Z:\Digital Library\Conservatorio Benedetto Marcello\Conservatorio\B083.012"),
        Path(r"Z:\Digital Library\Conservatorio Benedetto Marcello\Conservatorio\B088.001"),
    }

    KEYPOINT_PATHS = {
        Path(r"Z:\Digital Library\Conservatorio Benedetto Marcello\Conservatorio\B002\B002.034"),
        Path(r"Z:\Digital Library\Conservatorio Benedetto Marcello\Conservatorio\B008\B008.003"),
        Path(r"Z:\Digital Library\Conservatorio Benedetto Marcello\Conservatorio\B012\B012.020"),
        Path(r"Z:\Digital Library\Conservatorio Benedetto Marcello\Conservatorio\B020\B020.001"),
        Path(r"Z:\Digital Library\Conservatorio Benedetto Marcello\Conservatorio\B020\B020.002"),
    }

    CORRECTED_DIR = ["B001.001", "B001.002"]
    RESIZED_DIR = Path(r"Z:\Digital Library\Conservatorio Benedetto Marcello\resize_lossless\B001")

    ANALYZE_DIR = CONSERVATORIO_DIR

    for folder in ANALYZE_DIR.rglob("*"):

        if not folder.is_dir():
            continue

        if Path(folder) in PATHS_TO_SKIP_FOR_NOW:
            print(f"Skipped {folder} for now (check code)")
            continue
            
        """
        if folder.name in CORRECTED_DIR:
            folder = RESIZED_DIR / folder.name
            ANALYZE_DIR = RESIZED_DIR
        """
            
        # --- raccolta immagini valide ---
        images_in_folder = [p for p in folder.glob("*")
                            if p.is_file() and p.name.lower() != "thumbs.db" and is_valid_image_file(p)[0]]
        
        print(f"\n📌 Analisi del percorso: {folder} con {len(images_in_folder)} immagini")

        if not images_in_folder:
            logger.log(folder.name, "no_images_to_process", success=False,
                    error="Nessuna immagine valida trovata", full_path=str(folder))
            continue

        # --- check immagini già processate ---
        relative_folder = folder.relative_to(ANALYZE_DIR)
        super_resolution_dir, downscaling_dir = find_output_dir_appr(relative_folder)
        print(f"Searching all processed images in {downscaling_dir}")

        all_exist = all((downscaling_dir / img.name).exists() for img in images_in_folder)
        if all_exist:
            logger.log(folder.name, "all_images_processed", success=False,
                    error="Tutte le immagini sono già state processate", full_path=str(folder))
            print("✅ Tutte le immagini sono già state processate")
            continue

        # --- ricerca banda cromatica ---
        try:
            chromatic_band_path = find_chromatic_band_in_folder(folder)
            if chromatic_band_path is None:
                logger.log(folder.name, "chromatic_band_search", success=False,
                        error="Nessuna banda cromatica trovata", full_path=str(folder))
                print("❌ Nessuna banda cromatica trovata")
                continue

            # --- analisi banda cromatica ---
            if Path(folder) in KEYPOINT_PATHS:
                res = analyze_chromatic_band_keypoint_approximated(chromatic_band_path, logger)
            else:
                res = analyze_chromatic_band_approximated(chromatic_band_path, unet_model, logger)

            if res is None:
                logger.log_failure(chromatic_band_path.name, "full_analysis",
                                "Analisi banda cromatica fallita", str(chromatic_band_path))
                print("❌ Analisi banda cromatica fallita")
                continue

            chromatic_band_path = Path(chromatic_band_path)
            save_path = TMP_SEGMENTATION_MINUS_PERCENT_DIR / "json" / f"{chromatic_band_path.parent.name}_{chromatic_band_path.stem}_analysis.json"
            save_results_to_json(res, save_path)
            print(f"✅ Risultati salvati in {save_path}")

            continue

            # --- stima PPI ---
            ppi = res.get('ppi', None)
            if not ppi:
                logger.log(folder.name, "estimate_ppi", success=False,
                        error="Impossibile stimare PPI", full_path=str(folder))
                print("❌ Impossibile stimare PPI")
                continue
            else:
                logger.log(folder.name, "estimate_ppi", success=True,
                        error=f"PPI stimati: {ppi}", full_path=str(folder))

        except Exception as e:
            logger.log_crash(f"Errore inatteso durante l'analisi di {folder}: {e}\n{traceback.format_exc()}",
                            full_path=str(folder))
            continue

        # --- multiprocessing chunking ---
        chunk_size = ceil(len(images_in_folder) / processes)
        chunks = list(chunked(images_in_folder, chunk_size))

        target = partial(
            process_batch,
            threads=threads,
            super_resolution_dir=super_resolution_dir,
            downscaling_dir=downscaling_dir,
            model_path=SR_SCRIPT_MODEL_DIR,
            logger_path=CSV_LOG_APPR_PATH,
            progress_queue=progress_queue,
            analysis_result=res
        )

        try:
            set_start_method("spawn", force=True)
            with Pool(processes) as pool:
                result = pool.map_async(target, chunks)
                completed = 0
                with tqdm(total=len(images_in_folder), desc="📷 Immagini elaborate", ncols=80) as pbar:
                    while completed < len(images_in_folder):
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


        # --- conteggio successi e fallimenti ---
        folder_error_count = 0
        if CSV_LOG_APPR_PATH.exists():
            with open(CSV_LOG_APPR_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                folder_error_count = sum(
                    1 for row in reader
                    if row["status"] == "false" and Path(row["full_path"]).parent.name == folder.name
                )

        folder_success = len(images_in_folder) - folder_error_count
        total_success += folder_success
        total_error += folder_error_count

        print(f"   ✅ Successi: {folder_success} | ❌ Errori: {folder_error_count}\n")
    # --- riepilogo finale ---
    logger.sort_itslef()

    

    print("\n📊 Risultato finale:")
    print(f"✅ Immagini processate con successo: {total_success}")
    print(f"❌ Immagini con errore:              {total_error}")

    logger.sort_itslef()

# ---------- Main ----------
def main():
    if CSV_LOG_APPR_PATH.exists():
        print(f"📜 Log esistente trovato: {CSV_LOG_APPR_PATH}. Rimuovo per una nuova esecuzione.")
        CSV_LOG_APPR_PATH.unlink()
    logger = CSVLogger(CSV_LOG_APPR_PATH)

    # ---------- Check or run benchmark ----------
    if not JSON_BENCHMARK_BEST_CONFIG_PATH.exists():
        print("⚠️ Nessuna configurazione ottimale trovata. Eseguo benchmark...")
        benchmark()

    # ---------- Load best config ----------
    with JSON_BENCHMARK_BEST_CONFIG_PATH.open("r", encoding="utf-8") as f:
        best_config = json.load(f)

    processes = int(best_config["processes"])
    threads = int(best_config["threads"])
    device_type = best_config["device"]
    print(f"\n📌 Uso della configurazione ottimale: {device_type} device, {processes} processi, {threads} thread")

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
                super_resolution_dir, downscaling_dir = find_output_dir_appr()
                print(f"🗑️ Pulizia della directory temporanea: {super_resolution_dir}")
                shutil.rmtree(super_resolution_dir, ignore_errors=True)

    logger.stop()


if __name__ == "__main__":
    main()