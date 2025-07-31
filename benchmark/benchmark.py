import os
import time
import json
import shutil
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from more_itertools import chunked
from tqdm import tqdm
import csv

from src.paths import *
from src.utils import *
from src.worker import ImageWorker
from src.config import *
from logs.logger import CSVLogger
from model.SR_Script.super_resolution import SA_SuperResolution


def process_batch(images, threads, super_resolution_dir, downscaling_dir, model_path, use_gpu):
    gpu_id = 0 if use_gpu else -1
    model = SA_SuperResolution(
        models_dir=model_path,
        model_scale=SUPER_RESOLUTION_PAR,
        tile_size=128,
        gpu_id=gpu_id,
        verbosity=False,
    )
    logger = CSVLogger(CSV_LOG_PATH)
    worker = ImageWorker(logger, super_resolution_dir, downscaling_dir, model)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(worker.run, img): img for img in images}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Elaborazione immagini"):
            try:
                future.result()
            except Exception:
                pass
    logger.stop()


def benchmark():
    print("🔍 Avvio benchmark per identificare la configurazione ottimale di device, processi e thread...")

    devices = ["CPU", "GPU"]
    cpu_exceeded = False
    completed = set()

    if CSV_BENCHMARK_LOG_PATH.exists():
        with CSV_BENCHMARK_LOG_PATH.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                device = row["device"]
                processes = int(row["processes"])
                threads = int(row["threads"])
                completed.add((device, processes, threads))

                if device == "CPU":
                    try:
                        avg_time = float(row["avg_time"])
                        if avg_time > 60:
                            cpu_exceeded = True
                    except ValueError:
                        pass  # ignore malformed entries
    else:
        with CSV_BENCHMARK_LOG_PATH.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "device", "processes", "threads", "total_time", "avg_time"])


    images = sorted(count_all_images(INPUT_IMAGES_DIR), key=lambda p: p.stat().st_size, reverse=True)[:5]
    if not images:
        print("⚠️ Nessuna immagine trovata per benchmark.")
        return

    best_config = None
    best_avg_time = float("inf")

    for device in devices:
        if device == "CPU" and cpu_exceeded:
            print("🚫 CPU già troppo lenta, skippo ulteriori test su CPU.\n")
            continue

        use_gpu = device == "GPU"

        for processes in [1, 2, 4, 8]:
            for threads in [1, 2, 4, 8]:
                if (device, processes, threads) in completed:
                    print(f"⏭️  Combinazione già testata: {device}, {processes} processi, {threads} thread\n")
                    continue

                if device == "CPU" and cpu_exceeded:
                    print("🚫 CPU già troppo lenta, skip test con  {processes} processi, {threads} thread su CPU.\n")
                    continue

                print(f"⚙️  Testando: {device}, {processes} processi, {threads} thread\n")

                super_resolution_dir = BENCHMARK_IMAGES_DIR / f"SR_{device}_p{processes}_t{threads}"
                downscaling_dir = BENCHMARK_IMAGES_DIR / f"DS_{device}_p{processes}_t{threads}"
                super_resolution_dir.mkdir(parents=True, exist_ok=True)
                downscaling_dir.mkdir(parents=True, exist_ok=True)

                chunks = list(chunked(images, max(1, len(images) // processes)))

                target = partial(
                    process_batch,
                    threads=threads,
                    super_resolution_dir=super_resolution_dir,
                    downscaling_dir=downscaling_dir,
                    model_path=SR_SCRIPT_MODEL_DIR,
                    use_gpu=use_gpu,
                )

                start_time = time.time()
                with Pool(processes) as pool:
                    for _ in tqdm(pool.imap(target, chunks), total=len(chunks), desc=f"Processi {device} ({processes})"):
                        pass
                total_time = time.time() - start_time
                avg_time = total_time / len(images)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with CSV_BENCHMARK_LOG_PATH.open("a", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, device, processes, threads, f"{total_time:.2f}", f"{avg_time:.4f}"])

                print(f"⏱️  Tempo totale: {total_time:.2f}s | Tempo medio per immagine: {avg_time:.4f}s")

                # Stop further CPU tests if too slow
                if device == "CPU" and avg_time > 60:
                    cpu_exceeded = True
                    print("⚠️  Tempo medio per immagine con CPU superiore a 60s. Interrompo test CPU.\n")
                    break

                if avg_time < best_avg_time:
                    best_avg_time = avg_time
                    best_config = {"device": device, "processes": processes, "threads": threads}

    if best_config:
        with JSON_BENCHMARK_BEST_CONFIG_PATH.open("w", encoding="utf-8") as f:
            json.dump(best_config, f, indent=4)
        print(f"🏁 Configurazione migliore: {best_config['device']} | "
              f"{best_config['processes']} processi, {best_config['threads']} thread")

    if BENCHMARK_IMAGES_DIR.exists():
        shutil.rmtree(BENCHMARK_IMAGES_DIR)
        print("🧹 Pulizia delle cartelle temporanee di benchmark completata.")


if __name__ == "__main__":
    benchmark()