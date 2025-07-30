import os
import time
import json
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from more_itertools import chunked
from tqdm import tqdm
import csv

from mutliprocess_and_multithread.paths import *
from mutliprocess_and_multithread.utils import count_all_images, find_output_dir
from mutliprocess_and_multithread.config import *
from logs.logger import CSVLogger
from model.SR_Script.super_resolution import SA_SuperResolution

def get_custom_output_dirs(device: str, processes: int, threads: int):
    base_sr_dir, base_ds_dir = BENCHMARK_IMAGES_DIR
    suffix = f"{device}_p{processes}_t{threads}"

    custom_sr_dir = base_sr_dir / f"{base_sr_dir.name}_{suffix}"
    custom_ds_dir = base_ds_dir / f"{base_ds_dir.name}_{suffix}"

    custom_sr_dir.mkdir(parents=True, exist_ok=True)
    custom_ds_dir.mkdir(parents=True, exist_ok=True)

    return custom_sr_dir, custom_ds_dir

def create_sr_model(use_gpu: bool):
    try:
        gpu_id = 0 if use_gpu else -1
        model = SA_SuperResolution(
            models_dir=SR_SCRIPT_MODEL_DIR,
            model_scale=SUPER_RESOLUTION_PAR,
            tile_size=128,
            gpu_id=gpu_id,
            verbosity=True,
        )
        return model
    except Exception as e:
        print(f"[ERRORE] Caricamento modello {'GPU' if use_gpu else 'CPU'} fallito: {type(e).__name__}: {e}")
        with open("model_load_errors.log", "a") as f:
            f.write(f"Errore caricamento modello {'GPU' if use_gpu else 'CPU'}: {type(e).__name__}: {e}\n")
        raise

def worker_process_batch(image_paths: list[str], super_resolution_dir: Path, downscaling_dir: Path, use_gpu: bool, num_threads: int) -> list[str]:
    from mutliprocess_and_multithread.worker import ImageWorker
    from concurrent.futures import ThreadPoolExecutor, as_completed

    sr_model = create_sr_model(use_gpu)
    logger = CSVLogger(CSV_LOG_PATH.with_name(f"{CSV_LOG_PATH.stem}_{os.getpid()}.csv"))
    worker = ImageWorker(logger, super_resolution_dir, downscaling_dir, sr_model)

    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(worker.run, Path(p)): p for p in image_paths}
        for future in tqdm(as_completed(futures), total=len(image_paths), desc=f"Process {os.getpid()}"):
            img_name = Path(futures[future]).name
            try:
                future.result()
                results.append(f"Success: {img_name}")
            except Exception as e:
                results.append(f"Failed: {img_name} with error {e}")

    logger.stop()
    return results

def benchmark(devices=["gpu", "cpu"], process_list=[1, 2, 4, 8], thread_list=[1, 2, 3, 4, 8]):
    print("\n🧪 Avvio benchmark multiprocessing + multithreading...\n")

    images = sorted(count_all_images(INPUT_IMAGES_DIR), key=lambda p: p.stat().st_size, reverse=True)[:10]
    if not images:
        print("⚠️ Nessuna immagine trovata.")
        return

    all_results_path = CSV_BENCHMARK_LOG_PATH
    header = [
        "timestamp", "device", "processes", "threads",
        "total_time", "avg_time_per_image", "success", "errors", "images_count"
    ]

    existing_combinations = set()
    if all_results_path.exists():
        with all_results_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()[1:]
            for line in lines:
                parts = line.strip().split(",")
                if len(parts) >= 4:
                    device, processes, threads = parts[1], int(parts[2]), int(parts[3])
                    existing_combinations.add((device, processes, threads))
    else:
        with all_results_path.open("w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")

    best_result = None
    all_results = []

    for device in devices:
        use_gpu = device == "gpu"
        print(f"\n🚩 Benchmark su {device.upper()}")

        for processes in process_list:
            for threads in thread_list:
                key = (device, processes, threads)
                if key in existing_combinations:
                    print(f"⏭️  Già eseguito: {device} | {processes} processi | {threads} thread. Skipping.")
                    continue

                print(f"\n🔧 Test con {processes} processi e {threads} thread per processo...")

                super_resolution_dir, downscaling_dir = get_custom_output_dirs(device, processes, threads)

                for img in images:
                    for out_dir in [super_resolution_dir, downscaling_dir]:
                        out_file = out_dir / img.name
                        if out_file.exists():
                            out_file.unlink()

                start = time.time()
                chunk_size = (len(images) + processes - 1) // processes
                chunks = list(chunked([str(img) for img in images], chunk_size))

                with Pool(processes=processes) as pool:
                    func = partial(
                        worker_process_batch,
                        super_resolution_dir=super_resolution_dir,
                        downscaling_dir=downscaling_dir,
                        use_gpu=use_gpu,
                        num_threads=threads,
                    )
                    results_batches = list(
                        tqdm(pool.imap(func, chunks), total=len(chunks),
                             desc=f"{device.upper()} {processes}x{threads}")
                    )

                end = time.time()
                total_time = end - start

                flat_results = [r for batch in results_batches for r in batch]
                errors = sum("Failed" in r for r in flat_results)
                success = len(images) - errors
                avg_time = total_time / len(images)

                print(f"\n📊 Risultati {device.upper()} con {processes} processi e {threads} thread:")
                print(f"   🕒 Tempo totale: {total_time:.2f} s")
                print(f"   ✅ Successi: {success}")
                print(f"   ❌ Errori: {errors}")
                print(f"   ⏱️ Tempo medio per immagine: {avg_time:.2f} s\n")

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                current_config = {
                    "timestamp": timestamp,
                    "device": device,
                    "processes": processes,
                    "threads": threads,
                    "total_time": total_time,
                    "avg_time_per_image": avg_time,
                    "success": success,
                    "errors": errors,
                    "images_count": len(images),
                }
                all_results.append(current_config)

                with all_results_path.open("a", encoding="utf-8") as f:
                    f.write(",".join(str(current_config[col]) for col in header) + "\n")

                if best_result is None or avg_time < best_result["avg_time_per_image"]:
                    best_result = current_config

    print("📁 Riordinamento dei risultati...")
    with all_results_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    rows_sorted = sorted(rows, key=lambda r: (r["device"], int(r["threads"]), int(r["processes"])))
    with all_results_path.open("w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows_sorted)

    best_path = JSON_BENCHMARK_BEST_CONFIG_PATH
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best_result, f, indent=4)

    print(f"✅ Configurazione migliore salvata in {best_path.absolute()}")
    print(f"🏁 Migliore: {best_result}")
    print(f"🗂️  Tutti i risultati salvati in {all_results_path.absolute()}")

if __name__ == "__main__":
    benchmark()
