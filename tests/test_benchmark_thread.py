import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.paths import *
from src.worker import ImageWorker
from src.utils import *
from logs.logger import CSVLogger
from src.utils import count_all_images
from model.SR_Script.super_resolution import SA_SuperResolution

from tqdm import tqdm

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
        # Qui stampi su console e salvi su file di log generale
        print(f"[ERRORE] Caricamento modello {'GPU' if use_gpu else 'CPU'} fallito: {type(e).__name__}: {e}")
        with open("model_load_errors.log", "a") as f:
            f.write(f"Errore caricamento modello {'GPU' if use_gpu else 'CPU'}: {type(e).__name__}: {e}\n")
        raise


def benchmark(max_workers_list=[1, 2, 4, 8], devices=["gpu", "cpu"]):
    print("\n🧪 Avvio benchmark thread...\n")

    images = count_all_images(INPUT_IMAGES_DIR)
    if not images:
        print("⚠️ Nessuna immagine trovata.")
        return

    images = images[:100]  # Usa un subset per benchmark veloce

    for device in devices:
        use_gpu = device == "gpu"
        print(f"\n🚩 Benchmark su {device.upper()}")

        sr_model = create_sr_model(use_gpu=use_gpu)

        for workers in max_workers_list:
            print(f"\n🔧 Test con {workers} thread...")

            # Reset logger e output per test pulito
            super_resolution_dir, downscaling_dir = find_output_dir()
            logger = CSVLogger(CSV_LOG_PATH)
            worker = ImageWorker(logger, super_resolution_dir, downscaling_dir, sr_model)

            # Pulisci eventuali output precedenti
            for img in images:
                for out_dir in [super_resolution_dir, downscaling_dir]:
                    out_file = out_dir / img.name
                    if out_file.exists():
                        out_file.unlink()

            start = time.time()

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(worker.run, img): img for img in images}
                for _ in tqdm(as_completed(futures), total=len(futures), desc=f"{device.upper()} Threads={workers}"):
                    pass

            end = time.time()
            total_time = end - start
            errors = len(logger.rows)
            success = len(images) - errors

            print(f"\n📊 Risultati su {device.upper()} con {workers} thread:")
            print(f"   🕒 Tempo totale: {total_time:.2f} secondi")
            print(f"   ✅ Successi: {success}")
            print(f"   ❌ Errori: {errors}")
            print(f"   ⏱️ Tempo medio per immagine: {total_time / len(images):.2f} s\n")


if __name__ == "__main__":
    benchmark(max_workers_list=[1, 2, 3, 4], devices=["gpu", "cpu"])