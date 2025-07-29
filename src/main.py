import sys
import time
import torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils import *
from src.paths import *
from src.config import *
from src.worker import ImageWorker
from logs.logger import CSVLogger
from model.SR_Script.super_resolution import SA_SuperResolution

MAX_ATTEMPTS = 10
RETRY_DELAY = 5  # seconds
MAX_WORKERS = 4  # Numero di thread concorrenti (adatta alla tua macchina)

def main():

    print("caricamento del modello...")

    try:
        model = SA_SuperResolution(
            models_dir=SR_SCRIPT_MODEL_DIR,
            model_scale=SUPER_RESOLUTION_PAR,
            tile_size=128,
            gpu_id=0,
            verbosity=True,
        )
    except Exception as e:
        raise RuntimeError(f"Errore durante il caricamento del modello di super-risoluzione: {e}")
    print("🚀 Avvio del processo di elaborazione immagini...\n")

    super_resolution_dir, downscaling_dir = find_output_dir()
    logger = CSVLogger(CSV_LOG_PATH)
    worker = ImageWorker(logger, super_resolution_dir, downscaling_dir)

    input_images = count_all_images(INPUT_IMAGES_DIR)
    if not input_images:
        print("⚠️ Nessuna immagine trovata nella cartella di input. Uscita.")
        return

    # Filtra le immagini non ancora elaborate
    images_to_process = [
        img for img in input_images if not (downscaling_dir / img.name).exists()
    ]

    print(f"📦 Totale immagini trovate:        {len(input_images)}")
    print(f"✅ Immagini già elaborate:        {len(input_images) - len(images_to_process)}")
    print(f"🕐 Immagini da elaborare ora:     {len(images_to_process)}\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(worker.run, img): img for img in images_to_process}

        for future in tqdm(as_completed(futures), total=len(futures), desc="🔍 Elaborazione immagini"):
            img = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.log(img.name, "run", success=False, error=f"Thread error: {e}")

    logger.stop()
    logger.print_status(input_images)

    success_count = len(input_images) - len(logger.rows)
    error_count = len(logger.rows)
    print("\n📊 Risultato finale:")
    print(f"✅ Immagini processate con successo: {success_count}")
    print(f"❌ Immagini con errore:              {error_count}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("✅ GPU disponibile e usata:", torch.cuda.get_device_name(0))
    else:
        print("⚠️ GPU non disponibile, si usa CPU")

    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                print(f"\n🔁 Tentativo {attempt} di {MAX_ATTEMPTS}...\n")
                main()
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
