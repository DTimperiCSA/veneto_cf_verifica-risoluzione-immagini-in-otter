import torch
import os
import numpy as np
import time
import sys

from tqdm import tqdm

from src.utils import *
from src.paths import *
from src.config import *
from src.worker import ImageWorker
from logs.logger import CSVLogger

def main():
    super_resolution_dir, downscaling_dir = find_output_dir()
    logger = CSVLogger(CSV_LOG_PATH)
    worker = ImageWorker(
        logger = logger,
        output_sr_dir = super_resolution_dir,
        output_final_dir = downscaling_dir
    )

    for current_dir, _, filenames in os.walk(INPUT_IMAGES_DIR):

        for filename in tqdm(filenames, desc=f"🔍 Scanning file in {current_dir}"):
            full_path = os.path.join(current_dir, filename)
            try:
                worker.run(full_path)
            except Exception as e:
                logger.log_crash(f"Fatal error in main loop with image {filename}: {e}")

            logger.stop()

if __name__ == "__main__":
    try:
        while True:
            try:
                print("🔄 Starting processing...")
                main()
                print("✅ Processing completed successfully.")
                break  # Esce se completa senza errori
            except Exception as e:
                print(f"[!] Error occurred: {e}. Retrying in 5 seconds...")
                time.sleep(5)
    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted by user. Shutting down gracefully.")
        sys.exit(0)