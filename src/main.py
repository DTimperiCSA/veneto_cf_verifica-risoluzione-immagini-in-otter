import torch
import os
import numpy as np

from PIL import Image
from pathlib import Path
from tqdm import tqdm

from src.utils import *
from src.paths import *
from src.config import *
from src.image_processing import *
from logs.logger import CSVLogger

if __name__ == "__main__":

    # Ensure output directories exist
    super_resolution_dir, downscaling_dir = find_output_dir()
    csv_logger = CSVLogger(CSV_LOG_PATH)

    for current_dir, subforlder, filenames in os.walk(INPUT_IMAGES_DIR):

        for filename in tqdm(filenames, desc=f"🔍 Scanning file in {current_dir}"):
            full_path = os.path.join(current_dir, filename)
            sr_out = apply_super_resolution_single(full_path, super_resolution_dir)
            ds_out = apply_personalized_downscaling_single(sr_out, downscaling_dir)
            """
            if true:
                print(f"Processing file: {full_path}")
                print("Checking if file is a valid image...")
                print("applying super-resolution...")
                print("checking if image is valid...")
                print("applying personalized downscaling...")
                print("check if image is valid...")
             """
            print(filename)
            success, error = is_valid_image_file(Path(full_path))
            if not success:
                #csv_error_writer.append((full_path, "time", "image problems", "super res problems", "downscaling problems", error))
                print("Invalid image file:", full_path, error)
                continue

        print(f"Process recap")
        print(f"directory exolored: {current_dir}")
        print(f"{len(filenames)} images processed.")
        print("number of images with problems:", 2)
        print("number of images without problems:", 0)

        #print(csv_error_writer, len(csv_error_writer))


"""

def main():


MAX_RETRIES = 10
RETRY_DELAY = 10  # seconds

def run_main():
    from src.main import main  # assumes your main script has a main() function
    main()

if __name__ == "__main__":
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"🚀 Attempt {attempt} to run the image processor.")
            run_main()
            print("✅ Processing completed successfully.")
            break  # Done
        except Exception as e:
            print(f"❌ Crash detected (Attempt {attempt}): {e}")
            traceback.print_exc()

"""