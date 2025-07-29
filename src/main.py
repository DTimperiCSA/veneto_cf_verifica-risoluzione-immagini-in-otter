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


def apply_super_resolution(image_paths: list[str], input_dir: Path, output_dir: Path):
    """
    Apply super-resolution model to a list of input images and save the results.
    
    Args:
        image_paths (list[str]): Filenames of images in the input directory.
        input_dir (Path): Directory containing original images.
        output_dir (Path): Directory to save super-resolved images.
    """
    print(f"Super-resolution output directory: {output_dir}")
    
    for img_filename in tqdm(image_paths, desc="Super-resolution"):
        img_path = input_dir / img_filename

        try:
            img_np = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
            continue

        upscaled_image_np = model.run(img_np)
        output_img = numpy_to_image(upscaled_image_np)

        output_path = output_dir / img_filename
        if output_path.exists():
            output_path.unlink()
        output_img.save(output_path)

def apply_personalized_downscaling(image_path: Path, output_dir: Path):
    """
    Resize super-resolved image to match target physical ruler dimensions
    for 400 PPI or 600 PPI output resolution.

    Args:
        image_path (Path): Path to the super-resolved image.
        output_dir (Path): Directory to save resized images.
    """
    print(f"Personalized downscaling output directory: {output_dir}")

    try:
        requested_PPI = int(image_path.name.split("_")[-1].split(".")[0])
    except ValueError:
        print(f"Skipped file with invalid PPI in name: {image_path.name}")
        return

    if requested_PPI == 400:
        chromatic_ruler = CROMATIC_SCALE_RULER_400_PPI
        target_ruler_px = TARGET_RULER_PX_400_PPI
    elif requested_PPI == 600:
        chromatic_ruler = CROMATIC_SCALE_RULER_600_PPI
        target_ruler_px = TARGET_RULER_PX_600_PPI
    else:
        print(f"Unsupported PPI: {requested_PPI}")
        return

    original_ruler_px = SUPER_RESOLUTION_PAR * chromatic_ruler["width_pixel"]
    scale_factor = target_ruler_px / original_ruler_px

    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Failed to load {image_path}: {e}")
        return

    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)

    resized_img = image.resize((new_width, new_height), resample=Image.LANCZOS)
    output_path = os.path.join(output_dir, image_path.name)
    os.makedirs(output_path.parent, exist_ok=True)
    resized_img.save(output_path, dpi=(requested_PPI, requested_PPI))


if __name__ == "__main__":
    # Load super-resolution model



    # Ensure output directories exist
    super_resolution_dir, downscaling_dir = find_output_dir()
    csv_logger = CSVLogger(CSV_LOG_PATH)

    for current_dir, subforlder, filenames in os.walk(INPUT_IMAGES_DIR):

        for filename in tqdm(filenames, desc=f"🔍 Scanning file in {current_dir}"):



                full_path = os.path.join(current_dir, filename)
                """
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