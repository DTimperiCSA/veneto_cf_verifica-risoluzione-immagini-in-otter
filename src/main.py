import torch
import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from utils import *
from model.SR_Script.super_resolution import SA_SuperResolution


def numpy_to_image(array: np.ndarray) -> Image.Image:
    """
    Convert a NumPy array (H x W x C) to a PIL Image.
    
    Args:
        array (np.ndarray): Image in NumPy format.

    Returns:
        Image.Image: PIL Image object.
    """
    return Image.fromarray(array)


def check_super_resolution_output_dir() -> tuple[Path, Path]:
    """
    Determine the appropriate super-resolution output directory based on the scale factor.
    
    Returns:
        tuple[Path, Path]: The corresponding super-resolution and downscaling directories.
    
    Raises:
        ValueError: If scale factor is unsupported.
    """
    if SUPER_RESOLUTION_PAR == 2:
        super_res_dir = IMAGES_SR_X2_DIR
        downscaling_dir = IMAGES_DOWNSIZED_X2_DIR
    elif SUPER_RESOLUTION_PAR == 3:
        super_res_dir = IMAGES_SR_X3_DIR
        downscaling_dir = IMAGES_DOWNSIZED_X3_DIR
    elif SUPER_RESOLUTION_PAR == 4:
        super_res_dir = IMAGES_SR_X4_DIR
        downscaling_dir = IMAGES_DOWNSIZED_X4_DIR
    else:
        raise ValueError("Unsupported super resolution scale. Supported values are 2, 3, or 4.")
    
    os.makedirs(super_res_dir, exist_ok=True)
    os.makedirs(downscaling_dir, exist_ok=True)

    return super_res_dir, downscaling_dir


def create_super_resolution_images(image_paths: list[str], input_dir: Path, output_dir: Path):
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


def create_personalized_downscaling(image_paths: list[str], super_resolution_dir: Path, downscaling_dir: Path):
    """
    Resize super-resolved images to match target physical ruler dimensions
    for 400 PPI or 600 PPI output resolution.

    Args:
        image_paths (list[str]): Filenames of super-resolved images.
        super_resolution_dir (Path): Where to find SR images.
        downscaling_dir (Path): Where to save resized images.
    """
    print(f"Personalized downscaling output directory: {downscaling_dir}")
    
    for img_filename in tqdm(image_paths, desc="Downscaling"):
        img_path = super_resolution_dir / img_filename

        try:
            # CHANGE THIS! I HAD ONLY TWO IMAGES AND I BASED MYSELF ON THEIR NAMES
            requested_PPI = int(img_filename.split("_")[-1].split(".")[0])
        except ValueError:
            print(f"Skipped file with invalid PPI in name: {img_filename}")
            continue

        if requested_PPI == 400:
            chromatic_ruler = CROMATIC_SCALE_RULER_400_PPI
            target_ruler_px = TARGET_RULER_PX_400_PPI
        elif requested_PPI == 600:
            chromatic_ruler = CROMATIC_SCALE_RULER_600_PPI
            target_ruler_px = TARGET_RULER_PX_600_PPI
        else:
            print(f"Unsupported PPI: {requested_PPI}")
            continue

        original_ruler_px = SUPER_RESOLUTION_PAR * chromatic_ruler["width_pixel"]
        scale_factor = target_ruler_px / original_ruler_px

        try:
            image = Image.open(img_path)
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
            continue

        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)

        resized_img = image.resize((new_width, new_height), resample=Image.LANCZOS)
        output_path = downscaling_dir / img_filename
        os.makedirs(output_path.parent, exist_ok=True)
        resized_img.save(output_path, dpi=(requested_PPI, requested_PPI))


if __name__ == "__main__":
    # Load super-resolution model
    model = SA_SuperResolution(
        models_dir=SR_SCRIPT_MODEL_DIR,
        model_scale=SUPER_RESOLUTION_PAR,
        tile_size=128,
        gpu_id=0,
        verbosity=True,
    )

    # Define input directory (can be set from Colab memory or copied drive)
    input_dir = IMAGES_INPUT_DIR

    # Ensure output directories exist
    super_resolution_dir, downscaling_dir = check_super_resolution_output_dir()

    # Apply super-resolution to input images
    image_paths = os.listdir(input_dir)
    create_super_resolution_images(image_paths, input_dir, super_resolution_dir)

    # Apply personalized downscaling to super-resolved images
    sr_images = os.listdir(super_resolution_dir)
    create_personalized_downscaling(sr_images, super_resolution_dir, downscaling_dir)