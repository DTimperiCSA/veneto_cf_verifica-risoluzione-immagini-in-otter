# processing.py

import os
import numpy as np

from pathlib import Path
from PIL import Image

from src.utils import *
from src.paths import *
from src.config import *
from model.SR_Script.super_resolution import SA_SuperResolution

model = SA_SuperResolution(
    models_dir=SR_SCRIPT_MODEL_DIR,
    model_scale=SUPER_RESOLUTION_PAR,
    tile_size=128,
    gpu_id=0,
    verbosity=True,
)

def apply_super_resolution_single(image_path: Path, output_dir: Path) -> Path:
    """
    Apply super-resolution model to a single image.

    Args:
        image_path (Path): Path to input image.
        output_dir (Path): Directory to save super-resolved image.

    Returns:
        Path: Output path of the super-resolved image.
    """
    try:
        img_np = np.array(Image.open(image_path).convert("RGB"))
    except Exception as e:
        raise RuntimeError(f"Failed to load {image_path}: {e}")

    upscaled_image_np = model.run(img_np)
    output_img = numpy_to_image(upscaled_image_np)

    output_path = output_dir / image_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    output_img.save(output_path)

    return output_path

def apply_personalized_downscaling_single(image_path: Path, output_dir: Path) -> Path:
    """
    Resize a super-resolved image based on PPI info in filename.

    Args:
        image_path (Path): Path to the super-resolved image.
        output_dir (Path): Directory to save the resized image.

    Returns:
        Path: Output path of the resized image.
    """
    try:
        requested_PPI = int(image_path.name.split("_")[-1].split(".")[0])
    except ValueError:
        raise ValueError(f"Invalid PPI in filename: {image_path.name}")

    if requested_PPI == 400:
        chromatic_ruler = CROMATIC_SCALE_RULER_400_PPI
        target_ruler_px = TARGET_RULER_PX_400_PPI
    elif requested_PPI == 600:
        chromatic_ruler = CROMATIC_SCALE_RULER_600_PPI
        target_ruler_px = TARGET_RULER_PX_600_PPI
    else:
        raise ValueError(f"Unsupported PPI value in filename: {requested_PPI}")

    original_ruler_px = SUPER_RESOLUTION_PAR * chromatic_ruler["width_pixel"]
    scale_factor = target_ruler_px / original_ruler_px

    try:
        image = Image.open(image_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load {image_path}: {e}")

    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    resized_img = image.resize(new_size, resample=Image.LANCZOS)

    output_path = output_dir / image_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resized_img.save(output_path, dpi=(requested_PPI, requested_PPI))

    return output_path