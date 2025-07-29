# processing.py

import numpy as np
from PIL import Image
from pathlib import Path

from src.utils import *
from src.paths import *
from src.config import *
from model.SR_Script.super_resolution import SA_SuperResolution

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


def apply_super_resolution_single(image_path: Path, output_dir: Path) -> Path:
    """
    Apply super-resolution model to a single image.

    Args:
        image_path (Path): Path to input image.
        output_dir (Path): Directory to save super-resolved image.

    Returns:
        Path: Output path of the super-resolved image.

    Raises:
        RuntimeError: If image loading or saving fails.
    """
    try:
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
            img_np = np.array(img_rgb)
    except Exception as e:
        raise RuntimeError(f"Failed to load or convert image {image_path}: {e}")

    try:
        upscaled_image_np = model.run(img_np)
        output_img = numpy_to_image(upscaled_image_np)
    except Exception as e:
        raise RuntimeError(f"Super-resolution model failed for {image_path}: {e}")

    output_path = output_dir / image_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if output_path.exists():
            output_path.unlink()
        output_img.save(output_path)
    except Exception as e:
        raise RuntimeError(f"Failed to save super-resolved image to {output_path}: {e}")

    return output_path


def apply_personalized_downscaling_single(image_path: Path, output_dir: Path) -> Path:
    """
    Resize a super-resolved image based on PPI info in filename.

    Args:
        image_path (Path): Path to the super-resolved image.
        output_dir (Path): Directory to save the resized image.

    Returns:
        Path: Output path of the resized image.

    Raises:
        ValueError: If PPI is invalid or unsupported.
        RuntimeError: If image loading or saving fails.
    """
    name = image_path.stem
    if "400" in name:
        requested_PPI = 400
    elif "600" in name:
        requested_PPI = 600
    else:
        raise ValueError(f"Unsupported or missing PPI value in filename: {image_path.name}")

    if requested_PPI == 400:
        chromatic_ruler = CROMATIC_SCALE_RULER_400_PPI
        target_ruler_px = TARGET_RULER_PX_400_PPI
    elif requested_PPI == 600:
        chromatic_ruler = CROMATIC_SCALE_RULER_600_PPI
        target_ruler_px = TARGET_RULER_PX_600_PPI

    original_ruler_px = SUPER_RESOLUTION_PAR * chromatic_ruler["width_pixel"]
    scale_factor = target_ruler_px / original_ruler_px

    try:
        with Image.open(image_path) as image:
            new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
            resized_img = image.resize(new_size, resample=Image.LANCZOS)
    except Exception as e:
        raise RuntimeError(f"Failed to load or resize image {image_path}: {e}")

    output_path = output_dir / image_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        resized_img.save(output_path, dpi=(requested_PPI, requested_PPI))
    except Exception as e:
        raise RuntimeError(f"Failed to save resized image to {output_path}: {e}")

    return output_path
