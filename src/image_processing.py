# processing.py

import numpy as np
from PIL import Image
from pathlib import Path

from src.utils import *
from src.paths import *
from src.config import *
from model.SR_Script.super_resolution import SA_SuperResolution


def apply_super_resolution_single(image_path: Path, output_dir: Path, sr_model: SA_SuperResolution) -> Path:
    """
    Apply super-resolution model to a single image.

    Args:
        image_path (Path): Path to input image.
        output_dir (Path): Directory to save super-resolved image.
        sr_model (SA_SuperResolution): Preloaded super-resolution model instance.

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
        upscaled_image_np = sr_model.run(img_np)
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


def apply_personalized_downscaling_single(image_path: Path, output_dir: Path, ppi = int) -> Path:
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

    if ppi == 400:
        chromatic_ruler = CHROMATIC_BAND_400_PPI
        target_ruler_px = TARGET_RULER_PX_400_PPI
    elif ppi == 600:
        chromatic_ruler = CHROMATIC_BAND_600_PPI
        target_ruler_px = TARGET_RULER_PX_600_PPI

    original_ruler_inch = chromatic_ruler["width_mm"] / INCH_CONVERSION
    original_ruler_px = SUPER_RESOLUTION_PAR * chromatic_ruler["ppi"] * original_ruler_inch

    scale_factor = target_ruler_px / original_ruler_px
    scale_factor *= chromatic_ruler["correction_factor"]

    try:
        with Image.open(image_path) as image:
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)
            new_size = (new_width, new_height)
            resized_img = image.resize(new_size, resample=Image.LANCZOS)
    except Exception as e:
        raise RuntimeError(f"Failed to load or resize image {image_path}: {e}")

    output_path = output_dir / image_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        resized_img.save(output_path, dpi=(ppi, ppi))
    except Exception as e:
        raise RuntimeError(f"Failed to save resized image to {output_path}: {e}")
    
    return output_path