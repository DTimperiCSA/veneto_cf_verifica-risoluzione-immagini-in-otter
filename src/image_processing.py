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


def apply_personalized_downscaling_single(image_path: Path, output_dir: Path, analysis_res) -> Path:
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
    ppi = analysis_res.get('ppi', None)
    img_mm = analysis_res.get('img_px', None)
    img_long_side_mm, img_short_side_mm = max(img_mm), min(img_mm)

    pixel_per_mm = ppi / INCH_CONVERSION

    img_long_side_target_px = img_long_side_mm * pixel_per_mm
    img_short_side_target_px = img_short_side_mm * pixel_per_mm

    try:
        with Image.open(image_path) as image:
            width, height = image.size

            if width >= height:  # Lato lungo = width
                new_width = int(img_long_side_target_px)
                new_height = int(img_short_side_target_px)
            else:  # Lato lungo = height
                new_width = int(img_short_side_target_px)
                new_height = int(img_long_side_target_px)
                
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