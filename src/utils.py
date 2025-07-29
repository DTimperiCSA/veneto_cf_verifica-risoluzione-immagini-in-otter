import os

import numpy as np

from PIL import Image
from pathlib import Path
from typing import Tuple

from src.paths import *
from src.config import *

def numpy_to_image(array: np.ndarray) -> Image.Image:
    """
    Convert a NumPy array (H x W x C) to a PIL Image.
    
    Args:
        array (np.ndarray): Image in NumPy format.

    Returns:
        Image.Image: PIL Image object.
    """
    return Image.fromarray(array)


def find_output_dir() -> tuple[Path, Path]:
    """
    Determine the appropriate super-resolution output directory based on the scale factor.
    
    Returns:
        tuple[Path, Path]: The corresponding super-resolution and downscaling directories.
    
    Raises:
        ValueError: If scale factor is unsupported.
    """
    if 2 <= SUPER_RESOLUTION_PAR <= 4:
        final_super_res_suffix = f"x{SUPER_RESOLUTION_PAR}"
    else:
        raise ValueError("SUPER_RESOLUTION_PAR must be 2, 3, or 4.")

    # Dynamically construct directories
    super_res_dir = os.path.join(OUTPUT_IMAGES_DIR, f"sr_{final_super_res_suffix}")
    downscaling_dir = os.path.join(OUTPUT_IMAGES_DIR, f"downscaled_{final_super_res_suffix}")

    # Create directories if they don’t exist
    os.makedirs(super_res_dir, exist_ok=True)
    os.makedirs(downscaling_dir, exist_ok=True)

    return super_res_dir, downscaling_dir

def is_valid_image_file(file_path: Path) -> Tuple[bool, str]:
    """
    Checks if the file is a valid image using PIL verification only.

    Args:
        file_path (Path): Path to the file.

    Returns:
        Tuple[bool, str]: (True, "") if valid image, (False, error message) otherwise.
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True, ""
    except Exception as e:
        return False, f"Invalid image: {e}"
