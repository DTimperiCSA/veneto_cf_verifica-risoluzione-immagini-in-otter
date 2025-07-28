import numpy as np
import os

from PIL import Image

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