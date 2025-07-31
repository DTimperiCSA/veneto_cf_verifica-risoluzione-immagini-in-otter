import os
import traceback
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
    super_res_dir = Path(OUTPUT_IMAGES_DIR) / f"sr_{final_super_res_suffix}"
    downscaling_dir = Path(OUTPUT_IMAGES_DIR) / f"downscaled_{final_super_res_suffix}"


    # Create directories if they don’t exist
    os.makedirs(super_res_dir, exist_ok=True)
    os.makedirs(downscaling_dir, exist_ok=True)

    return super_res_dir, downscaling_dir


def count_all_images(directory: Path) -> list[Path]:
    all_images = []
    for current_dir, _, filenames in os.walk(directory):
        current_dir = Path(current_dir)
        for filename in filenames:
            full_path = current_dir / filename
            all_images.append(full_path)
    return all_images

def is_valid_image_file(file_path: Path) -> Tuple[bool, str]:
    """
    Verifica approfondita se il file è un'immagine valida:
    - Controlla esistenza, tipo file, dimensione.
    - Esegue Image.open, verify e load.
    - Restituisce messaggi dettagliati SOLO in caso di errore.

    Args:
        file_path (Path): Percorso del file immagine.

    Returns:
        Tuple[bool, str]: (True, "") se valida, (False, errore descrittivo) se fallisce.
    """
    try:
        if not file_path.exists():
            return False, "File non trovato (path inesistente)"
        
        if not file_path.is_file():
            return False, "Il path non è un file"

        try:
            file_size = file_path.stat().st_size
            if file_size < 10_000:
                return False, f"File troppo piccolo ({file_size} byte), probabilmente corrotto"
        except Exception as e:
            return False, f"Errore durante lettura dimensione file: {e}"

        # Step 1: Apertura iniziale
        try:
            img = Image.open(file_path)
        except Exception as e:
            return False, f"[OPEN FAIL] Errore in Image.open(): {e}"

        # Step 2: Verifica struttura (senza caricare pixel)
        try:
            img.verify()
        except Exception as e:
            return False, f"[VERIFY FAIL] Errore in img.verify(): {e}"

        # Step 3: Riapertura dopo verify per forzare il load
        try:
            img = Image.open(file_path)
        except Exception as e:
            return False, f"[REOPEN FAIL] Errore riaprendo dopo verify: {e}"

        # Step 4: Caricamento reale dei dati
        try:
            img.load()
        except Exception as e:
            return False, f"[LOAD FAIL] Errore in img.load(): {e}"

        return True, ""

    except FileNotFoundError:
        return False, "File non trovato (FileNotFoundError)"
    except PermissionError:
        return False, "Permessi negati per accedere al file (PermissionError)"
    except OSError as e:
        return False, f"Errore di sistema operativo (OSError): {e}"
    except Exception as e:
        tb = traceback.format_exc(limit=1)
        return False, f"[UNEXPECTED] Errore sconosciuto: {e} | Traceback: {tb}"
