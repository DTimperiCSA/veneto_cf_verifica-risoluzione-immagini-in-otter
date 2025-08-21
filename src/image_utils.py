import os
import traceback
import numpy as np
import csv
import cv2
import shutil
import time

from PIL import Image
from pathlib import Path
from typing import Tuple

from src.paths import *
from src.config import *
from src.image_utils import *

def safe_imread(path: Path, retries=3, delay=0.5):
    for attempt in range(retries):
        img = cv2.imread(str(path))
        if img is not None:
            return img
        time.sleep(delay)
    raise IOError(f"Impossibile leggere immagine {path} dopo {retries} tentativi")


def safe_copy(src: Path, dst: Path, retries=3, delay=0.5):
    for attempt in range(retries):
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return
        except (PermissionError, OSError) as e:
            if hasattr(e, 'winerror') and e.winerror in (32, 1224):
                time.sleep(delay)
            else:
                raise
    raise IOError(f"Impossibile copiare il file {src} in {dst} dopo {retries} tentativi")

def numpy_to_image(array: np.ndarray) -> Image.Image:
    """
    Convert a NumPy array (H x W x C) to a PIL Image.
    
    Args:
        array (np.ndarray): Image in NumPy format.

    Returns:
        Image.Image: PIL Image object.
    """
    return Image.fromarray(array)


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


def find_chromatic_band_in_folder(
    folder: Path
) -> str | None:
    folder = Path(folder)
    template_path = TEMPLATE_IMG_PATH

    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder {folder} does not exist or is not a directory.")
    if not template_path.exists():
        raise ValueError(f"Template {template_path} does not exist.")

    # Load template (grayscale)
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise ValueError(f"Could not load template image: {template_path}")
    t_h, t_w = template.shape[:2]

    best_match = None
    best_val = -1
    total_images = 0
    skipped_small = 0

    for img_path in folder.iterdir():
        if img_path.is_file():
            if img_path.name.lower() == "thumbs.db":
                continue
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]:
                continue

            total_images += 1
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            i_h, i_w = img.shape[:2]
            if i_h < t_h or i_w < t_w:
                continue

            # Template matching
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            if max_val > best_val:
                best_val = max_val
                best_match = img_path

    if best_match is not None and best_val > 0.5:
        return str(best_match)
    else:
        return None

def measure_chromatic_band_dimension(path_input: Path):
    img = safe_imread(path_input)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_gray = np.array([0, 0, 40])
    upper_gray = np.array([180, 50, 100])
    mask = cv2.inRange(hsv, lower_gray, upper_gray)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    best_area = 0
    best_rect = None

    for c in contours:
        if cv2.arcLength(c, True) < 200:
            continue

        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), angle = rect
        area = w * h

        if area < 1000:
            continue

        aspect_ratio = max(w, h) / min(w, h)

        if aspect_ratio < 3:
            continue

        if area > best_area:
            best_area = area
            best_contour = c
            best_rect = rect

    if best_contour is not None:
        box = cv2.boxPoints(best_rect)
        box = box.astype(int)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

        w, h = best_rect[1]
        long_side = max(w, h)
        short_side = min(w, h)
        return (long_side, short_side)
    else:
        print(f"⚠️ Nessun righello Tiffen identificato in {path_input}.")
        return None
    
def binaryize_image(image_path: Path, threshold: int = 50) -> Path | None:
    valid, msg = is_valid_image_file(image_path)
    if not valid:
        print(f"⚠️ File non valido: {image_path} | {msg}")
        return None

    img = safe_imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    subfolder = image_path.parent.name
    dest_folder = OUTPUT_TMP_DIR / subfolder
    dest_folder.mkdir(parents=True, exist_ok=True)
    dest_path = dest_folder / image_path.name

    cv2.imwrite(str(dest_path), binary)
    return dest_path

def estimate_ppi_from_chromatic_band(chromatic_band_path: Path) -> int | None:
    chromatic_band_path = Path(chromatic_band_path)

    chromatic_band_dim_px = measure_chromatic_band_dimension(chromatic_band_path)
    if not chromatic_band_dim_px:
        return None
    
    bin_img = binaryize_image(chromatic_band_path)
    bin_img_dim_px = measure_document_from_binary(bin_img)

    img_long_side_px, img_short_side_px = max(bin_img_dim_px), min(bin_img_dim_px)
    chromatic_band_long_side_px, chromatic_band_short_side_px = max(chromatic_band_dim_px), min(chromatic_band_dim_px)

    scale_factor = CHROMATIC_BAND_MM / chromatic_band_long_side_px

    img_long_side_mm = img_long_side_px * scale_factor
    img_short_side_mm = img_short_side_px * scale_factor

    print("Measueres")
    print(f"chromatic px: {chromatic_band_long_side_px} ")
    print(f"img px: {img_long_side_px} ")
    print(f"chromatic mm: {CHROMATIC_BAND_MM} ")
    print(f"img mm: {img_long_side_mm} ")
    print(f"scale factor: {scale_factor}")

    width_ok = min(img_long_side_mm, img_short_side_mm) <= A4_WIDTH_MM
    height_ok = max(img_long_side_mm, img_short_side_mm) <= A4_HEIGHT_MM

    print(f"Dimensioni stimate: ")
    print(f"{img_long_side_mm:.2f} minore di {A4_HEIGHT_MM}? {'✅' if height_ok else '❌'}")
    print(f"{img_short_side_mm:.2f} minore di {A4_WIDTH_MM}? {'✅' if width_ok else '❌'}")
    print(f"Il file è un A4? {'✅' if (width_ok and height_ok) else '❌'}")

    ppi = 600

    if width_ok and height_ok:
        ppi = 400
    
    print(f"PPI stimati: {ppi}")
    return ppi
    
def measure_document_from_binary(binary_image_path: Path) -> tuple[float, float] | None:
    img = safe_imread(binary_image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    (cx, cy), (w, h), angle = rect

    long_side_px = max(w, h)
    short_side_px = min(w, h)

    return (long_side_px, short_side_px)


    







