import cv2
import shutil
import numpy as np
import time
from pathlib import Path

from src.utils import *
from src.config import *
from src.paths import *
from logs.logger import CSVLogger


def safe_imread(path: Path, retries=3, delay=0.5):
    """
    Tenta a leggere un'immagine più volte con delay, gestendo eventuali lock temporanei.
    """
    for attempt in range(retries):
        img = cv2.imread(str(path))
        if img is not None:
            return img
        time.sleep(delay)
    raise IOError(f"Impossibile leggere immagine {path} dopo {retries} tentativi")

def safe_copy(src: Path, dst: Path, retries=3, delay=0.5):
    """
    Tenta a copiare un file più volte con delay per evitare lock file temporanei.
    """
    for attempt in range(retries):
        try:
            shutil.copy2(src, dst)
            return
        except (PermissionError, OSError) as e:
            if hasattr(e, 'winerror') and e.winerror in (32, 1224):
                time.sleep(delay)
            else:
                raise
    raise IOError(f"Impossibile copiare il file {src} in {dst} dopo {retries} tentativi")


def measure_chromatic_band_dimension(path_input: Path):
    # Carica immagine con safe_imread
    img = safe_imread(path_input)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Maschera per grigio scuro (esclude il documento beige)
    lower_gray = np.array([0, 0, 40])
    upper_gray = np.array([180, 50, 100])
    mask = cv2.inRange(hsv, lower_gray, upper_gray)

    # Trova contorni
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

        # Disegna rettangolo in rosso
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

        # Misure lato lungo e corto
        w, h = best_rect[1]
        long_side = max(w, h)
        short_side = min(w, h)
        return (long_side, short_side)
    else:
        print(f"⚠️ Nessun righello Tiffen identificato in {path_input}.")
        return None


def find_chromatic_band_in_folder(folder_path: Path) -> Path:
    """
    Trova l'ultima immagine valida nella cartella e la copia in OUTPUT_TMP_DIR.
    Restituisce il percorso della copia.
    """
    images = [f for f in folder_path.iterdir() if f.is_file() and is_valid_image_file(f)]
    if not images:
        raise FileNotFoundError(f"Nessuna immagine valida trovata in {folder_path}")

    images.sort()
    last_image = images[-1]

    tmp_dir = OUTPUT_TMP_DIR
    tmp_dir.mkdir(parents=True, exist_ok=True)

    dest_path = tmp_dir / f"chromatic_band_{last_image.name}"
    safe_copy(last_image, dest_path)

    return dest_path


def binaryize_image(image_path: Path, threshold: int = 50) -> Path | None:
    """
    Binarizza una singola immagine e la salva in OUTPUT_TMP_DIR.
    Ritorna il path della nuova immagine salvata o None in caso di errore.
    """
    if not is_valid_image_file(image_path):
        return None

    img = safe_imread(image_path)
    if img is None:
        return None

    # Se l'immagine è a colori, converti in grigio
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    tmp_dir = OUTPUT_TMP_DIR
    tmp_dir.mkdir(parents=True, exist_ok=True)

    output_path = tmp_dir / image_path.name
    cv2.imwrite(str(output_path), binary)
    return output_path


def measure_document_from_binary(binary_image_path: Path) -> tuple[float, float] | None:
    """
    Trova il contorno più grande in un'immagine binaria e misura
    lato lungo e lato corto del rettangolo ruotato che lo approssima.
    Ritorna (long_side_px, short_side_px) o None se fallisce.
    """
    img = safe_imread(binary_image_path)
    if img is None:
        print(f"⚠️ Impossibile leggere {binary_image_path}")
        return None

    # Se è a colori, converti in grigio (per sicurezza)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"⚠️ Nessun contorno trovato in {binary_image_path}")
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    (cx, cy), (w, h), angle = rect

    long_side_px = max(w, h)
    short_side_px = min(w, h)

    # Opzionale: salva immagine con rettangolo disegnato
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    output_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_img, [box], 0, (0, 255, 0), 2)

    out_path = OUTPUT_TMP_DIR / binary_image_path.name
    cv2.imwrite(str(out_path), output_img)
    print(f"[💾] Rettangolo salvato: {out_path}")

    return (long_side_px, short_side_px)


def calculate_mm_from_px(img_band_px: float, scale_factor: float) -> float:
    return scale_factor * img_band_px


def estimate_ppi_from_dimensions(image_rect_dim_px, chromatic_band_dim_px) -> int:
    img_long_side_px, img_short_side_px = max(image_rect_dim_px), min(image_rect_dim_px)
    chromatic_band_long_side_px, chromatic_band_short_side_px = max(chromatic_band_dim_px), min(chromatic_band_dim_px)

    # Calcola il fattore di scala (mm/pixel)
    scale_factor = CHROMATIC_BAND_MM / chromatic_band_long_side_px

    img_long_side_mm = calculate_mm_from_px(img_long_side_px, scale_factor)
    img_short_side_mm = calculate_mm_from_px(img_short_side_px, scale_factor)

    # Se il documento è minore o uguale ad A4, PPI=400, altrimenti 600
    width_ok = min(img_long_side_mm, img_short_side_mm) <= A4_WIDTH_MM
    height_ok = max(img_long_side_mm, img_short_side_mm) <= A4_HEIGHT_MM

    if width_ok and height_ok:
        return 400
    else:
        return 600
