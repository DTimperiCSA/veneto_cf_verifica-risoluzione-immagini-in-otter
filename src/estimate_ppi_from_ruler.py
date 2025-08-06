import cv2
import shutil
import numpy as np
import time
from pathlib import Path

from src.utils import *
from src.config import *
from src.paths import *
from logs.logger import CSVLogger


def estimate_ppi_for_folder(folder_path: Path) -> int | None:
    try:
        chromatic_band_img = find_chromatic_band_in_folder(folder_path)
        if not chromatic_band_img:
            return None

        chromatic_band_dim_px = measure_chromatic_band_dimension(chromatic_band_img)
        if not chromatic_band_dim_px:
            return None

        all_images = sorted([img for img in folder_path.iterdir() if is_valid_image_file(img)])
        document_images = all_images[:-2] if len(all_images) > 2 else []

        measured_dims = []
        for img in document_images:
            bin_img = binaryize_image(img)
            if not bin_img:
                continue
            dims = measure_document_from_binary(bin_img)
            if dims:
                measured_dims.append(dims)

        if not measured_dims:
            return None

        avg_long = sum(max(w, h) for (w, h) in measured_dims) / len(measured_dims)
        avg_short = sum(min(w, h) for (w, h) in measured_dims) / len(measured_dims)

        estimated_ppi = estimate_ppi_from_dimensions((avg_long, avg_short), chromatic_band_dim_px)
        print(f"[✅] PPI stimato per {folder_path.name}: {estimated_ppi}")
        return estimated_ppi
    except Exception as e:
        print(f"[⚠️] Errore durante la stima PPI in {folder_path}: {e}")
        return None


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


def find_chromatic_band_in_folder(folder_path: Path) -> Path:
    images = [f for f in folder_path.iterdir() if f.is_file() and is_valid_image_file(f)]
    if not images:
        raise FileNotFoundError(f"Nessuna immagine valida trovata in {folder_path}")

    images.sort()
    last_image = images[-1]

    relative_path = folder_path.name
    dest_folder = OUTPUT_TMP_DIR / relative_path
    dest_folder.mkdir(parents=True, exist_ok=True)

    dest_path = dest_folder / f"chromatic_band_{last_image.name}"
    safe_copy(last_image, dest_path)

    return dest_path


def binaryize_image(image_path: Path, threshold: int = 50) -> Path | None:
    if not is_valid_image_file(image_path):
        return None

    img = safe_imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # 🔧 Fix: salvataggio in tmp/cartella_originale
    subfolder = image_path.parent.name
    dest_folder = OUTPUT_TMP_DIR / subfolder
    dest_folder.mkdir(parents=True, exist_ok=True)
    dest_path = dest_folder / image_path.name

    cv2.imwrite(str(dest_path), binary)
    return dest_path



def measure_document_from_binary(binary_image_path: Path) -> tuple[float, float] | None:
    img = safe_imread(binary_image_path)
    if img is None:
        print(f"⚠️ Impossibile leggere {binary_image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"⚠️ Nessun contorno trovato in {binary_image_path}")
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    (cx, cy), (w, h), angle = rect

    long_side_px = max(w, h)
    short_side_px = min(w, h)

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

    scale_factor = CHROMATIC_BAND_MM / chromatic_band_long_side_px

    img_long_side_mm = calculate_mm_from_px(img_long_side_px, scale_factor)
    img_short_side_mm = calculate_mm_from_px(img_short_side_px, scale_factor)

    width_ok = min(img_long_side_mm, img_short_side_mm) <= A4_WIDTH_MM
    height_ok = max(img_long_side_mm, img_short_side_mm) <= A4_HEIGHT_MM

    print(f"Dimensioni stimate: ")
    print(f"{img_long_side_mm:.2f} minore di {A4_HEIGHT_MM}? {'✅' if height_ok else '❌'}")
    print(f"{img_short_side_mm:.2f} minore di {A4_WIDTH_MM}? {'✅' if width_ok else '❌'}")
    print(f"Il file è un A4? {'✅' if (width_ok and height_ok) else '❌'}")

    if width_ok and height_ok:
        return 400
    else:
        return 600
