import os
import traceback
import numpy as np
import csv
import cv2
import shutil
import time
import torch

from PIL import Image
from pathlib import Path
from typing import Tuple

from src.paths import *
from src.config import *
from src.image_utils import *
from src.segmentation.unet import UNet

COL_KP = (255, 0, 0)

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL = None  # global cached model

def load_unet(model_path: Path, model_class):
    global MODEL
    if MODEL is None:  # lazy load only once
        model = model_class().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        MODEL = model
    return MODEL


def measure_chromatic_band_dimension(path_input: Path, input_size=(256, 256)):
    """Return (long_side, short_side) in pixels using UNet segmentation and save visualization."""
    global MODEL

    if MODEL is None:
        # You can put SAVE_PATH in config.py
        MODEL = load_unet(SAVE_PATH, UNet)

    # --- Load image ---
    img = cv2.imread(str(path_input))
    if img is None:
        print(f"⚠️ Could not read {path_input}")
        return None
    orig_h, orig_w = img.shape[:2]

    # --- Preprocess ---
    img_resized = cv2.resize(img, input_size)
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(DEVICE)

    # --- Inference ---
    with torch.no_grad():
        pred = MODEL(img_tensor)
        pred = torch.sigmoid(pred)
        mask = (pred > 0.5).float().cpu().numpy()[0, 0]

    mask = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # --- Get bounding box ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"⚠️ No Tiffen ruler found in {path_input}.")
        return None

    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    (cx, cy), (w, h), angle = rect
    box = cv2.boxPoints(rect).astype(int)

    # --- Save visualization ---
    debug = img.copy()
    # overlay mask
    colored_mask = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(debug, 0.7, colored_mask, 0.3, 0)
    # draw bounding box
    cv2.drawContours(overlay, [box], 0, (0, 255, 0), 2)

    out_dir = OUTPUT_TMP_DIR / "segmentation_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_TMP_DIR / f"{path_input.stem}_segmented.png"
    cv2.imwrite(str(out_path), overlay)

    print(f"💾 Saved segmentation result: {out_path}")

    return (max(w, h), min(w, h))

    
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

def largest_component_mask(mask):
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    largest = max(cnts, key=cv2.contourArea)
    big_mask = np.zeros_like(mask)
    cv2.drawContours(big_mask, [largest], -1, 255, thickness=-1)
    return big_mask, largest

def method_keypoint_orb(img, template):    
    try:
        orb = cv2.ORB_create(3000)
        kp1, des1 = orb.detectAndCompute(template, None)
        kp2, des2 = orb.detectAndCompute(img, None)
        if des1 is None or des2 is None:
            return None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:3000]  # cap
        if len(matches) < 8:
            return None
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, inliers_mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if M is None:
            return None
        h, w = template.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 2)
        transformed = cv2.transform(np.array([corners]), M)[0].astype(int)
        # clamp coords
        transformed[:,0] = np.clip(transformed[:,0], 0, img.shape[1]-1)
        transformed[:,1] = np.clip(transformed[:,1], 0, img.shape[0]-1)
        mask_poly = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_poly, [transformed], 255)
        # keep largest connected comp (to remove tiny noisy polygons)
        big, largest = largest_component_mask(mask_poly)
        if big is not None:
            mask_poly = big
        overlay = img.copy()
        cv2.drawContours(overlay, [transformed], -1, COL_KP, 3)
        # roi bounding rect from transformed poly
        minx = np.min(transformed[:,0]); maxx = np.max(transformed[:,0])
        miny = np.min(transformed[:,1]); maxy = np.max(transformed[:,1])
        if maxx<=minx or maxy<=miny:
            roi = None
        else:
            roi = img[miny:maxy+1, minx:maxx+1].copy()
        meta = {"matches": len(matches)}
        return {"method": "keypoint", "mask": mask_poly, "overlay": overlay, "roi": roi, "meta": meta}
    except Exception:
        return None
    







