import cv2
import numpy as np
from pathlib import Path
from src.utils import *
from src.paths import *
from src.config import *
from src.estimate_ppi_from_ruler import *
from src.image_processing import *

# Cartelle di output
HSV_DIR = OUTPUT_TMP_DIR / "chromatic_bands" / "hsv"
TMPL_DIR = OUTPUT_TMP_DIR / "chromatic_bands" / "template_matching"
HSV_DIR.mkdir(parents=True, exist_ok=True)
TMPL_DIR.mkdir(parents=True, exist_ok=True)

# Template path
TEMPLATE_PATH = Path("images/input/assets/tiffen_template.jpg")
TEMPLATE = cv2.imread(str(TEMPLATE_PATH))
if TEMPLATE is None:
    raise FileNotFoundError(f"Template non trovato: {TEMPLATE_PATH}")
TEMPLATE_GRAY = cv2.cvtColor(TEMPLATE, cv2.COLOR_BGR2GRAY)

def detect_chromatic_band_hsv_adaptive(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Calcola intervallo HSV dinamico
    h_mean, s_mean, v_mean = np.mean(h), np.mean(s), np.mean(v)
    lower = np.array([max(0, h_mean - 20), max(0, s_mean - 30), max(0, v_mean - 40)], dtype=np.uint8)
    upper = np.array([min(180, h_mean + 20), min(255, s_mean + 30), min(255, v_mean + 40)], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect = None
    best_area = 0

    for c in contours:
        if cv2.arcLength(c, True) < 200:
            continue
        rect = cv2.minAreaRect(c)
        (_, _), (w, h), _ = rect
        area = w * h
        if area < 1000:
            continue
        if max(w, h) / min(w, h) < 3:
            continue
        if area > best_area:
            best_area = area
            best_rect = rect

    if best_rect is not None:
        result_img = img.copy()
        box = cv2.boxPoints(best_rect).astype(int)
        cv2.drawContours(result_img, [box], 0, (255, 0, 255), 2)
        return result_img
    return None

def detect_chromatic_band_hsv_from_template_region(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(img_gray, TEMPLATE_GRAY, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < 0.6:
        return None

    top_left = max_loc
    h_temp, w_temp = TEMPLATE_GRAY.shape
    roi = img[top_left[1]:top_left[1]+h_temp, top_left[0]:top_left[0]+w_temp]

    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 40])
    upper = np.array([180, 50, 100])
    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect = None
    best_area = 0

    for c in contours:
        if cv2.arcLength(c, True) < 100:
            continue
        rect = cv2.minAreaRect(c)
        (_, _), (w, h), _ = rect
        area = w * h
        if area < 300:
            continue
        if max(w, h) / min(w, h) < 3:
            continue
        if area > best_area:
            best_area = area
            best_rect = rect

    if best_rect is not None:
        box = cv2.boxPoints(best_rect).astype(int)
        box += np.array([top_left[0], top_left[1]])
        result_img = img.copy()
        cv2.drawContours(result_img, [box], 0, (0, 165, 255), 2)
        return result_img
    return None

def detect_template_matching_multiscale(img, scales=[0.5, 0.75, 1.0, 1.25, 1.5], threshold=0.6):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    best_max_val = -1
    best_top_left = None
    best_scale = None
    best_template_resized = None

    for scale in scales:
        # Ridimensiona il template
        w = int(TEMPLATE_GRAY.shape[1] * scale)
        h = int(TEMPLATE_GRAY.shape[0] * scale)
        if w < 10 or h < 10:
            continue
        template_resized = cv2.resize(TEMPLATE_GRAY, (w, h), interpolation=cv2.INTER_AREA)

        result = cv2.matchTemplate(img_gray, template_resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_max_val:
            best_max_val = max_val
            best_top_left = max_loc
            best_scale = scale
            best_template_resized = template_resized

    if best_max_val < threshold or best_top_left is None:
        return None

    top_left = best_top_left
    h, w = best_template_resized.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)

    matched_img = img.copy()
    cv2.rectangle(matched_img, top_left, bottom_right, (0, 255, 0), 2)
    return matched_img


def process_image(img_path: Path, folder_name: str):
    img = safe_imread(img_path)
    if img is None:
        print(f"[❌] Impossibile leggere l'immagine {img_path}")
        return

    # Metodo 1 – HSV adattivo
    hsv_adaptive = detect_chromatic_band_hsv_adaptive(img)
    if hsv_adaptive is not None:
        out_folder = HSV_DIR / "adaptive"
        out_folder.mkdir(parents=True, exist_ok=True)
        out_path = out_folder / f"hsv_adaptive_{folder_name}_{img_path.name}"
        cv2.imwrite(str(out_path), hsv_adaptive)
        print(f"[HSV_ADAPT ✅] {out_path}")
    else:
        print(f"[HSV_ADAPT ❌] {img_path.name} – non trovato")

    # Metodo 2 – HSV nella regione TMPL
    hsv_tmpl_roi = detect_chromatic_band_hsv_from_template_region(img)
    if hsv_tmpl_roi is not None:
        out_folder = HSV_DIR / "tmpl_roi"
        out_folder.mkdir(parents=True, exist_ok=True)
        out_path = out_folder / f"hsv_tmpl_roi_{folder_name}_{img_path.name}"
        cv2.imwrite(str(out_path), hsv_tmpl_roi)
        print(f"[HSV_TMPL_ROI ✅] {out_path}")
    else:
        print(f"[HSV_TMPL_ROI ❌] {img_path.name} – non trovato")

    # Template matching classico
    tmpl_result = detect_template_matching_multiscale(img)
    if tmpl_result is not None:
        out_folder = TMPL_DIR
        out_folder.mkdir(parents=True, exist_ok=True)
        out_path = out_folder / f"chromatic_band_{folder_name}_{img_path.name}"
        cv2.imwrite(str(out_path), tmpl_result)
        print(f"[TMPL ✅] {out_path}")
    else:
        print(f"[TMPL ❌] {img_path.name} – non trovato")

def process_all_folders(root: Path):
    folders_processed = 0

    for folder in root.rglob("*"):
        if not folder.is_dir():
            continue

        image_files = [
            f for f in folder.iterdir()
            if f.is_file() and f.name.lower() != "thumbs.db" and is_valid_image_file(f)[0]
        ]

        if not image_files:
            continue

        image_files = sorted(image_files)
        last_image = image_files[-1]

        print(f"\n📁 Elaborazione cartella: {folder}")
        try:
            process_image(last_image, folder.name)
        except Exception as e:
            print(f"[💥] Errore su {last_image.name}: {e}")

        folders_processed += 1

    print(f"\n✅ Cartelle processate: {folders_processed}")

if __name__ == "__main__":
    print(f"📂 Directory di input: {INPUT_IMAGES_DIR}")
    process_all_folders(INPUT_IMAGES_DIR)
