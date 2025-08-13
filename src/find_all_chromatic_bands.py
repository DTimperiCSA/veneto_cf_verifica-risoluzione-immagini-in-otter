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
TIFFEN_SEGMENTATION_DIR = OUTPUT_TMP_DIR / "chromatic_bands" / "tiffen_segmentation"

HSV_DIR.mkdir(parents=True, exist_ok=True)
TMPL_DIR.mkdir(parents=True, exist_ok=True)
TIFFEN_SEGMENTATION_DIR.mkdir(parents=True, exist_ok=True)

# Percorso template
TEMPLATE_PATH = TEMPLATE_IMG_PATH
TEMPLATE = cv2.imread(str(TEMPLATE_PATH))
if TEMPLATE is None:
    raise FileNotFoundError(f"Template non trovato: {TEMPLATE_PATH}")
TEMPLATE_GRAY = cv2.cvtColor(TEMPLATE, cv2.COLOR_BGR2GRAY)

def safe_match_template(img_gray, template_gray, method=cv2.TM_CCOEFF_NORMED):
    try:
        if img_gray.shape[0] < template_gray.shape[0] or img_gray.shape[1] < template_gray.shape[1]:
            return None
        return cv2.matchTemplate(img_gray, template_gray, method)
    except cv2.error as e:
        if '(-215:Assertion failed)' in str(e):
            print(f"[💥] Errore OpenCV matchTemplate: immagine più piccola del template")
            return None
        else:
            raise

def detect_chromatic_band_hsv_precise_from_template(img, folder_name, img_name):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = safe_match_template(img_gray, TEMPLATE_GRAY)
    if result is None:
        return None

    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val < 0.6:
        return None

    top_left = max_loc
    h_temp, w_temp = TEMPLATE_GRAY.shape
    roi = img[top_left[1]:top_left[1]+h_temp, top_left[0]:top_left[0]+w_temp]

    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return None

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask_non_white = cv2.inRange(roi_gray, 10, 240)
    mask_low_saturation = cv2.inRange(roi_hsv[:, :, 1], 0, 60)
    combined_mask = cv2.bitwise_and(mask_non_white, mask_low_saturation)
    pixels = roi_gray[combined_mask > 0]

    if pixels.size == 0:
        print(f"[⚠️] Nessun pixel valido per media grigio trovato in {img_name}")
        return None

    mean_gray = np.mean(pixels)

    lower_v = max(0, mean_gray - 40)
    upper_v = min(255, mean_gray + 40)

    lower_hsv = np.array([0, 0, lower_v])
    upper_hsv = np.array([180, 60, upper_v])
    mask_hsv = cv2.inRange(roi_hsv, lower_hsv, upper_hsv)

    contours, _ = cv2.findContours(mask_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect = None
    best_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < 300:
            continue
        rect = cv2.minAreaRect(c)
        (_, _), (w, h), _ = rect
        if min(w, h) == 0:
            continue
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio < 1.5:
            continue
        if area > best_area:
            best_area = area
            best_rect = rect

    # Salvataggio ROI originale da template + maschera della segmentazione
    roi_save_dir = HSV_DIR / "precise_from_template" / "roi"
    roi_save_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(roi_save_dir / f"roi_{folder_name}_{img_name}"), roi)

    if best_rect is not None:
        box = cv2.boxPoints(best_rect).astype(int)
        box += np.array([top_left[0], top_left[1]])
        result_img = img.copy()
        cv2.drawContours(result_img, [box], 0, (255, 0, 255), 2)
        return result_img

    print(f"[ℹ️] Nessun rettangolo adatto trovato in {img_name}")
    return None


def detect_tiffen_segmentation(img, folder_name=None, img_name=None):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = safe_match_template(img_gray, TEMPLATE_GRAY)
    if result is None:
        return None

    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    h_temp, w_temp = TEMPLATE_GRAY.shape

    if max_val >= 0.6:
        top_left = max_loc
        roi = img_gray[top_left[1]:top_left[1]+h_temp, top_left[0]:top_left[0]+w_temp]

        if roi.size == 0:
            return None

        mean_gray = np.mean(roi)

        lower_thresh = max(0, mean_gray - 20)
        upper_thresh = min(255, mean_gray + 20)

        mask = cv2.inRange(img_gray, lower_thresh, upper_thresh)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_rect = None
        best_area = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area < 500:
                continue
            rect = cv2.minAreaRect(c)
            (_, _), (w, h), _ = rect
            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
            if aspect_ratio < 2:
                continue
            if area > best_area:
                best_area = area
                best_rect = rect

        # Salvataggio ROI
        roi_save_dir = TIFFEN_SEGMENTATION_DIR / "roi"
        roi_save_dir.mkdir(parents=True, exist_ok=True)
        if folder_name and img_name:
            roi_color = img[top_left[1]:top_left[1]+h_temp, top_left[0]:top_left[0]+w_temp]
            cv2.imwrite(str(roi_save_dir / f"roi_{folder_name}_{img_name}"), roi_color)

        if best_rect is not None:
            box = cv2.boxPoints(best_rect).astype(int)
            result_img = img.copy()
            cv2.drawContours(result_img, [box], 0, (255, 0, 0), 2)
            return result_img

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(img, 0.7, mask_bgr, 0.3, 0)
        return overlay

    else:
        lower_generic = 50
        upper_generic = 120
        mask = cv2.inRange(img_gray, lower_generic, upper_generic)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_rect = None
        best_area = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area < 500:
                continue
            rect = cv2.minAreaRect(c)
            (_, _), (w, h), _ = rect
            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
            if aspect_ratio < 2:
                continue
            if area > best_area:
                best_area = area
                best_rect = rect

        if best_rect is not None:
            box = cv2.boxPoints(best_rect).astype(int)
            result_img = img.copy()
            cv2.drawContours(result_img, [box], 0, (0, 0, 255), 2)
            return result_img

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(img, 0.7, mask_bgr, 0.3, 0)
        return overlay


def detect_chromatic_band_hsv_from_template_region(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = safe_match_template(img_gray, TEMPLATE_GRAY)
    if result is None:
        return None

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

    if best_rect is None:
        return None

    box = cv2.boxPoints(best_rect).astype(int)
    # Ritorna solo la ROI ritagliata per salvare:
    x_min = np.min(box[:, 0])
    y_min = np.min(box[:, 1])
    x_max = np.max(box[:, 0])
    y_max = np.max(box[:, 1])

    # Clip per sicurezza dentro roi:
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(roi.shape[1], x_max)
    y_max = min(roi.shape[0], y_max)

    roi_final = roi[y_min:y_max, x_min:x_max]
    return roi_final


def detect_template_matching_multiscale(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = safe_match_template(img_gray, TEMPLATE_GRAY)
    if result is None:
        return None

    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val < 0.6:
        return None

    top_left = max_loc
    h_temp, w_temp = TEMPLATE_GRAY.shape
    bottom_right = (top_left[0] + w_temp, top_left[1] + h_temp)

    result_img = img.copy()
    cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 2)
    return result_img


def process_image(img_path: Path, folder_name: str):
    img = safe_imread(img_path)
    if img is None:
        print(f"[❌] Impossibile leggere l'immagine {img_path}")
        return

    print(f"[ℹ️] Elaborazione {img_path.name}")

    # Metodo 2 – HSV nella regione TMPL: salva ROI
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

    # Metodo 3 – Segmentazione righello Tiffen basata su template matching + soglia dinamica
    tiffen_seg = detect_tiffen_segmentation(img, folder_name, img_path.name)
    if tiffen_seg is not None:
        out_folder = TIFFEN_SEGMENTATION_DIR
        out_folder.mkdir(parents=True, exist_ok=True)
        out_path = out_folder / f"tiffen_segmentation_{folder_name}_{img_path.name}"
        cv2.imwrite(str(out_path), tiffen_seg)
        print(f"[TIFFEN_SEG ✅] {out_path}")
    else:
        print(f"[TIFFEN_SEG ❌] {img_path.name} – non trovato")

    # Metodo 4 – Template matching + media grigi + segmentazione HSV precisa
    precise_hsv_result = detect_chromatic_band_hsv_precise_from_template(img, folder_name, img_path.name)
    if precise_hsv_result is not None:
        out_folder = HSV_DIR / "precise_from_template"
        out_folder.mkdir(parents=True, exist_ok=True)
        out_path = out_folder / f"precise_hsv_tmpl_{folder_name}_{img_path.name}"
        cv2.imwrite(str(out_path), precise_hsv_result)
        print(f"[PRECISE_HSV_TMPL ✅] {out_path}")
    else:
        print(f"[PRECISE_HSV_TMPL ❌] {img_path.name} – non trovato")


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

        print(f"\n🚀 Elaborazione cartella: {folder.name} immagini: {len(image_files)}")

        process_image(last_image, folder.name)


        folders_processed += 1

    print(f"\n✅ Finito! Cartelle elaborate: {folders_processed}")



if __name__ == "__main__":
    process_all_folders(INPUT_IMAGES_DIR)
