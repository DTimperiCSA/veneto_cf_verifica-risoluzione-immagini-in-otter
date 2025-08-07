import cv2
import numpy as np
from pathlib import Path
from src.utils import safe_imread, is_valid_image_file
from src.paths import INPUT_IMAGES_DIR, OUTPUT_TMP_DIR

TEMPLATE_PATH = Path("images/input/assets/tiffen_template.jpg")
OUTPUT_DIR = OUTPUT_TMP_DIR / "ruler_detection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE = cv2.imread(str(TEMPLATE_PATH))
if TEMPLATE is None:
    raise FileNotFoundError(f"Template non trovato: {TEMPLATE_PATH}")
TEMPLATE_GRAY = cv2.cvtColor(TEMPLATE, cv2.COLOR_BGR2GRAY)

def match_template(img_gray):
    result = cv2.matchTemplate(img_gray, TEMPLATE_GRAY, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_val, max_loc

def get_average_gray_excluding_patches(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    low_sat_mask = cv2.inRange(hsv, (0, 0, 30), (180, 40, 255))
    gray_pixels = roi_bgr[low_sat_mask > 0]
    if gray_pixels.size == 0:
        return None
    avg_gray = np.mean(gray_pixels, axis=0)
    return avg_gray.astype(np.uint8)

def segment_by_gray(img_bgr, target_bgr, threshold=30):
    diff = np.linalg.norm(img_bgr.astype(float) - target_bgr.astype(float), axis=2)
    mask = (diff < threshold).astype(np.uint8) * 255
    return mask

def process_image(img_path: Path):
    img = safe_imread(img_path)
    if img is None:
        print(f"[❌] Impossibile leggere: {img_path.name}")
        return

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    max_val, top_left = match_template(img_gray)

    if max_val < 0.6:
        print(f"[❌] Template match troppo debole ({max_val:.2f}) in {img_path.name}")
        return

    h_temp, w_temp = TEMPLATE_GRAY.shape
    roi = img[top_left[1]:top_left[1]+h_temp, top_left[0]:top_left[0]+w_temp]
    if roi.size == 0:
        print(f"[❌] ROI vuota per {img_path.name}")
        return

    avg_gray = get_average_gray_excluding_patches(roi)
    if avg_gray is None:
        print(f"[❌] Nessun pixel grigio trovato in ROI di {img_path.name}")
        return

    mask = segment_by_gray(img, avg_gray, threshold=30)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"[❌] Nessun contorno trovato in {img_path.name}")
        return

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect).astype(int)

    img_out = img.copy()
    cv2.drawContours(img_out, [box], 0, (0, 255, 255), 2)

    width_px, height_px = rect[1]
    print(f"[✅] {img_path.name} – Dimensioni righello: {width_px:.1f}px × {height_px:.1f}px")

    out_path = OUTPUT_DIR / f"ruler_box_{img_path.name}"
    cv2.imwrite(str(out_path), img_out)

def process_all_folders(root: Path):
    for folder in root.rglob("*"):
        if not folder.is_dir():
            continue
        image_files = [
            f for f in folder.iterdir()
            if f.is_file() and f.name.lower() != "thumbs.db" and is_valid_image_file(f)[0]
        ]
        if not image_files:
            continue
        last_image = sorted(image_files)[-1]
        print(f"\n📁 Cartella: {folder.name}")
        try:
            process_image(last_image)
        except Exception as e:
            print(f"[💥] Errore su {last_image.name}: {e}")

if __name__ == "__main__":
    print(f"📂 Input: {INPUT_IMAGES_DIR}")
    process_all_folders(INPUT_IMAGES_DIR)
