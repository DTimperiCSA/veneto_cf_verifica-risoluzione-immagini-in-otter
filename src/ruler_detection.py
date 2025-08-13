import cv2
import numpy as np
from pathlib import Path
from src.paths import *
from src.utils import *
from src.image_processing import *

# === CONFIG ===
MATCH_THRESHOLD = 0.6
CASE1_DIR = OUTPUT_TMP_DIR / "ruler_detection/strength_template_matching"
CASE2_DIR =  OUTPUT_TMP_DIR / "ruler_detection/fallback_hsv"
CASE1_DIR.mkdir(parents=True, exist_ok=True)
CASE2_DIR.mkdir(parents=True, exist_ok=True)

# === FUNZIONI ===
def preprocess_for_template_matching(img_gray):
    """Equalizzazione e blur per migliorare il template matching."""
    img_eq = cv2.equalizeHist(img_gray)
    img_blur = cv2.GaussianBlur(img_eq, (3, 3), 0)
    return img_blur

def draw_bounding_box(img, top_left, w, h, color=(0, 255, 0), thickness=2):
    """Disegna un rettangolo sull'immagine."""
    cv2.rectangle(img, top_left, (top_left[0] + w, top_left[1] + h), color, thickness)

def measure_and_draw(img, mask, color=(0, 0, 255)):
    """Trova il bounding box dal mask, lo disegna e stampa le dimensioni."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("⚠ Nessun contorno trovato")
        return img
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    print(f"📏 Dimensioni rilevate: {w}px x {h}px")
    return img

def process_image(img_path: Path, template_gray):
    img = cv2.imread(str(img_path))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Preprocess per matching
    img_gray_proc = preprocess_for_template_matching(img_gray)
    template_gray_proc = preprocess_for_template_matching(template_gray)

    # Template matching (non scalato)
    result = cv2.matchTemplate(img_gray_proc, template_gray_proc, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    th, tw = template_gray_proc.shape[:2]

    output_filename = f"{img_path.parent.name}_{img_path.name}"

    if max_val >= MATCH_THRESHOLD:
        # === CASO 1: HSV dinamico derivato dalla ROI, applicato a tutta l'immagine ===
        roi = img[max_loc[1]:max_loc[1]+th, max_loc[0]:max_loc[0]+tw]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Calcolo soglie dinamiche dalla ROI
        mean_hsv = np.mean(hsv_roi.reshape(-1, 3), axis=0)
        lower = np.array([max(mean_hsv[0]-10, 0), max(mean_hsv[1]-30, 0), max(mean_hsv[2]-30, 0)], dtype=np.uint8)
        upper = np.array([min(mean_hsv[0]+10, 179), min(mean_hsv[1]+30, 255), min(mean_hsv[2]+30, 255)], dtype=np.uint8)

        # Applico soglie a tutta l'immagine
        hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_full, lower, upper)

        img_out = measure_and_draw(img.copy(), mask, color=(0, 255, 0))
        os.makedirs(CASE1_DIR, exist_ok=True)


        out_path = CASE1_DIR / output_filename
        cv2.imwrite(str(out_path), img_out)
        print(f"[✅ Caso1] {output_filename} match={max_val:.2f} salvato in {out_path}")

    else:
        # === CASO 2: HSV statico su tutta l'immagine ===
        hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 30], dtype=np.uint8)   # soglie statiche per grigio
        upper = np.array([180, 40, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv_full, lower, upper)

        img_out = measure_and_draw(img.copy(), mask, color=(255, 0, 0))
        os.makedirs(CASE2_DIR, exist_ok=True)

        out_path = CASE2_DIR / output_filename
        cv2.imwrite(str(out_path), img_out)
        print(f"[⚠ Caso2] {output_filename} match={max_val:.2f} salvato in {out_path}")

def process_folder(images_dir: Path, template_path: Path):
    # Carico template
    template_gray = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if template_gray is None:
        raise FileNotFoundError(f"Template non trovato: {template_path}")

    # Prendo tutte le sottocartelle (inclusa la radice)
    all_dirs = {p.parent for p in images_dir.rglob("*") if p.is_file()} | {images_dir}

    for folder in sorted(all_dirs):
        # Filtra immagini valide nella cartella
        image_files = [
            f for f in folder.iterdir()
            if f.is_file() and f.name.lower() != "thumbs.db" and is_valid_image_file(f)[0]
        ]

        if not image_files:
            continue

        # Ordina alfabeticamente e prendo l’ultima
        image_files = sorted(image_files)
        last_image = image_files[-1]

        print(f"📌 Cartella: {folder.name} → Ultima immagine: {last_image.name}")

        process_image(last_image, template_gray)

# === ESEMPIO DI ESECUZIONE ===
if __name__ == "__main__":
    process_folder(INPUT_IMAGES_DIR, TEMPLATE_IMG_PATH)
