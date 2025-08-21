import cv2
import numpy as np
import time
from pathlib import Path
from PIL import Image

# === CONFIG (adatta ai tuoi percorsi) ===
from src.paths import *

TEMPLATE_IMG_PATH = TEMPLATE_IMG_PATH
INPUT_DIR = INPUT_IMAGES_DIR
OUTPUT_TMP_DIR = OUTPUT_TMP_DIR
CHROMATIC_BAND_MM = 200  # lunghezza reale banda cromatica (esempio)
A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297


# === UTILS ===
def safe_imread(path: Path, retries=3, delay=0.5):
    for attempt in range(retries):
        img = cv2.imread(str(path))
        if img is not None:
            return img
        time.sleep(delay)
    raise IOError(f"Impossibile leggere immagine {path} dopo {retries} tentativi")


def is_valid_image_file(file_path: Path):
    try:
        if not file_path.exists():
            return False, "File non trovato"
        if not file_path.is_file():
            return False, "Il path non è un file"
        img = Image.open(file_path)
        img.verify()
        return True, ""
    except Exception as e:
        return False, str(e)


def find_chromatic_band_in_folder(folder: Path) -> str | None:
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

    for img_path in folder.iterdir():
        if img_path.is_file():
            if img_path.name.lower() == "thumbs.db":
                continue
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]:
                continue

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

    best_rect = None
    best_area = 0
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
            best_rect = rect

    return best_rect


def binaryize_image(image_path: Path, threshold: int = 50) -> Path | None:
    valid, msg = is_valid_image_file(image_path)
    if not valid:
        print(f"⚠️ File non valido: {image_path} | {msg}")
        return None

    img = safe_imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    dest_folder = OUTPUT_TMP_DIR / "binary"
    dest_folder.mkdir(parents=True, exist_ok=True)
    dest_path = dest_folder / image_path.name
    cv2.imwrite(str(dest_path), binary)
    return dest_path


def measure_document_from_binary(binary_image_path: Path):
    img = safe_imread(binary_image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    return rect


def draw_bounding_boxes_with_measures(
    original_path: Path,
    chromatic_rect,
    document_rect,
    output_dir: Path
) -> Path:
    img = safe_imread(original_path).copy()

    def draw_rect(img, rect, color, label):
        (cx, cy), (w, h), angle = rect
        box = cv2.boxPoints(rect)
        box = box.astype(int)  # FIXED np.int0 → astype(int)
        cv2.drawContours(img, [box], 0, color, 2)

        long_side, short_side = (max(w, h), min(w, h))
        x, y = box[0]
        text = f"{label}: {int(long_side)}px x {int(short_side)}px (rot {angle:.1f}°)"
        cv2.putText(
            img, text, (int(x), int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
        )

    if chromatic_rect:
        draw_rect(img, chromatic_rect, (0, 0, 255), "ChromaticBand")
    if document_rect:
        draw_rect(img, document_rect, (0, 255, 0), "Document")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{original_path.stem}_bbox.png"
    cv2.imwrite(str(out_path), img)
    print(f"[💾] Bounding boxes salvati: {out_path}")
    return out_path


# === MAIN ===
def main():
    OUTPUT_TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Ricorsivamente in tutte le cartelle
    INPUT_DIR = Path(r"C:\Users\andre\Desktop\test_all_mask")
    for folder in INPUT_DIR.rglob("*"):
        if not folder.is_dir():
            continue

        print(f"🔎 Cerco banda cromatica in {folder}")
        chromatic_path = find_chromatic_band_in_folder(folder)
        if not chromatic_path:
            print("⚠️ Nessuna banda cromatica trovata in questa cartella.")
            continue

        chromatic_path = Path(chromatic_path)
        print(f"👉 Banda cromatica trovata: {chromatic_path.name}")

        chromatic_rect = measure_chromatic_band_dimension(chromatic_path)
        if not chromatic_rect:
            print("⚠️ Banda cromatica non rilevata con segmentazione HSV.")
            continue

        bin_img = binaryize_image(chromatic_path)
        if not bin_img:
            continue
        document_rect = measure_document_from_binary(bin_img)
        if not document_rect:
            print("⚠️ Documento non rilevato.")
            continue

        draw_bounding_boxes_with_measures(chromatic_path, chromatic_rect, document_rect, OUTPUT_TMP_DIR)


if __name__ == "__main__":
    main()
